# -*- coding: utf-8 -*-
"""
train_student_mnv3_kd_ablation_full_fixed.py

Ablation + finetune script that is CHECKPOINT-COMPATIBLE with train_student_mnv3_kd_fixed.py.

Fixes / guarantees:
- Resume from student_mnv3_kd_fixed_best.pth will NOT error:
  * MobileNetV3 backbone.proj is lazy -> we run a dummy forward to instantiate it BEFORE strict loading
  * kd_adapter is Identity (matches your fixed student ckpt)
- AMP GradScaler: call scaler.update() ONLY if scaler.step() happened (avoid "No inf checks were recorded")
- A0 (no KD) will NOT load teacher (speed up)
- Optional KD schedule (warmup/ramp) + EXP presets (A0/A1/A2/A3/A4)

Env vars (optional):
  EXP=A0|A1|A2|A3|A4
  RESUME_STUDENT=student_mnv3_kd_fixed_best.pth
  TEACHER_CKPT=teacher_paper_best.pth
  CSV=results_A0.csv
  SAVE_PREFIX=student_A0
  LR=5e-5
  EPOCHS=4
  KD_WARMUP_EPOCHS=0
  KD_RAMP_EPOCHS=0
  BATCH_SIZE=4
  GRAD_ACCUM=2
"""

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128,garbage_collection_threshold:0.8")

import sys, gc, math, time, csv
from typing import List, Tuple, Optional

import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from torch.utils.checkpoint import checkpoint
from tqdm import tqdm

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.utils.data_classes import Quaternion, LidarPointCloud
from nuscenes.utils.geometry_utils import view_points


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_DIR = os.path.dirname(SCRIPT_DIR)


def _first_existing_path(candidates):
    for p in candidates:
        if p and os.path.exists(p):
            return p
    return candidates[0]


# ============================================================
# 0) Config (base, can be overridden by ENV)
# ============================================================
DATA_ROOT = os.getenv("DATA_ROOT", _first_existing_path([
    os.path.join(WORKSPACE_DIR, "datasets", "nuscenes"),
    os.path.join(SCRIPT_DIR, "..", "datasets", "nuscenes"),
    "/root/autodl-tmp/BEVnext/datasets/nuscenes",
]))
NUSC_VERSION = os.getenv("NUSC_VERSION", "v1.0-trainval")

EXP = os.getenv("EXP", "A0")  # A0..A4
CSV_PATH = os.getenv("CSV", f"results_{EXP}.csv")
SAVE_PREFIX = os.getenv("SAVE_PREFIX", f"student_{EXP}")

RESUME_STUDENT = os.getenv("RESUME_STUDENT", "")  # path to *.pth
TEACHER_CKPT = os.getenv("TEACHER_CKPT", _first_existing_path([
    os.path.join(SCRIPT_DIR, "模型文件", "teacher_paper_best.pth"),
    os.path.join(SCRIPT_DIR, "teacher_paper_best.pth"),
    "teacher_paper_best.pth",
]))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CAM_CHANNELS = ["CAM_FRONT","CAM_FRONT_LEFT","CAM_FRONT_RIGHT","CAM_BACK","CAM_BACK_LEFT","CAM_BACK_RIGHT"]
STUDENT_CAM = "CAM_FRONT"

# teacher fixed input (match teacher training)
TCH_W, TCH_H = 1280, 720
# student input
IMG_W, IMG_H = 960, 384
FEAT_W, FEAT_H = IMG_W // 8, IMG_H // 8

NUM_DEPTH_BINS = 32
DEPTH_MIN, DEPTH_MAX = 1.0, 50.0

# BEV
BEV_X_MIN, BEV_X_MAX = 0.0, 50.0
BEV_Y_MIN, BEV_Y_MAX = -25.0, 25.0
BEV_RES = 0.25
BEV_H = int((BEV_X_MAX - BEV_X_MIN) / BEV_RES)
BEV_W = int((BEV_Y_MAX - BEV_Y_MIN) / BEV_RES)

# Student explicit Z voxel
Z_MIN, Z_MAX = -5.0, 3.0
Z_BINS = 16

OCC_DILATE_CELLS = int(os.getenv("OCC_DILATE_CELLS", "1"))
OCC_GT_MODE = os.getenv("OCC_GT_MODE", "rotated").strip().lower()

# training
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "4"))
GRAD_ACCUM = int(os.getenv("GRAD_ACCUM", "2"))
EPOCHS = int(os.getenv("EPOCHS", "4"))
LR = float(os.getenv("LR", "5e-5"))
WEIGHT_DECAY = float(os.getenv("WEIGHT_DECAY", "1e-4"))

# GT weights
W_OCC, W_HM, W_REG, W_DEP = 1.0, 0.3, 0.3, 0.2
USE_DEPTH_SUP = True

# KD base weights (will be set by EXP)
W_KD_OCC, W_KD_HM, W_KD_REG, W_KD_FEAT, W_KD_DEP = 0.0, 0.0, 0.0, 0.0, 0.0
KD_T = 2.0
KD_POS_THR = float(os.getenv("KD_POS_THR", "0.70"))
KD_NEG_THR = float(os.getenv("KD_NEG_THR", "0.30"))
KD_REQUIRE_GT_AGREEMENT = os.getenv("KD_REQUIRE_GT_AGREEMENT", "1") == "1"
KD_FEAT_POS_THR = float(os.getenv("KD_FEAT_POS_THR", "0.55"))
KD_DEPTH_CONF_THR = float(os.getenv("KD_DEPTH_CONF_THR", "0.55"))
KD_MASK_EMPTY_OK = os.getenv("KD_MASK_EMPTY_OK", "1") == "1"

# KD schedule
KD_WARMUP_EPOCHS = int(os.getenv("KD_WARMUP_EPOCHS", "0"))
KD_RAMP_EPOCHS = int(os.getenv("KD_RAMP_EPOCHS", "0"))

GAUSS_BETA = 4.0
FOCAL_ALPHA = 2.0

AMP = True
GRAD_CLIP_NORM = 5.0
SKIP_NAN_BATCH = True

USE_CHECKPOINT_STUDENT = False
USE_SPLAT_FP32 = True

NUM_WORKERS = int(os.getenv("NUM_WORKERS", "4"))
PIN_MEMORY = True
PERSISTENT_WORKERS = True if NUM_WORKERS > 0 else False
PREFETCH_FACTOR = 4 if NUM_WORKERS > 0 else None
TRAIN_MAX_SAMPLES = int(os.getenv("TRAIN_MAX_SAMPLES", "0"))
VAL_MAX_SAMPLES = int(os.getenv("VAL_MAX_SAMPLES", "0"))

SHOW_TRAIN_PBAR = True
SHOW_VAL_PBAR = True
PBAR_MININTERVAL = 0.2
PBAR_LEAVE = False

THR_LIST = [0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.50,0.60]

POS_WEIGHT_INIT = 40.0
AUTO_EST_POS_WEIGHT = True
ESTIMATE_SAMPLES = 200

SEED = 0
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.benchmark = True
cv2.setNumThreads(0)


# ============================================================
# 0.1) EXP presets
# ============================================================
def apply_exp_preset(exp: str):
    """
    A0: GT only (finetune) -> all KD weights 0, teacher not needed
    A1: KD occ/hm/reg
    A2: KD feat only
    A3: KD depth only
    A4: KD all
    A5: confidence-masked, lighter KD all
    A6: confidence-masked occ + feat + depth
    """
    global W_KD_OCC, W_KD_HM, W_KD_REG, W_KD_FEAT, W_KD_DEP

    if exp == "A0":
        W_KD_OCC, W_KD_HM, W_KD_REG, W_KD_FEAT, W_KD_DEP = 0,0,0,0,0
    elif exp == "A1":
        W_KD_OCC, W_KD_HM, W_KD_REG, W_KD_FEAT, W_KD_DEP = 0.8,0.2,0.2,0,0
    elif exp == "A2":
        W_KD_OCC, W_KD_HM, W_KD_REG, W_KD_FEAT, W_KD_DEP = 0,0,0,0.6,0
    elif exp == "A3":
        W_KD_OCC, W_KD_HM, W_KD_REG, W_KD_FEAT, W_KD_DEP = 0,0,0,0,0.2
    elif exp == "A4":
        W_KD_OCC, W_KD_HM, W_KD_REG, W_KD_FEAT, W_KD_DEP = 0.8,0.2,0.2,0.6,0.2
    elif exp == "A5":
        W_KD_OCC, W_KD_HM, W_KD_REG, W_KD_FEAT, W_KD_DEP = 0.20,0.05,0.05,0.15,0.03
    elif exp == "A6":
        W_KD_OCC, W_KD_HM, W_KD_REG, W_KD_FEAT, W_KD_DEP = 0.20,0.00,0.00,0.15,0.03
    else:
        # fallback: GT only
        W_KD_OCC, W_KD_HM, W_KD_REG, W_KD_FEAT, W_KD_DEP = 0,0,0,0,0

apply_exp_preset(EXP)
if EXP in {"A5", "A6"}:
    if "KD_WARMUP_EPOCHS" not in os.environ:
        KD_WARMUP_EPOCHS = 2
    if "KD_RAMP_EPOCHS" not in os.environ:
        KD_RAMP_EPOCHS = 2


def need_teacher() -> bool:
    return (W_KD_OCC + W_KD_HM + W_KD_REG + W_KD_FEAT + W_KD_DEP) > 0.0


# ============================================================
# 1) Losses / KD
# ============================================================
class DiceLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps
    def forward(self, logits, targets):
        targets = targets.float()
        prob = torch.sigmoid(logits).flatten(1)
        targets = targets.flatten(1)
        inter = (prob * targets).sum(dim=1)
        union = prob.sum(dim=1) + targets.sum(dim=1)
        dice = (2 * inter + self.eps) / (union + self.eps)
        return (1 - dice).mean()

class BCEDiceLoss(nn.Module):
    def __init__(self, pos_weight: float = 40.0, w_bce: float = 1.0, w_dice: float = 1.0):
        super().__init__()
        self.w_bce = w_bce
        self.w_dice = w_dice
        self.dice = DiceLoss()
        self.register_buffer("pos_weight", torch.tensor([float(pos_weight)], dtype=torch.float32))
    def set_pos_weight(self, v: float):
        v = float(max(1.0, v))
        self.pos_weight.data = torch.tensor([v], dtype=self.pos_weight.dtype, device=self.pos_weight.device)
    def forward(self, logits, targets):
        targets = targets.float()
        bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=self.pos_weight, reduction="mean")
        dice = self.dice(logits, targets)
        return self.w_bce * bce + self.w_dice * dice

def gaussian_focal_loss(pred, gt, alpha=2.0, beta=4.0, eps=1e-6):
    pred = torch.nan_to_num(pred, nan=0.0, posinf=1.0, neginf=0.0).clamp(eps, 1 - eps)
    gt = torch.nan_to_num(gt, nan=0.0, posinf=1.0, neginf=0.0)
    pos = (gt == 1).float()
    neg = (gt < 1).float()
    pos_loss = -torch.log(pred) * ((1 - pred) ** alpha) * pos
    neg_loss = -torch.log(1 - pred) * (pred ** alpha) * ((1 - gt) ** beta) * neg
    num_pos = pos.sum().clamp(min=1.0)
    return (pos_loss.sum() + neg_loss.sum()) / num_pos

def smooth_l1_reg_loss(pred, target, mask, beta=1.0):
    mask = mask.float()
    pred = torch.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
    target = torch.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)
    diff = F.smooth_l1_loss(pred, target, reduction="none", beta=beta)
    diff = diff * mask
    denom = mask.sum().clamp(min=1.0)
    return diff.sum() / denom

def safe_depth_ce_loss(depth_logits: torch.Tensor, depth_gt: torch.Tensor, ignore_index: int = -1):
    depth_gt = depth_gt.long()
    valid = (depth_gt != ignore_index)
    if valid.sum().item() == 0:
        return depth_logits.new_tensor(0.0)
    B, D, Hf, Wf = depth_logits.shape
    logits = depth_logits.permute(0,2,3,1).reshape(-1, D)
    gt = depth_gt.reshape(-1)
    v = valid.reshape(-1)
    return F.cross_entropy(logits[v], gt[v], reduction="mean")

def kd_bce_with_logits(student_logits, teacher_logits, T: float = 2.0):
    s = student_logits / T
    with torch.no_grad():
        t_prob = torch.sigmoid(teacher_logits / T)
    return F.binary_cross_entropy_with_logits(s, t_prob, reduction="mean") * (T * T)

def kd_mse(student_feat, teacher_feat):
    return F.mse_loss(student_feat, teacher_feat)


def masked_mean(value_map: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    if mask is None:
        return value_map.mean()
    mask = mask.to(dtype=value_map.dtype)
    denom = mask.sum().clamp(min=1.0)
    if denom.item() <= 0 and not KD_MASK_EMPTY_OK:
        return value_map.new_tensor(0.0)
    return (value_map * mask).sum() / denom


def build_occ_kd_mask(teacher_logits: torch.Tensor, occ_gt: Optional[torch.Tensor] = None, T: float = 2.0) -> torch.Tensor:
    with torch.no_grad():
        t_prob = torch.sigmoid(teacher_logits / T)
        conf_mask = (t_prob >= KD_POS_THR) | (t_prob <= KD_NEG_THR)
        if occ_gt is not None and KD_REQUIRE_GT_AGREEMENT:
            gt_bin = (occ_gt >= 0.5)
            teacher_bin = (t_prob >= 0.5)
            conf_mask = conf_mask & (teacher_bin == gt_bin)
    return conf_mask.float()


def kd_bce_with_logits_masked(student_logits, teacher_logits, mask: Optional[torch.Tensor], T: float = 2.0):
    s = student_logits / T
    with torch.no_grad():
        t_prob = torch.sigmoid(teacher_logits / T)
    loss_map = F.binary_cross_entropy_with_logits(s, t_prob, reduction="none") * (T * T)
    return masked_mean(loss_map, mask)


def kd_feat_occ_mask(teacher_occ_logits: Optional[torch.Tensor], feat: torch.Tensor) -> Optional[torch.Tensor]:
    if teacher_occ_logits is None:
        return None
    with torch.no_grad():
        occ_prob = torch.sigmoid(teacher_occ_logits)
        occ_mask = (occ_prob >= KD_FEAT_POS_THR).float()
    if occ_mask.shape[-2:] != feat.shape[-2:]:
        occ_mask = F.interpolate(occ_mask, size=feat.shape[-2:], mode="nearest")
    return occ_mask


def kd_mse_masked(student_feat, teacher_feat, mask: Optional[torch.Tensor]):
    loss_map = (student_feat - teacher_feat) ** 2
    if mask is not None and mask.dim() == 4 and mask.shape[1] == 1:
        mask = mask.expand(-1, loss_map.shape[1], -1, -1)
    return masked_mean(loss_map, mask)


def kd_depth_mask(depth_logits_teacher: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        prob = F.softmax(depth_logits_teacher, dim=1)
        conf = prob.max(dim=1, keepdim=True).values
    return (conf >= KD_DEPTH_CONF_THR).float()


# ============================================================
# 2) Dataset / targets
# ============================================================
def _yaw_from_quat(q: Quaternion) -> float:
    R = q.rotation_matrix
    return float(math.atan2(R[1,0], R[0,0]))

def gaussian_radius(w: float, h: float, min_overlap: float = 0.7) -> int:
    a1 = 1; b1 = (h+w); c1 = w*h*(1-min_overlap)/(1+min_overlap)
    r1 = (b1 + math.sqrt(max(0.0, b1*b1 - 4*a1*c1))) / 2
    a2 = 4; b2 = 2*(h+w); c2 = (1-min_overlap)*w*h
    r2 = (b2 + math.sqrt(max(0.0, b2*b2 - 4*a2*c2))) / 2
    a3 = 4*min_overlap; b3 = -2*min_overlap*(h+w); c3 = (min_overlap-1)*w*h
    r3 = (b3 + math.sqrt(max(0.0, b3*b3 - 4*a3*c3))) / 2
    return max(0, int(min(r1,r2,r3)))

def draw_gaussian(heatmap: np.ndarray, center: Tuple[int,int], radius: int):
    x, y = center
    H, W = heatmap.shape
    d = 2*radius + 1
    sigma = d / 6.0
    xs = np.arange(d) - radius
    ys = np.arange(d) - radius
    yy, xx = np.meshgrid(ys, xs)
    g = np.exp(-(xx**2 + yy**2) / (2*sigma*sigma)).astype(np.float32)

    x0 = max(0, x-radius); x1 = min(H, x+radius+1)
    y0 = max(0, y-radius); y1 = min(W, y+radius+1)
    gx0 = x0 - (x-radius); gy0 = y0 - (y-radius)
    gx1 = gx0 + (x1-x0);  gy1 = gy0 + (y1-y0)

    patch = heatmap[x0:x1, y0:y1]
    heatmap[x0:x1, y0:y1] = np.maximum(patch, g[gx0:gx1, gy0:gy1])

def dilate_occ(occ: np.ndarray, cells: int) -> np.ndarray:
    if cells <= 0: return occ
    k = 2*cells + 1
    kernel = np.ones((k,k), dtype=np.uint8)
    occ_u8 = (occ > 0.5).astype(np.uint8) * 255
    occ_d = cv2.dilate(occ_u8, kernel, iterations=1)
    return (occ_d > 0).astype(np.float32)


def fill_rotated_box_on_occ(occ: np.ndarray, cx: float, cy: float, length: float, width: float, yaw: float):
    local = np.array([
        [ length / 2.0,  width / 2.0],
        [ length / 2.0, -width / 2.0],
        [-length / 2.0, -width / 2.0],
        [-length / 2.0,  width / 2.0],
    ], dtype=np.float32)
    c, s = math.cos(yaw), math.sin(yaw)
    rot = np.array([[c, -s], [s, c]], dtype=np.float32)
    corners = local @ rot.T
    corners[:, 0] += cx
    corners[:, 1] += cy

    rows = (corners[:, 0] - BEV_X_MIN) / BEV_RES
    cols = (corners[:, 1] - BEV_Y_MIN) / BEV_RES
    poly = np.stack([cols, rows], axis=1).astype(np.int32)
    cv2.fillConvexPoly(occ, poly, 1.0)

class DistillMonoNuScenesDataset(Dataset):
    """
    Outputs:
      teacher: imgs6_tch, Ks6_tch, R6, t6   (6 cams, 1280x720)
      student: img1_stu, K1_stu, R1, t1    (front cam, 960x384)
      GT: occ, hm, reg, m
      depth_gt: (FEAT_H, FEAT_W)
    """
    def __init__(self, dataroot, split="train",
                 transform_teacher=None, transform_student=None,
                 verbose=False, max_samples=None, use_depth_sup=True):
        self.dataroot = dataroot
        self.nusc = NuScenes(version=NUSC_VERSION, dataroot=dataroot, verbose=verbose)
        self.transform_teacher = transform_teacher
        self.transform_student = transform_student
        self.use_depth_sup = use_depth_sup

        split_scenes = set(create_splits_scenes()[split])
        self.sample_tokens: List[str] = []
        self.sample_camera_paths: List[List[str]] = []

        print(f"[INFO] Building {split} split (require 6 cams exist) ...")
        for scene in self.nusc.scene:
            if scene["name"] not in split_scenes:
                continue
            token = scene["first_sample_token"]
            while token:
                sample = self.nusc.get("sample", token)
                ok = True
                img_paths = []
                for cam in CAM_CHANNELS:
                    if cam not in sample["data"]:
                        ok = False; break
                    sd = self.nusc.get("sample_data", sample["data"][cam])
                    path = os.path.join(self.dataroot, sd["filename"])
                    if not os.path.exists(path):
                        ok = False; break
                    img_paths.append(path)

                if ok:
                    self.sample_tokens.append(sample["token"])
                    self.sample_camera_paths.append(img_paths)
                    if max_samples is not None and len(self.sample_tokens) >= max_samples:
                        break

                token = sample["next"]
            if max_samples is not None and len(self.sample_tokens) >= max_samples:
                break

        print(f"[INFO] {split}: valid samples = {len(self.sample_tokens)}")
        self.student_cam_idx = CAM_CHANNELS.index(STUDENT_CAM)

    def __len__(self): return len(self.sample_tokens)

    def _make_occ_bev(self, sample) -> np.ndarray:
        occ = np.zeros((BEV_H, BEV_W), dtype=np.float32)

        cam_front_sd = self.nusc.get("sample_data", sample["data"]["CAM_FRONT"])
        ego_pose = self.nusc.get("ego_pose", cam_front_sd["ego_pose_token"])
        ego_t = np.array(ego_pose["translation"], dtype=np.float32)
        ego_R = Quaternion(ego_pose["rotation"]).rotation_matrix
        ego_yaw = _yaw_from_quat(Quaternion(ego_pose["rotation"]))

        for ann_token in sample["anns"]:
            ann = self.nusc.get("sample_annotation", ann_token)
            pos_global = np.array(ann["translation"], dtype=np.float32)
            rel = pos_global - ego_t
            pos_ego = ego_R.T.dot(rel)

            x, y = float(pos_ego[0]), float(pos_ego[1])
            if not (BEV_X_MIN <= x <= BEV_X_MAX and BEV_Y_MIN <= y <= BEV_Y_MAX):
                continue

            sx, sy, _ = ann["size"]
            length = float(max(sx, sy))
            width = float(min(sx, sy))

            if OCC_GT_MODE == "box":
                dx = int((length/2)/BEV_RES)
                dy = int((width/2)/BEV_RES)
                ix = int((x - BEV_X_MIN)/BEV_RES)
                iy = int((y - BEV_Y_MIN)/BEV_RES)
                x0 = max(ix-dx, 0)
                x1 = min(ix+dx+1, BEV_H)
                y0 = max(iy-dy, 0)
                y1 = min(iy+dy+1, BEV_W)
                occ[x0:x1, y0:y1] = 1.0
                continue

            ann_yaw = _yaw_from_quat(Quaternion(ann["rotation"]))
            yaw_ego = ann_yaw - ego_yaw
            while yaw_ego > math.pi:
                yaw_ego -= 2 * math.pi
            while yaw_ego < -math.pi:
                yaw_ego += 2 * math.pi
            fill_rotated_box_on_occ(occ, x, y, length, width, yaw_ego)

        if OCC_DILATE_CELLS > 0:
            occ = dilate_occ(occ, OCC_DILATE_CELLS)
        return occ

    def _make_det_targets(self, sample):
        hm = np.zeros((1, BEV_H, BEV_W), dtype=np.float32)
        reg = np.zeros((6, BEV_H, BEV_W), dtype=np.float32)
        m = np.zeros((1, BEV_H, BEV_W), dtype=np.float32)

        cam_front_sd = self.nusc.get("sample_data", sample["data"]["CAM_FRONT"])
        ego_pose = self.nusc.get("ego_pose", cam_front_sd["ego_pose_token"])
        ego_t = np.array(ego_pose["translation"], dtype=np.float32)
        ego_R = Quaternion(ego_pose["rotation"]).rotation_matrix
        ego_yaw = _yaw_from_quat(Quaternion(ego_pose["rotation"]))

        for ann_token in sample["anns"]:
            ann = self.nusc.get("sample_annotation", ann_token)
            pos_global = np.array(ann["translation"], dtype=np.float32)
            rel = pos_global - ego_t
            pos_ego = ego_R.T.dot(rel)

            x, y = float(pos_ego[0]), float(pos_ego[1])
            if not (BEV_X_MIN <= x <= BEV_X_MAX and BEV_Y_MIN <= y <= BEV_Y_MAX):
                continue

            sx, sy, _ = ann["size"]
            length = float(max(sx, sy))
            width  = float(min(sx, sy))

            ann_yaw = _yaw_from_quat(Quaternion(ann["rotation"]))
            yaw_ego = ann_yaw - ego_yaw
            while yaw_ego > math.pi: yaw_ego -= 2*math.pi
            while yaw_ego < -math.pi: yaw_ego += 2*math.pi

            gx = (x - BEV_X_MIN)/BEV_RES
            gy = (y - BEV_Y_MIN)/BEV_RES
            ix, iy = int(gx), int(gy)
            if ix < 0 or ix >= BEV_H or iy < 0 or iy >= BEV_W:
                continue

            dx = gx - ix; dy = gy - iy
            rad = min(gaussian_radius(width/BEV_RES, length/BEV_RES, 0.7), 10)
            draw_gaussian(hm[0], (ix, iy), rad)

            reg[0, ix, iy] = dx
            reg[1, ix, iy] = dy
            reg[2, ix, iy] = math.log(max(width,  1e-3))
            reg[3, ix, iy] = math.log(max(length, 1e-3))
            reg[4, ix, iy] = math.sin(yaw_ego)
            reg[5, ix, iy] = math.cos(yaw_ego)
            m[0, ix, iy] = 1.0

        return hm, reg, m

    def _build_depth_gt_camfront(self, sample, K_scaled_np: np.ndarray) -> np.ndarray:
        depth_gt = np.full((FEAT_H, FEAT_W), -1, dtype=np.int64)
        depth_min = np.full((FEAT_H, FEAT_W), np.inf, dtype=np.float32)

        if "LIDAR_TOP" not in sample["data"]:
            return depth_gt

        lidar_sd = self.nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
        lidar_cs = self.nusc.get("calibrated_sensor", lidar_sd["calibrated_sensor_token"])
        lidar_pose = self.nusc.get("ego_pose", lidar_sd["ego_pose_token"])

        lidar_path = os.path.join(self.dataroot, lidar_sd["filename"])
        if not os.path.exists(lidar_path):
            return depth_gt

        pc = LidarPointCloud.from_file(lidar_path)

        # lidar -> ego
        R_l2e = Quaternion(lidar_cs["rotation"]).rotation_matrix
        t_l2e = np.array(lidar_cs["translation"], dtype=np.float32)
        pc.rotate(R_l2e); pc.translate(t_l2e)

        # ego -> global
        R_e2g = Quaternion(lidar_pose["rotation"]).rotation_matrix
        t_e2g = np.array(lidar_pose["translation"], dtype=np.float32)
        pc.rotate(R_e2g); pc.translate(t_e2g)

        pts_global = pc.points[:3, :]

        # global -> ego(cam)
        cam_sd = self.nusc.get("sample_data", sample["data"][STUDENT_CAM])
        cam_cs = self.nusc.get("calibrated_sensor", cam_sd["calibrated_sensor_token"])
        cam_pose = self.nusc.get("ego_pose", cam_sd["ego_pose_token"])

        R_e2g_cam = Quaternion(cam_pose["rotation"]).rotation_matrix
        t_e2g_cam = np.array(cam_pose["translation"], dtype=np.float32)
        pts_ego = R_e2g_cam.T.dot(pts_global - t_e2g_cam.reshape(3, 1))

        # ego -> cam
        R_c2e = Quaternion(cam_cs["rotation"]).rotation_matrix
        t_c2e = np.array(cam_cs["translation"], dtype=np.float32)
        pts_cam = R_c2e.T.dot(pts_ego - t_c2e.reshape(3, 1))

        z = pts_cam[2, :]
        valid = z > 0.1
        if valid.sum() == 0:
            return depth_gt
        pts_cam = pts_cam[:, valid]
        z = z[valid].astype(np.float32)

        pts_img = view_points(pts_cam, K_scaled_np, normalize=True)
        u = pts_img[0, :].astype(np.float32)
        v = pts_img[1, :].astype(np.float32)

        in_img = (u >= 0) & (u < IMG_W) & (v >= 0) & (v < IMG_H)
        if in_img.sum() == 0:
            return depth_gt
        u = u[in_img]; v = v[in_img]; z = z[in_img]

        in_d = (z >= DEPTH_MIN) & (z <= DEPTH_MAX)
        if in_d.sum() == 0:
            return depth_gt
        u = u[in_d]; v = v[in_d]; z = z[in_d]

        uf = (u / 8.0).astype(np.int32)
        vf = (v / 8.0).astype(np.int32)

        ok = (uf >= 0) & (uf < FEAT_W) & (vf >= 0) & (vf < FEAT_H)
        if ok.sum() == 0:
            return depth_gt
        uf = uf[ok]; vf = vf[ok]; z = z[ok]

        bin_idx = ((z - DEPTH_MIN) / (DEPTH_MAX - DEPTH_MIN) * NUM_DEPTH_BINS).astype(np.int32)
        bin_idx = np.clip(bin_idx, 0, NUM_DEPTH_BINS - 1)

        for uu, vv, zz, bb in zip(uf, vf, z, bin_idx):
            if zz < depth_min[vv, uu]:
                depth_min[vv, uu] = zz
                depth_gt[vv, uu] = int(bb)

        return depth_gt

    def __getitem__(self, idx):
        token = self.sample_tokens[idx]
        sample = self.nusc.get("sample", token)

        imgs6_tch, Ks6_tch, R6, t6 = [], [], [], []
        imgs6_stu, Ks6_stu = [], []

        for vi, (cam, img_path) in enumerate(zip(CAM_CHANNELS, self.sample_camera_paths[idx])):
            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                raise FileNotFoundError(img_path)
            orig_h, orig_w = img_bgr.shape[:2]
            img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            sd = self.nusc.get("sample_data", sample["data"][cam])
            cs = self.nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])

            K = np.array(cs["camera_intrinsic"], dtype=np.float32)
            R_c2e = Quaternion(cs["rotation"]).rotation_matrix.astype(np.float32)
            t_c2e = np.array(cs["translation"], dtype=np.float32)

            # teacher K scaled
            sx_t = TCH_W / float(orig_w); sy_t = TCH_H / float(orig_h)
            K_t = K.copy()
            K_t[0,0] *= sx_t; K_t[1,1] *= sy_t
            K_t[0,2] *= sx_t; K_t[1,2] *= sy_t

            # student K scaled
            sx_s = IMG_W / float(orig_w); sy_s = IMG_H / float(orig_h)
            K_s = K.copy()
            K_s[0,0] *= sx_s; K_s[1,1] *= sy_s
            K_s[0,2] *= sx_s; K_s[1,2] *= sy_s

            if self.transform_teacher is not None:
                img_tch = self.transform_teacher(img)
            else:
                img_rs = cv2.resize(img, (TCH_W, TCH_H))
                img_tch = torch.from_numpy(img_rs).permute(2,0,1).float()/255.0

            if self.transform_student is not None:
                img_stu = self.transform_student(img)
            else:
                img_rs = cv2.resize(img, (IMG_W, IMG_H))
                img_stu = torch.from_numpy(img_rs).permute(2,0,1).float()/255.0

            imgs6_tch.append(img_tch)
            Ks6_tch.append(K_t)

            imgs6_stu.append(img_stu)
            Ks6_stu.append(K_s)

            R6.append(R_c2e)
            t6.append(t_c2e)

        imgs6_tch = torch.stack(imgs6_tch, dim=0)  # (6,3,TCH_H,TCH_W)
        Ks6_tch = torch.from_numpy(np.stack(Ks6_tch, 0))

        imgs6_stu = torch.stack(imgs6_stu, dim=0)  # (6,3,IMG_H,IMG_W)
        Ks6_stu = torch.from_numpy(np.stack(Ks6_stu, 0))

        R6 = torch.from_numpy(np.stack(R6, 0))
        t6 = torch.from_numpy(np.stack(t6, 0))

        si = self.student_cam_idx
        img1_stu = imgs6_stu[si]
        K1_stu = Ks6_stu[si]
        R1 = R6[si]
        t1 = t6[si]

        occ = torch.from_numpy(self._make_occ_bev(sample)).unsqueeze(0).float()
        hm, reg, m = self._make_det_targets(sample)
        hm = torch.from_numpy(hm).float()
        reg = torch.from_numpy(reg).float()
        m = torch.from_numpy(m).float()

        if self.use_depth_sup:
            depth_gt = self._build_depth_gt_camfront(sample, K1_stu.numpy())
            depth_gt = torch.from_numpy(depth_gt).long()
        else:
            depth_gt = torch.full((FEAT_H, FEAT_W), -1, dtype=torch.long)

        return imgs6_tch, Ks6_tch, R6, t6, img1_stu, K1_stu, R1, t1, occ, hm, reg, m, depth_gt


# ============================================================
# 3) Splat utils
# ============================================================
def bilinear_splat_3d(vox_flat, feat_flat, ix_f, iy_f, iz, Z: int):
    ix_f = ix_f.detach(); iy_f = iy_f.detach(); iz = iz.detach()
    B, C, _ = vox_flat.shape
    H, W = BEV_H, BEV_W
    HW = H * W

    ix0 = torch.floor(ix_f).long()
    iy0 = torch.floor(iy_f).long()
    ix1 = ix0 + 1
    iy1 = iy0 + 1
    wx1 = (ix_f - ix0.float()).clamp(0, 1)
    wy1 = (iy_f - iy0.float()).clamp(0, 1)
    wx0 = 1.0 - wx1
    wy0 = 1.0 - wy1

    def add(ix, iy, w):
        valid = (iz >= 0) & (iz < Z) & (ix >= 0) & (ix < H) & (iy >= 0) & (iy < W) & (w > 0)
        if valid.sum().item() == 0:
            return
        idx2d = (ix * W + iy).clamp(0, HW - 1)
        idx3d = (iz * HW + idx2d).clamp(0, Z * HW - 1)

        for b in range(B):
            vb = valid[b]
            if vb.sum().item() == 0:
                continue
            idx_b = idx3d[b, vb]
            w_b = w[b, vb].unsqueeze(0).to(feat_flat.dtype)
            feat_b = (feat_flat[b, :, vb] * w_b).to(vox_flat.dtype)
            vox_flat[b].index_add_(1, idx_b, feat_b)

    add(ix0, iy0, wx0 * wy0)
    add(ix0, iy1, wx0 * wy1)
    add(ix1, iy0, wx1 * wy0)
    add(ix1, iy1, wx1 * wy1)


# ============================================================
# 4) Teacher (must match teacher ckpt)
# ============================================================
class ResNet50FPN(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.resnet50(weights=None)
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.lat2 = nn.Conv2d(512, 256, 1)
        self.lat3 = nn.Conv2d(1024, 256, 1)
        self.lat4 = nn.Conv2d(2048, 256, 1)

    def forward(self, x):
        x = self.stem(x)
        c4 = self.layer1(x)
        c8 = self.layer2(c4)
        c16 = self.layer3(c8)
        c32 = self.layer4(c16)
        p8 = self.lat2(c8)
        p16 = F.interpolate(self.lat3(c16), size=p8.shape[-2:], mode="bilinear", align_corners=False)
        p32 = F.interpolate(self.lat4(c32), size=p8.shape[-2:], mode="bilinear", align_corners=False)
        return p8 + p16 + p32

class DepthHeadTeacher(nn.Module):
    def __init__(self, in_ch=256, D=NUM_DEPTH_BINS, out_ch=256):
        super().__init__()
        self.depth = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, D, 1)
        )
        self.feat = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        depth_logits = self.depth(x)
        depth_prob = F.softmax(depth_logits, dim=1)
        feat = self.feat(x)
        return depth_prob, depth_logits, feat

class LiftSplatTeacher2D(nn.Module):
    def forward(self, feat2d, depth_prob, K, R_c2e, t_c2e, img_w: int, img_h: int):
        B, C, Hf, Wf = feat2d.shape
        D = depth_prob.shape[1]
        device = feat2d.device

        if USE_SPLAT_FP32:
            feat2d_ = feat2d.float()
            depth_prob_ = depth_prob.float()
            bev_flat = torch.zeros((B, C, BEV_H * BEV_W), device=device, dtype=torch.float32)
        else:
            feat2d_ = feat2d
            depth_prob_ = depth_prob
            bev_flat = torch.zeros((B, C, BEV_H * BEV_W), device=device, dtype=feat2d.dtype)

        feat_pix = feat2d_.view(B, C, -1)

        with torch.no_grad():
            u = torch.arange(Wf, device=device).view(1, 1, Wf).float()
            v = torch.arange(Hf, device=device).view(1, Hf, 1).float()
            u_img = (u + 0.5) * (img_w / Wf)
            v_img = (v + 0.5) * (img_h / Hf)
            u_img = u_img.expand(1, Hf, Wf).reshape(1, -1).expand(B, -1)
            v_img = v_img.expand(1, Hf, Wf).reshape(1, -1).expand(B, -1)

            fx = K[:, 0, 0].view(B, 1).float()
            fy = K[:, 1, 1].view(B, 1).float()
            cx = K[:, 0, 2].view(B, 1).float()
            cy = K[:, 1, 2].view(B, 1).float()

            depth_edges = torch.linspace(DEPTH_MIN, DEPTH_MAX, steps=D + 1, device=device).float()
            depth_centers = 0.5 * (depth_edges[:-1] + depth_edges[1:])
            R = R_c2e.float()
            t = t_c2e.float()

        H, W = BEV_H, BEV_W

        def bilinear_splat_2d(ix_f, iy_f, feat_flat_local):
            ix0 = torch.floor(ix_f).long()
            iy0 = torch.floor(iy_f).long()
            ix1 = ix0 + 1
            iy1 = iy0 + 1
            wx1 = (ix_f - ix0.float()).clamp(0, 1)
            wy1 = (iy_f - iy0.float()).clamp(0, 1)
            wx0 = 1.0 - wx1
            wy0 = 1.0 - wy1

            def add(ix, iy, w):
                valid = (ix >= 0) & (ix < H) & (iy >= 0) & (iy < W) & (w > 0)
                if valid.sum().item() == 0:
                    return
                idx = (ix * W + iy).clamp(0, H * W - 1)
                for b in range(B):
                    vb = valid[b]
                    if vb.sum().item() == 0:
                        continue
                    idx_b = idx[b, vb]
                    w_b = w[b, vb].unsqueeze(0).to(feat_flat_local.dtype)
                    feat_b = (feat_flat_local[b, :, vb] * w_b).to(bev_flat.dtype)
                    bev_flat[b].index_add_(1, idx_b, feat_b)

            add(ix0, iy0, wx0 * wy0)
            add(ix0, iy1, wx0 * wy1)
            add(ix1, iy0, wx1 * wy0)
            add(ix1, iy1, wx1 * wy1)

        for di in range(D):
            w = depth_prob_[:, di].view(B, -1)
            feat_w = feat_pix * w.unsqueeze(1)
            with torch.no_grad():
                z = depth_centers[di]
                x_cam = (u_img - cx) / fx * z
                y_cam = (v_img - cy) / fy * z
                z_cam = torch.full_like(x_cam, z)
                pts_cam = torch.stack([x_cam, y_cam, z_cam], dim=1)
                pts_ego = R @ pts_cam + t.view(B, 3, 1)
                x_ego = pts_ego[:, 0]
                y_ego = pts_ego[:, 1]
                ix_f = (x_ego - BEV_X_MIN) / BEV_RES
                iy_f = (y_ego - BEV_Y_MIN) / BEV_RES
            bilinear_splat_2d(ix_f, iy_f, feat_w)

        bev = bev_flat.view(B, C, BEV_H, BEV_W)
        if USE_SPLAT_FP32:
            bev = bev.to(feat2d.dtype)
        return bev

class BEVEncoderTeacher(nn.Module):
    def __init__(self, ch=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1), nn.BatchNorm2d(ch), nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1), nn.BatchNorm2d(ch), nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1), nn.BatchNorm2d(ch), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)

class SegHeadTeacher(nn.Module):
    def __init__(self, ch=256):
        super().__init__()
        self.out = nn.Conv2d(ch, 1, 1)
    def forward(self, x): return self.out(x)

class DetHeadTeacher(nn.Module):
    def __init__(self, ch=256):
        super().__init__()
        self.hm = nn.Sequential(nn.Conv2d(ch, ch, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(ch, 1, 1))
        self.reg = nn.Sequential(nn.Conv2d(ch, ch, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(ch, 6, 1))
    def forward(self, x): return self.hm(x), self.reg(x)

class MultiViewTeacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = ResNet50FPN()
        self.depth_head = DepthHeadTeacher(in_ch=256, D=NUM_DEPTH_BINS, out_ch=256)
        self.lss2d = LiftSplatTeacher2D()
        self.bev_encoder = BEVEncoderTeacher(256)
        self.seg_head = SegHeadTeacher(256)
        self.det_head = DetHeadTeacher(256)

    def _one_view(self, img, K, R, t):
        feat = self.backbone(img)
        depth_prob, depth_logits, feat2d = self.depth_head(feat)
        bev = self.lss2d(feat2d, depth_prob, K, R, t, img_w=TCH_W, img_h=TCH_H)
        return bev, depth_logits

    def forward(self, imgs6, Ks6, R6, t6, return_bev=False):
        B, V, _, _, _ = imgs6.shape
        bev_acc = torch.zeros((B, 256, BEV_H, BEV_W), device=imgs6.device, dtype=imgs6.dtype)
        depth_logits_all = []
        for v in range(V):
            bev_v, depth_logits = self._one_view(imgs6[:, v], Ks6[:, v], R6[:, v], t6[:, v])
            bev_acc = bev_acc + bev_v
            depth_logits_all.append(depth_logits)
        depth_logits_all = torch.stack(depth_logits_all, dim=1)  # (B,6,D,Hf,Wf)
        bev_ctx = self.bev_encoder(bev_acc)
        occ = self.seg_head(bev_ctx)
        hm, reg = self.det_head(bev_ctx)
        if return_bev:
            return occ, hm, reg, depth_logits_all, bev_ctx
        return occ, hm, reg, depth_logits_all


# ============================================================
# 5) Student (MUST match train_student_mnv3_kd_fixed.py)
# ============================================================
class MobileNetV3Stride8Extractor(nn.Module):
    """
    IMPORTANT: proj is LAZY. For resume strict loading, we must instantiate proj before load.
    """
    def __init__(self, pretrained=True, out_ch=256):
        super().__init__()
        weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        net = models.mobilenet_v3_large(weights=weights)
        self.features = net.features
        self.proj = None
        self.out_ch = out_ch

    def _ensure_proj(self, in_ch: int, device):
        if self.proj is None:
            self.proj = nn.Conv2d(in_ch, self.out_ch, 1).to(device)

    def forward(self, x):
        B, _, H0, W0 = x.shape
        y = x
        feat = None
        for layer in self.features:
            y = layer(y)
            sH = H0 / max(1, y.shape[-2])
            if feat is None and sH >= 8.0:
                feat = y
        if feat is None:
            feat = y

        target_h = H0 // 8
        target_w = W0 // 8
        if feat.shape[-2:] != (target_h, target_w):
            feat = F.interpolate(feat, size=(target_h, target_w), mode="bilinear", align_corners=False)

        self._ensure_proj(feat.shape[1], feat.device)
        return self.proj(feat)

class DepthHeadStudent(nn.Module):
    def __init__(self, in_ch=256, D=NUM_DEPTH_BINS, feat_ch=256):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_ch, in_ch, 3, padding=1), nn.ReLU(inplace=True))
        self.depth_logits = nn.Conv2d(in_ch, D, 1)
        self.feat = nn.Conv2d(in_ch, feat_ch, 1)
    def forward(self, x):
        x = self.conv(x)
        depth_logits = self.depth_logits(x)
        depth_prob = F.softmax(depth_logits, dim=1)
        feat = self.feat(x)
        return depth_prob, depth_logits, feat

class VoxelLiftSplatZMean(nn.Module):
    def __init__(self, z_bins=Z_BINS, z_min=Z_MIN, z_max=Z_MAX):
        super().__init__()
        self.z_bins = int(z_bins)
        self.z_min = float(z_min)
        self.z_max = float(z_max)

    def forward(self, feat2d, depth_prob, K, R_c2e, t_c2e):
        B, C, Hf, Wf = feat2d.shape
        D = depth_prob.shape[1]
        device = feat2d.device
        Z = self.z_bins
        H, W = BEV_H, BEV_W
        HW = H * W

        if USE_SPLAT_FP32:
            feat2d_ = feat2d.float()
            depth_prob_ = depth_prob.float()
            vox_flat = torch.zeros((B, C, Z * HW), device=device, dtype=torch.float32)
        else:
            feat2d_ = feat2d
            depth_prob_ = depth_prob
            vox_flat = torch.zeros((B, C, Z * HW), device=device, dtype=feat2d.dtype)

        feat_pix = feat2d_.view(B, C, -1)

        with torch.no_grad():
            u = torch.arange(Wf, device=device).view(1, 1, Wf).float()
            v = torch.arange(Hf, device=device).view(1, Hf, 1).float()
            u_img = (u + 0.5) * (IMG_W / Wf)
            v_img = (v + 0.5) * (IMG_H / Hf)
            u_img = u_img.expand(1, Hf, Wf).reshape(1, -1).expand(B, -1)
            v_img = v_img.expand(1, Hf, Wf).reshape(1, -1).expand(B, -1)

            fx = K[:, 0, 0].view(B, 1).float()
            fy = K[:, 1, 1].view(B, 1).float()
            cx = K[:, 0, 2].view(B, 1).float()
            cy = K[:, 1, 2].view(B, 1).float()

            depth_edges = torch.linspace(DEPTH_MIN, DEPTH_MAX, steps=D + 1, device=device).float()
            depth_centers = 0.5 * (depth_edges[:-1] + depth_edges[1:])
            R = R_c2e.float()
            t = t_c2e.float()
            z_scale = Z / (self.z_max - self.z_min + 1e-6)

        for di in range(D):
            w = depth_prob_[:, di].view(B, -1)
            feat_w = feat_pix * w.unsqueeze(1)
            with torch.no_grad():
                z = depth_centers[di]
                x_cam = (u_img - cx) / fx * z
                y_cam = (v_img - cy) / fy * z
                z_cam = torch.full_like(x_cam, z)
                pts_cam = torch.stack([x_cam, y_cam, z_cam], dim=1)
                pts_ego = R @ pts_cam + t.view(B, 3, 1)
                x_ego, y_ego, z_ego = pts_ego[:,0], pts_ego[:,1], pts_ego[:,2]
                ix_f = (x_ego - BEV_X_MIN) / BEV_RES
                iy_f = (y_ego - BEV_Y_MIN) / BEV_RES
                iz = torch.floor((z_ego - self.z_min) * z_scale).long()
            bilinear_splat_3d(vox_flat, feat_w, ix_f, iy_f, iz, Z=Z)

        vox = vox_flat.view(B, C, Z, H, W)
        bev2d = vox.mean(dim=2)
        if USE_SPLAT_FP32:
            bev2d = bev2d.to(feat2d.dtype)
        return bev2d

class BEVEncoderStudent(nn.Module):
    def __init__(self, ch=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1), nn.BatchNorm2d(ch), nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1), nn.BatchNorm2d(ch), nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1), nn.BatchNorm2d(ch), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)

class SegHeadStudent(nn.Module):
    def __init__(self, ch=256):
        super().__init__()
        self.out = nn.Conv2d(ch, 1, 1)
    def forward(self, x): return self.out(x)

class DetHeadStudent(nn.Module):
    def __init__(self, ch=256):
        super().__init__()
        self.hm = nn.Sequential(nn.Conv2d(ch, ch, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(ch, 1, 1))
        self.reg = nn.Sequential(nn.Conv2d(ch, ch, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(ch, 6, 1))
    def forward(self, x): return self.hm(x), self.reg(x)

class MonocularStudentMobileNetV3(nn.Module):
    """
    MUST match your fixed training script:
      kd_adapter = Identity
      backbone.proj = lazy
    """
    def __init__(self):
        super().__init__()
        self.backbone = MobileNetV3Stride8Extractor(pretrained=True, out_ch=256)
        self.depth_head = DepthHeadStudent(in_ch=256, D=NUM_DEPTH_BINS, feat_ch=256)
        self.voxel = VoxelLiftSplatZMean(z_bins=Z_BINS, z_min=Z_MIN, z_max=Z_MAX)
        self.bev_encoder = BEVEncoderStudent(ch=256)
        self.seg_head = SegHeadStudent(ch=256)
        self.det_head = DetHeadStudent(ch=256)
        self.kd_adapter = nn.Identity()

    def _forward_impl(self, img, K, R, t):
        feat8 = self.backbone(img)
        depth_prob, depth_logits, feat2d = self.depth_head(feat8)
        bev = self.voxel(feat2d, depth_prob, K, R, t)
        bev_ctx = self.bev_encoder(bev)
        occ = self.seg_head(bev_ctx)
        hm, reg = self.det_head(bev_ctx)
        return occ, hm, reg, depth_logits, self.kd_adapter(bev_ctx)

    def forward(self, img, K, R, t):
        if USE_CHECKPOINT_STUDENT:
            return checkpoint(lambda a,b,c,d: self._forward_impl(a,b,c,d), img, K, R, t, use_reentrant=False)
        return self._forward_impl(img, K, R, t)


# ============================================================
# 6) Metrics / utils / CSV
# ============================================================
@torch.no_grad()
def seg_counts_at_thr(occ_logits, occ_gt, thr: float):
    prob = torch.sigmoid(occ_logits)
    pred = (prob >= thr)
    gt = (occ_gt >= 0.5)
    tp = (pred & gt).sum().item()
    fp = (pred & (~gt)).sum().item()
    fn = ((~pred) & gt).sum().item()
    union = (pred | gt).sum().item()
    pm = prob.mean().item()
    return tp, fp, fn, union, pm

def iou_pr_from_counts(tp, fp, fn, union, eps=1e-6):
    iou = tp / (union + eps) if union > 0 else 0.0
    p = tp / (tp + fp + eps) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn + eps) if (tp + fn) > 0 else 0.0
    return iou, p, r

def pbar(desc, loader, enable=True):
    return tqdm(loader, total=len(loader), desc=desc, dynamic_ncols=True,
                leave=PBAR_LEAVE, mininterval=PBAR_MININTERVAL, file=sys.stdout,
                disable=not enable)

def get_lr(optimizer): return float(optimizer.param_groups[0]["lr"])

def kd_scale_at_epoch(ep: int) -> float:
    if KD_WARMUP_EPOCHS > 0 and ep <= KD_WARMUP_EPOCHS:
        return 0.0
    if KD_RAMP_EPOCHS <= 0:
        return 1.0 if need_teacher() else 0.0
    x = (ep - KD_WARMUP_EPOCHS) / float(KD_RAMP_EPOCHS)
    return float(np.clip(x, 0.0, 1.0))

def csv_init_if_needed(path: str):
    if os.path.exists(path): return
    header = [
        "epoch","lr","pos_weight","kd_scale",
        "train_total","train_occ","train_hm","train_reg","train_dep",
        "train_kd_occ","train_kd_hm","train_kd_reg","train_kd_feat","train_kd_dep",
        "val_best_miou","val_best_thr","val_best_p","val_best_r",
        "val_miou@0.20","val_p@0.20","val_r@0.20","val_pm",
        "sec_per_epoch"
    ]
    with open(path, "w", newline="") as f:
        csv.writer(f).writerow(header)

def csv_append(path: str, row: dict):
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            row["epoch"], row["lr"], row["pos_weight"], row["kd_scale"],
            row["train_total"], row["train_occ"], row["train_hm"], row["train_reg"], row["train_dep"],
            row["train_kd_occ"], row["train_kd_hm"], row["train_kd_reg"], row["train_kd_feat"], row["train_kd_dep"],
            row["val_best_miou"], row["val_best_thr"], row["val_best_p"], row["val_best_r"],
            row["val_miou20"], row["val_p20"], row["val_r20"], row["val_pm"],
            row["sec_per_epoch"]
        ])

@torch.no_grad()
def estimate_pos_weight_from_trainset(train_set: DistillMonoNuScenesDataset, n_samples: int = 200, seed: int = 0) -> float:
    n = len(train_set)
    if n == 0: return POS_WEIGHT_INIT
    rng = np.random.RandomState(seed)
    idxs = rng.choice(n, size=min(n_samples, n), replace=False)
    pos = 0.0; tot = 0.0
    for i in idxs:
        token = train_set.sample_tokens[int(i)]
        sample = train_set.nusc.get("sample", token)
        occ = train_set._make_occ_bev(sample)
        pos += float((occ > 0.5).sum())
        tot += float(occ.size)
    r = pos / max(1.0, tot)
    r = float(np.clip(r, 1e-4, 0.2))
    pw = (1.0 - r) / r
    return float(np.clip(pw, 5.0, 200.0))


# ============================================================
# 7) Train / Eval
# ============================================================
def train_epoch(student, teacher, loader, optimizer, occ_criterion, scaler, epoch: int):
    student.train()
    if teacher is not None:
        teacher.eval()

    optimizer.zero_grad(set_to_none=True)

    kd_scale = kd_scale_at_epoch(epoch) if teacher is not None else 0.0

    sum_total=sum_occ=sum_hm=sum_reg=sum_dep=0.0
    sum_kd_occ=sum_kd_hm=sum_kd_reg=sum_kd_feat=sum_kd_dep=0.0
    n=0

    # robust AMP scaler usage
    did_step = False
    accum_counter = 0

    bar = pbar(f"Train ep{epoch:02d}", loader, enable=SHOW_TRAIN_PBAR)
    for it, batch in enumerate(bar, 1):
        imgs6_tch, Ks6_tch, R6, t6, img1, K1, R1, t1, occ_gt, hm_gt, reg_gt, reg_m, depth_gt = batch

        img1 = img1.to(DEVICE, non_blocking=True)
        K1   = K1.to(DEVICE, non_blocking=True)
        R1   = R1.to(DEVICE, non_blocking=True)
        t1   = t1.to(DEVICE, non_blocking=True)

        occ_gt = occ_gt.to(DEVICE, non_blocking=True)
        hm_gt  = hm_gt.to(DEVICE, non_blocking=True)
        reg_gt = reg_gt.to(DEVICE, non_blocking=True)
        reg_m  = reg_m.to(DEVICE, non_blocking=True)
        depth_gt = depth_gt.to(DEVICE, non_blocking=True)

        # teacher forward only if needed
        if teacher is not None and kd_scale > 0:
            imgs6_tch = imgs6_tch.to(DEVICE, non_blocking=True)
            Ks6_tch   = Ks6_tch.to(DEVICE, non_blocking=True)
            R6        = R6.to(DEVICE, non_blocking=True)
            t6        = t6.to(DEVICE, non_blocking=True)
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=(AMP and DEVICE.type=="cuda")):
                    occ_t, hm_t, reg_t, depth_all_t, bev_t = teacher(imgs6_tch, Ks6_tch, R6, t6, return_bev=True)
        else:
            occ_t = hm_t = reg_t = depth_all_t = bev_t = None

        with torch.cuda.amp.autocast(enabled=(AMP and DEVICE.type=="cuda")):
            occ_s, hm_s, reg_s, depth_s, bev_s = student(img1, K1, R1, t1)

            # GT losses
            l_occ = occ_criterion(occ_s, occ_gt)
            l_hm  = gaussian_focal_loss(torch.sigmoid(hm_s), hm_gt, alpha=FOCAL_ALPHA, beta=GAUSS_BETA)
            l_reg = smooth_l1_reg_loss(reg_s, reg_gt, reg_m)
            l_dep = occ_s.new_tensor(0.0)
            if USE_DEPTH_SUP:
                l_dep = safe_depth_ce_loss(depth_s, depth_gt, ignore_index=-1)

            # KD losses (default 0)
            kd_occ = occ_s.new_tensor(0.0)
            kd_hm  = occ_s.new_tensor(0.0)
            kd_reg = occ_s.new_tensor(0.0)
            kd_feat= occ_s.new_tensor(0.0)
            kd_dep = occ_s.new_tensor(0.0)

            if teacher is not None and kd_scale > 0:
                occ_mask = build_occ_kd_mask(occ_t, occ_gt=occ_gt, T=KD_T)
                hm_mask = build_occ_kd_mask(hm_t, occ_gt=hm_gt, T=KD_T)
                feat_mask = kd_feat_occ_mask(occ_t, bev_s)

                if W_KD_OCC > 0:
                    kd_occ = kd_bce_with_logits_masked(occ_s, occ_t, mask=occ_mask, T=KD_T)
                if W_KD_HM  > 0:
                    kd_hm  = kd_bce_with_logits_masked(hm_s, hm_t, mask=hm_mask, T=KD_T)
                if W_KD_REG > 0: kd_reg = smooth_l1_reg_loss(reg_s, reg_t.detach(), reg_m)
                if W_KD_FEAT> 0:
                    kd_feat = kd_mse_masked(bev_s, bev_t.detach(), feat_mask)
                if W_KD_DEP > 0:
                    depth_front_t = depth_all_t[:, 0].detach()  # (B,D,Ht,Wt)
                    depth_front_t_rs = F.interpolate(depth_front_t, size=(FEAT_H, FEAT_W), mode="bilinear", align_corners=False)
                    depth_mask = kd_depth_mask(depth_front_t_rs)
                    kd_dep = kd_mse_masked(depth_s, depth_front_t_rs, depth_mask)

            total = (W_OCC*l_occ + W_HM*l_hm + W_REG*l_reg + W_DEP*l_dep) + \
                    kd_scale * (W_KD_OCC*kd_occ + W_KD_HM*kd_hm + W_KD_REG*kd_reg + W_KD_FEAT*kd_feat + W_KD_DEP*kd_dep)

        if not torch.isfinite(total):
            if SKIP_NAN_BATCH:
                optimizer.zero_grad(set_to_none=True)
                bar.set_postfix_str("skip(nonfinite)")
                continue
            raise RuntimeError(f"Non-finite loss: {total.item()}")

        scaler.scale(total / GRAD_ACCUM).backward()
        accum_counter += 1

        if accum_counter >= GRAD_ACCUM:
            scaler.unscale_(optimizer)
            if GRAD_CLIP_NORM and GRAD_CLIP_NORM > 0:
                torch.nn.utils.clip_grad_norm_(student.parameters(), GRAD_CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()
            did_step = True
            optimizer.zero_grad(set_to_none=True)
            accum_counter = 0

        sum_total += float(total.item())
        sum_occ   += float(l_occ.item())
        sum_hm    += float(l_hm.item())
        sum_reg   += float(l_reg.item())
        sum_dep   += float(l_dep.item())
        sum_kd_occ  += float(kd_occ.item())
        sum_kd_hm   += float(kd_hm.item())
        sum_kd_reg  += float(kd_reg.item())
        sum_kd_feat += float(kd_feat.item())
        sum_kd_dep  += float(kd_dep.item())
        n += 1

        bar.set_postfix(loss=sum_total/max(1,n), kd_scale=kd_scale, kd_feat=sum_kd_feat/max(1,n))

    # flush remainder grads if any
    if accum_counter > 0:
        scaler.unscale_(optimizer)
        if GRAD_CLIP_NORM and GRAD_CLIP_NORM > 0:
            torch.nn.utils.clip_grad_norm_(student.parameters(), GRAD_CLIP_NORM)
        scaler.step(optimizer)
        scaler.update()
        did_step = True
        optimizer.zero_grad(set_to_none=True)

    # if never stepped, do NOT call scaler.update() anymore (already avoided)
    if n == 0:
        return {k: float("nan") for k in ["total","occ","hm","reg","dep","kd_occ","kd_hm","kd_reg","kd_feat","kd_dep"]}, kd_scale

    return dict(
        total=sum_total/n, occ=sum_occ/n, hm=sum_hm/n, reg=sum_reg/n, dep=sum_dep/n,
        kd_occ=sum_kd_occ/n, kd_hm=sum_kd_hm/n, kd_reg=sum_kd_reg/n, kd_feat=sum_kd_feat/n, kd_dep=sum_kd_dep/n
    ), kd_scale

@torch.no_grad()
def eval_epoch(student, loader, epoch: int):
    student.eval()
    counts = {thr: {"tp":0.0,"fp":0.0,"fn":0.0,"union":0.0} for thr in THR_LIST}
    pm_sum=0.0
    n=0
    bar = pbar(f"Val   ep{epoch:02d}", loader, enable=SHOW_VAL_PBAR)
    for batch in bar:
        _, _, _, _, img1, K1, R1, t1, occ_gt, _, _, _, _ = batch
        img1 = img1.to(DEVICE, non_blocking=True)
        K1   = K1.to(DEVICE, non_blocking=True)
        R1   = R1.to(DEVICE, non_blocking=True)
        t1   = t1.to(DEVICE, non_blocking=True)
        occ_gt = occ_gt.to(DEVICE, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=(AMP and DEVICE.type=="cuda")):
            occ_s, _, _, _, _ = student(img1, K1, R1, t1)

        for thr in THR_LIST:
            tp, fp, fn, union, pm = seg_counts_at_thr(occ_s, occ_gt, thr)
            counts[thr]["tp"] += tp; counts[thr]["fp"] += fp; counts[thr]["fn"] += fn; counts[thr]["union"] += union

        pm_sum += pm
        n += 1

        best_thr = THR_LIST[0]; best_iou=-1.0
        for thr in THR_LIST:
            iou, p, r = iou_pr_from_counts(counts[thr]["tp"], counts[thr]["fp"], counts[thr]["fn"], counts[thr]["union"])
            if iou > best_iou:
                best_iou, best_thr = iou, thr
        bar.set_postfix(best_iou=best_iou, best_thr=best_thr, pm=pm_sum/max(1,n))

    best_thr=THR_LIST[0]; best_iou=-1.0; best_p=best_r=0.0
    for thr in THR_LIST:
        iou, p, r = iou_pr_from_counts(counts[thr]["tp"], counts[thr]["fp"], counts[thr]["fn"], counts[thr]["union"])
        if iou > best_iou:
            best_iou, best_thr, best_p, best_r = iou, thr, p, r

    miou20, p20, r20 = 0.0, 0.0, 0.0
    if 0.20 in counts:
        miou20, p20, r20 = iou_pr_from_counts(counts[0.20]["tp"], counts[0.20]["fp"], counts[0.20]["fn"], counts[0.20]["union"])

    return dict(best_miou=best_iou, best_thr=best_thr, best_p=best_p, best_r=best_r,
                miou20=miou20, p20=p20, r20=r20, pm=pm_sum/max(1,n))


# ============================================================
# 8) Resume helper (CRITICAL: instantiate lazy proj BEFORE strict load)
# ============================================================
def instantiate_student_lazy_modules(student: MonocularStudentMobileNetV3):
    """
    Run a dummy forward to create backbone.proj (lazy) so that strict load won't fail
    with Unexpected key(s): backbone.proj.*
    """
    student.eval()
    with torch.no_grad():
        img = torch.zeros((1,3,IMG_H,IMG_W), device=DEVICE)
        K = torch.eye(3, device=DEVICE).unsqueeze(0)
        R = torch.eye(3, device=DEVICE).unsqueeze(0)
        t = torch.zeros((1,3), device=DEVICE)
        _ = student(img, K, R, t)  # create proj
    student.train()


# ============================================================
# 9) Main
# ============================================================
def main():
    print(f"[ENV] EXP={EXP} CSV={CSV_PATH} SAVE_PREFIX={SAVE_PREFIX} RESUME_STUDENT={RESUME_STUDENT if RESUME_STUDENT else '(none)'}")
    print(f"[ENV] LR={LR} EPOCHS={EPOCHS} KD_WARMUP_EPOCHS={KD_WARMUP_EPOCHS} KD_RAMP_EPOCHS={KD_RAMP_EPOCHS}")
    print(f"[ENV] KD: T={KD_T} W=[occ{W_KD_OCC},hm{W_KD_HM},reg{W_KD_REG},feat{W_KD_FEAT},dep{W_KD_DEP}]  need_teacher={need_teacher()}")
    print(f"[ENV] DATA_ROOT={DATA_ROOT} OCC_GT_MODE={OCC_GT_MODE} dilate={OCC_DILATE_CELLS} train_max={TRAIN_MAX_SAMPLES} val_max={VAL_MAX_SAMPLES}")
    print(f"[ENV] KD mask: pos>={KD_POS_THR:.2f} neg<={KD_NEG_THR:.2f} gt_agree={KD_REQUIRE_GT_AGREEMENT} feat_pos>={KD_FEAT_POS_THR:.2f} depth_conf>={KD_DEPTH_CONF_THR:.2f}")

    transform_teacher = T.Compose([
        T.ToTensor(),
        T.Resize((TCH_H, TCH_W), antialias=True),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    transform_student = T.Compose([
        T.ToTensor(),
        T.Resize((IMG_H, IMG_W), antialias=True),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])

    print("====== Build datasets ======")
    train_set = DistillMonoNuScenesDataset(DATA_ROOT, split="train",
        transform_teacher=transform_teacher, transform_student=transform_student,
        verbose=False,
        max_samples=(TRAIN_MAX_SAMPLES if TRAIN_MAX_SAMPLES > 0 else None),
        use_depth_sup=USE_DEPTH_SUP)
    val_set = DistillMonoNuScenesDataset(DATA_ROOT, split="val",
        transform_teacher=transform_teacher, transform_student=transform_student,
        verbose=False,
        max_samples=(VAL_MAX_SAMPLES if VAL_MAX_SAMPLES > 0 else None),
        use_depth_sup=USE_DEPTH_SUP)

    train_kwargs = dict(batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
                        persistent_workers=PERSISTENT_WORKERS, drop_last=True)
    val_kwargs = dict(batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
                      persistent_workers=PERSISTENT_WORKERS, drop_last=False)
    if NUM_WORKERS > 0 and PREFETCH_FACTOR is not None:
        train_kwargs["prefetch_factor"] = PREFETCH_FACTOR
        val_kwargs["prefetch_factor"] = PREFETCH_FACTOR

    train_loader = DataLoader(train_set, **train_kwargs)
    val_loader = DataLoader(val_set, **val_kwargs)

    teacher = None
    if need_teacher():
        assert os.path.exists(TEACHER_CKPT), f"Teacher ckpt not found: {TEACHER_CKPT}"
        teacher = MultiViewTeacher().to(DEVICE)
        sd_t = torch.load(TEACHER_CKPT, map_location="cpu")
        teacher.load_state_dict(sd_t, strict=True)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        print(f"[OK] Loaded teacher: {TEACHER_CKPT} (teacher_img={TCH_W}x{TCH_H})")
    else:
        print("[OK] EXP has no KD -> teacher is NOT loaded (finetune GT only).")

    student = MonocularStudentMobileNetV3().to(DEVICE)

    # IMPORTANT: instantiate lazy proj BEFORE resume strict load
    instantiate_student_lazy_modules(student)

    if RESUME_STUDENT:
        assert os.path.exists(RESUME_STUDENT), f"RESUME_STUDENT not found: {RESUME_STUDENT}"
        sd_s = torch.load(RESUME_STUDENT, map_location="cpu")
        print(f"[RESUME] Loading student from {RESUME_STUDENT} (strict=True)")
        student.load_state_dict(sd_s, strict=True)

    occ_criterion = BCEDiceLoss(pos_weight=POS_WEIGHT_INIT, w_bce=1.0, w_dice=1.0).to(DEVICE)
    if AUTO_EST_POS_WEIGHT:
        print(f"[INFO] Estimating pos_weight from {min(ESTIMATE_SAMPLES, len(train_set))} samples ...")
        pw = estimate_pos_weight_from_trainset(train_set, n_samples=ESTIMATE_SAMPLES, seed=SEED)
        occ_criterion.set_pos_weight(pw)
        print(f"[INFO] Estimated pos_weight = {pw:.2f}")

    optimizer = torch.optim.AdamW(student.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, EPOCHS), eta_min=LR*0.05)

    scaler = torch.cuda.amp.GradScaler(enabled=(AMP and DEVICE.type=="cuda"))

    csv_init_if_needed(CSV_PATH)

    print("\n========= TRAIN Student (ablation/fine-tune fixed) ==========")
    print(f"[CFG] teacher_img=({TCH_W}x{TCH_H})  student_img=({IMG_W}x{IMG_H}) depth_bins={NUM_DEPTH_BINS} "
          f"bev=({BEV_H}x{BEV_W}) z_bins={Z_BINS} bs={BATCH_SIZE} accum={GRAD_ACCUM} amp={AMP} "
          f"pos_weight={float(occ_criterion.pos_weight.item()):.2f}")
    print(f"[KD] T={KD_T} warmup={KD_WARMUP_EPOCHS} ramp={KD_RAMP_EPOCHS} "
          f"W=[occ{W_KD_OCC},hm{W_KD_HM},reg{W_KD_REG},feat{W_KD_FEAT},dep{W_KD_DEP}]")
    print(f"[GT] W=[occ{W_OCC},hm{W_HM},reg{W_REG},dep{W_DEP}]  depth_sup={USE_DEPTH_SUP}  dilate={OCC_DILATE_CELLS}")

    best = -1.0
    for ep in range(1, EPOCHS+1):
        t0 = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        tr, kd_scale = train_epoch(student, teacher, train_loader, optimizer, occ_criterion, scaler, epoch=ep)
        va = eval_epoch(student, val_loader, epoch=ep)

        tqdm.write(f"[Train ep{ep:02d}] total={tr['total']:.4f} (kd_scale={kd_scale:.2f}) "
                   f"occ={tr['occ']:.4f} hm={tr['hm']:.4f} reg={tr['reg']:.4f} dep={tr['dep']:.4f} | "
                   f"kd_occ={tr['kd_occ']:.4f} kd_hm={tr['kd_hm']:.4f} kd_reg={tr['kd_reg']:.4f} kd_feat={tr['kd_feat']:.4f} kd_dep={tr['kd_dep']:.4f}")
        tqdm.write(f"[Val   ep{ep:02d}] best_mIoU={va['best_miou']:.4f}@thr={va['best_thr']:.2f} "
                   f"(P={va['best_p']:.4f},R={va['best_r']:.4f})  mIoU@0.20={va['miou20']:.4f} pm={va['pm']:.4f}")

        if va["best_miou"] > best:
            best = va["best_miou"]
            best_path = f"{SAVE_PREFIX}_best.pth"
            torch.save(student.state_dict(), best_path)
            tqdm.write(f"[SAVED BEST] {best_path} (best_mIoU={best:.4f})")

        last_path = f"{SAVE_PREFIX}_last.pth"
        torch.save(student.state_dict(), last_path)

        sec = time.time() - t0
        csv_append(CSV_PATH, dict(
            epoch=ep, lr=get_lr(optimizer), pos_weight=float(occ_criterion.pos_weight.item()), kd_scale=float(kd_scale),
            train_total=tr["total"], train_occ=tr["occ"], train_hm=tr["hm"], train_reg=tr["reg"], train_dep=tr["dep"],
            train_kd_occ=tr["kd_occ"], train_kd_hm=tr["kd_hm"], train_kd_reg=tr["kd_reg"],
            train_kd_feat=tr["kd_feat"], train_kd_dep=tr["kd_dep"],
            val_best_miou=va["best_miou"], val_best_thr=va["best_thr"], val_best_p=va["best_p"], val_best_r=va["best_r"],
            val_miou20=va["miou20"], val_p20=va["p20"], val_r20=va["r20"], val_pm=va["pm"],
            sec_per_epoch=sec
        ))

        scheduler.step()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        tqdm.write(f"[ep{ep:02d} end] sec={sec:.1f}s lr={get_lr(optimizer):.2e} csv={CSV_PATH}")

    print(f"[DONE] Saved {SAVE_PREFIX}_best.pth / {SAVE_PREFIX}_last.pth  | csv={CSV_PATH}")


if __name__ == "__main__":
    main()
