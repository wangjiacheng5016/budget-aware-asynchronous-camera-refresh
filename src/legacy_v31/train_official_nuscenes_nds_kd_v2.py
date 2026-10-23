# -*- coding: utf-8 -*-
"""
Official nuScenes 10-class detection training/evaluation for NDS-aligned experiments.

This script is intentionally separate from the original mIoU occupancy scripts.
It trains:
  1) a 6-camera teacher with a nuScenes 10-class CenterPoint-style head;
  2) a monocular MobileNetV3 student with the same official detection head;
  3) optional KD from teacher to student;
  4) official nuScenes detection JSON export and NuScenesEval NDS.

The detection head predicts:
  heatmap: 10 official nuScenes detection classes
  reg: dx, dy, z, log(w), log(l), log(h), sin(yaw), cos(yaw), vx, vy

Run on the 3090 server, for example:
  pip install nuscenes-devkit opencv-python pyquaternion tqdm torchvision

  python train50/train_official_nuscenes_nds_kd.py train-teacher \
    --data-root datasets/nuscenes --epochs 12 --batch-size 1 --grad-accum 4 \
    --out-dir train50/official_nds/teacher

  python train50/train_official_nuscenes_nds_kd.py train-student \
    --data-root datasets/nuscenes --epochs 12 --batch-size 4 --grad-accum 2 \
    --teacher-ckpt train50/official_nds/teacher/teacher_best.pth \
    --out-dir train50/official_nds/student_kd

  python train50/train_official_nuscenes_nds_kd.py eval \
    --model student --ckpt train50/official_nds/student_kd/student_best.pth \
    --data-root datasets/nuscenes --out-dir train50/official_nds/student_kd/eval_val
"""

import argparse
import csv
import importlib.util
import json
import math
import os
import random
import sqlite3
import sys
import time
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision import models
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm import tqdm

try:
    from nuscenes.eval.common.config import config_factory
except Exception:
    from nuscenes.eval.detection.config import config_factory
from nuscenes.eval.detection.evaluate import NuScenesEval
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import BoxVisibility, box_in_image
from pyquaternion import Quaternion as PyQuaternion


THIS_DIR = Path(__file__).resolve().parent
BASE_SCRIPT = Path(os.getenv("BEV_BASE_SCRIPT", str(THIS_DIR / "train_student_mnv3_kd_ablation_full.py")))

DETECTION_NAMES = [
    "car", "truck", "construction_vehicle", "bus", "trailer",
    "barrier", "motorcycle", "bicycle", "pedestrian", "traffic_cone",
]
NAME_TO_ID = {name: i for i, name in enumerate(DETECTION_NAMES)}

ATTRIBUTE_NAMES = [
    "cycle.with_rider", "cycle.without_rider",
    "pedestrian.moving", "pedestrian.sitting_lying_down", "pedestrian.standing",
    "vehicle.moving", "vehicle.parked", "vehicle.stopped",
]
ATTRIBUTE_TO_ID = {name: i for i, name in enumerate(ATTRIBUTE_NAMES)}
VALID_ATTRIBUTES_BY_DETECTION = {
    "car": ("vehicle.moving", "vehicle.parked", "vehicle.stopped"),
    "truck": ("vehicle.moving", "vehicle.parked", "vehicle.stopped"),
    "construction_vehicle": ("vehicle.moving", "vehicle.parked", "vehicle.stopped"),
    "bus": ("vehicle.moving", "vehicle.parked", "vehicle.stopped"),
    "trailer": ("vehicle.moving", "vehicle.parked", "vehicle.stopped"),
    "motorcycle": ("cycle.with_rider", "cycle.without_rider"),
    "bicycle": ("cycle.with_rider", "cycle.without_rider"),
    "pedestrian": ("pedestrian.moving", "pedestrian.sitting_lying_down", "pedestrian.standing"),
    "barrier": (),
    "traffic_cone": (),
}

# Mean [width, length, height] in metres for the nuScenes training classes.  These
# are only used to normalize the class-aware size loss; decoded boxes remain in
# the official nuScenes convention.  A repository tool regenerates the values
# from the official train split so experiments can audit the training-only prior.
CLASS_SIZE_MEANS = {
    "car": (1.960029, 4.634432, 1.737557),
    "truck": (2.517862, 6.936331, 2.850624),
    "construction_vehicle": (2.824728, 6.561963, 3.198066),
    "bus": (2.950115, 11.194943, 3.491828),
    "trailer": (2.923130, 12.275775, 3.870860),
    "barrier": (2.508320, 0.496818, 0.989223),
    "motorcycle": (0.772890, 2.106671, 1.463524),
    "bicycle": (0.608169, 1.700587, 1.302928),
    "pedestrian": (0.669412, 0.729165, 1.770168),
    "traffic_cone": (0.409679, 0.415949, 1.076954),
}

MOVING_ATTRIBUTES_BY_DETECTION = {
    "car": ("vehicle.moving",),
    "truck": ("vehicle.moving",),
    "construction_vehicle": ("vehicle.moving",),
    "bus": ("vehicle.moving",),
    "trailer": ("vehicle.moving",),
    "motorcycle": ("cycle.with_rider",),
    "bicycle": ("cycle.with_rider",),
    "pedestrian": ("pedestrian.moving",),
    "barrier": (),
    "traffic_cone": (),
}

CATEGORY_TO_DETECTION = {
    "vehicle.car": "car",
    "vehicle.truck": "truck",
    "vehicle.construction": "construction_vehicle",
    "vehicle.bus.bendy": "bus",
    "vehicle.bus.rigid": "bus",
    "vehicle.trailer": "trailer",
    "movable_object.barrier": "barrier",
    "vehicle.motorcycle": "motorcycle",
    "vehicle.bicycle": "bicycle",
    "human.pedestrian.adult": "pedestrian",
    "human.pedestrian.child": "pedestrian",
    "human.pedestrian.construction_worker": "pedestrian",
    "human.pedestrian.police_officer": "pedestrian",
    "movable_object.trafficcone": "traffic_cone",
}


def load_base_module():
    spec = importlib.util.spec_from_file_location("bev_base_module", BASE_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def fast_mobilenet_stride8_forward(self, x):
    """Return the exact stride-8 feature used by the legacy forward without computing discarded stages."""
    _batch, _channels, h0, w0 = x.shape
    y = x
    feat = None
    for layer in self.features:
        y = layer(y)
        if h0 / max(1, y.shape[-2]) >= 8.0:
            feat = y
            break
    if feat is None:
        feat = y
    target_h = h0 // 8
    target_w = w0 // 8
    if feat.shape[-2:] != (target_h, target_w):
        feat = F.interpolate(feat, size=(target_h, target_w), mode="bilinear", align_corners=False)
    self._ensure_proj(feat.shape[1], feat.device)
    return self.proj(feat)


class FastVoxelLiftSplatZMean(nn.Module):
    """
    Algebraically equivalent replacement for the legacy 3D voxel splat.

    The legacy implementation scatters into Z separate planes and immediately averages those
    planes. Linearity lets us accumulate valid points directly in 2D and divide by Z. Processing
    multiple depth bins per chunk also removes hundreds of Python loops and CUDA synchronizations.
    """

    def __init__(self, mod, z_bins, z_min, z_max, depth_chunk=4):
        super().__init__()
        self.mod = mod
        self.z_bins = int(z_bins)
        self.z_min = float(z_min)
        self.z_max = float(z_max)
        self.depth_chunk = max(1, int(depth_chunk))

    def forward(self, feat2d, depth_prob, k, r_cam_to_ego, t_cam_to_ego):
        mod = self.mod
        batch, channels, feat_h, feat_w = feat2d.shape
        depth_bins = depth_prob.shape[1]
        device = feat2d.device
        bev_h, bev_w = int(mod.BEV_H), int(mod.BEV_W)
        bev_hw = bev_h * bev_w

        accumulate_fp32 = bool(getattr(mod, "USE_SPLAT_FP32", True))
        work_dtype = torch.float32 if accumulate_fp32 else feat2d.dtype
        feat = feat2d.to(work_dtype)
        depth = depth_prob.to(work_dtype)
        feat_pixels = feat.reshape(batch, channels, -1)
        bev_flat = torch.zeros((batch, channels, bev_hw), device=device, dtype=work_dtype)

        with torch.no_grad():
            u = torch.arange(feat_w, device=device, dtype=torch.float32).view(1, 1, feat_w)
            v = torch.arange(feat_h, device=device, dtype=torch.float32).view(1, feat_h, 1)
            u = ((u + 0.5) * (float(mod.IMG_W) / feat_w)).expand(1, feat_h, feat_w).reshape(1, -1)
            v = ((v + 0.5) * (float(mod.IMG_H) / feat_h)).expand(1, feat_h, feat_w).reshape(1, -1)
            u = u.expand(batch, -1)
            v = v.expand(batch, -1)
            fx = k[:, 0, 0].float().view(batch, 1)
            fy = k[:, 1, 1].float().view(batch, 1)
            cx = k[:, 0, 2].float().view(batch, 1)
            cy = k[:, 1, 2].float().view(batch, 1)
            ray_x = (u - cx) / fx
            ray_y = (v - cy) / fy
            edges = torch.linspace(
                float(mod.DEPTH_MIN), float(mod.DEPTH_MAX),
                steps=depth_bins + 1, device=device, dtype=torch.float32,
            )
            centers = 0.5 * (edges[:-1] + edges[1:])
            r = r_cam_to_ego.float()
            t = t_cam_to_ego.float()

        for start in range(0, depth_bins, self.depth_chunk):
            stop = min(depth_bins, start + self.depth_chunk)
            chunk_depth = centers[start:stop]
            chunk_size = stop - start
            with torch.no_grad():
                z_cam = chunk_depth.view(1, chunk_size, 1).expand(batch, -1, ray_x.shape[1])
                x_cam = ray_x.unsqueeze(1) * z_cam
                y_cam = ray_y.unsqueeze(1) * z_cam
                points_cam = torch.stack((x_cam, y_cam, z_cam), dim=2)
                points_ego = torch.einsum("bij,bdjn->bdin", r, points_cam) + t[:, None, :, None]
                x_ego = points_ego[:, :, 0]
                y_ego = points_ego[:, :, 1]
                z_ego = points_ego[:, :, 2]
                ix = (x_ego - float(mod.BEV_X_MIN)) / float(mod.BEV_RES)
                iy = (y_ego - float(mod.BEV_Y_MIN)) / float(mod.BEV_RES)

                ix0 = torch.floor(ix).long()
                iy0 = torch.floor(iy).long()
                ix1 = ix0 + 1
                iy1 = iy0 + 1
                wx1 = (ix - ix0.float()).clamp(0.0, 1.0)
                wy1 = (iy - iy0.float()).clamp(0.0, 1.0)
                wx0 = 1.0 - wx1
                wy0 = 1.0 - wy1
                # Match the legacy epsilon exactly at the upper Z boundary.
                z_scale = self.z_bins / (self.z_max - self.z_min + 1e-6)
                iz = torch.floor((z_ego - self.z_min) * z_scale).long()
                z_valid = (iz >= 0) & (iz < self.z_bins)

            probabilities = depth[:, start:stop].reshape(batch, chunk_size, -1)
            weighted_features = (
                feat_pixels.unsqueeze(2) * probabilities.unsqueeze(1)
            ).reshape(batch, channels, -1)

            def scatter(ix_neighbor, iy_neighbor, weights):
                valid = (
                    z_valid & (ix_neighbor >= 0) & (ix_neighbor < bev_h) &
                    (iy_neighbor >= 0) & (iy_neighbor < bev_w) & (weights > 0)
                )
                indices = (ix_neighbor * bev_w + iy_neighbor).clamp(0, bev_hw - 1).reshape(batch, -1)
                source = weighted_features * (weights * valid).reshape(batch, 1, -1).to(work_dtype)
                bev_flat.scatter_add_(2, indices.unsqueeze(1).expand(-1, channels, -1), source)

            scatter(ix0, iy0, wx0 * wy0)
            scatter(ix0, iy1, wx0 * wy1)
            scatter(ix1, iy0, wx1 * wy0)
            scatter(ix1, iy1, wx1 * wy1)

        bev = bev_flat.reshape(batch, channels, bev_h, bev_w) / float(self.z_bins)
        return bev.to(feat2d.dtype) if accumulate_fp32 else bev


def configure_fast_runtime(mod, args):
    if not getattr(args, "fast_runtime", True):
        print("[RUNTIME] reference MobileNet and voxel splat enabled")
        return
    mod.MobileNetV3Stride8Extractor.forward = fast_mobilenet_stride8_forward
    depth_chunk = int(getattr(args, "splat_depth_chunk", 4))
    mod.VoxelLiftSplatZMean = lambda z_bins, z_min, z_max: FastVoxelLiftSplatZMean(
        mod, z_bins=z_bins, z_min=z_min, z_max=z_max, depth_chunk=depth_chunk,
    )
    print(f"[RUNTIME] fast stride-8 backbone + direct-Z splat enabled depth_chunk={depth_chunk}")


def compact_student_backbone(model, enabled: bool = True):
    """Physically remove MobileNet stages after the first stride-8 feature.

    The legacy extractor saved the first stride-8 tensor but continued through all later
    MobileNet stages before returning that saved tensor. Those later stages therefore cannot
    affect the output or receive gradients. Keeping stages 0..4 is exact for torchvision's
    MobileNetV3-Large and makes the state size/parameter count reflect the executed student.
    """
    if not enabled:
        return
    if getattr(model, "backbone_variant", "stride8") != "stride8":
        print(f"[MODEL] compact backbone skipped for variant={model.backbone_variant}")
        return
    features = list(model.backbone.features.children())
    keep_stages = 5
    if len(features) <= keep_stages:
        return
    before = sum(p.numel() for p in model.parameters())
    removed = sum(p.numel() for layer in features[keep_stages:] for p in layer.parameters())
    model.backbone.features = nn.Sequential(*features[:keep_stages])
    after = sum(p.numel() for p in model.parameters())
    if before - after != removed:
        raise RuntimeError("unexpected parameter count while compacting MobileNetV3")
    print(f"[MODEL] compact stride-8 backbone removed_params={removed} params_before={before} params_after={after}")


def load_model_state_compatible(model, state, strict: bool = True):
    """Load normal or legacy-full-backbone checkpoints without hiding unrelated mismatches."""
    state_dict = unwrap_state_dict(state)
    if not isinstance(state_dict, dict):
        raise TypeError(f"checkpoint does not contain a state dict: {type(state_dict)!r}")
    expected = model.state_dict()
    extras = [key for key in state_dict if key not in expected]
    allowed_extras = []
    if hasattr(model, "backbone") and hasattr(model.backbone, "features"):
        stage_count = len(model.backbone.features)
        prefix = "backbone.features."
        for key in extras:
            if not key.startswith(prefix):
                continue
            suffix = key[len(prefix):]
            stage_text = suffix.split(".", 1)[0]
            if stage_text.isdigit() and int(stage_text) >= stage_count:
                allowed_extras.append(key)
    training_only_prefixes = ("kd_adapter.", "hop_history_decoder.", "hop_det_head.")
    architecture_upgrade_prefixes = (
        "class_motion_head.", "temporal_spatial_fusion.",
        "azimuth_bin_head.", "class_size_head.", "temporal_correlation_motion_head.",
        "class_center_head.",
        "image_yaw_flip_head.",
        "image_radial_center_head.",
        "image_object_geometry_head.",
        "object_velocity_refiner.", "residual_azimuth_multibin_head.",
        "residual_yaw_vector_head.",
        "class_attribute_velocity_calibrator.", "temporal_stereo_depth_matcher.",
        "privileged_residual_adapter.",
    )
    training_only_extras = [key for key in extras if key.startswith(training_only_prefixes)]
    allowed_extras.extend(training_only_extras)
    unexpected = sorted(set(extras) - set(allowed_extras))
    if strict and unexpected:
        raise RuntimeError(f"unexpected checkpoint keys: {unexpected[:10]}")
    filtered = {key: value for key, value in state_dict.items() if key in expected}
    missing, remaining_unexpected = model.load_state_dict(filtered, strict=False)
    allowed_missing = [
        key for key in missing
        if key.startswith(training_only_prefixes + architecture_upgrade_prefixes)
    ]
    disallowed_missing = sorted(set(missing) - set(allowed_missing))
    if strict and (disallowed_missing or remaining_unexpected):
        raise RuntimeError(
            f"checkpoint mismatch missing={disallowed_missing[:10]} unexpected={list(remaining_unexpected)[:10]}"
        )
    history_head_missing = any(key.startswith("hop_det_head.") for key in allowed_missing)
    if history_head_missing and getattr(model, "hop_det_head", None) is not None:
        # The auxiliary head predicts the same classes and box code as the deployed head.
        # Reusing its trained prior avoids a destructive random-loss spike at fine-tune start.
        model.hop_det_head.load_state_dict(model.det_head.state_dict(), strict=True)
        print("[CKPT] initialized training-only HoP detector from the deployed detection head")
    motion_head_missing = any(key.startswith("class_motion_head.") for key in allowed_missing)
    if motion_head_missing and getattr(model, "class_motion_head", None) is not None:
        print("[CKPT] initialized zero-residual class-aware motion head for architecture upgrade")
    if allowed_extras:
        print(f"[CKPT] ignored {len(allowed_extras)} legacy or training-only tensors")
    if allowed_missing:
        print(f"[CKPT] initialized {len(allowed_missing)} training-only tensors")
    return missing, remaining_unexpected


def unwrap_state_dict(state):
    if isinstance(state, dict):
        state = state.get("state_dict", state.get("model_state_dict", state.get("model", state)))
    if isinstance(state, dict) and any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    return state


def load_model_state_partial(model, state):
    """Load tensors whose names and shapes match, for controlled architecture upgrades."""
    state_dict = unwrap_state_dict(state)
    if not isinstance(state_dict, dict):
        raise TypeError(f"checkpoint does not contain a state dict: {type(state_dict)!r}")
    expected = model.state_dict()
    compatible = {
        key: value for key, value in state_dict.items()
        if key in expected and tuple(value.shape) == tuple(expected[key].shape)
    }
    shape_mismatch = sum(
        1 for key, value in state_dict.items()
        if key in expected and tuple(value.shape) != tuple(expected[key].shape)
    )
    unexpected = sum(1 for key in state_dict if key not in expected)
    missing, _ = model.load_state_dict(compatible, strict=False)
    report = {
        "loaded": len(compatible),
        "shape_mismatch": shape_mismatch,
        "unexpected": unexpected,
        "missing": len(missing),
    }
    print(
        "[PARTIAL_INIT] "
        f"loaded={report['loaded']} shape_mismatch={shape_mismatch} "
        f"unexpected={unexpected} missing={len(missing)}"
    )
    return report


def yaw_from_quat(rotation) -> float:
    rot = PyQuaternion(rotation).rotation_matrix
    return float(math.atan2(rot[1, 0], rot[0, 0]))


def wrap_pi(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def gaussian_radius(w: float, h: float, min_overlap: float = 0.7) -> int:
    a1 = 1
    b1 = h + w
    c1 = w * h * (1 - min_overlap) / (1 + min_overlap)
    sq1 = b1 ** 2 - 4 * a1 * c1
    r1 = (b1 + math.sqrt(max(0.0, sq1))) / 2

    a2 = 4
    b2 = 2 * (h + w)
    c2 = (1 - min_overlap) * w * h
    sq2 = b2 ** 2 - 4 * a2 * c2
    r2 = (b2 + math.sqrt(max(0.0, sq2))) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (h + w)
    c3 = (min_overlap - 1) * w * h
    sq3 = b3 ** 2 - 4 * a3 * c3
    r3 = (b3 + math.sqrt(max(0.0, sq3))) / 2
    return max(1, int(min(r1, r2, r3)))


def draw_gaussian(heatmap: np.ndarray, center: Tuple[int, int], radius: int):
    x, y = center
    h, w = heatmap.shape
    diameter = 2 * radius + 1
    sigma = diameter / 6.0
    xs = np.arange(diameter, dtype=np.float32) - radius
    ys = np.arange(diameter, dtype=np.float32) - radius
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    g = np.exp(-(xx * xx + yy * yy) / (2 * sigma * sigma))

    left, right = min(y, radius), min(w - y - 1, radius)
    top, bottom = min(x, radius), min(h - x - 1, radius)
    if left < 0 or right < 0 or top < 0 or bottom < 0:
        return
    patch = heatmap[x - top:x + bottom + 1, y - left:y + right + 1]
    gp = g[radius - top:radius + bottom + 1, radius - left:radius + right + 1]
    np.maximum(patch, gp, out=patch)


def draw_gaussian_scaled(heatmap: np.ndarray, center: Tuple[int, int], radius: int, scale: float):
    x, y = center
    h, w = heatmap.shape
    diameter = 2 * radius + 1
    sigma = diameter / 6.0
    xs = np.arange(diameter, dtype=np.float32) - radius
    ys = np.arange(diameter, dtype=np.float32) - radius
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    gaussian = np.exp(-(xx * xx + yy * yy) / (2 * sigma * sigma)) * float(scale)
    left, right = min(y, radius), min(w - y - 1, radius)
    top, bottom = min(x, radius), min(h - x - 1, radius)
    if left < 0 or right < 0 or top < 0 or bottom < 0:
        return
    patch = heatmap[x - top:x + bottom + 1, y - left:y + right + 1]
    gp = gaussian[radius - top:radius + bottom + 1, radius - left:radius + right + 1]
    np.maximum(patch, gp, out=patch)


def gaussian_focal_loss(pred: torch.Tensor, gt: torch.Tensor, alpha=2.0, beta=4.0, eps=1e-6,
                        pos_class_weights: Optional[torch.Tensor] = None):
    # Keep the focal-loss probability arithmetic in FP32 even under autocast.
    # In FP16, 1 - 1e-6 rounds back to 1.0; a saturated sigmoid can then make
    # the masked negative term evaluate as log(0) * 0 -> NaN.  This conversion
    # is a numerical implementation detail shared by every training arm and
    # does not change the model, targets, sampling, optimizer, or inference.
    pred = pred.float()
    gt = gt.float()
    pred = torch.nan_to_num(pred, nan=0.0, posinf=1.0, neginf=0.0).clamp(eps, 1 - eps)
    gt = torch.nan_to_num(gt, nan=0.0, posinf=1.0, neginf=0.0)
    pos = (gt == 1).float()
    neg = (gt < 1).float()
    pos_loss = -torch.log(pred) * ((1 - pred) ** alpha) * pos
    neg_loss = -torch.log(1 - pred) * (pred ** alpha) * ((1 - gt) ** beta) * neg
    if pos_class_weights is not None:
        weights = pos_class_weights.to(device=pred.device, dtype=pred.dtype)
        if weights.ndim == 1:
            weights = weights.view(1, -1, 1, 1)
        if weights.shape[1] != pred.shape[1]:
            raise ValueError(
                f"positive class weights have {weights.shape[1]} entries, expected {pred.shape[1]}"
            )
        pos_loss = pos_loss * weights
    num_pos = pos.sum().clamp(min=1.0)
    return (pos_loss.sum() + neg_loss.sum()) / num_pos


def masked_reg_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, vel_mask: torch.Tensor,
                  channel_scales: Optional[torch.Tensor] = None):
    mask10 = mask.expand_as(pred).float().clone()
    mask10[:, 8:10] = mask10[:, 8:10] * vel_mask.expand(-1, 2, -1, -1).float()
    if channel_scales is not None:
        mask10 = mask10 * channel_scales.to(dtype=mask10.dtype, device=mask10.device)
    diff = F.smooth_l1_loss(pred, target, reduction="none", beta=1.0)
    diff = torch.nan_to_num(diff, nan=0.0, posinf=0.0, neginf=0.0) * mask10
    return diff.sum() / mask10.sum().clamp(min=1.0)


def safe_depth_ce_loss(depth_logits: torch.Tensor, depth_gt: torch.Tensor, ignore_index: int = -1):
    valid = depth_gt != ignore_index
    if valid.sum().item() == 0:
        return depth_logits.new_tensor(0.0)
    b, d, h, w = depth_logits.shape
    logits = depth_logits.permute(0, 2, 3, 1).reshape(-1, d)
    gt = depth_gt.reshape(-1).long()
    v = valid.reshape(-1)
    return F.cross_entropy(logits[v], gt[v], reduction="mean")


def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sanitize_batchnorm_running_stats(model: nn.Module) -> int:
    """Repair non-finite BatchNorm buffers without changing learned weights."""
    repaired = 0
    for module in model.modules():
        if not isinstance(module, nn.modules.batchnorm._BatchNorm):
            continue
        if module.running_mean is not None:
            bad_mean = ~torch.isfinite(module.running_mean)
            repaired += int(bad_mean.sum().item())
            if bad_mean.any():
                module.running_mean[bad_mean] = 0.0
        if module.running_var is not None:
            bad_var = (~torch.isfinite(module.running_var)) | (module.running_var <= 0)
            repaired += int(bad_var.sum().item())
            if bad_var.any():
                module.running_var[bad_var] = 1.0
    return repaired


def snapshot_named_trainable_modules(
    model: nn.Module, module_names: Sequence[str],
) -> Dict[str, Dict[str, torch.Tensor]]:
    """Take compact CPU snapshots used to prove declared modules really update."""
    snapshots: Dict[str, Dict[str, torch.Tensor]] = {}
    modules = dict(model.named_modules())
    for module_name in module_names:
        if module_name not in modules:
            raise ValueError(f"cannot audit missing module {module_name!r}")
        values = {
            name: parameter.detach().cpu().clone()
            for name, parameter in modules[module_name].named_parameters()
            if parameter.requires_grad
        }
        if not values:
            raise ValueError(f"module {module_name!r} has no trainable parameters to audit")
        snapshots[module_name] = values
    return snapshots


def assert_named_modules_updated(
    model: nn.Module,
    snapshots: Dict[str, Dict[str, torch.Tensor]],
    context: str,
) -> Dict[str, float]:
    """Fail fast when AMP/optimizer issues silently prevent an expected update."""
    modules = dict(model.named_modules())
    deltas: Dict[str, float] = {}
    for module_name, before in snapshots.items():
        after = dict(modules[module_name].named_parameters())
        maximum = 0.0
        for parameter_name, old_value in before.items():
            new_value = after[parameter_name].detach().cpu()
            if not torch.isfinite(new_value).all():
                raise FloatingPointError(
                    f"{context}: audited module {module_name}.{parameter_name} became non-finite"
                )
            maximum = max(
                maximum,
                float((new_value.float() - old_value.float()).abs().max().item()),
            )
        deltas[module_name] = maximum
        if maximum == 0.0:
            raise RuntimeError(
                f"{context}: audited trainable module {module_name!r} did not update; "
                "this usually indicates skipped optimizer steps or a detached loss path"
            )
    return deltas


def assert_finite_model_parameters(model: nn.Module, context: str = "model"):
    """Fail before saving/evaluating a checkpoint whose learned weights are invalid."""
    bad = []
    bad_values = 0
    for name, parameter in model.named_parameters():
        if not parameter.is_floating_point():
            continue
        count = int((~torch.isfinite(parameter.detach())).sum().item())
        if count:
            bad.append(name)
            bad_values += count
    if bad:
        preview = ", ".join(bad[:5])
        raise FloatingPointError(
            f"{context} has {bad_values} non-finite learned values in {len(bad)} tensors; "
            f"first tensors: {preview}"
        )


def prepare_model_for_eval(model: nn.Module, checkpoint_state) -> int:
    """Load an evaluation checkpoint and repair invalid BatchNorm buffers."""
    load_model_state_compatible(model, checkpoint_state, strict=True)
    repaired = sanitize_batchnorm_running_stats(model)
    assert_finite_model_parameters(model, context="evaluation checkpoint")
    return repaired


def replace_batchnorm_with_groupnorm(module: nn.Module, max_groups: int = 32) -> int:
    """Replace BatchNorm recursively while preserving compatible affine parameters."""
    replaced = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.modules.batchnorm._BatchNorm):
            groups = min(int(max_groups), int(child.num_features))
            while child.num_features % groups != 0:
                groups -= 1
            norm = nn.GroupNorm(groups, child.num_features, eps=child.eps, affine=child.affine)
            if child.affine:
                with torch.no_grad():
                    norm.weight.copy_(child.weight)
                    norm.bias.copy_(child.bias)
            setattr(module, name, norm)
            replaced += 1
        else:
            replaced += replace_batchnorm_with_groupnorm(child, max_groups=max_groups)
    return replaced


def masked_mean(value: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6):
    mask = mask.to(dtype=value.dtype, device=value.device)
    while mask.dim() < value.dim():
        mask = mask.unsqueeze(1)
    if mask.shape[1] == 1 and value.dim() >= 2 and value.shape[1] != 1:
        mask = mask.expand(-1, value.shape[1], *mask.shape[2:])
    return (value * mask).sum() / mask.sum().clamp(min=eps)


def dilate_mask(mask: torch.Tensor, kernel: int):
    if kernel <= 1:
        return mask
    # Same-padding max pooling requires an odd kernel.  Several experiment
    # configs historically used ``2`` to mean a small dilation; rounding it to
    # the next odd kernel keeps the spatial grid unchanged instead of silently
    # expanding 200x200 to 201x201.
    if kernel % 2 == 0:
        kernel += 1
    pad = kernel // 2
    return F.max_pool2d(mask.float(), kernel_size=kernel, stride=1, padding=pad)


def kd_scale_for_epoch(args, epoch: int) -> float:
    warmup = int(getattr(args, "kd_warmup_epochs", 0))
    scale = (
        1.0 if warmup <= 0 else
        min(1.0, max(0.0, float(epoch) / float(warmup)))
    )
    decay_start = int(getattr(args, "kd_early_decay_start_epoch", 0))
    decay_end = int(getattr(args, "kd_early_decay_end_epoch", 0))
    if decay_start > 0 or decay_end > 0:
        if decay_start <= 0 or decay_end < decay_start:
            raise ValueError("invalid early KD decay epoch range")
        if epoch >= decay_start:
            if epoch > decay_end:
                return 0.0
            decay_steps = decay_end - decay_start + 1
            remaining = decay_end - epoch + 1
            scale *= float(remaining) / float(decay_steps + 1)
    return scale


def kd_balance_target_for_epoch(args, epoch: int) -> float:
    """Piecewise-linear KD budget that stays active through late training."""
    fallback = float(getattr(args, "kd_balance_target", 0.0))
    start = float(getattr(args, "kd_balance_target_start", -1.0))
    middle = float(getattr(args, "kd_balance_target_middle", -1.0))
    end = float(getattr(args, "kd_balance_target_end", -1.0))
    if start < 0.0 and middle < 0.0 and end < 0.0:
        return fallback
    start = fallback if start < 0.0 else start
    middle = fallback if middle < 0.0 else middle
    end = fallback if end < 0.0 else end
    total_epochs = max(1, int(getattr(args, "schedule_total_epochs", 0) or args.epochs))
    progress = min(1.0, max(0.0, float(epoch - 1) / max(1, total_epochs - 1)))
    if progress <= 0.5:
        alpha = progress / 0.5
        return start + alpha * (middle - start)
    alpha = (progress - 0.5) / 0.5
    return middle + alpha * (end - middle)


def adaptive_kd_balance_scale(
    det_loss: torch.Tensor,
    kd_loss: torch.Tensor,
    target_fraction: float,
    minimum: float,
    maximum: float,
) -> torch.Tensor:
    """Keep KD numerically relevant without allowing it to dominate detection.

    The scale is computed from detached losses, so it changes only the relative
    gradient magnitude of the already-defined objectives.  A zero target keeps
    the historical behaviour exactly.
    """
    target_fraction = float(target_fraction)
    minimum = float(minimum)
    maximum = float(maximum)
    if target_fraction <= 0.0:
        return kd_loss.new_tensor(1.0)
    if minimum <= 0.0 or maximum < minimum:
        raise ValueError("invalid adaptive KD balance bounds")
    ratio = (
        target_fraction * det_loss.detach().float() /
        kd_loss.detach().float().clamp(min=1e-8)
    )
    return ratio.clamp(min=minimum, max=maximum).to(dtype=kd_loss.dtype)


def quality_gated_logit_target(
    privileged_logits: torch.Tensor,
    reference_logits: torch.Tensor,
    supervised_target: torch.Tensor,
    alpha: float = 0.5,
    margin: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Transfer only privileged binary-logit residuals that beat the camera path.

    The privileged and reference predictions come from the same teacher weights;
    the only difference is whether the custom LiDAR completion/refinement path is
    enabled.  Ground truth therefore provides a direct per-response quality test,
    preventing LiDAR-only errors from becoming student targets.
    """
    if privileged_logits.shape != reference_logits.shape or privileged_logits.shape != supervised_target.shape:
        raise ValueError("quality-gated logit tensors must have identical shapes")
    alpha = float(alpha)
    margin = float(margin)
    if not 0.0 <= alpha <= 1.0 or margin < 0.0:
        raise ValueError("quality-gated residual alpha/margin are invalid")
    privileged_error = F.binary_cross_entropy_with_logits(
        privileged_logits.float(), supervised_target.float(), reduction="none",
    )
    reference_error = F.binary_cross_entropy_with_logits(
        reference_logits.float(), supervised_target.float(), reduction="none",
    )
    accepted = privileged_error + margin < reference_error
    candidate = reference_logits.float() + alpha * (
        privileged_logits.float() - reference_logits.float()
    )
    bridge = torch.where(accepted, candidate, reference_logits.float())
    return bridge.to(dtype=privileged_logits.dtype), accepted


def quality_gated_regression_target(
    privileged_prediction: torch.Tensor,
    reference_prediction: torch.Tensor,
    supervised_target: torch.Tensor,
    valid_mask: torch.Tensor,
    alpha: float = 0.5,
    margin: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Gate each regression channel independently at supervised object centres."""
    if privileged_prediction.shape != reference_prediction.shape or privileged_prediction.shape != supervised_target.shape:
        raise ValueError("quality-gated regression tensors must have identical shapes")
    alpha = float(alpha)
    margin = float(margin)
    if not 0.0 <= alpha <= 1.0 or margin < 0.0:
        raise ValueError("quality-gated residual alpha/margin are invalid")
    valid = valid_mask.bool().expand_as(privileged_prediction)
    privileged_error = (privileged_prediction.float() - supervised_target.float()).abs()
    reference_error = (reference_prediction.float() - supervised_target.float()).abs()
    accepted = valid & (privileged_error + margin < reference_error)
    candidate = reference_prediction.float() + alpha * (
        privileged_prediction.float() - reference_prediction.float()
    )
    bridge = torch.where(accepted, candidate, reference_prediction.float())
    return bridge.to(dtype=privileged_prediction.dtype), accepted


def hybrid_quality_gated_regression_target(
    privileged_prediction: torch.Tensor,
    reference_prediction: torch.Tensor,
    supervised_target: torch.Tensor,
    valid_mask: torch.Tensor,
    privileged_channels: Sequence[int] = (0, 1, 2),
    alpha: float = 0.5,
    margin: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Route mature camera knowledge and privileged geometry by channel.

    Only explicitly declared channels may receive a LiDAR-on residual.  Every
    other channel stays exactly on the same teacher's LiDAR-off camera output,
    preserving the V38 assistant's useful size/attribute behaviour while
    preventing the weaker privileged yaw/velocity responses from leaking in.
    """
    bridge, accepted = quality_gated_regression_target(
        privileged_prediction, reference_prediction, supervised_target,
        valid_mask, alpha=alpha, margin=margin,
    )
    channel_route = torch.zeros(
        privileged_prediction.shape[1], dtype=torch.bool,
        device=privileged_prediction.device,
    )
    for channel in privileged_channels:
        channel = int(channel)
        if channel < 0 or channel >= privileged_prediction.shape[1]:
            raise ValueError(f"invalid privileged regression channel {channel}")
        channel_route[channel] = True
    channel_route = channel_route.view(1, -1, 1, 1)
    accepted = accepted & channel_route
    bridge = torch.where(
        channel_route, bridge, reference_prediction.to(dtype=bridge.dtype),
    )
    return bridge, accepted


def privileged_residual_bridge_target(
    privileged_prediction: torch.Tensor,
    camera_prediction: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """Add the privileged-only residual to an independent camera teacher.

    V43 incorrectly used the LiDAR-off branch of the end-to-end privileged
    teacher as its camera reference and then discarded privileged BEV/depth.
    V44 instead anchors on an independently trained camera detector and transfers
    only the residual introduced by the project-native privileged pathway.
    """
    if privileged_prediction.shape != camera_prediction.shape:
        raise ValueError(
            "privileged and camera predictions must have identical shapes, got "
            f"{tuple(privileged_prediction.shape)} and {tuple(camera_prediction.shape)}"
        )
    if not 0.0 <= float(alpha) <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    privileged_prediction = privileged_prediction.detach()
    camera_prediction = camera_prediction.detach()
    return camera_prediction + float(alpha) * (
        privileged_prediction - camera_prediction
    )


def quality_gated_categorical_target(
    privileged_logits: torch.Tensor,
    reference_logits: torch.Tensor,
    supervised_target: torch.Tensor,
    alpha: float = 0.5,
    margin: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Select a complete categorical logit vector when privileged CE is lower."""
    if privileged_logits.shape != reference_logits.shape or privileged_logits.ndim != 4:
        raise ValueError("categorical teacher logits must be matching BCHW tensors")
    if supervised_target.shape != privileged_logits.shape[:1] + privileged_logits.shape[2:]:
        raise ValueError("categorical target must have shape BHW")
    alpha = float(alpha)
    margin = float(margin)
    if not 0.0 <= alpha <= 1.0 or margin < 0.0:
        raise ValueError("quality-gated residual alpha/margin are invalid")
    valid = supervised_target >= 0
    safe_target = supervised_target.long().clamp(0, privileged_logits.shape[1] - 1)
    privileged_error = F.cross_entropy(
        privileged_logits.float(), safe_target, reduction="none",
    )
    reference_error = F.cross_entropy(
        reference_logits.float(), safe_target, reduction="none",
    )
    accepted = valid & (privileged_error + margin < reference_error)
    candidate = reference_logits.float() + alpha * (
        privileged_logits.float() - reference_logits.float()
    )
    bridge = torch.where(
        accepted.unsqueeze(1), candidate, reference_logits.float(),
    )
    return bridge.to(dtype=privileged_logits.dtype), accepted.unsqueeze(1)


def quality_gated_attribute_target(
    privileged_logits: torch.Tensor,
    reference_logits: torch.Tensor,
    supervised_target: torch.Tensor,
    alpha: float = 0.5,
    margin: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return quality_gated_categorical_target(
        privileged_logits, reference_logits, supervised_target, alpha, margin,
    )


def gradient_balanced_kd_sum(
    branches: Sequence[Tuple[str, torch.Tensor, float]],
    anchor: torch.Tensor,
    minimum: float = 0.1,
    maximum: float = 10.0,
    exact: bool = False,
    reference_loss: Optional[torch.Tensor] = None,
    skip_negative_cosine: bool = False,
    protected_reference_loss: Optional[torch.Tensor] = None,
    project_negative_protected_cosine: bool = False,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Balance active KD branches by their gradient norm on the shared BEV.

    Loss magnitudes are a poor proxy for influence: V38 devoted 87% of its
    weighted scalar objective to heatmap KD while feature/relation supervision
    was nearly inert.  For every active branch, this routine measures the
    gradient norm with respect to the shared student BEV tensor, then rescales
    it so the resulting branch norms follow predeclared shares.  The scales are
    detached; this is a stable GradNorm-style controller rather than a second-
    order objective.  Branches without a path to the BEV anchor are returned at
    unit scale and can still be controlled by the outer total-KD budget.
    """
    if anchor.ndim != 4 or not anchor.requires_grad:
        raise ValueError("gradient-balanced KD requires a differentiable BCHW BEV anchor")
    minimum = float(minimum)
    maximum = float(maximum)
    if minimum <= 0.0 or maximum < minimum:
        raise ValueError("invalid gradient-balance scale bounds")

    active = [(str(name), loss, float(share)) for name, loss, share in branches
              if float(share) > 0.0]
    if not active:
        return anchor.new_tensor(0.0), {}
    if any(loss.ndim != 0 for _, loss, _ in active):
        raise ValueError("KD branch losses must be scalar tensors")

    reference_gradient = None
    if skip_negative_cosine:
        if reference_loss is None or not reference_loss.requires_grad:
            raise ValueError("gradient conflict control requires reference_loss")
        reference_gradient = torch.autograd.grad(
            reference_loss, anchor, retain_graph=True, create_graph=False,
            allow_unused=True,
        )[0]
        if reference_gradient is None:
            raise ValueError("reference loss has no path to BEV anchor")
        reference_gradient = reference_gradient.detach().float()

    protected_gradient = None
    if project_negative_protected_cosine:
        if protected_reference_loss is None or not protected_reference_loss.requires_grad:
            raise ValueError("protected gradient projection requires protected_reference_loss")
        protected_gradient = torch.autograd.grad(
            protected_reference_loss, anchor, retain_graph=True,
            create_graph=False, allow_unused=True,
        )[0]
        if protected_gradient is None:
            raise ValueError("protected reference loss has no path to BEV anchor")
        protected_gradient = protected_gradient.detach().float()
        protected_norm_sq = protected_gradient.square().sum().clamp(min=1e-12)

    norms: Dict[str, torch.Tensor] = {}
    differentiable = []
    for name, loss, share in active:
        if not loss.requires_grad:
            norms[name] = loss.detach().new_tensor(0.0)
            continue
        gradient = torch.autograd.grad(
            loss, anchor, retain_graph=True, create_graph=False,
            allow_unused=True,
        )[0]
        if gradient is None:
            norms[name] = loss.detach().new_tensor(0.0)
            continue
        gradient = gradient.detach().float()
        effective_loss = loss
        if protected_gradient is not None:
            protected_cosine = torch.sum(gradient * protected_gradient) / (
                gradient.norm(p=2).clamp(min=1e-12) *
                protected_gradient.norm(p=2).clamp(min=1e-12)
            )
            norms[f"{name}_protected_cosine"] = protected_cosine
            if float(protected_cosine) < 0.0:
                projected_gradient = gradient - (
                    torch.sum(gradient * protected_gradient) /
                    protected_norm_sq
                ) * protected_gradient
                # Zero-valued first-order surrogate: the scalar objective is
                # unchanged, while its gradient at the shared BEV anchor is
                # replaced by the PCGrad projection.
                correction = torch.sum(
                    (projected_gradient - gradient).to(anchor.dtype).detach() * anchor
                )
                effective_loss = loss + correction - correction.detach()
                gradient = projected_gradient
        norm = gradient.norm(p=2).clamp(min=1e-12)
        norms[name] = norm
        if reference_gradient is not None:
            cosine = torch.sum(gradient * reference_gradient) / (
                norm * reference_gradient.norm(p=2).clamp(min=1e-12)
            )
            scales_key = f"{name}_cosine"
            # Stored below with branch scales for deterministic diagnostics.
            norms[scales_key] = cosine
            if float(cosine) < 0.0:
                continue
        differentiable.append((name, effective_loss, share, norm))

    scales: Dict[str, torch.Tensor] = {}
    for key, value in norms.items():
        if key.endswith("_cosine"):
            scales[key] = value.to(dtype=anchor.dtype)
    if differentiable:
        share_total = sum(item[2] for item in differentiable)
        norm_total = torch.stack([item[3] for item in differentiable]).sum()
        for name, _loss, share, norm in differentiable:
            target_norm = norm_total * (share / max(share_total, 1e-12))
            scale = target_norm / norm
            if not exact:
                scale = scale.clamp(min=minimum, max=maximum)
            scales[name] = scale.to(dtype=anchor.dtype)

    total = anchor.new_tensor(0.0)
    effective_losses = {name: loss for name, loss, _share in active}
    for name, effective_loss, _share, _norm in differentiable:
        effective_losses[name] = effective_loss
    for name, loss, _share in active:
        rejected = (
            skip_negative_cosine and
            f"{name}_cosine" in scales and
            float(scales[f"{name}_cosine"]) < 0.0
        )
        scale = (
            loss.detach().new_tensor(0.0) if rejected else
            scales.get(name, loss.detach().new_tensor(1.0))
        )
        scales.setdefault(name, scale)
        total = total + scale.detach() * effective_losses[name]
    return total, scales


_natural_anchor_mema_ema: Dict[str, float] = {}


def per_branch_mema_gate(
    branch_losses: Dict[str, torch.Tensor],
    branch_anchors: Dict[str, Optional[torch.Tensor]],
    det_loss: torch.Tensor,
    decay: float = 0.90,
    floor: float = 0.15,
    ceiling: float = 1.0,
    project_conflicts: bool = False,
    max_gradient_ratio: float = 0.0,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
    """Control each KD branch at the tensor where it actually attaches.

    V50 introduced the punish-only MEMA scalar because low-level feature and
    depth losses do not reach the final BEV tensor used by the common gradient
    balancer.  V54 additionally performs PCGrad at that natural attachment
    point and can cap the KD gradient relative to the detection gradient.  The
    gradient replacement is value preserving, so logging and the outer KD
    budget still observe the original scaled loss.
    """
    global _natural_anchor_mema_ema
    scaled: Dict[str, torch.Tensor] = {}
    diagnostics: Dict[str, float] = {}

    for name, loss in branch_losses.items():
        anchor = branch_anchors.get(name)
        if (
            anchor is None
            or not loss.requires_grad
            or anchor.ndim < 2
            or not anchor.requires_grad
        ):
            scaled[name] = loss
            diagnostics[f"{name}_cosine"] = 1.0
            diagnostics[f"{name}_mema_scale"] = 1.0
            diagnostics[f"{name}_projected"] = 0.0
            diagnostics[f"{name}_gradient_ratio"] = 0.0
            continue

        kd_grad = torch.autograd.grad(
            loss, anchor, retain_graph=True, create_graph=False,
            allow_unused=True,
        )[0]
        det_grad = torch.autograd.grad(
            det_loss, anchor, retain_graph=True, create_graph=False,
            allow_unused=True,
        )[0]
        if kd_grad is None or det_grad is None:
            scaled[name] = loss
            diagnostics[f"{name}_cosine"] = 1.0
            diagnostics[f"{name}_mema_scale"] = 1.0
            diagnostics[f"{name}_projected"] = 0.0
            diagnostics[f"{name}_gradient_ratio"] = 0.0
            continue

        kd_grad_float = kd_grad.detach().float()
        det_grad_float = det_grad.detach().float()
        kd_flat = kd_grad_float.flatten()
        det_flat = det_grad_float.flatten()
        kd_norm = kd_flat.norm(p=2).clamp(min=1e-12)
        det_norm = det_flat.norm(p=2).clamp(min=1e-12)
        dot = torch.dot(kd_flat, det_flat)
        cosine = float(dot / (kd_norm * det_norm))

        ema_key = f"natural_anchor_{name}"
        if ema_key not in _natural_anchor_mema_ema:
            _natural_anchor_mema_ema[ema_key] = cosine
        else:
            _natural_anchor_mema_ema[ema_key] = (
                decay * _natural_anchor_mema_ema[ema_key] +
                (1.0 - decay) * cosine
            )
        baseline = _natural_anchor_mema_ema[ema_key]
        if cosine < 0.0:
            mema_scale = floor
        elif baseline > 1e-6 and cosine < baseline:
            mema_scale = max(floor, min(ceiling, cosine / baseline))
        else:
            mema_scale = ceiling

        controlled_grad = kd_grad_float * float(mema_scale)
        projected = 0.0
        if project_conflicts and dot < 0:
            controlled_grad = controlled_grad - (
                (controlled_grad.flatten().dot(det_flat) /
                 det_flat.square().sum().clamp(min=1e-12)) * det_grad_float
            )
            projected = 1.0

        controlled_norm = controlled_grad.flatten().norm(p=2).clamp(min=1e-12)
        if max_gradient_ratio > 0.0:
            maximum_norm = det_norm * float(max_gradient_ratio)
            controlled_grad = controlled_grad * torch.minimum(
                controlled_norm.new_tensor(1.0), maximum_norm / controlled_norm,
            )
            controlled_norm = controlled_grad.flatten().norm(p=2).clamp(min=1e-12)

        diagnostics[f"{name}_cosine"] = cosine
        diagnostics[f"{name}_mema_scale"] = float(mema_scale)
        diagnostics[f"{name}_projected"] = projected
        diagnostics[f"{name}_gradient_ratio"] = float(controlled_norm / det_norm)

        value = loss * float(mema_scale)
        if project_conflicts or max_gradient_ratio > 0.0:
            surrogate = (anchor.float() * controlled_grad).sum()
            scaled[name] = value.detach() + surrogate - surrogate.detach()
        else:
            scaled[name] = value

    return scaled, diagnostics


def multiscale_region_feature_kd(
    student_feature: torch.Tensor,
    teacher_feature: torch.Tensor,
    fov_mask: torch.Tensor,
    foreground_mask: torch.Tensor,
    false_positive_mask: torch.Tensor,
    scales: Sequence[int] = (1, 2, 4),
    foreground_weight: float = 1.0,
    false_positive_weight: float = 0.5,
    background_weight: float = 0.05,
    attention_strength: float = 1.0,
) -> torch.Tensor:
    """Region-balanced multi-scale BEV feature distillation.

    This compact adaptation follows DistillBEV's central observation that BEV
    foreground, false-positive and empty-background regions should not be
    treated equally.  Pooling the same aligned BEV representation at several
    scales transfers both local object detail and wider spatial context.
    """
    if student_feature.shape != teacher_feature.shape:
        raise ValueError(
            f"feature shapes differ: {tuple(student_feature.shape)} vs "
            f"{tuple(teacher_feature.shape)}"
        )
    if student_feature.ndim != 4:
        raise ValueError("BEV features must be BCHW tensors")
    valid_scales = tuple(dict.fromkeys(int(scale) for scale in scales))
    if not valid_scales or any(scale <= 0 for scale in valid_scales):
        raise ValueError("feature KD scales must contain positive integers")

    student_feature = student_feature.float()
    teacher_feature = teacher_feature.detach().float()
    fov_mask = fov_mask.float().clamp(0.0, 1.0)
    foreground_mask = foreground_mask.float().clamp(0.0, 1.0) * fov_mask
    false_positive_mask = (
        false_positive_mask.float().clamp(0.0, 1.0) *
        fov_mask * (1.0 - foreground_mask)
    )
    background_mask = (
        fov_mask - torch.maximum(foreground_mask, false_positive_mask)
    ).clamp(0.0, 1.0)

    total = student_feature.new_tensor(0.0)
    for scale in valid_scales:
        if scale == 1:
            student_level, teacher_level = student_feature, teacher_feature
            fov_level = fov_mask
            foreground_level = foreground_mask
            false_positive_level = false_positive_mask
            background_level = background_mask
        else:
            pool = dict(kernel_size=scale, stride=scale)
            student_level = F.avg_pool2d(student_feature, **pool)
            teacher_level = F.avg_pool2d(teacher_feature, **pool)
            fov_level = F.avg_pool2d(fov_mask, **pool)
            foreground_level = F.avg_pool2d(foreground_mask, **pool)
            false_positive_level = F.avg_pool2d(false_positive_mask, **pool)
            background_level = F.avg_pool2d(background_mask, **pool)

        region_weight = (
            float(foreground_weight) * foreground_level +
            float(false_positive_weight) * false_positive_level +
            float(background_weight) * background_level
        ) * fov_level
        teacher_attention = teacher_level.abs().mean(dim=1, keepdim=True)
        attention_mean = masked_mean(teacher_attention, fov_level).detach().clamp(min=1e-6)
        teacher_attention = (teacher_attention / attention_mean).clamp(max=4.0)
        region_weight = region_weight * (
            1.0 + float(attention_strength) * teacher_attention
        )
        student_normalized = F.normalize(student_level, dim=1, eps=1e-6)
        teacher_normalized = F.normalize(teacher_level, dim=1, eps=1e-6)
        level_loss = F.smooth_l1_loss(
            student_normalized, teacher_normalized, reduction="none", beta=0.5,
        )
        total = total + masked_mean(level_loss, region_weight)
    return total / float(len(valid_scales))


def bridge_relation_target(
    assistant_relations: torch.Tensor,
    lidar_relations: torch.Tensor,
    residual_alpha: float,
) -> torch.Tensor:
    """Inject only a controlled LiDAR residual into the camera-domain relation."""
    if assistant_relations.shape != lidar_relations.shape:
        raise ValueError("assistant and LiDAR relation shapes differ")
    alpha = float(residual_alpha)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("relation residual alpha must be in [0, 1]")
    assistant_relations = assistant_relations.detach().float()
    lidar_relations = lidar_relations.detach().float()
    return assistant_relations + alpha * (lidar_relations - assistant_relations)


def build_oriented_foreground_extent_mask(
    regression_target: torch.Tensor,
    regression_mask: torch.Tensor,
    bev_resolution: float,
    ellipse_scale: float = 1.0,
) -> torch.Tensor:
    """Rasterize full oriented GT extents as soft elliptical BEV foreground.

    The detector heatmap intentionally supervises only object centres.  Region
    distillation needs a different target: the area occupied by the object.  We
    reconstruct each box from the sparse regression target and take the maximum
    of its oriented elliptical kernels.  This is training-only and adds no
    deployable parameters.
    """
    if regression_target.ndim != 4 or regression_target.shape[1] < 8:
        raise ValueError("regression_target must be [B, >=8, H, W]")
    if regression_mask.shape != regression_target[:, :1].shape:
        raise ValueError("regression_mask shape does not match regression target")
    if float(bev_resolution) <= 0 or float(ellipse_scale) <= 0:
        raise ValueError("BEV resolution and ellipse scale must be positive")

    batch, _channels, height, width = regression_target.shape
    result = regression_target.new_zeros((batch, 1, height, width), dtype=torch.float32)
    target = regression_target.detach().float()
    active = regression_mask[:, 0].detach() > 0.5
    for batch_index in range(batch):
        centres = active[batch_index].nonzero(as_tuple=False)
        for centre in centres:
            ix, iy = (int(centre[0]), int(centre[1]))
            values = target[batch_index, :, ix, iy]
            centre_x = float(ix) + float(values[0])
            centre_y = float(iy) + float(values[1])
            box_width = max(
                float(torch.exp(values[3]).clamp(max=100.0)) /
                float(bev_resolution), 1.0,
            )
            box_length = max(
                float(torch.exp(values[4]).clamp(max=100.0)) /
                float(bev_resolution), 1.0,
            )
            sine, cosine = float(values[6]), float(values[7])
            yaw_norm = max((sine * sine + cosine * cosine) ** 0.5, 1e-6)
            sine, cosine = sine / yaw_norm, cosine / yaw_norm
            radius_x = max(0.5 * box_length * float(ellipse_scale), 0.75)
            radius_y = max(0.5 * box_width * float(ellipse_scale), 0.75)
            support_x = int(math.ceil(2.5 * max(radius_x, radius_y)))
            x0, x1 = max(0, ix - support_x), min(height, ix + support_x + 1)
            y0, y1 = max(0, iy - support_x), min(width, iy + support_x + 1)
            grid_x = torch.arange(x0, x1, device=target.device, dtype=torch.float32)
            grid_y = torch.arange(y0, y1, device=target.device, dtype=torch.float32)
            delta_x = grid_x[:, None] - centre_x
            delta_y = grid_y[None, :] - centre_y
            local_length = cosine * delta_x + sine * delta_y
            local_width = -sine * delta_x + cosine * delta_y
            squared_radius = (
                (local_length / radius_x) ** 2 +
                (local_width / radius_y) ** 2
            )
            kernel = torch.exp(-0.5 * squared_radius) * (squared_radius <= 6.25)
            result[batch_index, 0, x0:x1, y0:y1] = torch.maximum(
                result[batch_index, 0, x0:x1, y0:y1], kernel,
            )
    return result


def resolve_target_scope(args) -> str:
    scope = getattr(args, "target_scope", "auto")
    if scope != "auto":
        return scope
    return "student_fov" if getattr(args, "cmd", "") == "train-student" else "full"


def apply_sample_manifest(
    sample_tokens: Sequence[str],
    sample_camera_paths: Sequence[Sequence[str]],
    manifest_path: str,
) -> Tuple[List[str], List[List[str]]]:
    """Select a deterministic token manifest while preserving split order."""
    path = Path(manifest_path)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    requested = payload.get("sample_tokens") if isinstance(payload, dict) else payload
    if not isinstance(requested, list) or not requested:
        raise ValueError(f"sample manifest must contain a non-empty sample_tokens list: {path}")
    requested = [str(token) for token in requested]
    if len(requested) != len(set(requested)):
        raise ValueError(f"sample manifest contains duplicate tokens: {path}")

    available = set(sample_tokens)
    missing = sorted(set(requested) - available)
    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(
            f"sample manifest contains {len(missing)} missing split tokens: {preview}"
        )
    keep = set(requested)
    selected_tokens: List[str] = []
    selected_paths: List[List[str]] = []
    for token, paths in zip(sample_tokens, sample_camera_paths):
        if token in keep:
            selected_tokens.append(token)
            selected_paths.append(list(paths))
    return selected_tokens, selected_paths


class NuScenesOfficialDetectionDataset(Dataset):
    def __init__(self, mod, dataroot: str, version: str, split: str,
                 transform_teacher=None, transform_student=None,
                 max_samples: int = 0, verbose: bool = False,
                 build_depth: bool = True,
                 need_teacher_inputs: bool = True,
                 need_student_inputs: bool = True,
                 need_teacher_depth: Optional[bool] = None,
                 need_student_depth: Optional[bool] = None,
                 need_temporal_inputs: bool = False,
                 need_history_targets: bool = False,
                 build_detection_targets: bool = True,
                 target_scope: str = "full",
                 student_view_mode: str = "front",
                 student_camera: str = "CAM_FRONT",
                 pseudo_labels: str = "",
                 pseudo_score_thr: float = 0.2,
                 pseudo_class_score_thrs: Optional[Sequence[float]] = None,
                 subset_mode: str = "first",
                 subset_seed: int = 0,
                 subset_manifest: str = "",
                 student_cameras: Optional[Sequence[str]] = None,
                 need_lidar_relation_inputs: bool = False,
                 lidar_relation_keypoints_only: bool = False,
                 lidar_relation_max_points: int = 35000,
                 lidar_relation_max_objects: int = 128,
                 lidar_relation_cache: str = ""):
        self.mod = mod
        self.dataroot = dataroot
        self.version = version
        self.split = split
        self.transform_teacher = transform_teacher
        self.transform_student = transform_student
        self.target_scope = target_scope
        self.need_teacher_inputs = need_teacher_inputs
        self.need_student_inputs = need_student_inputs
        self.need_teacher_depth = build_depth if need_teacher_depth is None else need_teacher_depth
        self.need_student_depth = build_depth if need_student_depth is None else need_student_depth
        self.need_temporal_inputs = bool(need_temporal_inputs)
        self.need_history_targets = bool(need_history_targets)
        if self.need_history_targets and not self.need_temporal_inputs:
            raise ValueError("history targets require temporal inputs")
        self.build_detection_targets = bool(build_detection_targets)
        self.need_lidar_relation_inputs = bool(need_lidar_relation_inputs)
        self.lidar_relation_keypoints_only = bool(lidar_relation_keypoints_only)
        self.lidar_relation_max_points = int(lidar_relation_max_points)
        self.lidar_relation_max_objects = int(lidar_relation_max_objects)
        self.lidar_relation_cache_path = str(lidar_relation_cache or "")
        self.lidar_relation_cache: Optional[Dict[str, np.ndarray]] = None
        if self.need_lidar_relation_inputs:
            if self.lidar_relation_max_points <= 0:
                raise ValueError("lidar_relation_max_points must be positive")
            if self.lidar_relation_max_objects <= 0:
                raise ValueError("lidar_relation_max_objects must be positive")
        if student_view_mode not in {"front", "balanced", "all"}:
            raise ValueError(f"unsupported student_view_mode={student_view_mode}")
        if student_camera not in mod.CAM_CHANNELS:
            raise ValueError(f"unsupported student_camera={student_camera}")
        self.student_view_mode = student_view_mode
        self.student_camera = student_camera
        requested_cameras = list(student_cameras or mod.CAM_CHANNELS)
        invalid_cameras = sorted(set(requested_cameras) - set(mod.CAM_CHANNELS))
        if invalid_cameras:
            raise ValueError(f"unsupported student cameras: {invalid_cameras}")
        if len(requested_cameras) != len(set(requested_cameras)):
            raise ValueError(f"duplicate student cameras: {requested_cameras}")
        if not requested_cameras:
            raise ValueError("student_cameras must not be empty")
        self.student_camera_indices = [
            mod.CAM_CHANNELS.index(camera) for camera in requested_cameras
        ]
        self.pseudo_score_thr = float(pseudo_score_thr)
        if pseudo_class_score_thrs is None:
            self.pseudo_class_score_thrs = [self.pseudo_score_thr] * len(DETECTION_NAMES)
        else:
            if len(pseudo_class_score_thrs) != len(DETECTION_NAMES):
                raise ValueError(
                    f"pseudo_class_score_thrs must contain {len(DETECTION_NAMES)} values"
                )
            self.pseudo_class_score_thrs = [float(value) for value in pseudo_class_score_thrs]
        if subset_mode not in {"first", "random"}:
            raise ValueError(f"unsupported subset_mode={subset_mode}")
        self.pseudo_results = None
        if pseudo_labels:
            pseudo_path = Path(pseudo_labels)
            if not pseudo_path.exists():
                raise FileNotFoundError(f"pseudo-label file not found: {pseudo_path}")
            with pseudo_path.open("r", encoding="utf-8") as f:
                pseudo_payload = json.load(f)
            self.pseudo_results = pseudo_payload.get("results", pseudo_payload)
            if not isinstance(self.pseudo_results, dict):
                raise TypeError("pseudo-label file must contain a token-to-boxes mapping")
            print(
                f"[PSEUDO] loaded {len(self.pseudo_results)} sample tokens from {pseudo_path} "
                f"score_thr={self.pseudo_score_thr} class_score_thrs="
                f"{dict(zip(DETECTION_NAMES, self.pseudo_class_score_thrs))}"
            )
        self._ann_visibility_cache: Dict[Tuple[str, str], bool] = {}
        self._fov_mask_cache: Dict[Tuple, np.ndarray] = {}
        self.nusc = mod.NuScenes(version=version, dataroot=dataroot, verbose=verbose)
        self.sample_tokens: List[str] = []
        self.sample_camera_paths: List[List[str]] = []

        split_scenes = set(mod.create_splits_scenes()[split])
        print(f"[INFO] Building {split} split for official detection ...")
        for scene in self.nusc.scene:
            if scene["name"] not in split_scenes:
                continue
            token = scene["first_sample_token"]
            while token:
                sample = self.nusc.get("sample", token)
                img_paths = []
                ok = True
                for cam in mod.CAM_CHANNELS:
                    if cam not in sample["data"]:
                        ok = False
                        break
                    sd = self.nusc.get("sample_data", sample["data"][cam])
                    path = os.path.join(dataroot, sd["filename"])
                    if not os.path.exists(path):
                        ok = False
                        break
                    img_paths.append(path)
                if ok:
                    self.sample_tokens.append(sample["token"])
                    self.sample_camera_paths.append(img_paths)
                token = sample["next"]
        if subset_manifest:
            self.sample_tokens, self.sample_camera_paths = apply_sample_manifest(
                self.sample_tokens, self.sample_camera_paths, subset_manifest,
            )
        if max_samples and len(self.sample_tokens) > max_samples:
            if subset_mode == "random":
                rng = np.random.default_rng(int(subset_seed))
                selected = rng.choice(len(self.sample_tokens), size=int(max_samples), replace=False).tolist()
            else:
                selected = list(range(int(max_samples)))
            self.sample_tokens = [self.sample_tokens[index] for index in selected]
            self.sample_camera_paths = [self.sample_camera_paths[index] for index in selected]
        print(f"[INFO] {split}: valid samples = {len(self.sample_tokens)} target_scope={self.target_scope}")
        if self.need_lidar_relation_inputs and self.lidar_relation_cache_path:
            cache_path = Path(self.lidar_relation_cache_path)
            if not cache_path.is_file():
                raise FileNotFoundError(f"LiDAR relation cache not found: {cache_path}")
            connection = sqlite3.connect(str(cache_path))
            try:
                sample_set = set(self.sample_tokens)
                loaded: Dict[str, np.ndarray] = {}
                for sample_token, ann_token, blob in connection.execute(
                    "SELECT sample_token, ann_token, relation FROM relations"
                ):
                    if sample_token in sample_set:
                        loaded[str(ann_token)] = np.frombuffer(
                            blob, dtype=np.float16,
                        ).reshape(9, 9).astype(np.float32)
                self.lidar_relation_cache = loaded
            finally:
                connection.close()
            if not self.lidar_relation_cache:
                raise RuntimeError(
                    f"relation cache has no entries for {split} subset: {cache_path}"
                )
            print(
                f"[RELATION_CACHE] loaded {len(self.lidar_relation_cache)} objects "
                f"for {split} from {cache_path}"
            )
        self.student_cam_idx = mod.CAM_CHANNELS.index(student_camera)

    def __len__(self):
        if self.student_view_mode == "all":
            return len(self.sample_tokens) * len(self.student_camera_indices)
        return len(self.sample_tokens)

    def _resolve_item_index(self, idx):
        if isinstance(idx, (tuple, list)):
            sample_idx, cam_idx = int(idx[0]), int(idx[1])
        elif self.student_view_mode == "all":
            sample_idx, local_cam_idx = divmod(
                int(idx), len(self.student_camera_indices)
            )
            cam_idx = self.student_camera_indices[local_cam_idx]
        else:
            sample_idx, cam_idx = int(idx), self.student_cam_idx
        if cam_idx < 0 or cam_idx >= len(self.mod.CAM_CHANNELS):
            raise IndexError(f"camera index out of range: {cam_idx}")
        return sample_idx, cam_idx

    def _ann_visible_in_camera(self, sample, ann, cam_name: str) -> bool:
        key = (ann["token"], cam_name)
        if key in self._ann_visibility_cache:
            return self._ann_visibility_cache[key]

        sd = self.nusc.get("sample_data", sample["data"][cam_name])
        cs = self.nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
        pose = self.nusc.get("ego_pose", sd["ego_pose_token"])
        box = Box(
            center=ann["translation"],
            size=ann["size"],
            orientation=PyQuaternion(ann["rotation"]),
        )
        box.translate(-np.array(pose["translation"]))
        box.rotate(PyQuaternion(pose["rotation"]).inverse)
        box.translate(-np.array(cs["translation"]))
        box.rotate(PyQuaternion(cs["rotation"]).inverse)
        intrinsic = np.array(cs["camera_intrinsic"], dtype=np.float32)
        imsize = (int(sd["width"]), int(sd["height"]))
        visible = bool(box_in_image(box, intrinsic, imsize, vis_level=BoxVisibility.ANY))
        self._ann_visibility_cache[key] = visible
        return visible

    def _make_student_fov_mask(self, sample, cam_name: str):
        mod = self.mod
        sd = self.nusc.get("sample_data", sample["data"][cam_name])
        cs = self.nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
        key = (
            sd["calibrated_sensor_token"], mod.BEV_X_MIN, mod.BEV_X_MAX,
            mod.BEV_Y_MIN, mod.BEV_Y_MAX, mod.BEV_RES, mod.BEV_H, mod.BEV_W,
            int(sd["width"]), int(sd["height"]),
        )
        if key in self._fov_mask_cache:
            return self._fov_mask_cache[key].copy()

        xs = mod.BEV_X_MIN + (np.arange(mod.BEV_H, dtype=np.float32) + 0.5) * mod.BEV_RES
        ys = mod.BEV_Y_MIN + (np.arange(mod.BEV_W, dtype=np.float32) + 0.5) * mod.BEV_RES
        xx, yy = np.meshgrid(xs, ys, indexing="ij")
        r_cam_to_ego = PyQuaternion(cs["rotation"]).rotation_matrix.astype(np.float32)
        t_cam_to_ego = np.array(cs["translation"], dtype=np.float32).reshape(3, 1)
        k = np.array(cs["camera_intrinsic"], dtype=np.float32)
        mask = np.zeros((mod.BEV_H, mod.BEV_W), dtype=bool)
        for z_value in (0.0, 0.75, 1.5):
            zz = np.full_like(xx, z_value, dtype=np.float32)
            pts_ego = np.stack([xx.reshape(-1), yy.reshape(-1), zz.reshape(-1)], axis=0)
            pts_cam = r_cam_to_ego.T.dot(pts_ego - t_cam_to_ego)
            depth_ok = pts_cam[2] > 0.1
            pts_img = mod.view_points(pts_cam, k, normalize=True)
            u = pts_img[0]
            v = pts_img[1]
            in_img = (
                depth_ok & (u >= 0) & (u < float(sd["width"])) &
                (v >= 0) & (v < float(sd["height"]))
            )
            mask |= in_img.reshape(mod.BEV_H, mod.BEV_W)
        mask = cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), dtype=np.uint8), iterations=1).astype(np.float32)
        mask = mask[None]
        self._fov_mask_cache[key] = mask
        return mask.copy()

    def _make_det_targets(self, sample, cam_name: str):
        mod = self.mod
        hm = np.zeros((len(DETECTION_NAMES), mod.BEV_H, mod.BEV_W), dtype=np.float32)
        reg = np.zeros((10, mod.BEV_H, mod.BEV_W), dtype=np.float32)
        reg_mask = np.zeros((1, mod.BEV_H, mod.BEV_W), dtype=np.float32)
        vel_mask = np.zeros((1, mod.BEV_H, mod.BEV_W), dtype=np.float32)
        attr_target = np.full((mod.BEV_H, mod.BEV_W), -1, dtype=np.int64)

        ref_sd = self.nusc.get("sample_data", sample["data"][cam_name])
        ego_pose = self.nusc.get("ego_pose", ref_sd["ego_pose_token"])
        ego_t = np.array(ego_pose["translation"], dtype=np.float32)
        ego_R = PyQuaternion(ego_pose["rotation"]).rotation_matrix.astype(np.float32)
        ego_yaw = yaw_from_quat(ego_pose["rotation"])

        for ann_token in sample["anns"]:
            ann = self.nusc.get("sample_annotation", ann_token)
            det_name = CATEGORY_TO_DETECTION.get(ann["category_name"])
            if det_name is None:
                continue
            if ann.get("num_lidar_pts", 1) + ann.get("num_radar_pts", 1) <= 0:
                continue
            if self.target_scope == "student_fov" and not self._ann_visible_in_camera(sample, ann, cam_name):
                continue

            cls_id = NAME_TO_ID[det_name]
            pos_global = np.array(ann["translation"], dtype=np.float32)
            pos_ego = ego_R.T.dot(pos_global - ego_t)
            x, y, z = float(pos_ego[0]), float(pos_ego[1]), float(pos_ego[2])
            if not (mod.BEV_X_MIN <= x < mod.BEV_X_MAX and mod.BEV_Y_MIN <= y < mod.BEV_Y_MAX):
                continue

            w, l, h = [float(v) for v in ann["size"]]
            if w <= 0 or l <= 0 or h <= 0:
                continue
            yaw_ego = wrap_pi(yaw_from_quat(ann["rotation"]) - ego_yaw)
            gx = (x - mod.BEV_X_MIN) / mod.BEV_RES
            gy = (y - mod.BEV_Y_MIN) / mod.BEV_RES
            ix, iy = int(gx), int(gy)
            if ix < 0 or ix >= mod.BEV_H or iy < 0 or iy >= mod.BEV_W:
                continue

            radius = min(gaussian_radius(l / mod.BEV_RES, w / mod.BEV_RES, 0.7), 12)
            draw_gaussian(hm[cls_id], (ix, iy), radius)

            vx, vy = self.nusc.box_velocity(ann_token)[:2]
            velocity_valid = np.isfinite(vx) and np.isfinite(vy)
            if velocity_valid:
                vel_ego = ego_R.T.dot(np.array([vx, vy, 0.0], dtype=np.float32))
                vx_ego, vy_ego = float(vel_ego[0]), float(vel_ego[1])
            else:
                vx_ego, vy_ego = 0.0, 0.0

            reg[:, ix, iy] = np.array([
                gx - ix, gy - iy, z,
                math.log(max(w, 1e-3)), math.log(max(l, 1e-3)), math.log(max(h, 1e-3)),
                math.sin(yaw_ego), math.cos(yaw_ego),
                vx_ego, vy_ego,
            ], dtype=np.float32)
            reg_mask[0, ix, iy] = 1.0
            vel_mask[0, ix, iy] = 1.0 if velocity_valid else 0.0

            attribute_tokens = ann.get("attribute_tokens", [])
            if attribute_tokens:
                attribute_name = self.nusc.get("attribute", attribute_tokens[0])["name"]
                attr_target[ix, iy] = ATTRIBUTE_TO_ID.get(attribute_name, -1)

        return hm, reg, reg_mask, vel_mask, attr_target

    def _pseudo_box_visible_in_camera(self, sample, prediction, cam_name: str) -> bool:
        sd = self.nusc.get("sample_data", sample["data"][cam_name])
        cs = self.nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
        pose = self.nusc.get("ego_pose", sd["ego_pose_token"])
        box = Box(
            center=prediction["translation"],
            size=prediction["size"],
            orientation=PyQuaternion(prediction["rotation"]),
        )
        box.translate(-np.asarray(pose["translation"], dtype=np.float64))
        box.rotate(PyQuaternion(pose["rotation"]).inverse)
        box.translate(-np.asarray(cs["translation"], dtype=np.float64))
        box.rotate(PyQuaternion(cs["rotation"]).inverse)
        intrinsic = np.asarray(cs["camera_intrinsic"], dtype=np.float64)
        imsize = (int(sd["width"]), int(sd["height"]))
        return bool(box_in_image(box, intrinsic, imsize, vis_level=BoxVisibility.ANY))

    def _make_pseudo_targets(self, sample, cam_name: str):
        mod = self.mod
        hm = np.zeros((len(DETECTION_NAMES), mod.BEV_H, mod.BEV_W), dtype=np.float32)
        reg = np.zeros((10, mod.BEV_H, mod.BEV_W), dtype=np.float32)
        reg_mask = np.zeros((1, mod.BEV_H, mod.BEV_W), dtype=np.float32)
        score_map = np.zeros((1, mod.BEV_H, mod.BEV_W), dtype=np.float32)
        if self.pseudo_results is None:
            return hm, reg, reg_mask, score_map

        sd = self.nusc.get("sample_data", sample["data"][cam_name])
        pose = self.nusc.get("ego_pose", sd["ego_pose_token"])
        ego_t = np.asarray(pose["translation"], dtype=np.float32)
        ego_r = PyQuaternion(pose["rotation"]).rotation_matrix.astype(np.float32)
        ego_yaw = yaw_from_quat(pose["rotation"])

        for pred in self.pseudo_results.get(sample["token"], []):
            score = float(pred.get("detection_score", 0.0))
            det_name = pred.get("detection_name")
            if det_name not in NAME_TO_ID:
                continue
            class_id = NAME_TO_ID[det_name]
            if score < self.pseudo_class_score_thrs[class_id]:
                continue
            if not self._pseudo_box_visible_in_camera(sample, pred, cam_name):
                continue
            position_global = np.asarray(pred["translation"], dtype=np.float32)
            position_ego = ego_r.T.dot(position_global - ego_t)
            x, y, z = [float(value) for value in position_ego]
            if not (mod.BEV_X_MIN <= x < mod.BEV_X_MAX and mod.BEV_Y_MIN <= y < mod.BEV_Y_MAX):
                continue
            width, length, height = [float(value) for value in pred["size"]]
            if min(width, length, height) <= 0:
                continue
            yaw = wrap_pi(yaw_from_quat(pred["rotation"]) - ego_yaw)
            gx = (x - mod.BEV_X_MIN) / mod.BEV_RES
            gy = (y - mod.BEV_Y_MIN) / mod.BEV_RES
            ix, iy = int(gx), int(gy)
            if not (0 <= ix < mod.BEV_H and 0 <= iy < mod.BEV_W):
                continue
            radius = min(gaussian_radius(length / mod.BEV_RES, width / mod.BEV_RES, 0.7), 12)
            draw_gaussian_scaled(hm[class_id], (ix, iy), radius, score)

            velocity = pred.get("velocity", [0.0, 0.0])
            velocity_global = np.asarray([velocity[0], velocity[1], 0.0], dtype=np.float32)
            velocity_ego = ego_r.T.dot(velocity_global)
            if score >= score_map[0, ix, iy]:
                reg[:, ix, iy] = np.asarray([
                    gx - ix, gy - iy, z,
                    math.log(max(width, 1e-3)), math.log(max(length, 1e-3)), math.log(max(height, 1e-3)),
                    math.sin(yaw), math.cos(yaw), float(velocity_ego[0]), float(velocity_ego[1]),
                ], dtype=np.float32)
                reg_mask[0, ix, iy] = 1.0
                score_map[0, ix, iy] = score
        return hm, reg, reg_mask, score_map

    def _build_depth_gt(self, sample, cam_name: str, k_scaled_np: np.ndarray,
                        img_w: int, img_h: int, feat_w: int, feat_h: int):
        mod = self.mod
        depth_gt = np.full((feat_h, feat_w), -1, dtype=np.int64)
        depth_min = np.full((feat_h, feat_w), np.inf, dtype=np.float32)
        if "LIDAR_TOP" not in sample["data"]:
            return depth_gt

        lidar_sd = self.nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
        lidar_path = os.path.join(self.dataroot, lidar_sd["filename"])
        if not os.path.exists(lidar_path):
            return depth_gt

        lidar_cs = self.nusc.get("calibrated_sensor", lidar_sd["calibrated_sensor_token"])
        lidar_pose = self.nusc.get("ego_pose", lidar_sd["ego_pose_token"])
        pc = mod.LidarPointCloud.from_file(lidar_path)

        pc.rotate(PyQuaternion(lidar_cs["rotation"]).rotation_matrix)
        pc.translate(np.array(lidar_cs["translation"], dtype=np.float32))
        pc.rotate(PyQuaternion(lidar_pose["rotation"]).rotation_matrix)
        pc.translate(np.array(lidar_pose["translation"], dtype=np.float32))
        pts_global = pc.points[:3, :]

        cam_sd = self.nusc.get("sample_data", sample["data"][cam_name])
        cam_cs = self.nusc.get("calibrated_sensor", cam_sd["calibrated_sensor_token"])
        cam_pose = self.nusc.get("ego_pose", cam_sd["ego_pose_token"])

        r_ego_to_global = PyQuaternion(cam_pose["rotation"]).rotation_matrix
        t_ego_to_global = np.array(cam_pose["translation"], dtype=np.float32)
        pts_ego = r_ego_to_global.T.dot(pts_global - t_ego_to_global.reshape(3, 1))

        r_cam_to_ego = PyQuaternion(cam_cs["rotation"]).rotation_matrix
        t_cam_to_ego = np.array(cam_cs["translation"], dtype=np.float32)
        pts_cam = r_cam_to_ego.T.dot(pts_ego - t_cam_to_ego.reshape(3, 1))

        z = pts_cam[2, :]
        valid = z > 0.1
        if valid.sum() == 0:
            return depth_gt
        pts_cam = pts_cam[:, valid]
        z = z[valid].astype(np.float32)
        pts_img = mod.view_points(pts_cam, k_scaled_np, normalize=True)
        u, v = pts_img[0].astype(np.float32), pts_img[1].astype(np.float32)

        keep = (u >= 0) & (u < img_w) & (v >= 0) & (v < img_h) & (z >= mod.DEPTH_MIN) & (z <= mod.DEPTH_MAX)
        if keep.sum() == 0:
            return depth_gt
        u, v, z = u[keep], v[keep], z[keep]
        uf = (u / 8.0).astype(np.int32)
        vf = (v / 8.0).astype(np.int32)
        ok = (uf >= 0) & (uf < feat_w) & (vf >= 0) & (vf < feat_h)
        if ok.sum() == 0:
            return depth_gt
        uf, vf, z = uf[ok], vf[ok], z[ok]
        bin_idx = ((z - mod.DEPTH_MIN) / (mod.DEPTH_MAX - mod.DEPTH_MIN) * mod.NUM_DEPTH_BINS).astype(np.int32)
        bin_idx = np.clip(bin_idx, 0, mod.NUM_DEPTH_BINS - 1)
        for uu, vv, zz, bb in zip(uf, vf, z, bin_idx):
            if zz < depth_min[vv, uu]:
                depth_min[vv, uu] = zz
                depth_gt[vv, uu] = int(bb)
        return depth_gt

    def _make_previous_student_input(self, sample, cam_name: str, current_tensors):
        """Load the previous keyframe of the same camera and its ego-motion transform."""
        img1, k1, r1, t1 = current_tensors
        if not self.need_temporal_inputs or not sample.get("prev"):
            return (
                img1.clone(), k1.clone(), r1.clone(), t1.clone(),
                torch.eye(3, dtype=torch.float32), torch.zeros(3, dtype=torch.float32),
                torch.tensor(0.0, dtype=torch.float32), torch.tensor(0.5, dtype=torch.float32),
            )

        prev_sample = self.nusc.get("sample", sample["prev"])
        if cam_name not in prev_sample["data"]:
            return (
                img1.clone(), k1.clone(), r1.clone(), t1.clone(),
                torch.eye(3, dtype=torch.float32), torch.zeros(3, dtype=torch.float32),
                torch.tensor(0.0, dtype=torch.float32), torch.tensor(0.5, dtype=torch.float32),
            )
        prev_sd = self.nusc.get("sample_data", prev_sample["data"][cam_name])
        prev_path = os.path.join(self.dataroot, prev_sd["filename"])
        prev_bgr = cv2.imread(prev_path)
        if prev_bgr is None:
            raise FileNotFoundError(prev_path)
        prev_h, prev_w = prev_bgr.shape[:2]
        prev_rgb = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2RGB)
        prev_cs = self.nusc.get("calibrated_sensor", prev_sd["calibrated_sensor_token"])
        prev_k = np.asarray(prev_cs["camera_intrinsic"], dtype=np.float32).copy()
        prev_k[0, 0] *= self.mod.IMG_W / float(prev_w)
        prev_k[1, 1] *= self.mod.IMG_H / float(prev_h)
        prev_k[0, 2] *= self.mod.IMG_W / float(prev_w)
        prev_k[1, 2] *= self.mod.IMG_H / float(prev_h)
        prev_r = PyQuaternion(prev_cs["rotation"]).rotation_matrix.astype(np.float32)
        prev_t = np.asarray(prev_cs["translation"], dtype=np.float32)
        prev_img = self.transform_student(prev_rgb) if self.transform_student else torch.from_numpy(
            cv2.resize(prev_rgb, (self.mod.IMG_W, self.mod.IMG_H))
        ).permute(2, 0, 1).float() / 255.0

        current_sd = self.nusc.get("sample_data", sample["data"][cam_name])
        current_pose = self.nusc.get("ego_pose", current_sd["ego_pose_token"])
        prev_pose = self.nusc.get("ego_pose", prev_sd["ego_pose_token"])
        current_r_global = PyQuaternion(current_pose["rotation"]).rotation_matrix.astype(np.float32)
        prev_r_global = PyQuaternion(prev_pose["rotation"]).rotation_matrix.astype(np.float32)
        current_t_global = np.asarray(current_pose["translation"], dtype=np.float32)
        prev_t_global = np.asarray(prev_pose["translation"], dtype=np.float32)
        prev_to_current_r = current_r_global.T.dot(prev_r_global)
        prev_to_current_t = current_r_global.T.dot(prev_t_global - current_t_global)
        delta_t = max(1e-3, (float(current_sd["timestamp"]) - float(prev_sd["timestamp"])) / 1e6)
        return (
            prev_img, torch.from_numpy(prev_k).float(), torch.from_numpy(prev_r).float(),
            torch.from_numpy(prev_t).float(), torch.from_numpy(prev_to_current_r).float(),
            torch.from_numpy(prev_to_current_t).float(), torch.tensor(1.0, dtype=torch.float32),
            torch.tensor(delta_t, dtype=torch.float32),
        )

    def _make_history_targets(self, sample, cam_name: str):
        """Build previous-keyframe targets in the previous ego coordinate frame."""
        if sample.get("prev"):
            prev_sample = self.nusc.get("sample", sample["prev"])
            if cam_name in prev_sample.get("data", {}):
                hm, reg, reg_mask, vel_mask, _attr = self._make_det_targets(prev_sample, cam_name)
                return hm, reg, reg_mask, vel_mask
        return (
            np.zeros((len(DETECTION_NAMES), self.mod.BEV_H, self.mod.BEV_W), dtype=np.float32),
            np.zeros((10, self.mod.BEV_H, self.mod.BEV_W), dtype=np.float32),
            np.zeros((1, self.mod.BEV_H, self.mod.BEV_W), dtype=np.float32),
            np.zeros((1, self.mod.BEV_H, self.mod.BEV_W), dtype=np.float32),
        )

    def _make_lidar_relation_inputs(self, sample, cam_name: str):
        """Build current LiDAR input and paired ego/LiDAR GT box keypoints.

        The nine points are the box centre, four inset corners and four inset
        edge midpoints.  Each point is represented in the selected camera's
        ego frame for sampling the student BEV and in the LIDAR_TOP frame for
        sampling CenterPoint.  Only GT objects visible in the selected camera
        are included, preventing 360-degree LiDAR context from supervising an
        unobservable single-view region.
        """
        max_points = self.lidar_relation_max_points
        max_objects = self.lidar_relation_max_objects
        cached_relations = None
        cached_relation_mask = None
        # V45 consumes only the paired GT instance keypoints.  Do not read or
        # batch the raw point cloud in that path: the privileged teacher gets
        # its sparse depth through the regular selected-camera depth input.
        point_slots = 1 if self.lidar_relation_keypoints_only else max_points
        points_padded = np.zeros((point_slots, 5), dtype=np.float32)
        paired_keypoints = np.zeros((max_objects, 9, 4), dtype=np.float32)
        object_mask = np.zeros((max_objects,), dtype=np.float32)

        lidar_sd = self.nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
        lidar_path = os.path.join(self.dataroot, lidar_sd["filename"])
        if self.lidar_relation_keypoints_only:
            point_count = 0
        elif self.lidar_relation_cache is None:
            raw_points = np.fromfile(lidar_path, dtype=np.float32)
            if raw_points.size % 5:
                raise RuntimeError(f"invalid nuScenes LiDAR file: {lidar_path}")
            raw_points = raw_points.reshape(-1, 5)
            if len(raw_points) > max_points:
                raise RuntimeError(
                    f"LiDAR frame has {len(raw_points)} points, exceeding audited "
                    f"padding limit {max_points}: {lidar_path}"
                )
            point_count = len(raw_points)
            points_padded[:point_count] = raw_points
            # The public CenterPoint checkpoint interprets channel 4 as time lag.
            points_padded[:point_count, 4] = 0.0
        else:
            point_count = 0
            cached_relations = np.zeros((max_objects, 9, 9), dtype=np.float32)
            cached_relation_mask = np.zeros((max_objects,), dtype=np.float32)

        cam_sd = self.nusc.get("sample_data", sample["data"][cam_name])
        cam_pose = self.nusc.get("ego_pose", cam_sd["ego_pose_token"])
        cam_global_r = PyQuaternion(cam_pose["rotation"]).rotation_matrix.astype(np.float32)
        cam_global_t = np.asarray(cam_pose["translation"], dtype=np.float32)

        lidar_pose = self.nusc.get("ego_pose", lidar_sd["ego_pose_token"])
        lidar_global_r = PyQuaternion(lidar_pose["rotation"]).rotation_matrix.astype(np.float32)
        lidar_global_t = np.asarray(lidar_pose["translation"], dtype=np.float32)
        lidar_cs = self.nusc.get("calibrated_sensor", lidar_sd["calibrated_sensor_token"])
        lidar_to_ego_r = PyQuaternion(lidar_cs["rotation"]).rotation_matrix.astype(np.float32)
        lidar_to_ego_t = np.asarray(lidar_cs["translation"], dtype=np.float32)

        object_index = 0
        for ann_token in sample["anns"]:
            ann = self.nusc.get("sample_annotation", ann_token)
            if CATEGORY_TO_DETECTION.get(ann["category_name"]) is None:
                continue
            if ann.get("num_lidar_pts", 1) + ann.get("num_radar_pts", 1) <= 0:
                continue
            if not self._ann_visible_in_camera(sample, ann, cam_name):
                continue
            if object_index >= max_objects:
                raise RuntimeError(
                    f"visible object count exceeds {max_objects} for sample {sample['token']}"
                )

            width, length, _height = [float(value) for value in ann["size"]]
            # Inset the perimeter points so they sample target interior rather
            # than unstable box boundaries after BEV downsampling.
            half_length = 0.4 * length
            half_width = 0.4 * width
            local_xy = np.asarray([
                [0.0, 0.0],
                [half_length, half_width], [half_length, -half_width],
                [-half_length, half_width], [-half_length, -half_width],
                [half_length, 0.0], [-half_length, 0.0],
                [0.0, half_width], [0.0, -half_width],
            ], dtype=np.float32)
            local_xyz = np.concatenate(
                [local_xy, np.zeros((len(local_xy), 1), dtype=np.float32)], axis=1,
            )
            box_global_r = PyQuaternion(ann["rotation"]).rotation_matrix.astype(np.float32)
            box_global_t = np.asarray(ann["translation"], dtype=np.float32)
            points_global = local_xyz.dot(box_global_r.T) + box_global_t

            points_cam_ego = (points_global - cam_global_t).dot(cam_global_r)
            points_lidar_ego = (points_global - lidar_global_t).dot(lidar_global_r)
            points_lidar = (points_lidar_ego - lidar_to_ego_t).dot(lidar_to_ego_r)
            paired_keypoints[object_index, :, :2] = points_cam_ego[:, :2]
            paired_keypoints[object_index, :, 2:] = points_lidar[:, :2]
            object_mask[object_index] = 1.0
            if self.lidar_relation_cache is not None:
                relation = self.lidar_relation_cache.get(ann_token)
                if relation is None:
                    raise RuntimeError(
                        f"missing relation cache entry for annotation {ann_token}"
                    )
                cached_relations[object_index] = relation
                cached_relation_mask[object_index] = 1.0
            object_index += 1

        if cached_relations is not None:
            return cached_relations, cached_relation_mask, paired_keypoints, object_mask
        return points_padded, point_count, paired_keypoints, object_mask

    def __getitem__(self, idx):
        mod = self.mod
        sample_idx, si = self._resolve_item_index(idx)
        token = self.sample_tokens[sample_idx]
        sample = self.nusc.get("sample", token)

        imgs6_tch, ks6_tch, r6, t6 = [], [], [], []
        depth6 = []
        img1 = k1 = r1 = t1 = depth1 = None
        student_cam = mod.CAM_CHANNELS[si]

        if self.need_teacher_inputs or self.need_teacher_depth:
            cam_items = list(zip(mod.CAM_CHANNELS, self.sample_camera_paths[sample_idx]))
        else:
            cam_items = [(student_cam, self.sample_camera_paths[sample_idx][si])]

        for cam, img_path in cam_items:
            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                raise FileNotFoundError(img_path)
            orig_h, orig_w = img_bgr.shape[:2]
            img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            sd = self.nusc.get("sample_data", sample["data"][cam])
            cs = self.nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
            k = np.array(cs["camera_intrinsic"], dtype=np.float32)
            r_cam_to_ego = PyQuaternion(cs["rotation"]).rotation_matrix.astype(np.float32)
            t_cam_to_ego = np.array(cs["translation"], dtype=np.float32)

            if self.need_teacher_inputs or self.need_teacher_depth:
                sx_t, sy_t = mod.TCH_W / float(orig_w), mod.TCH_H / float(orig_h)
                k_t = k.copy()
                k_t[0, 0] *= sx_t; k_t[1, 1] *= sy_t
                k_t[0, 2] *= sx_t; k_t[1, 2] *= sy_t
                if self.need_teacher_inputs:
                    img_tch = self.transform_teacher(img) if self.transform_teacher else torch.from_numpy(
                        cv2.resize(img, (mod.TCH_W, mod.TCH_H))).permute(2, 0, 1).float() / 255.0
                    imgs6_tch.append(img_tch)
                    ks6_tch.append(k_t)
                    r6.append(r_cam_to_ego)
                    t6.append(t_cam_to_ego)
                if self.need_teacher_depth:
                    depth6.append(self._build_depth_gt(sample, cam, k_t, mod.TCH_W, mod.TCH_H, mod.TCH_W // 8, mod.TCH_H // 8))

            if self.need_student_inputs and cam == student_cam:
                sx_s, sy_s = mod.IMG_W / float(orig_w), mod.IMG_H / float(orig_h)
                k_s = k.copy()
                k_s[0, 0] *= sx_s; k_s[1, 1] *= sy_s
                k_s[0, 2] *= sx_s; k_s[1, 2] *= sy_s
                img1 = self.transform_student(img) if self.transform_student else torch.from_numpy(
                    cv2.resize(img, (mod.IMG_W, mod.IMG_H))).permute(2, 0, 1).float() / 255.0
                k1 = torch.from_numpy(k_s).float()
                r1 = torch.from_numpy(r_cam_to_ego).float()
                t1 = torch.from_numpy(t_cam_to_ego).float()
                if self.need_student_depth:
                    depth1 = torch.from_numpy(self._build_depth_gt(sample, student_cam, k_s, mod.IMG_W, mod.IMG_H, mod.FEAT_W, mod.FEAT_H)).long()

        if self.build_detection_targets:
            hm, reg, reg_mask, vel_mask, attr_target = self._make_det_targets(sample, student_cam)
        else:
            hm = np.empty((0, 0, 0), dtype=np.float32)
            reg = np.empty((0, 0, 0), dtype=np.float32)
            reg_mask = np.empty((0, 0, 0), dtype=np.float32)
            vel_mask = np.empty((0, 0, 0), dtype=np.float32)
            attr_target = np.empty((0, 0), dtype=np.int64)
        bev_fov_mask = self._make_student_fov_mask(sample, student_cam)

        if self.need_teacher_inputs:
            imgs6_tensor = torch.stack(imgs6_tch, dim=0)
            ks6_tensor = torch.from_numpy(np.stack(ks6_tch, 0)).float()
            r6_tensor = torch.from_numpy(np.stack(r6, 0)).float()
            t6_tensor = torch.from_numpy(np.stack(t6, 0)).float()
        else:
            imgs6_tensor = torch.empty(0, dtype=torch.float32)
            ks6_tensor = torch.empty(0, dtype=torch.float32)
            r6_tensor = torch.empty(0, dtype=torch.float32)
            t6_tensor = torch.empty(0, dtype=torch.float32)

        if not self.need_student_inputs:
            img1 = torch.empty(0, dtype=torch.float32)
            k1 = torch.empty(0, dtype=torch.float32)
            r1 = torch.empty(0, dtype=torch.float32)
            t1 = torch.empty(0, dtype=torch.float32)
        if not self.need_teacher_depth:
            depth6_tensor = torch.empty(0, dtype=torch.long)
        else:
            depth6_tensor = torch.from_numpy(np.stack(depth6, 0)).long()
        if not self.need_student_depth:
            depth1 = torch.empty(0, dtype=torch.long)

        if self.need_temporal_inputs:
            temporal_tensors = self._make_previous_student_input(sample, student_cam, (img1, k1, r1, t1))
        else:
            temporal_tensors = tuple(torch.empty(0, dtype=torch.float32) for _ in range(8))

        output = (
            token,
            imgs6_tensor, ks6_tensor, r6_tensor, t6_tensor,
            img1, k1, r1, t1,
            torch.from_numpy(hm).float(), torch.from_numpy(reg).float(),
            torch.from_numpy(reg_mask).float(), torch.from_numpy(vel_mask).float(),
            depth6_tensor, depth1,
            torch.from_numpy(bev_fov_mask).float(),
            torch.tensor(si, dtype=torch.long),
            torch.from_numpy(attr_target).long(),
        ) + temporal_tensors
        if self.need_history_targets:
            history_hm, history_reg, history_reg_mask, history_vel_mask = self._make_history_targets(
                sample, student_cam,
            )
            output = output + (
                torch.from_numpy(history_hm).float(),
                torch.from_numpy(history_reg).float(),
                torch.from_numpy(history_reg_mask).float(),
                torch.from_numpy(history_vel_mask).float(),
            )
        if self.pseudo_results is not None:
            pseudo_hm, pseudo_reg, pseudo_reg_mask, pseudo_score = self._make_pseudo_targets(sample, student_cam)
            output = output + (
                torch.from_numpy(pseudo_hm).float(),
                torch.from_numpy(pseudo_reg).float(),
                torch.from_numpy(pseudo_reg_mask).float(),
                torch.from_numpy(pseudo_score).float(),
            )
        if self.need_lidar_relation_inputs:
            lidar_points, lidar_count, relation_keypoints, relation_object_mask = (
                self._make_lidar_relation_inputs(sample, student_cam)
            )
            output = output + (
                torch.from_numpy(lidar_points).float(),
                torch.tensor(lidar_count, dtype=torch.long),
                torch.from_numpy(relation_keypoints).float(),
                torch.from_numpy(relation_object_mask).float(),
            )
        return output


class BalancedRandomViewSampler(Sampler):
    """One uniformly sampled camera per scene and epoch, without multiplying epoch length by six."""

    def __init__(self, dataset: NuScenesOfficialDetectionDataset, seed: int = 0):
        self.dataset = dataset
        self.seed = int(seed)
        self.epoch = 0

    def __len__(self):
        return len(self.dataset.sample_tokens)

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        count = len(self.dataset.sample_tokens)
        order = torch.randperm(count, generator=generator).tolist()
        camera_indices = torch.randint(
            0, len(self.dataset.mod.CAM_CHANNELS), (count,), generator=generator,
        ).tolist()
        self.epoch += 1
        return iter(zip(order, camera_indices))


class ClassBalancedRandomViewSampler(BalancedRandomViewSampler):
    """CBGS-inspired replacement sampling while retaining one random camera per sample."""

    def __init__(self, dataset: NuScenesOfficialDetectionDataset, seed: int = 0,
                 max_oversample: float = 3.0):
        super().__init__(dataset, seed=seed)
        self.max_oversample = max(1.0, float(max_oversample))
        sample_classes = []
        class_counts = torch.zeros(len(DETECTION_NAMES), dtype=torch.float64)
        for token in tqdm(dataset.sample_tokens, desc="Build class-balanced sampler", dynamic_ncols=True):
            sample = dataset.nusc.get("sample", token)
            present = set()
            for ann_token in sample["anns"]:
                ann = dataset.nusc.get("sample_annotation", ann_token)
                det_name = CATEGORY_TO_DETECTION.get(ann["category_name"])
                if det_name is None:
                    continue
                if ann.get("num_lidar_pts", 1) + ann.get("num_radar_pts", 1) <= 0:
                    continue
                present.add(NAME_TO_ID[det_name])
            sample_classes.append(present)
            for cls_id in present:
                class_counts[cls_id] += 1.0

        nonzero = class_counts[class_counts > 0]
        reference_count = float(nonzero.max().item()) if len(nonzero) else 1.0
        class_weights = torch.ones_like(class_counts)
        valid = class_counts > 0
        class_weights[valid] = torch.clamp(
            reference_count / class_counts[valid], max=self.max_oversample,
        )
        sample_weights = []
        for present in sample_classes:
            if present:
                sample_weights.append(max(float(class_weights[cls_id]) for cls_id in present))
            else:
                sample_weights.append(1.0)
        self.sample_weights = torch.tensor(sample_weights, dtype=torch.float64)
        print(
            "[CBGS] sample class counts="
            f"{dict(zip(DETECTION_NAMES, [int(value) for value in class_counts.tolist()]))} "
            f"class weights={dict(zip(DETECTION_NAMES, [round(value, 3) for value in class_weights.tolist()]))}"
        )

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        count = len(self.dataset.sample_tokens)
        order = torch.multinomial(
            self.sample_weights, count, replacement=True, generator=generator,
        ).tolist()
        camera_indices = torch.randint(
            0, len(self.dataset.mod.CAM_CHANNELS), (count,), generator=generator,
        ).tolist()
        self.epoch += 1
        return iter(zip(order, camera_indices))


class TargetClassViewSampler(Sampler):
    """Sample camera views that actually contain a selected long-tail class.

    Class-balanced sampling at sample level still wastes most motorcycle and bus
    examples when the independently chosen camera does not see the object.  This
    sampler builds visible (sample, camera) pairs once and samples the requested
    classes uniformly.  It is intended for frozen-base, class-specific head
    training and therefore does not change deployment or evaluation.
    """

    def __init__(self, dataset: NuScenesOfficialDetectionDataset,
                 class_names: Sequence[str], seed: int = 0, epoch_size: int = 0,
                 state_balanced_classes: Sequence[str] = ()):
        self.dataset = dataset
        self.seed = int(seed)
        self.epoch = 0
        self.epoch_size = int(epoch_size) if int(epoch_size) > 0 else len(dataset.sample_tokens)
        invalid = sorted(set(class_names) - set(DETECTION_NAMES))
        if invalid:
            raise ValueError(f"unknown target view sampler classes: {invalid}")
        self.class_names = tuple(dict.fromkeys(str(name) for name in class_names))
        if not self.class_names:
            raise ValueError("TargetClassViewSampler requires at least one class")
        self.state_balanced_classes = set(str(name) for name in state_balanced_classes)
        invalid_state_classes = sorted(self.state_balanced_classes - set(self.class_names))
        if invalid_state_classes:
            raise ValueError(
                "state-balanced classes must also be target classes: "
                f"{invalid_state_classes}"
            )

        stratum_names = []
        for name in self.class_names:
            if name in self.state_balanced_classes:
                stratum_names.extend((f"{name}:static", f"{name}:moving"))
            else:
                stratum_names.append(name)
        pairs_by_class: Dict[str, List[Tuple[int, int]]] = {
            name: [] for name in stratum_names
        }
        for sample_index, token in enumerate(tqdm(
            dataset.sample_tokens, desc="Build target-class view sampler", dynamic_ncols=True,
        )):
            sample = dataset.nusc.get("sample", token)
            annotations_by_class: Dict[str, List[Mapping]] = {
                name: [] for name in stratum_names
            }
            for ann_token in sample["anns"]:
                annotation = dataset.nusc.get("sample_annotation", ann_token)
                det_name = CATEGORY_TO_DETECTION.get(annotation["category_name"])
                if det_name not in self.class_names:
                    continue
                if annotation.get("num_lidar_pts", 1) + annotation.get("num_radar_pts", 1) <= 0:
                    continue
                if det_name in self.state_balanced_classes:
                    velocity = np.asarray(dataset.nusc.box_velocity(ann_token)[:2], dtype=np.float64)
                    speed = float(np.linalg.norm(np.nan_to_num(velocity, nan=0.0)))
                    state = "moving" if speed >= 0.2 else "static"
                    annotations_by_class[f"{det_name}:{state}"].append(annotation)
                else:
                    annotations_by_class[det_name].append(annotation)
            if not any(annotations_by_class.values()):
                continue
            for camera_index, camera_name in enumerate(dataset.mod.CAM_CHANNELS):
                for class_name, annotations in annotations_by_class.items():
                    if any(
                        dataset._ann_visible_in_camera(sample, annotation, camera_name)
                        for annotation in annotations
                    ):
                        pairs_by_class[class_name].append((sample_index, camera_index))

        self.pairs_by_class = {
            name: pairs for name, pairs in pairs_by_class.items() if pairs
        }
        missing = sorted(set(stratum_names) - set(self.pairs_by_class))
        if missing:
            raise ValueError(f"no visible training views found for classes: {missing}")
        print(
            "[TARGET_VIEW_SAMPLER] visible pairs="
            f"{dict((name, len(pairs)) for name, pairs in self.pairs_by_class.items())} "
            f"epoch_size={self.epoch_size}"
        )

    def __len__(self):
        return self.epoch_size

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        class_names = tuple(self.pairs_by_class)
        class_choices = torch.randint(
            0, len(class_names), (self.epoch_size,), generator=generator,
        ).tolist()
        output = []
        for class_choice in class_choices:
            pairs = self.pairs_by_class[class_names[class_choice]]
            pair_index = int(torch.randint(
                0, len(pairs), (1,), generator=generator,
            ).item())
            output.append(pairs[pair_index])
        self.epoch += 1
        return iter(output)


class OfficialCenterHead(nn.Module):
    def __init__(self, in_ch=256, num_classes=10, reg_dim=10):
        super().__init__()
        self.hm = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(in_ch, num_classes, 1)
        )
        self.reg = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(in_ch, reg_dim, 1)
        )
        nn.init.constant_(self.hm[-1].bias, -2.19)

    def forward(self, x):
        return self.hm(x), self.reg(x)


class ImageYawFlipHead(nn.Module):
    """Classify front/back pi flips directly from stride-8 image appearance.

    The BEV detector already predicts an orientation axis reasonably well, but
    lift/splat discards much of the asymmetric front/rear appearance.  This tiny
    image-space head predicts only whether the frozen yaw should be negated. Its
    zero initialization leaves every existing S1 prediction unchanged.
    """

    def __init__(self, in_ch=256, num_classes=10, hidden_ch=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(min(16, hidden_ch), hidden_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_ch, num_classes, kernel_size=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, image_feature):
        return self.net(image_feature)


class ImageRadialCenterHead(nn.Module):
    """Predict a bounded class-specific depth-axis correction in image space."""

    def __init__(self, in_ch=256, num_classes=10, hidden_ch=32, max_residual=2.0):
        super().__init__()
        self.max_residual = float(max_residual)
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(min(16, hidden_ch), hidden_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_ch, num_classes, kernel_size=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, image_feature):
        return self.max_residual * torch.tanh(self.net(image_feature))


class ImageObjectGeometryHead(nn.Module):
    """Joint object-region depth and front/back refinement in image space.

    V24/V25 sampled a single stride-8 feature with only a 3x3 receptive field.
    This head fuses the explicitly supervised depth distribution with a small
    dilated appearance pyramid, so every output cell observes an object-sized
    region while keeping the deployed student lightweight.  Zero initialization
    preserves the frozen S1 centre and yaw exactly.
    """

    def __init__(self, in_ch=256, depth_bins=32, num_classes=10,
                 hidden_ch=64, max_residual=2.0):
        super().__init__()
        self.max_residual = float(max_residual)
        image_ch = 3 * int(hidden_ch) // 4
        depth_ch = int(hidden_ch) - image_ch
        self.image_projection = nn.Conv2d(in_ch, image_ch, kernel_size=1, bias=False)
        self.depth_projection = nn.Conv2d(depth_bins, depth_ch, kernel_size=1, bias=False)
        self.pre = nn.Sequential(
            nn.GroupNorm(16, hidden_ch),
            nn.SiLU(inplace=True),
        )
        branch_ch = hidden_ch // 2
        self.branches = nn.ModuleList([
            nn.Conv2d(
                hidden_ch, branch_ch, kernel_size=3,
                padding=dilation, dilation=dilation, bias=False,
            )
            for dilation in (1, 2, 4)
        ])
        self.fuse = nn.Sequential(
            nn.Conv2d(3 * branch_ch, hidden_ch, kernel_size=1, bias=False),
            nn.GroupNorm(16, hidden_ch),
            nn.SiLU(inplace=True),
        )
        self.output = nn.Conv2d(hidden_ch, 2 * num_classes, kernel_size=1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        self.num_classes = int(num_classes)

    def forward(self, image_feature, depth_logits):
        depth_probability = torch.softmax(depth_logits.float(), dim=1).to(image_feature.dtype)
        x = torch.cat((
            self.image_projection(image_feature),
            self.depth_projection(depth_probability),
        ), dim=1)
        x = self.pre(x)
        x = self.fuse(torch.cat([F.silu(branch(x)) for branch in self.branches], dim=1))
        raw = self.output(x)
        flip_logits = raw[:, :self.num_classes]
        radial = self.max_residual * torch.tanh(raw[:, self.num_classes:])
        return flip_logits, radial


class ClassAwareMotionResidualHead(nn.Module):
    """Small class-conditioned yaw/velocity residual head for deployed NDS refinement."""

    def __init__(self, in_ch=256, num_classes=10, hidden_ch=64):
        super().__init__()
        self.num_classes = int(num_classes)
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(min(32, hidden_ch), hidden_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_ch, self.num_classes * 4, kernel_size=1),
        )
        # Exact backward-compatible initialization: the upgraded model initially emits
        # the proven base head's yaw and velocity without any perturbation.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        batch, _channels, height, width = x.shape
        return self.net(x).reshape(batch, self.num_classes, 4, height, width)


class AzimuthBinHead(nn.Module):
    """Tiny per-class two-bin classifier for azimuth-invariant yaw regression."""

    def __init__(self, in_ch=256, num_classes=10):
        super().__init__()
        self.num_classes = int(num_classes)
        self.proj = nn.Conv2d(in_ch, self.num_classes * 2, kernel_size=1)
        # Equal bin probabilities preserve a neutral architecture-upgrade state.
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        batch, _channels, height, width = x.shape
        return self.proj(x).reshape(batch, self.num_classes, 2, height, width)


class ClassAwareSizeResidualHead(nn.Module):
    """Low-cost class-conditioned correction for the base log-size prediction."""

    def __init__(self, in_ch=256, num_classes=10, hidden_ch=32):
        super().__init__()
        self.num_classes = int(num_classes)
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden_ch, kernel_size=1, bias=False),
            nn.GroupNorm(min(16, hidden_ch), hidden_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_ch, self.num_classes * 3, kernel_size=1),
        )
        # Start from the proven base detector exactly.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        batch, _channels, height, width = x.shape
        return self.net(x).reshape(batch, self.num_classes, 3, height, width)


class ClassAwareCenterResidualHead(nn.Module):
    """Bounded class-conditioned correction for [dx, dy, z]."""

    def __init__(self, in_ch=256, num_classes=10, hidden_ch=32,
                 active_class_indices=tuple(range(10)),
                 max_residual=(1.5, 1.5, 1.0)):
        super().__init__()
        self.num_classes = int(num_classes)
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden_ch, kernel_size=1, bias=False),
            nn.GroupNorm(min(16, hidden_ch), hidden_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_ch, self.num_classes * 3, kernel_size=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        active = torch.zeros(self.num_classes, dtype=torch.float32)
        for class_index in active_class_indices:
            active[int(class_index)] = 1.0
        self.register_buffer(
            "active_class_mask", active.view(1, self.num_classes, 1, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "max_residual", torch.tensor(max_residual, dtype=torch.float32).view(1, 1, 3, 1, 1),
            persistent=False,
        )

    def forward(self, x):
        batch, _channels, height, width = x.shape
        raw = self.net(x.detach()).reshape(batch, self.num_classes, 3, height, width)
        return torch.tanh(raw.float()).to(raw) * self.max_residual.to(raw) * self.active_class_mask.to(raw)


class PrivilegedSparseDepthCompletion(nn.Module):
    """Project-native LiDAR privileged depth for the selected camera only.

    The deployable monocular depth logits remain the base prediction.  Sparse
    LiDAR bins are used as exact anchors where available, while a zero-initialized
    context residual learns to propagate their geometry to nearby image features.
    No public detector, teacher architecture, or teacher checkpoint is involved.
    """

    def __init__(self, depth_bins: int, hidden_ch: int = 64,
                 training_anchor_dropout: float = 0.35):
        super().__init__()
        self.depth_bins = int(depth_bins)
        self.training_anchor_dropout = float(training_anchor_dropout)
        if not 0.0 <= self.training_anchor_dropout < 1.0:
            raise ValueError("training_anchor_dropout must be in [0, 1)")
        groups = min(16, int(hidden_ch))
        self.context = nn.Sequential(
            nn.Conv2d(self.depth_bins * 2 + 1, hidden_ch, 3, padding=1, bias=False),
            nn.GroupNorm(groups, hidden_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(
                hidden_ch, hidden_ch, 3, padding=2, dilation=2,
                bias=False,
            ),
            nn.GroupNorm(groups, hidden_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_ch, self.depth_bins, 1),
        )
        nn.init.zeros_(self.context[-1].weight)
        nn.init.zeros_(self.context[-1].bias)

    def forward(self, monocular_logits, sparse_depth_bins):
        if sparse_depth_bins is None:
            raise ValueError("privileged depth teacher requires selected-camera LiDAR depth bins")
        if sparse_depth_bins.ndim == 4 and sparse_depth_bins.shape[1] == 1:
            sparse_depth_bins = sparse_depth_bins[:, 0]
        if sparse_depth_bins.ndim != 3:
            raise ValueError(
                "sparse depth bins must have shape [B,H,W] or [B,1,H,W], "
                f"got {tuple(sparse_depth_bins.shape)}"
            )
        if tuple(sparse_depth_bins.shape[-2:]) != tuple(monocular_logits.shape[-2:]):
            sparse_depth_bins = F.interpolate(
                sparse_depth_bins[:, None].float(),
                size=monocular_logits.shape[-2:], mode="nearest",
            )[:, 0]
        valid = (sparse_depth_bins >= 0) & (sparse_depth_bins < self.depth_bins)
        safe_bins = sparse_depth_bins.long().clamp(0, self.depth_bins - 1)
        one_hot = F.one_hot(safe_bins, num_classes=self.depth_bins).permute(0, 3, 1, 2)
        one_hot = one_hot.to(dtype=monocular_logits.dtype)
        support_valid = valid
        if self.training and self.training_anchor_dropout > 0.0:
            # Masked depth completion: held-out LiDAR points remain supervision
            # targets for the ordinary sparse depth loss but are not teacher input.
            support_valid = valid & (
                torch.rand(valid.shape, device=valid.device) >=
                self.training_anchor_dropout
            )
        valid_channel = support_valid[:, None].to(dtype=monocular_logits.dtype)
        one_hot = one_hot * valid_channel
        mono_probability = F.softmax(monocular_logits.float(), dim=1).to(monocular_logits.dtype)
        residual = self.context(torch.cat((mono_probability, one_hot, valid_channel), dim=1))
        completed = monocular_logits + residual

        # Exact LiDAR anchors are deliberately sharp but finite so AMP cannot
        # introduce infinities in the subsequent softmax or focal losses.
        oracle = torch.full_like(completed, -8.0)
        oracle.scatter_(1, safe_bins[:, None], 8.0)
        return torch.where(support_valid[:, None], oracle, completed)


class PrivilegedBEVResidualBlock(nn.Module):
    """Identity-initialized wide-context BEV block used only by the teacher."""

    def __init__(self, channels: int, dilation: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(
                channels, channels, 3, padding=dilation, dilation=dilation,
                groups=channels, bias=False,
            ),
            nn.GroupNorm(16, channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return x + self.net(x)


class PrivilegedBEVRefiner(nn.Module):
    """Extra training-only BEV capacity with 1/2/4-cell receptive fields."""

    def __init__(self, channels: int = 256):
        super().__init__()
        self.blocks = nn.Sequential(*(
            PrivilegedBEVResidualBlock(channels, dilation)
            for dilation in (1, 2, 4)
        ))

    def forward(self, x):
        return self.blocks(x)


class OfficialTeacher(nn.Module):
    def __init__(self, mod):
        super().__init__()
        self.mod = mod
        self.backbone = mod.ResNet50FPN()
        self.depth_head = mod.DepthHeadTeacher(in_ch=256, D=mod.NUM_DEPTH_BINS, out_ch=256)
        self.lss2d = mod.LiftSplatTeacher2D()
        self.bev_encoder = mod.BEVEncoderTeacher(256)
        self.det_head = OfficialCenterHead(256, len(DETECTION_NAMES), 10)

    def _one_view(self, img, k, r, t):
        feat = self.backbone(img)
        depth_prob, depth_logits, feat2d = self.depth_head(feat)
        bev = self.lss2d(feat2d, depth_prob, k, r, t, img_w=self.mod.TCH_W, img_h=self.mod.TCH_H)
        return bev, depth_logits

    def forward(self, imgs6, ks6, r6, t6, return_bev=False):
        b, v = imgs6.shape[:2]
        bev_acc = torch.zeros((b, 256, self.mod.BEV_H, self.mod.BEV_W), device=imgs6.device, dtype=imgs6.dtype)
        depth_logits_all = []
        for i in range(v):
            bev_i, depth_i = self._one_view(imgs6[:, i], ks6[:, i], r6[:, i], t6[:, i])
            bev_acc = bev_acc + bev_i
            depth_logits_all.append(depth_i)
        depth_logits_all = torch.stack(depth_logits_all, dim=1)
        bev_ctx = self.bev_encoder(bev_acc)
        hm, reg = self.det_head(bev_ctx)
        if return_bev:
            return hm, reg, depth_logits_all, bev_ctx
        return hm, reg, depth_logits_all


class MobileNetV3MultiScaleExtractor(nn.Module):
    """Fuse the deepest stride-8/16/32 MobileNetV3 features at stride 8.

    The legacy student returns the first stride-8 activation and therefore discards most
    ImageNet semantic stages. This variant keeps the full lightweight backbone while the
    downstream BEV network and single-view deployment protocol remain unchanged.
    """

    def __init__(self, pretrained=True, out_ch=256):
        super().__init__()
        weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        self.features = models.mobilenet_v3_large(weights=weights).features
        self.proj8 = nn.Sequential(
            nn.Conv2d(40, 96, kernel_size=1, bias=False), nn.BatchNorm2d(96), nn.ReLU(inplace=True),
        )
        self.proj16 = nn.Sequential(
            nn.Conv2d(112, 80, kernel_size=1, bias=False), nn.BatchNorm2d(80), nn.ReLU(inplace=True),
        )
        self.proj32 = nn.Sequential(
            nn.Conv2d(960, 80, kernel_size=1, bias=False), nn.BatchNorm2d(80), nn.ReLU(inplace=True),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(256, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        _batch, _channels, image_h, image_w = x.shape
        y = x
        by_stride = {}
        for layer in self.features:
            y = layer(y)
            effective_stride = int(round(max(
                image_h / max(1, y.shape[-2]),
                image_w / max(1, y.shape[-1]),
            )))
            if effective_stride in {8, 16, 32}:
                by_stride[effective_stride] = y
        missing = sorted({8, 16, 32} - set(by_stride))
        if missing:
            raise RuntimeError(f"MobileNetV3 did not produce expected strides: missing={missing}")
        target_size = (image_h // 8, image_w // 8)
        feat8 = self.proj8(by_stride[8])
        feat16 = F.interpolate(self.proj16(by_stride[16]), size=target_size, mode="bilinear", align_corners=False)
        feat32 = F.interpolate(self.proj32(by_stride[32]), size=target_size, mode="bilinear", align_corners=False)
        return self.fuse(torch.cat((feat8, feat16, feat32), dim=1))


class EfficientNetB2MultiScaleExtractor(nn.Module):
    """Fuse ImageNet EfficientNet-B2 stride-8/16/32 features at stride 8.

    The feature container intentionally uses ``efficient_features`` rather than
    ``features``.  A partial S1 initialization can therefore never copy a
    coincidentally shape-compatible MobileNet tensor into the new backbone.
    The downstream 256-channel depth/BEV interface stays unchanged.
    """

    def __init__(self, pretrained=True, out_ch=256):
        super().__init__()
        self.frozen_stages = 0
        weights = models.EfficientNet_B2_Weights.DEFAULT if pretrained else None
        self.efficient_features = models.efficientnet_b2(weights=weights).features
        self.proj8 = nn.Sequential(
            nn.Conv2d(48, 96, kernel_size=1, bias=False),
            nn.BatchNorm2d(96),
            nn.SiLU(inplace=True),
        )
        self.proj16 = nn.Sequential(
            nn.Conv2d(120, 80, kernel_size=1, bias=False),
            nn.BatchNorm2d(80),
            nn.SiLU(inplace=True),
        )
        self.proj32 = nn.Sequential(
            nn.Conv2d(1408, 80, kernel_size=1, bias=False),
            nn.BatchNorm2d(80),
            nn.SiLU(inplace=True),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(256, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        )

    def freeze_stages(self, count: int):
        """Freeze early ImageNet stages and keep their BN statistics fixed."""
        count = int(count)
        if count < 0 or count > len(self.efficient_features):
            raise ValueError(
                f"freeze stage count must be in [0, {len(self.efficient_features)}], got {count}"
            )
        self.frozen_stages = count
        for index, layer in enumerate(self.efficient_features):
            frozen = index < count
            for parameter in layer.parameters():
                parameter.requires_grad_(not frozen)
            if frozen:
                layer.eval()
        return self

    def train(self, mode: bool = True):
        super().train(mode)
        if mode and self.frozen_stages:
            for layer in list(self.efficient_features)[:self.frozen_stages]:
                layer.eval()
        return self

    def forward(self, x):
        _batch, _channels, image_h, image_w = x.shape
        y = x
        by_stride = {}
        for layer in self.efficient_features:
            y = layer(y)
            effective_stride = int(round(max(
                image_h / max(1, y.shape[-2]),
                image_w / max(1, y.shape[-1]),
            )))
            if effective_stride in {8, 16, 32}:
                by_stride[effective_stride] = y
        missing = sorted({8, 16, 32} - set(by_stride))
        if missing:
            raise RuntimeError(
                f"EfficientNet-B2 did not produce expected strides: missing={missing}"
            )
        target_size = by_stride[8].shape[-2:]
        feat8 = self.proj8(by_stride[8])
        feat16 = F.interpolate(
            self.proj16(by_stride[16]), size=target_size,
            mode="bilinear", align_corners=False,
        )
        feat32 = F.interpolate(
            self.proj32(by_stride[32]), size=target_size,
            mode="bilinear", align_corners=False,
        )
        return self.fuse(torch.cat((feat8, feat16, feat32), dim=1))


class CameraAwareDepthHeadStudent(nn.Module):
    """Condition the lightweight depth head on intrinsics and camera-to-ego calibration."""

    def __init__(self, in_ch=256, depth_bins=32, feat_ch=256, image_w=640, image_h=360):
        super().__init__()
        self.image_w = float(image_w)
        self.image_h = float(image_h)
        self.conv = nn.Sequential(nn.Conv2d(in_ch, in_ch, 3, padding=1), nn.ReLU(inplace=True))
        self.calibration_mlp = nn.Sequential(
            nn.Linear(16, 64), nn.ReLU(inplace=True), nn.Linear(64, 2 * in_ch),
        )
        nn.init.zeros_(self.calibration_mlp[-1].weight)
        nn.init.zeros_(self.calibration_mlp[-1].bias)
        self.depth_logits = nn.Conv2d(in_ch, depth_bins, 1)
        self.feat = nn.Conv2d(in_ch, feat_ch, 1)

    def calibration_vector(self, k, r, t):
        intrinsics = torch.stack((
            k[:, 0, 0] / self.image_w,
            k[:, 1, 1] / self.image_h,
            k[:, 0, 2] / self.image_w,
            k[:, 1, 2] / self.image_h,
        ), dim=1)
        rotation = r.reshape(r.shape[0], 9)
        translation = t / 10.0
        return torch.cat((intrinsics, rotation, translation), dim=1).float()

    def forward(self, x, k, r, t):
        x = self.conv(x)
        scale, bias = self.calibration_mlp(self.calibration_vector(k, r, t)).chunk(2, dim=1)
        scale = scale.to(dtype=x.dtype).unsqueeze(-1).unsqueeze(-1)
        bias = bias.to(dtype=x.dtype).unsqueeze(-1).unsqueeze(-1)
        x = x * (1.0 + 0.1 * torch.tanh(scale)) + 0.1 * bias
        depth_logits = self.depth_logits(x)
        depth_prob = F.softmax(depth_logits, dim=1)
        return depth_prob, depth_logits, self.feat(x)


def temporal_stereo_sampling_grid(
    feature_h, feature_w, image_h, image_w, depth_centers,
    current_k, current_r, current_t,
    previous_k, previous_r, previous_t,
    previous_to_current_r, previous_to_current_t,
):
    """Project current-view depth hypotheses into the previous image.

    Camera extrinsics map camera coordinates into the corresponding ego frame.
    ``previous_to_current_*`` maps the previous ego frame into the current ego
    frame.  The returned grid follows ``grid_sample(..., align_corners=False)``.
    """
    batch = current_k.shape[0]
    device = current_k.device
    dtype = torch.float32
    depths = depth_centers.to(device=device, dtype=dtype).reshape(1, -1, 1, 1)
    u = (torch.arange(feature_w, device=device, dtype=dtype) + 0.5) * (
        float(image_w) / float(feature_w)
    )
    v = (torch.arange(feature_h, device=device, dtype=dtype) + 0.5) * (
        float(image_h) / float(feature_h)
    )
    vv, uu = torch.meshgrid(v, u, indexing="ij")
    uu = uu.reshape(1, 1, feature_h, feature_w)
    vv = vv.reshape(1, 1, feature_h, feature_w)

    fx = current_k[:, 0, 0].float().reshape(batch, 1, 1, 1)
    fy = current_k[:, 1, 1].float().reshape(batch, 1, 1, 1)
    cx = current_k[:, 0, 2].float().reshape(batch, 1, 1, 1)
    cy = current_k[:, 1, 2].float().reshape(batch, 1, 1, 1)
    x_camera = (uu - cx) / fx.clamp(min=1e-6) * depths
    y_camera = (vv - cy) / fy.clamp(min=1e-6) * depths
    z_camera = depths.expand(batch, -1, feature_h, feature_w)
    points_camera = torch.stack((x_camera, y_camera, z_camera), dim=2)
    points_camera = points_camera.reshape(batch, depths.shape[1], 3, -1)

    current_rotation = current_r.float().reshape(batch, 1, 3, 3)
    current_translation = current_t.float().reshape(batch, 1, 3, 1)
    points_current_ego = current_rotation.matmul(points_camera) + current_translation
    previous_rotation_in_current = previous_to_current_r.float().reshape(batch, 1, 3, 3)
    previous_translation_in_current = previous_to_current_t.float().reshape(batch, 1, 3, 1)
    points_previous_ego = previous_rotation_in_current.transpose(-1, -2).matmul(
        points_current_ego - previous_translation_in_current
    )
    previous_camera_rotation = previous_r.float().reshape(batch, 1, 3, 3)
    previous_camera_translation = previous_t.float().reshape(batch, 1, 3, 1)
    points_previous_camera = previous_camera_rotation.transpose(-1, -2).matmul(
        points_previous_ego - previous_camera_translation
    )

    x_prev = points_previous_camera[:, :, 0]
    y_prev = points_previous_camera[:, :, 1]
    z_prev = points_previous_camera[:, :, 2]
    previous_fx = previous_k[:, 0, 0].float().reshape(batch, 1, 1)
    previous_fy = previous_k[:, 1, 1].float().reshape(batch, 1, 1)
    previous_cx = previous_k[:, 0, 2].float().reshape(batch, 1, 1)
    previous_cy = previous_k[:, 1, 2].float().reshape(batch, 1, 1)
    safe_z = z_prev.clamp(min=1e-6)
    projected_u = previous_fx * x_prev / safe_z + previous_cx
    projected_v = previous_fy * y_prev / safe_z + previous_cy
    grid_x = 2.0 * projected_u / float(image_w) - 1.0
    grid_y = 2.0 * projected_v / float(image_h) - 1.0
    grid = torch.stack((grid_x, grid_y), dim=-1).reshape(
        batch, depths.shape[1], feature_h, feature_w, 2,
    )
    valid = (
        (z_prev > 1e-4) &
        (projected_u >= 0.0) & (projected_u < float(image_w)) &
        (projected_v >= 0.0) & (projected_v < float(image_h))
    ).reshape(batch, depths.shape[1], feature_h, feature_w)
    return grid, valid


class TemporalStereoDepthMatcher(nn.Module):
    """Parameter-light temporal stereo evidence for the monocular depth logits.

    The matcher uses cosine correlation between the current feature and the
    previous feature sampled along each current-camera depth hypothesis.  A
    zero-initialized scalar gate makes an upgraded checkpoint exactly reproduce
    the original monocular depth distribution before fine-tuning.
    """

    def __init__(self, depth_bins, depth_min, depth_max, image_h, image_w,
                 dynamic_confidence=False):
        super().__init__()
        edges = torch.linspace(float(depth_min), float(depth_max), int(depth_bins) + 1)
        self.register_buffer("depth_centers", 0.5 * (edges[:-1] + edges[1:]), persistent=False)
        self.image_h = int(image_h)
        self.image_w = int(image_w)
        self.dynamic_confidence = bool(dynamic_confidence)
        self.gate = nn.Parameter(torch.zeros(()))
        if self.dynamic_confidence:
            # Mono/stereo entropy, stereo peak margin and geometric validity.
            # The final bias starts close to the scalar V19 behaviour; loading
            # a V19 checkpoint preserves its learned global gate while these
            # few parameters learn where temporal evidence is trustworthy.
            self.confidence_net = nn.Sequential(
                nn.Conv2d(4, 8, kernel_size=1),
                nn.SiLU(inplace=True),
                nn.Conv2d(8, 1, kernel_size=1),
            )
            nn.init.zeros_(self.confidence_net[-1].weight)
            nn.init.constant_(self.confidence_net[-1].bias, 2.0)
        else:
            self.confidence_net = None

    def cost_volume(
        self, current_feature, previous_feature,
        current_k, current_r, current_t,
        previous_k, previous_r, previous_t,
        previous_to_current_r, previous_to_current_t,
    ):
        batch, _channels, feature_h, feature_w = current_feature.shape
        grid, valid = temporal_stereo_sampling_grid(
            feature_h, feature_w, self.image_h, self.image_w, self.depth_centers,
            current_k, current_r, current_t,
            previous_k, previous_r, previous_t,
            previous_to_current_r, previous_to_current_t,
        )
        current = F.normalize(current_feature.float(), dim=1, eps=1e-6)
        previous = F.normalize(previous_feature.float(), dim=1, eps=1e-6)
        costs = []
        for depth_index in range(grid.shape[1]):
            warped = F.grid_sample(
                previous, grid[:, depth_index], mode="bilinear",
                padding_mode="zeros", align_corners=False,
            )
            costs.append((current * warped).sum(dim=1))
        cost = torch.stack(costs, dim=1)
        valid_float = valid.to(cost.dtype)
        count = valid_float.sum(dim=1, keepdim=True).clamp(min=1.0)
        mean = (cost * valid_float).sum(dim=1, keepdim=True) / count
        variance = (
            (cost - mean).square() * valid_float
        ).sum(dim=1, keepdim=True) / count
        normalized = (cost - mean) / torch.sqrt(variance + 1e-4)
        normalized = torch.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)
        return normalized * valid_float, valid

    def forward(
        self, monocular_logits, current_feature, previous_feature,
        current_k, current_r, current_t,
        previous_k, previous_r, previous_t,
        previous_to_current_r, previous_to_current_t, previous_valid,
    ):
        cost, valid = self.cost_volume(
            current_feature, previous_feature,
            current_k, current_r, current_t,
            previous_k, previous_r, previous_t,
            previous_to_current_r, previous_to_current_t,
        )
        sample_valid = previous_valid.to(cost).reshape(-1, 1, 1, 1)
        if self.confidence_net is None:
            delta = torch.tanh(self.gate) * cost * sample_valid
            return monocular_logits + delta.to(monocular_logits.dtype)

        mono_probability = F.softmax(monocular_logits.float(), dim=1)
        depth = self.depth_centers.to(mono_probability).reshape(1, -1, 1, 1)
        mono_mean = (mono_probability * depth).sum(dim=1, keepdim=True)
        mono_variance = (
            mono_probability * (depth - mono_mean).square()
        ).sum(dim=1, keepdim=True)
        bin_width = float(self.depth_centers[1] - self.depth_centers[0])
        search_radius = torch.sqrt(mono_variance + 1e-4).clamp(min=2.0 * bin_width)
        candidate = (depth - mono_mean).abs() <= 2.0 * search_radius
        candidate = candidate & valid
        candidate_float = candidate.to(cost.dtype)
        candidate_count = candidate_float.sum(dim=1, keepdim=True).clamp(min=2.0)

        masked_cost = cost.masked_fill(~candidate, -1e4)
        stereo_probability = F.softmax(masked_cost, dim=1) * candidate_float
        stereo_probability = stereo_probability / stereo_probability.sum(
            dim=1, keepdim=True,
        ).clamp(min=1e-6)
        mono_entropy = -(mono_probability * mono_probability.clamp(min=1e-6).log()).sum(
            dim=1, keepdim=True,
        ) / math.log(float(mono_probability.shape[1]))
        stereo_entropy = -(
            stereo_probability * stereo_probability.clamp(min=1e-6).log()
        ).sum(dim=1, keepdim=True) / candidate_count.log().clamp(min=1e-6)
        top_two = stereo_probability.topk(k=2, dim=1).values
        peak_margin = top_two[:, :1] - top_two[:, 1:2]
        valid_fraction = valid.to(cost.dtype).mean(dim=1, keepdim=True)
        confidence_features = torch.cat(
            (mono_entropy, stereo_entropy, peak_margin, valid_fraction), dim=1,
        )
        confidence = torch.sigmoid(self.confidence_net(confidence_features))
        reliable_cost = cost * candidate_float * confidence
        delta = torch.tanh(self.gate) * reliable_cost * sample_valid
        return monocular_logits + delta.to(monocular_logits.dtype)


def warp_previous_bev_to_current(bev, prev_to_current_r, prev_to_current_t, mod):
    """Warp a previous ego-frame BEV feature into the current ego frame."""
    batch, _channels, height, width = bev.shape
    xs = mod.BEV_X_MIN + (torch.arange(height, device=bev.device, dtype=torch.float32) + 0.5) * mod.BEV_RES
    ys = mod.BEV_Y_MIN + (torch.arange(width, device=bev.device, dtype=torch.float32) + 0.5) * mod.BEV_RES
    xx, yy = torch.meshgrid(xs, ys, indexing="ij")
    current_points = torch.stack((xx, yy, torch.zeros_like(xx)), dim=0).reshape(1, 3, -1)
    current_points = current_points.expand(batch, -1, -1)
    rotation = prev_to_current_r.float()
    translation = prev_to_current_t.float().unsqueeze(-1)
    previous_points = rotation.transpose(1, 2).bmm(current_points - translation)
    previous_x = (previous_points[:, 0] - mod.BEV_X_MIN) / mod.BEV_RES - 0.5
    previous_y = (previous_points[:, 1] - mod.BEV_Y_MIN) / mod.BEV_RES - 0.5
    grid_x = 2.0 * previous_y / max(1, width - 1) - 1.0
    grid_y = 2.0 * previous_x / max(1, height - 1) - 1.0
    grid = torch.stack((grid_x, grid_y), dim=-1).reshape(batch, height, width, 2)
    return F.grid_sample(
        bev.float(), grid, mode="bilinear", padding_mode="zeros", align_corners=True,
    ).to(bev.dtype)


def warp_current_bev_to_previous(bev, prev_to_current_r, prev_to_current_t, mod):
    """Warp a current-frame BEV into the previous ego frame without using the previous image."""
    current_to_previous_r = prev_to_current_r.float().transpose(1, 2)
    current_to_previous_t = -current_to_previous_r.bmm(
        prev_to_current_t.float().unsqueeze(-1)
    ).squeeze(-1)
    return warp_previous_bev_to_current(
        bev, current_to_previous_r, current_to_previous_t, mod,
    )


def bound_temporal_delta(delta: torch.Tensor, limit: float) -> torch.Tensor:
    """Prevent an AMP temporal residual from overflowing the downstream BEV encoder.

    A non-positive limit preserves the historical path bit-for-bit.  A positive
    limit is an explicit experimental setting and clips only the residual update,
    not the current or previous BEV features.
    """
    limit = float(limit)
    if limit <= 0.0:
        return delta
    bounded = torch.nan_to_num(
        delta.float(), nan=0.0, posinf=limit, neginf=-limit,
    ).clamp(min=-limit, max=limit)
    return bounded.to(delta.dtype)


class LightweightTemporalPreEncoder(nn.Module):
    """Shared residual refinement before temporal fusion, initialized as an identity."""

    def __init__(self, channels: int = 256):
        super().__init__()
        groups = min(32, int(channels))
        while channels % groups != 0:
            groups -= 1
        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.GroupNorm(groups, channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1),
        )
        nn.init.zeros_(self.refine[-1].weight)
        nn.init.zeros_(self.refine[-1].bias)

    def forward(self, bev):
        return bev + self.refine(bev)


class LightweightSpatialTemporalFusion(nn.Module):
    """Capture local inter-frame BEV displacement missed by per-cell 1x1 fusion."""

    def __init__(self, channels: int = 256, hidden_channels: int = 64):
        super().__init__()
        groups = min(32, hidden_channels)
        while hidden_channels % groups != 0:
            groups -= 1
        self.net = nn.Sequential(
            nn.Conv2d(channels * 3, hidden_channels, kernel_size=1, bias=False),
            nn.GroupNorm(groups, hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(
                hidden_channels, hidden_channels, kernel_size=3, padding=1,
                groups=hidden_channels, bias=False,
            ),
            nn.GroupNorm(groups, hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(
                hidden_channels, hidden_channels, kernel_size=3, padding=2, dilation=2,
                groups=hidden_channels, bias=False,
            ),
            nn.GroupNorm(groups, hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, kernel_size=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, temporal_features):
        return self.net(temporal_features)


class TemporalCorrelationMotionHead(nn.Module):
    """Tiny multi-scale correlation head for explicit inter-frame object motion.

    The previous BEV is already ego-motion aligned. Correlating it against the current
    BEV therefore exposes residual object displacement instead of asking a per-cell
    fusion layer to infer motion implicitly. Sparse multi-scale offsets cover both slow
    and fast objects while keeping the cost volume and deployment increment small.
    """

    def __init__(self, in_ch=256, num_classes=10, embedding_ch=16, hidden_ch=64,
                 offsets=(-8, -4, -2, -1, 0, 1, 2, 4, 8)):
        super().__init__()
        self.num_classes = int(num_classes)
        self.offsets = tuple(int(offset) for offset in offsets)
        if not self.offsets:
            raise ValueError("TemporalCorrelationMotionHead requires at least one offset")
        groups = min(8, int(embedding_ch))
        while embedding_ch % groups != 0:
            groups -= 1
        hidden_groups = min(32, int(hidden_ch))
        while hidden_ch % hidden_groups != 0:
            hidden_groups -= 1
        # A shared projection puts current and previous BEV cells in the same metric space.
        self.embedding = nn.Sequential(
            nn.Conv2d(in_ch, embedding_ch, kernel_size=1, bias=False),
            nn.GroupNorm(groups, embedding_ch),
            nn.SiLU(inplace=True),
        )
        correlation_ch = len(self.offsets) ** 2
        self.motion = nn.Sequential(
            nn.Conv2d(correlation_ch, hidden_ch, kernel_size=1, bias=False),
            nn.GroupNorm(hidden_groups, hidden_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(
                hidden_ch, hidden_ch, kernel_size=3, padding=1,
                groups=hidden_ch, bias=False,
            ),
            nn.GroupNorm(hidden_groups, hidden_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_ch, self.num_classes * 4, kernel_size=1),
        )
        # Loading an S1 checkpoint preserves its predictions exactly before fine-tuning.
        nn.init.zeros_(self.motion[-1].weight)
        nn.init.zeros_(self.motion[-1].bias)

    def forward(self, current_bev, previous_bev, previous_valid=None):
        if current_bev.shape != previous_bev.shape:
            raise ValueError(
                f"current/previous BEV shape mismatch: {tuple(current_bev.shape)} "
                f"vs {tuple(previous_bev.shape)}"
            )
        current = F.normalize(self.embedding(current_bev).float(), dim=1, eps=1e-6)
        previous = F.normalize(self.embedding(previous_bev).float(), dim=1, eps=1e-6)
        batch, _channels, height, width = current.shape
        padding = max(abs(offset) for offset in self.offsets)
        padded = F.pad(previous, (padding, padding, padding, padding))
        correlations = []
        for offset_x in self.offsets:
            start_x = padding + offset_x
            for offset_y in self.offsets:
                start_y = padding + offset_y
                shifted = padded[:, :, start_x:start_x + height, start_y:start_y + width]
                correlations.append((current * shifted).sum(dim=1))
        cost_volume = torch.stack(correlations, dim=1).to(dtype=current_bev.dtype)
        residual = self.motion(cost_volume).reshape(
            batch, self.num_classes, 4, height, width,
        )
        if previous_valid is not None:
            valid = previous_valid.to(residual).reshape(batch, 1, 1, 1, 1)
            residual = residual * valid
        return residual


class ClassAttributeVelocityCalibrator(nn.Module):
    """Metric-aligned class calibration with a tiny bounded object residual.

    The frozen S1 detector already predicts a useful velocity vector for common
    classes, but V15-P0 shows that its magnitude is systematically class biased.
    This module therefore learns a positive scale and a small rotation per dynamic
    class instead of replacing the vector.  A zero-initialized local residual can
    use the already temporal S1 BEV context, while a predicted-attribute moving
    probability reproduces the deployed velocity gate inside the trainable graph.

    At initialization the scale is one, rotation and residual are zero, so the
    output exactly matches S1 plus its existing attribute velocity gate.
    """

    def __init__(self, in_ch=256, num_classes=10, hidden_ch=16,
                 max_rotation=math.pi / 3.0, max_residual=0.75,
                 active_class_indices=(0, 1, 3, 6),
                 bypass_attribute_gate_indices=(), initial_scales=None):
        super().__init__()
        self.num_classes = int(num_classes)
        self.max_rotation = float(max_rotation)
        self.max_residual = float(max_residual)
        if initial_scales is None:
            initial_scales = [1.0] * self.num_classes
        if len(initial_scales) != self.num_classes:
            raise ValueError(
                f"expected {self.num_classes} velocity initial scales, got {len(initial_scales)}"
            )
        initial_scales_tensor = torch.tensor(initial_scales, dtype=torch.float32)
        if not torch.isfinite(initial_scales_tensor).all() or (initial_scales_tensor <= 0).any():
            raise ValueError("velocity initial scales must be finite and positive")
        self.log_scale = nn.Parameter(torch.log(initial_scales_tensor))
        self.rotation_control = nn.Parameter(torch.zeros(self.num_classes))
        groups = min(8, int(hidden_ch))
        while hidden_ch % groups != 0:
            groups -= 1
        self.local_residual = nn.Sequential(
            nn.Conv2d(in_ch, hidden_ch, kernel_size=1, bias=False),
            nn.GroupNorm(groups, hidden_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(
                hidden_ch, hidden_ch, kernel_size=3, padding=1,
                groups=hidden_ch, bias=False,
            ),
            nn.GroupNorm(groups, hidden_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_ch, self.num_classes * 2, kernel_size=1),
        )
        nn.init.zeros_(self.local_residual[-1].weight)
        nn.init.zeros_(self.local_residual[-1].bias)

        active_mask = torch.zeros(self.num_classes, dtype=torch.float32)
        for class_index in active_class_indices:
            active_mask[int(class_index)] = 1.0
        self.register_buffer(
            "active_class_mask", active_mask.view(1, self.num_classes, 1, 1, 1),
            persistent=False,
        )
        bypass_attribute_gate_mask = torch.zeros(self.num_classes, dtype=torch.float32)
        for class_index in bypass_attribute_gate_indices:
            bypass_attribute_gate_mask[int(class_index)] = 1.0
        self.register_buffer(
            "bypass_attribute_gate_mask",
            bypass_attribute_gate_mask.view(1, self.num_classes, 1, 1, 1),
            persistent=False,
        )

        valid_attribute_mask = torch.zeros(
            self.num_classes, len(ATTRIBUTE_NAMES), dtype=torch.bool,
        )
        moving_attribute_mask = torch.zeros_like(valid_attribute_mask)
        for class_name, valid_names in VALID_ATTRIBUTES_BY_DETECTION.items():
            class_index = NAME_TO_ID[class_name]
            for attribute_name in valid_names:
                attribute_index = ATTRIBUTE_TO_ID[attribute_name]
                valid_attribute_mask[class_index, attribute_index] = True
                if attribute_name in MOVING_ATTRIBUTES_BY_DETECTION[class_name]:
                    moving_attribute_mask[class_index, attribute_index] = True
        self.register_buffer(
            "valid_attribute_mask", valid_attribute_mask, persistent=False,
        )
        self.register_buffer(
            "moving_attribute_mask", moving_attribute_mask, persistent=False,
        )

    def moving_probability(self, attribute_logits, reference):
        batch, _classes, _xy, height, width = reference.shape
        probabilities = reference.new_ones(batch, self.num_classes, height, width)
        if attribute_logits is None:
            return probabilities
        detached_logits = attribute_logits.detach().float()
        for class_index in range(self.num_classes):
            valid_ids = self.valid_attribute_mask[class_index].nonzero(as_tuple=False).flatten()
            if valid_ids.numel() == 0:
                continue
            class_probabilities = torch.softmax(
                detached_logits[:, valid_ids], dim=1,
            )
            moving_local = self.moving_attribute_mask[class_index, valid_ids]
            if moving_local.any():
                probabilities[:, class_index] = class_probabilities[:, moving_local].sum(dim=1)
            else:
                probabilities[:, class_index] = 0.0
        return probabilities.to(reference.dtype)

    def forward(self, bev_context, base_displacement, attribute_logits=None):
        if base_displacement.ndim != 5 or base_displacement.shape[2] != 2:
            raise ValueError(
                f"expected base displacement [B,C,2,H,W], got {tuple(base_displacement.shape)}"
            )
        if base_displacement.shape[1] != self.num_classes:
            raise ValueError(
                f"expected {self.num_classes} classes, got {base_displacement.shape[1]}"
            )
        scale = torch.exp(self.log_scale.float().clamp(-2.3, 2.3)).view(
            1, self.num_classes, 1, 1,
        )
        angle = self.max_rotation * torch.tanh(self.rotation_control.float()).view(
            1, self.num_classes, 1, 1,
        )
        cosine = torch.cos(angle)
        sine = torch.sin(angle)
        base = base_displacement.float()
        base_x, base_y = base[:, :, 0], base[:, :, 1]
        rotated = torch.stack((
            scale * (cosine * base_x - sine * base_y),
            scale * (sine * base_x + cosine * base_y),
        ), dim=2)
        local = self.local_residual(bev_context.detach()).reshape(
            bev_context.shape[0], self.num_classes, 2,
            bev_context.shape[-2], bev_context.shape[-1],
        ).float()
        local = self.max_residual * torch.tanh(local)
        candidate = rotated + local
        calibrated = base + self.active_class_mask * (candidate - base)
        moving = self.moving_probability(attribute_logits, base).unsqueeze(2)
        # Bus and motorcycle attributes are especially unreliable in the long-tail
        # regime.  For explicitly selected classes the temporal residual must remain
        # trainable even when the frozen attribute head predicts parked/without-rider.
        moving = moving * (1.0 - self.bypass_attribute_gate_mask) + self.bypass_attribute_gate_mask
        calibrated = moving * calibrated
        return (calibrated - base).to(base_displacement.dtype)


class ObjectCentricVelocityRefiner(nn.Module):
    """Sparse-offset temporal displacement refiner with an exact identity start.

    The current and ego-aligned previous BEVs are correlated over a sparse local search
    window.  The correlation soft-argmax provides an explicit previous-to-current object
    displacement proposal.  A tiny class-conditioned head only learns how much of that
    proposal to trust and a bounded residual.  The detector, yaw, and size outputs are
    untouched; zero-initialized controls make an upgraded S1 checkpoint bit-compatible
    before fine-tuning.
    """

    def __init__(self, bev_resolution: float, in_ch=256, num_classes=10,
                 embedding_ch=16, hidden_ch=32,
                 offsets=(-12, -8, -6, -4, -2, -1, 0, 1, 2, 4, 6, 8, 12),
                 active_class_indices=(0, 1, 3, 6)):
        super().__init__()
        self.num_classes = int(num_classes)
        self.offsets = tuple(int(offset) for offset in offsets)
        if not self.offsets:
            raise ValueError("ObjectCentricVelocityRefiner requires at least one offset")
        groups = min(8, int(embedding_ch))
        while embedding_ch % groups != 0:
            groups -= 1
        hidden_groups = min(16, int(hidden_ch))
        while hidden_ch % hidden_groups != 0:
            hidden_groups -= 1
        self.embedding = nn.Sequential(
            nn.Conv2d(in_ch, embedding_ch, kernel_size=1, bias=False),
            nn.GroupNorm(groups, embedding_ch),
            nn.SiLU(inplace=True),
        )
        correlation_ch = len(self.offsets) ** 2
        self.controls = nn.Sequential(
            nn.Conv2d(correlation_ch + 2, hidden_ch, kernel_size=1, bias=False),
            nn.GroupNorm(hidden_groups, hidden_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(
                hidden_ch, hidden_ch, kernel_size=3, padding=1,
                groups=hidden_ch, bias=False,
            ),
            nn.GroupNorm(hidden_groups, hidden_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_ch, self.num_classes * 3, kernel_size=1),
        )
        nn.init.zeros_(self.controls[-1].weight)
        nn.init.zeros_(self.controls[-1].bias)

        offset_x = []
        offset_y = []
        for dx in self.offsets:
            for dy in self.offsets:
                # A match at previous[current + offset] means the object travelled
                # -offset cells from the previous timestamp to the current timestamp.
                offset_x.append(-float(dx) * float(bev_resolution))
                offset_y.append(-float(dy) * float(bev_resolution))
        self.register_buffer(
            "displacement_x", torch.tensor(offset_x, dtype=torch.float32).view(1, -1, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "displacement_y", torch.tensor(offset_y, dtype=torch.float32).view(1, -1, 1, 1),
            persistent=False,
        )
        active_mask = torch.zeros(self.num_classes, dtype=torch.float32)
        for class_index in active_class_indices:
            active_mask[int(class_index)] = 1.0
        self.register_buffer(
            "active_class_mask", active_mask.view(1, self.num_classes, 1, 1, 1),
            persistent=False,
        )

    def _cost_volume(self, current_bev, previous_bev):
        current = F.normalize(self.embedding(current_bev.detach()).float(), dim=1, eps=1e-6)
        previous = F.normalize(self.embedding(previous_bev.detach()).float(), dim=1, eps=1e-6)
        _batch, _channels, height, width = current.shape
        padding = max(abs(offset) for offset in self.offsets)
        padded = F.pad(previous, (padding, padding, padding, padding))
        correlations = []
        for offset_x in self.offsets:
            start_x = padding + offset_x
            for offset_y in self.offsets:
                start_y = padding + offset_y
                shifted = padded[:, :, start_x:start_x + height, start_y:start_y + width]
                correlations.append((current * shifted).sum(dim=1))
        return torch.stack(correlations, dim=1)

    def forward(self, current_bev, previous_bev, base_displacement, previous_valid=None):
        if current_bev.shape != previous_bev.shape:
            raise ValueError(
                f"current/previous BEV shape mismatch: {tuple(current_bev.shape)} "
                f"vs {tuple(previous_bev.shape)}"
            )
        if base_displacement.ndim != 5 or base_displacement.shape[2] != 2:
            raise ValueError(
                f"expected base displacement [B,C,2,H,W], got {tuple(base_displacement.shape)}"
            )
        cost_volume = self._cost_volume(current_bev, previous_bev)
        # A fixed sharpness keeps the physical proposal parameter-free. The learnable
        # controls below decide whether it improves each object class and location.
        probabilities = torch.softmax(10.0 * cost_volume, dim=1)
        proposal_x = (probabilities * self.displacement_x).sum(dim=1, keepdim=True)
        proposal_y = (probabilities * self.displacement_y).sum(dim=1, keepdim=True)
        proposal = torch.cat((proposal_x, proposal_y), dim=1)
        control_input = torch.cat((cost_volume, proposal), dim=1).to(current_bev.dtype)
        controls = self.controls(control_input).reshape(
            current_bev.shape[0], self.num_classes, 3,
            current_bev.shape[-2], current_bev.shape[-1],
        )
        blend = torch.tanh(controls[:, :, 0:1].float())
        bounded_residual = 2.0 * torch.tanh(controls[:, :, 1:3].float())
        proposal = proposal.float().unsqueeze(1)
        residual = blend * (proposal - base_displacement.detach().float()) + bounded_residual
        residual = residual * self.active_class_mask
        if previous_valid is not None:
            valid = previous_valid.to(residual).reshape(-1, 1, 1, 1, 1)
            residual = residual * valid
        return residual.to(base_displacement.dtype)


class ResidualAzimuthMultiBinHead(nn.Module):
    """Azimuth-conditioned MultiBin correction around a frozen base yaw.

    Unlike the earlier representation-replacement experiment, this head never re-encodes
    the proven S1 yaw. It predicts a class-specific angular correction and rotates the
    frozen base [sin(yaw), cos(yaw)] vector. Bin 0 is centred at zero and the final layer
    is zero-initialized, so an upgraded checkpoint initially reproduces S1 exactly.
    """

    def __init__(self, bev_x_min: float, bev_y_min: float, bev_resolution: float,
                 in_ch=256, num_classes=10, num_bins=8, hidden_ch=32,
                 active_class_indices=(1, 2, 4, 6, 7, 8)):
        super().__init__()
        self.num_classes = int(num_classes)
        self.num_bins = int(num_bins)
        self.bev_x_min = float(bev_x_min)
        self.bev_y_min = float(bev_y_min)
        self.bev_resolution = float(bev_resolution)
        if self.num_bins != 8:
            raise ValueError("ResidualAzimuthMultiBinHead currently uses the audited 8-bin layout")
        groups = min(16, int(hidden_ch))
        while hidden_ch % groups != 0:
            groups -= 1
        self.net = nn.Sequential(
            nn.Conv2d(in_ch + 2, hidden_ch, kernel_size=1, bias=False),
            nn.GroupNorm(groups, hidden_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(
                hidden_ch, hidden_ch, kernel_size=3, padding=1,
                groups=hidden_ch, bias=False,
            ),
            nn.GroupNorm(groups, hidden_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_ch, self.num_classes * self.num_bins * 3, kernel_size=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        centres = torch.tensor([
            0.0, math.pi / 4.0, math.pi / 2.0, 3.0 * math.pi / 4.0,
            math.pi, -3.0 * math.pi / 4.0, -math.pi / 2.0, -math.pi / 4.0,
        ], dtype=torch.float32)
        self.register_buffer("bin_centres", centres, persistent=False)
        active_mask = torch.zeros(self.num_classes, dtype=torch.float32)
        for class_index in active_class_indices:
            active_mask[int(class_index)] = 1.0
        self.register_buffer(
            "active_class_mask", active_mask.view(1, self.num_classes, 1, 1, 1),
            persistent=False,
        )

    def _ray_features(self, reference):
        batch, _channels, height, width = reference.shape
        xs = self.bev_x_min + (
            torch.arange(height, device=reference.device, dtype=torch.float32) + 0.5
        ) * self.bev_resolution
        ys = self.bev_y_min + (
            torch.arange(width, device=reference.device, dtype=torch.float32) + 0.5
        ) * self.bev_resolution
        xx, yy = torch.meshgrid(xs, ys, indexing="ij")
        radial = torch.atan2(yy, xx)
        rays = torch.stack((torch.sin(radial), torch.cos(radial)), dim=0)
        return rays.unsqueeze(0).expand(batch, -1, -1, -1).to(reference.dtype)

    def forward(self, bev_context, base_yaw_vectors):
        if base_yaw_vectors.ndim != 5 or base_yaw_vectors.shape[2] != 2:
            raise ValueError(
                f"expected base yaw vectors [B,C,2,H,W], got {tuple(base_yaw_vectors.shape)}"
            )
        features = torch.cat((bev_context.detach(), self._ray_features(bev_context)), dim=1)
        raw = self.net(features).reshape(
            bev_context.shape[0], self.num_classes, 3, self.num_bins,
            bev_context.shape[-2], bev_context.shape[-1],
        )
        logits = raw[:, :, 0]
        residual_sin = raw[:, :, 1]
        residual_cos_offset = raw[:, :, 2]
        predicted_bin = logits.argmax(dim=2, keepdim=True)
        selected_sin = residual_sin.gather(2, predicted_bin).squeeze(2).float()
        selected_cos_offset = residual_cos_offset.gather(2, predicted_bin).squeeze(2).float()
        centres = self.bin_centres[predicted_bin.squeeze(2)].float()
        intra_bin = torch.atan2(selected_sin, 1.0 + selected_cos_offset)
        correction = wrap_angle_tensor(centres + intra_bin)
        correction_sin = torch.sin(correction)
        correction_cos = torch.cos(correction)

        # Rotate the raw vector rather than normalizing it, preserving both components
        # bit-for-bit when the zero-centred correction is zero.
        base = base_yaw_vectors.float()
        base_sin = base[:, :, 0]
        base_cos = base[:, :, 1]
        refined_sin = base_sin * correction_cos + base_cos * correction_sin
        refined_cos = base_cos * correction_cos - base_sin * correction_sin
        refined = torch.stack((refined_sin, refined_cos), dim=2)
        vector_residual = (refined - base) * self.active_class_mask
        # The final two channels retain the actually applied vector correction so the
        # loss can reconstruct the frozen base yaw while supervising all target bins.
        audit_output = torch.cat(
            (raw.reshape(raw.shape[0], self.num_classes, 3 * self.num_bins,
                         raw.shape[-2], raw.shape[-1]), vector_residual),
            dim=2,
        )
        return vector_residual.to(base_yaw_vectors.dtype), audit_output


class ResidualYawVectorHead(nn.Module):
    """Smooth, zero-initialized yaw correction around a frozen detector.

    The previous V14-B head selected a hard direction bin, so one wrong rare-class
    decision could rotate a box abruptly.  This variant predicts a continuous
    residual unit vector and rotates S1 gradually; its zero initialization is an
    exact identity and unsupported classes remain bit-for-bit unchanged.
    """

    def __init__(self, bev_x_min: float, bev_y_min: float, bev_resolution: float,
                 in_ch=256, num_classes=10, hidden_ch=32,
                 active_class_indices=(1, 4, 6, 7, 8)):
        super().__init__()
        self.num_classes = int(num_classes)
        self.bev_x_min = float(bev_x_min)
        self.bev_y_min = float(bev_y_min)
        self.bev_resolution = float(bev_resolution)
        groups = min(16, int(hidden_ch))
        while hidden_ch % groups != 0:
            groups -= 1
        self.net = nn.Sequential(
            nn.Conv2d(in_ch + 2, hidden_ch, kernel_size=1, bias=False),
            nn.GroupNorm(groups, hidden_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(
                hidden_ch, hidden_ch, kernel_size=3, padding=1,
                groups=hidden_ch, bias=False,
            ),
            nn.GroupNorm(groups, hidden_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_ch, self.num_classes * 2, kernel_size=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        active_mask = torch.zeros(self.num_classes, dtype=torch.float32)
        for class_index in active_class_indices:
            active_mask[int(class_index)] = 1.0
        self.register_buffer(
            "active_class_mask", active_mask.view(1, self.num_classes, 1, 1, 1),
            persistent=False,
        )

    def _ray_features(self, reference):
        batch, _channels, height, width = reference.shape
        xs = self.bev_x_min + (
            torch.arange(height, device=reference.device, dtype=torch.float32) + 0.5
        ) * self.bev_resolution
        ys = self.bev_y_min + (
            torch.arange(width, device=reference.device, dtype=torch.float32) + 0.5
        ) * self.bev_resolution
        xx, yy = torch.meshgrid(xs, ys, indexing="ij")
        radial = torch.atan2(yy, xx)
        return torch.stack((torch.sin(radial), torch.cos(radial)), dim=0).unsqueeze(0).expand(
            batch, -1, -1, -1,
        ).to(reference.dtype)

    def forward(self, bev_context, base_yaw_vectors):
        features = torch.cat((bev_context.detach(), self._ray_features(bev_context)), dim=1)
        raw = self.net(features).reshape(
            bev_context.shape[0], self.num_classes, 2,
            bev_context.shape[-2], bev_context.shape[-1],
        )
        correction = torch.atan2(raw[:, :, 0].float(), 1.0 + raw[:, :, 1].float())
        correction_sin, correction_cos = torch.sin(correction), torch.cos(correction)
        base = base_yaw_vectors.float()
        refined = torch.stack((
            base[:, :, 0] * correction_cos + base[:, :, 1] * correction_sin,
            base[:, :, 1] * correction_cos - base[:, :, 0] * correction_sin,
        ), dim=2)
        vector_residual = (refined - base) * self.active_class_mask
        audit = torch.cat((raw, vector_residual.to(raw)), dim=2)
        return vector_residual.to(base_yaw_vectors.dtype), audit


class SceneAwarePrivilegedResidualAdapter(nn.Module):
    """Camera-only capacity dedicated to privileged geometric residuals.

    The adapter never consumes LiDAR.  LiDAR-on minus independent-camera-teacher
    residuals are training targets only.  Zero-initialised output projections make
    a newly attached adapter exactly preserve the audited V31 predictions.
    """

    def __init__(self, image_channels, bev_channels, depth_bins,
                 memory_slots=32, latent_dim=32):
        super().__init__()
        self.memory_slots = int(memory_slots)
        self.latent_dim = int(latent_dim)
        if self.memory_slots <= 0 or self.latent_dim <= 0:
            raise ValueError("memory_slots and latent_dim must be positive")

        self.depth_residual = nn.Sequential(
            nn.Conv2d(image_channels, 64, 3, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.SiLU(inplace=True),
            nn.Conv2d(64, depth_bins, 1),
        )
        self.query = nn.Sequential(
            nn.Conv2d(bev_channels, self.latent_dim, 1, bias=False),
            nn.GroupNorm(8 if self.latent_dim % 8 == 0 else 1, self.latent_dim),
            nn.SiLU(inplace=True),
        )
        self.memory_keys = nn.Parameter(
            torch.randn(self.memory_slots, self.latent_dim) /
            math.sqrt(float(self.latent_dim))
        )
        self.memory_values = nn.Parameter(
            torch.randn(self.memory_slots, self.latent_dim) /
            math.sqrt(float(self.latent_dim))
        )
        self.bev_residual = nn.Sequential(
            nn.Conv2d(self.latent_dim * 2, 64, 3, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.SiLU(inplace=True),
            nn.Conv2d(64, bev_channels, 1),
        )
        nn.init.zeros_(self.depth_residual[-1].weight)
        nn.init.zeros_(self.depth_residual[-1].bias)
        nn.init.zeros_(self.bev_residual[-1].weight)
        nn.init.zeros_(self.bev_residual[-1].bias)

    def predict_depth_residual(self, image_feature):
        # A privileged target can contain arbitrarily shifted logits.  Bounded
        # camera corrections preserve the calibrated V31 posterior while still
        # allowing approximately two depth-bin logits of local refinement.
        return 2.0 * torch.tanh(self.depth_residual(image_feature))

    def predict_bev_residual(self, bev_feature):
        query = self.query(bev_feature)
        batch, channels, height, width = query.shape
        query_tokens = query.flatten(2).transpose(1, 2)
        scores = torch.matmul(query_tokens.float(), self.memory_keys.float().t())
        scores = scores / math.sqrt(float(self.latent_dim))
        retrieved = torch.matmul(F.softmax(scores, dim=-1), self.memory_values.float())
        retrieved = retrieved.to(query.dtype).transpose(1, 2).reshape(
            batch, channels, height, width,
        )
        # Context residuals are deliberately small; their job is to repair object
        # geometry, not replace the mature V31 representation.
        return 0.25 * torch.tanh(
            self.bev_residual(torch.cat((query, retrieved), dim=1))
        )


class OfficialStudent(nn.Module):
    def __init__(self, mod, pretrained_backbone=True, use_kd_adapter=True,
                 backbone_variant="stride8", camera_aware_depth=False, bev_norm="batch",
                 attribute_head=False, temporal_bev=False, temporal_stereo_depth=False,
                 temporal_stereo_dynamic_confidence=False,
                 temporal_pre_encoder=False, temporal_delta_limit=0.0,
                 history_prediction=False, class_aware_motion=False,
                 temporal_spatial_fusion=False, azimuth_multibin=False,
                 class_aware_size=False, class_aware_center=False,
                 center_refiner_active_classes=None, temporal_correlation_motion=False,
                 image_yaw_flip=False,
                 image_radial_center=False, image_radial_center_max_residual=2.0,
                 image_object_geometry=False,
                 object_velocity_refiner=False, residual_azimuth_multibin=False,
                 residual_yaw_vector=False, residual_yaw_active_classes=None,
                 class_attribute_velocity_calibrator=False,
                 velocity_calibrator_motorcycle=True,
                 velocity_calibrator_active_classes=None,
                 velocity_calibrator_bypass_attribute_gate=False,
                 velocity_calibrator_max_residual=0.75,
                 velocity_calibrator_initial_scales=None,
                 privileged_depth_teacher=False,
                 camera_privileged_memory_adapter=False,
                 privileged_memory_slots=32,
                 privileged_memory_latent_dim=32):
        super().__init__()
        self.mod = mod
        self.backbone_variant = str(backbone_variant)
        self.camera_aware_depth = bool(camera_aware_depth)
        self.bev_norm = str(bev_norm)
        self.temporal_delta_limit = float(temporal_delta_limit)
        self.privileged_depth_teacher_enabled = bool(privileged_depth_teacher)
        self.camera_privileged_memory_adapter_enabled = bool(
            camera_privileged_memory_adapter
        )
        if self.backbone_variant == "stride8":
            self.backbone = mod.MobileNetV3Stride8Extractor(pretrained=pretrained_backbone, out_ch=256)
        elif self.backbone_variant == "multiscale":
            self.backbone = MobileNetV3MultiScaleExtractor(pretrained=pretrained_backbone, out_ch=256)
        elif self.backbone_variant == "efficientnet_b2":
            self.backbone = EfficientNetB2MultiScaleExtractor(
                pretrained=pretrained_backbone, out_ch=256,
            )
        else:
            raise ValueError(f"unsupported backbone_variant={self.backbone_variant}")
        if self.camera_aware_depth:
            self.depth_head = CameraAwareDepthHeadStudent(
                in_ch=256, depth_bins=mod.NUM_DEPTH_BINS, feat_ch=256,
                image_w=mod.IMG_W, image_h=mod.IMG_H,
            )
        else:
            self.depth_head = mod.DepthHeadStudent(in_ch=256, D=mod.NUM_DEPTH_BINS, feat_ch=256)
        self.lss3d = mod.VoxelLiftSplatZMean(z_bins=mod.Z_BINS, z_min=mod.Z_MIN, z_max=mod.Z_MAX)
        self.bev_encoder = mod.BEVEncoderStudent(256)
        if self.bev_norm == "group":
            replaced = replace_batchnorm_with_groupnorm(self.bev_encoder)
            print(f"[MODEL] BEV encoder GroupNorm enabled replaced_batchnorm={replaced}")
        elif self.bev_norm != "batch":
            raise ValueError(f"unsupported bev_norm={self.bev_norm}")
        self.privileged_depth_completion = (
            PrivilegedSparseDepthCompletion(mod.NUM_DEPTH_BINS)
            if self.privileged_depth_teacher_enabled else None
        )
        self.privileged_bev_refiner = (
            PrivilegedBEVRefiner(256)
            if self.privileged_depth_teacher_enabled else None
        )
        self.privileged_residual_adapter = (
            SceneAwarePrivilegedResidualAdapter(
                256, 256, mod.NUM_DEPTH_BINS,
                memory_slots=privileged_memory_slots,
                latent_dim=privileged_memory_latent_dim,
            )
            if self.camera_privileged_memory_adapter_enabled else None
        )
        self.det_head = OfficialCenterHead(256, len(DETECTION_NAMES), 10)
        self.class_aware_motion_enabled = bool(class_aware_motion)
        self.class_motion_head = (
            ClassAwareMotionResidualHead(256, len(DETECTION_NAMES))
            if self.class_aware_motion_enabled else None
        )
        self.azimuth_multibin_enabled = bool(azimuth_multibin)
        self.azimuth_bin_head = (
            AzimuthBinHead(256, len(DETECTION_NAMES))
            if self.azimuth_multibin_enabled else None
        )
        self.class_aware_size_enabled = bool(class_aware_size)
        self.class_size_head = (
            ClassAwareSizeResidualHead(256, len(DETECTION_NAMES))
            if self.class_aware_size_enabled else None
        )
        self.class_aware_center_enabled = bool(class_aware_center)
        center_classes = tuple(center_refiner_active_classes or DETECTION_NAMES)
        invalid_center_classes = sorted(set(center_classes) - set(DETECTION_NAMES))
        if invalid_center_classes:
            raise ValueError(f"unknown center refiner classes: {invalid_center_classes}")
        self.class_center_head = (
            ClassAwareCenterResidualHead(
                256, len(DETECTION_NAMES),
                active_class_indices=tuple(NAME_TO_ID[name] for name in center_classes),
            )
            if self.class_aware_center_enabled else None
        )
        self.image_yaw_flip_enabled = bool(image_yaw_flip)
        self.image_yaw_flip_head = (
            ImageYawFlipHead(256, len(DETECTION_NAMES))
            if self.image_yaw_flip_enabled else None
        )
        self.image_radial_center_enabled = bool(image_radial_center)
        self.image_radial_center_head = (
            ImageRadialCenterHead(
                256, len(DETECTION_NAMES),
                max_residual=float(image_radial_center_max_residual),
            )
            if self.image_radial_center_enabled else None
        )
        self.image_object_geometry_enabled = bool(image_object_geometry)
        self.image_object_geometry_head = (
            ImageObjectGeometryHead(
                256, mod.NUM_DEPTH_BINS, len(DETECTION_NAMES),
                max_residual=float(image_radial_center_max_residual),
            )
            if self.image_object_geometry_enabled else None
        )
        self.attribute_head_enabled = bool(attribute_head)
        self.temporal_bev = bool(temporal_bev)
        self.temporal_stereo_depth_enabled = bool(temporal_stereo_depth)
        self.temporal_stereo_dynamic_confidence_enabled = bool(
            temporal_stereo_dynamic_confidence
        )
        self.temporal_pre_encoder_enabled = bool(temporal_pre_encoder)
        self.temporal_spatial_fusion_enabled = bool(temporal_spatial_fusion)
        self.temporal_correlation_motion_enabled = bool(temporal_correlation_motion)
        self.object_velocity_refiner_enabled = bool(object_velocity_refiner)
        self.residual_azimuth_multibin_enabled = bool(residual_azimuth_multibin)
        self.residual_yaw_vector_enabled = bool(residual_yaw_vector)
        self.class_attribute_velocity_calibrator_enabled = bool(
            class_attribute_velocity_calibrator
        )
        self.history_prediction_enabled = bool(history_prediction)
        if self.temporal_pre_encoder_enabled and not self.temporal_bev:
            raise ValueError("temporal_pre_encoder requires temporal_bev")
        if self.temporal_stereo_depth_enabled and not self.temporal_bev:
            raise ValueError("temporal_stereo_depth requires temporal_bev")
        if (
            self.temporal_stereo_dynamic_confidence_enabled and
            not self.temporal_stereo_depth_enabled
        ):
            raise ValueError(
                "temporal_stereo_dynamic_confidence requires temporal_stereo_depth"
            )
        if self.history_prediction_enabled and not self.temporal_bev:
            raise ValueError("history_prediction requires temporal_bev")
        if self.temporal_spatial_fusion_enabled and not self.temporal_bev:
            raise ValueError("temporal_spatial_fusion requires temporal_bev")
        if self.temporal_correlation_motion_enabled and not self.temporal_bev:
            raise ValueError("temporal_correlation_motion requires temporal_bev")
        if self.temporal_correlation_motion_enabled and not self.class_aware_motion_enabled:
            raise ValueError("temporal_correlation_motion requires class_aware_motion")
        if self.object_velocity_refiner_enabled and not self.temporal_bev:
            raise ValueError("object_velocity_refiner requires temporal_bev")
        if self.object_velocity_refiner_enabled and not self.class_aware_motion_enabled:
            raise ValueError("object_velocity_refiner requires class_aware_motion")
        if self.class_attribute_velocity_calibrator_enabled and not self.class_aware_motion_enabled:
            raise ValueError("class_attribute_velocity_calibrator requires class_aware_motion")
        if self.class_attribute_velocity_calibrator_enabled and not self.attribute_head_enabled:
            raise ValueError("class_attribute_velocity_calibrator requires attribute_head")
        if self.residual_azimuth_multibin_enabled and not self.class_aware_motion_enabled:
            raise ValueError("residual_azimuth_multibin requires class_aware_motion")
        if self.residual_yaw_vector_enabled and not self.class_aware_motion_enabled:
            raise ValueError("residual_yaw_vector requires class_aware_motion")
        if self.residual_yaw_vector_enabled and self.residual_azimuth_multibin_enabled:
            raise ValueError("residual yaw vector and MultiBin heads are mutually exclusive")
        self.residual_azimuth_multibin_head = (
            ResidualAzimuthMultiBinHead(
                self.mod.BEV_X_MIN, self.mod.BEV_Y_MIN, self.mod.BEV_RES,
                256, len(DETECTION_NAMES),
                active_class_indices=tuple(
                    NAME_TO_ID[name] for name in (
                        "truck", "construction_vehicle", "trailer",
                        "motorcycle", "bicycle", "pedestrian",
                    )
                ),
            )
            if self.residual_azimuth_multibin_enabled else None
        )
        residual_yaw_classes = tuple(
            residual_yaw_active_classes or
            ("truck", "trailer", "pedestrian", "motorcycle", "bicycle")
        )
        invalid_yaw_classes = sorted(set(residual_yaw_classes) - set(DETECTION_NAMES))
        if invalid_yaw_classes:
            raise ValueError(f"unknown residual yaw classes: {invalid_yaw_classes}")
        self.residual_yaw_vector_head = (
            ResidualYawVectorHead(
                self.mod.BEV_X_MIN, self.mod.BEV_Y_MIN, self.mod.BEV_RES,
                256, len(DETECTION_NAMES),
                active_class_indices=tuple(NAME_TO_ID[name] for name in residual_yaw_classes),
            )
            if self.residual_yaw_vector_enabled else None
        )
        if self.attribute_head_enabled:
            self.attr_head = nn.Sequential(
                nn.Conv2d(256, 64, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, len(ATTRIBUTE_NAMES), kernel_size=1),
            )
        else:
            self.attr_head = None
        if self.temporal_bev:
            self.temporal_pre_encoder = (
                LightweightTemporalPreEncoder(256) if self.temporal_pre_encoder_enabled else None
            )
            self.temporal_fusion = nn.Sequential(
                nn.Conv2d(256 * 3, 64, kernel_size=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 256, kernel_size=1),
            )
            nn.init.zeros_(self.temporal_fusion[-1].weight)
            nn.init.zeros_(self.temporal_fusion[-1].bias)
            self.temporal_spatial_fusion = (
                LightweightSpatialTemporalFusion(256)
                if self.temporal_spatial_fusion_enabled else None
            )
            self.temporal_correlation_motion_head = (
                TemporalCorrelationMotionHead(256, len(DETECTION_NAMES))
                if self.temporal_correlation_motion_enabled else None
            )
            self.object_velocity_refiner = (
                ObjectCentricVelocityRefiner(
                    self.mod.BEV_RES, 256, len(DETECTION_NAMES),
                    active_class_indices=tuple(
                        NAME_TO_ID[name] for name in ("car", "truck", "bus", "motorcycle")
                    ),
                )
                if self.object_velocity_refiner_enabled else None
            )
        else:
            self.temporal_pre_encoder = None
            self.temporal_fusion = None
            self.temporal_spatial_fusion = None
            self.temporal_correlation_motion_head = None
            self.object_velocity_refiner = None
        self.temporal_stereo_depth_matcher = (
            TemporalStereoDepthMatcher(
                self.mod.NUM_DEPTH_BINS, self.mod.DEPTH_MIN, self.mod.DEPTH_MAX,
                self.mod.IMG_H, self.mod.IMG_W,
                dynamic_confidence=self.temporal_stereo_dynamic_confidence_enabled,
            )
            if self.temporal_stereo_depth_enabled else None
        )
        velocity_calibrator_class_names = tuple(
            velocity_calibrator_active_classes or (
                ("car", "truck", "bus", "motorcycle")
                if velocity_calibrator_motorcycle else
                ("car", "truck", "bus")
            )
        )
        invalid_velocity_classes = sorted(
            set(velocity_calibrator_class_names) - set(DETECTION_NAMES)
        )
        if invalid_velocity_classes:
            raise ValueError(
                f"unknown velocity calibrator classes: {invalid_velocity_classes}"
            )
        velocity_calibrator_class_indices = tuple(
            NAME_TO_ID[name] for name in velocity_calibrator_class_names
        )
        self.class_attribute_velocity_calibrator = (
            ClassAttributeVelocityCalibrator(
                256, len(DETECTION_NAMES),
                max_residual=float(velocity_calibrator_max_residual),
                active_class_indices=velocity_calibrator_class_indices,
                bypass_attribute_gate_indices=(
                    velocity_calibrator_class_indices
                    if velocity_calibrator_bypass_attribute_gate else ()
                ),
                initial_scales=velocity_calibrator_initial_scales,
            )
            if self.class_attribute_velocity_calibrator_enabled else None
        )
        if self.history_prediction_enabled:
            # Training-only short-term HoP branch. It reconstructs the previous-frame BEV
            # from the current BEV alone, then predicts previous-frame objects.
            self.hop_history_decoder = LightweightTemporalPreEncoder(256)
            self.hop_det_head = OfficialCenterHead(256, len(DETECTION_NAMES), 10)
        else:
            self.hop_history_decoder = None
            self.hop_det_head = None
        self.use_kd_adapter = bool(use_kd_adapter)
        if self.use_kd_adapter:
            self.kd_adapter = nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=1, bias=False),
            )
        else:
            self.kd_adapter = nn.Identity()

    def _depth_outputs_from_feature(self, feat, k, r, t):
        if self.camera_aware_depth:
            return self.depth_head(feat, k, r, t)
        return self.depth_head(feat)

    def _lift_feature(self, feat, k, r, t, depth_logits_override=None,
                      privileged_depth=None):
        depth_prob, depth_logits, feat2d = self._depth_outputs_from_feature(feat, k, r, t)
        if depth_logits_override is not None:
            depth_logits = depth_logits_override
        self._last_base_depth_logits = depth_logits
        if self.privileged_residual_adapter is not None:
            camera_depth_residual = (
                self.privileged_residual_adapter.predict_depth_residual(feat)
            )
            self._last_camera_depth_residual = camera_depth_residual
            depth_logits = depth_logits + camera_depth_residual
        else:
            self._last_camera_depth_residual = None
        if self.privileged_depth_completion is not None and privileged_depth is not None:
            depth_logits = self.privileged_depth_completion(depth_logits, privileged_depth)
        depth_prob = F.softmax(depth_logits, dim=1)
        bev = self.lss3d(feat2d, depth_prob, k, r, t)
        return bev, depth_logits, feat2d

    def _lift_view(self, img, k, r, t, return_feature=False, privileged_depth=None):
        bev, depth_logits, feat2d = self._lift_feature(
            self.backbone(img), k, r, t, privileged_depth=privileged_depth,
        )
        if return_feature:
            return bev, depth_logits, feat2d
        return bev, depth_logits

    def forward_outputs(self, img, k, r, t, prev_img=None, prev_k=None, prev_r=None, prev_t=None,
                        prev_to_current_r=None, prev_to_current_t=None, prev_valid=None,
                        return_history=False, privileged_depth=None,
                        bypass_privileged_path=False):
        if (
            self.privileged_depth_teacher_enabled and privileged_depth is None and
            not bypass_privileged_path
        ):
            raise ValueError(
                "project-native privileged teacher requires selected-camera LiDAR depth"
            )
        precomputed_previous_bev = None
        if self.temporal_stereo_depth_matcher is not None and prev_img is not None:
            if prev_valid is None:
                raise ValueError("temporal stereo depth requires prev_valid")
            current_feature = self.backbone(img)
            self._last_shared_image_feature = current_feature
            previous_feature = self.backbone(prev_img)
            _current_prob, mono_depth_logits, current_lift_feature = (
                self._depth_outputs_from_feature(current_feature, k, r, t)
            )
            previous_prob, _previous_logits, previous_lift_feature = (
                self._depth_outputs_from_feature(previous_feature, prev_k, prev_r, prev_t)
            )
            depth_logits = self.temporal_stereo_depth_matcher(
                mono_depth_logits, current_lift_feature, previous_lift_feature,
                k, r, t, prev_k, prev_r, prev_t,
                prev_to_current_r, prev_to_current_t, prev_valid,
            )
            self._last_base_depth_logits = depth_logits
            if self.privileged_residual_adapter is not None:
                camera_depth_residual = (
                    self.privileged_residual_adapter.predict_depth_residual(current_feature)
                )
                self._last_camera_depth_residual = camera_depth_residual
                depth_logits = depth_logits + camera_depth_residual
            else:
                self._last_camera_depth_residual = None
            if self.privileged_depth_completion is not None and not bypass_privileged_path:
                depth_logits = self.privileged_depth_completion(
                    depth_logits, privileged_depth,
                )
            bev = self.lss3d(
                current_lift_feature, F.softmax(depth_logits, dim=1), k, r, t,
            )
            precomputed_previous_bev = self.lss3d(
                previous_lift_feature, previous_prob, prev_k, prev_r, prev_t,
            )
        else:
            current_image_feature = self.backbone(img)
            self._last_shared_image_feature = current_image_feature
            bev, depth_logits, current_lift_feature = self._lift_feature(
                current_image_feature, k, r, t,
                privileged_depth=(None if bypass_privileged_path else privileged_depth),
            )
        history_outputs = None
        correlation_motion = None
        object_velocity_inputs = None
        residual_azimuth_output = None
        current_depth_residual_for_kd = self._last_camera_depth_residual
        current_base_depth_for_kd = self._last_base_depth_logits
        if return_history:
            if not self.history_prediction_enabled:
                raise ValueError("return_history requires history_prediction")
            if prev_to_current_r is None or prev_to_current_t is None:
                raise ValueError("history prediction requires the previous-to-current ego transform")
            pseudo_history_bev = warp_current_bev_to_previous(
                bev, prev_to_current_r, prev_to_current_t, self.mod,
            )
            pseudo_history_bev = self.hop_history_decoder(pseudo_history_bev)
            pseudo_history_ctx = self.bev_encoder(pseudo_history_bev)
            history_outputs = self.hop_det_head(pseudo_history_ctx)
        if self.temporal_fusion is not None and prev_img is not None:
            if precomputed_previous_bev is None:
                prev_bev, _prev_depth = self._lift_view(prev_img, prev_k, prev_r, prev_t)
            else:
                prev_bev = precomputed_previous_bev
            prev_bev = warp_previous_bev_to_current(
                prev_bev, prev_to_current_r, prev_to_current_t, self.mod,
            )
            if self.temporal_correlation_motion_head is not None:
                correlation_motion = self.temporal_correlation_motion_head(
                    bev, prev_bev, prev_valid,
                )
            if self.object_velocity_refiner is not None:
                object_velocity_inputs = (bev, prev_bev, prev_valid)
            if self.temporal_pre_encoder is not None:
                bev = self.temporal_pre_encoder(bev)
                prev_bev = self.temporal_pre_encoder(prev_bev)
            fusion_input = torch.cat((bev, prev_bev, torch.abs(bev - prev_bev)), dim=1)
            temporal_delta = self.temporal_fusion(fusion_input)
            if self.temporal_spatial_fusion is not None:
                temporal_delta = temporal_delta + self.temporal_spatial_fusion(fusion_input)
            temporal_delta = bound_temporal_delta(
                temporal_delta, self.temporal_delta_limit,
            )
            valid = prev_valid.to(dtype=bev.dtype).reshape(-1, 1, 1, 1)
            bev = bev + valid * temporal_delta
        # A previous-frame lift uses the same adapter but must not replace the
        # current-frame training seam consumed by privileged residual KD.
        self._last_camera_depth_residual = current_depth_residual_for_kd
        self._last_base_depth_logits = current_base_depth_for_kd
        # Training-only low-level BEV seam used by V45 instance residual KD.
        self._last_base_low_level_bev = bev
        self._last_low_level_bev = bev
        bev_ctx = self.bev_encoder(bev)
        if self.privileged_bev_refiner is not None and not bypass_privileged_path:
            bev_ctx = self.privileged_bev_refiner(bev_ctx)
        self._last_base_bev_ctx = bev_ctx
        if self.privileged_residual_adapter is not None:
            # The frozen V31 heatmap supplies a camera-only, inference-available
            # objectness gate. This prevents background memory from corrupting the
            # classification map, the failure observed in the first V55 gate.
            with torch.no_grad():
                base_hm_for_gate, _base_reg_for_gate = self.det_head(bev_ctx)
            object_gate = torch.sigmoid(base_hm_for_gate.detach()).amax(
                dim=1, keepdim=True,
            )
            object_gate = F.max_pool2d(object_gate, 7, stride=1, padding=3)
            memory_residual = (
                self.privileged_residual_adapter.predict_bev_residual(bev_ctx) *
                object_gate
            )
            self._last_memory_residual = memory_residual
            bev_ctx = bev_ctx + memory_residual
        else:
            self._last_memory_residual = None
        self._last_bev_ctx = bev_ctx
        hm, reg = self.det_head(bev_ctx)
        attr_logits = self.attr_head(bev_ctx) if self.attr_head is not None else None
        class_motion = self.class_motion_head(bev_ctx) if self.class_motion_head is not None else None
        if correlation_motion is not None:
            class_motion = class_motion + correlation_motion
        if object_velocity_inputs is not None:
            if class_motion is None:
                class_motion = reg.new_zeros(
                    reg.shape[0], len(DETECTION_NAMES), 4, reg.shape[-2], reg.shape[-1],
                )
            base_displacement = (
                reg[:, None, 8:10] + class_motion[:, :, 2:4]
            )
            velocity_residual = self.object_velocity_refiner(
                *object_velocity_inputs[:2], base_displacement,
                previous_valid=object_velocity_inputs[2],
            )
            class_motion = class_motion.clone()
            class_motion[:, :, 2:4] = class_motion[:, :, 2:4] + velocity_residual
        if self.class_attribute_velocity_calibrator is not None:
            if class_motion is None:
                class_motion = reg.new_zeros(
                    reg.shape[0], len(DETECTION_NAMES), 4, reg.shape[-2], reg.shape[-1],
                )
            base_displacement = reg[:, None, 8:10] + class_motion[:, :, 2:4]
            calibration_residual = self.class_attribute_velocity_calibrator(
                bev_ctx, base_displacement, attr_logits,
            )
            class_motion = class_motion.clone()
            class_motion[:, :, 2:4] = (
                class_motion[:, :, 2:4] + calibration_residual
            )
        if self.residual_azimuth_multibin_head is not None:
            if class_motion is None:
                class_motion = reg.new_zeros(
                    reg.shape[0], len(DETECTION_NAMES), 4, reg.shape[-2], reg.shape[-1],
                )
            base_yaw_vectors = reg[:, None, 6:8] + class_motion[:, :, 0:2]
            yaw_vector_residual, residual_azimuth_output = self.residual_azimuth_multibin_head(
                bev_ctx, base_yaw_vectors,
            )
            class_motion = class_motion.clone()
            class_motion[:, :, 0:2] = class_motion[:, :, 0:2] + yaw_vector_residual
        if self.residual_yaw_vector_head is not None:
            if class_motion is None:
                class_motion = reg.new_zeros(
                    reg.shape[0], len(DETECTION_NAMES), 4, reg.shape[-2], reg.shape[-1],
                )
            base_yaw_vectors = reg[:, None, 6:8] + class_motion[:, :, 0:2]
            yaw_vector_residual, residual_azimuth_output = self.residual_yaw_vector_head(
                bev_ctx, base_yaw_vectors,
            )
            class_motion = class_motion.clone()
            class_motion[:, :, 0:2] = class_motion[:, :, 0:2] + yaw_vector_residual
        azimuth_bins = self.azimuth_bin_head(bev_ctx) if self.azimuth_bin_head is not None else None
        class_size = self.class_size_head(bev_ctx) if self.class_size_head is not None else None
        class_center = self.class_center_head(bev_ctx) if self.class_center_head is not None else None
        image_yaw_flip_logits = (
            self.image_yaw_flip_head(current_lift_feature)
            if self.image_yaw_flip_head is not None else None
        )
        image_radial_center = (
            self.image_radial_center_head(current_lift_feature)
            if self.image_radial_center_head is not None else None
        )
        if self.image_object_geometry_head is not None:
            object_yaw_flip_logits, object_radial_center = (
                self.image_object_geometry_head(current_lift_feature, depth_logits)
            )
        else:
            object_yaw_flip_logits = object_radial_center = None
        outputs = (hm, reg, depth_logits, self.kd_adapter(bev_ctx), attr_logits, class_motion)
        # Preserve the historical six-output contract when both V12 heads are disabled.
        if self.azimuth_bin_head is not None or self.class_size_head is not None:
            outputs = outputs + (azimuth_bins, class_size)
        if class_center is not None:
            outputs = outputs + (class_center,)
        if image_yaw_flip_logits is not None:
            outputs = outputs + (image_yaw_flip_logits,)
        if image_radial_center is not None:
            outputs = outputs + (image_radial_center,)
        if object_yaw_flip_logits is not None:
            outputs = outputs + (object_yaw_flip_logits, object_radial_center)
        if residual_azimuth_output is not None:
            outputs = outputs + (residual_azimuth_output,)
        if history_outputs is not None:
            outputs = outputs + history_outputs
        return outputs

    def forward(self, img, k, r, t):
        return self.forward_outputs(img, k, r, t)[:4]


def fuse_bev_views(bev_views, occupancy_eps: float = 1e-6):
    """Fuse camera BEVs by a per-cell mean, preserving single-view feature scale."""
    if not bev_views:
        raise ValueError("at least one BEV view is required")
    fused = torch.zeros_like(bev_views[0], dtype=torch.float32)
    coverage = torch.zeros_like(bev_views[0][:, :1], dtype=torch.float32)
    for bev in bev_views:
        fused = fused + bev.float()
        occupied = bev.detach().float().abs().amax(dim=1, keepdim=True) > float(occupancy_eps)
        coverage = coverage + occupied.float()
    return fused / coverage.clamp(min=1.0)


class SixViewFusedTeacher(OfficialStudent):
    """A shared lightweight image encoder whose six lifted views are fused before BEV encoding."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.image_encoder_frozen = False

    def freeze_image_encoder(self):
        """Keep the proven single-view image/depth representation fixed while learning BEV fusion."""
        self.image_encoder_frozen = True
        for module in (self.backbone, self.depth_head):
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        if mode and self.image_encoder_frozen:
            self.backbone.eval()
            self.depth_head.eval()
        return self

    def forward(self, imgs6, ks6, r6, t6, return_bev=False):
        _batch, views = imgs6.shape[:2]
        bev_views = []
        depth_logits = []
        for view in range(views):
            bev_view, depth_view = self._lift_view(
                imgs6[:, view], ks6[:, view], r6[:, view], t6[:, view],
            )
            bev_views.append(bev_view)
            depth_logits.append(depth_view)
        fused_bev = fuse_bev_views(bev_views)
        bev_ctx = self.bev_encoder(fused_bev)
        hm, reg = self.det_head(bev_ctx)
        depth_all = torch.stack(depth_logits, dim=1)
        if return_bev:
            return hm, reg, depth_all, bev_ctx
        return hm, reg, depth_all


def instantiate_student_lazy(mod, model: OfficialStudent, device):
    model.to(device)
    model.eval()
    with torch.no_grad():
        img = torch.zeros((1, 3, mod.IMG_H, mod.IMG_W), device=device)
        k = torch.eye(3, device=device).unsqueeze(0)
        r = torch.eye(3, device=device).unsqueeze(0)
        t = torch.zeros((1, 3), device=device)
        privileged_depth = None
        if model.privileged_depth_teacher_enabled:
            privileged_depth = torch.full(
                (1, mod.FEAT_H, mod.FEAT_W), -1,
                dtype=torch.long, device=device,
            )
        model.forward_outputs(
            img, k, r, t, privileged_depth=privileged_depth,
        )
    model.train()


def build_official_student_from_args(mod, args, *, pretrained_backbone: bool, use_kd_adapter: bool):
    """Build the deployable student architecture from one audited argument set.

    The aligned self-distillation teacher must be structurally identical to the
    student.  Keeping construction in one helper prevents a silent teacher/student
    mismatch when a new temporal or class-aware head is added later.
    """
    return OfficialStudent(
        mod,
        pretrained_backbone=pretrained_backbone,
        use_kd_adapter=use_kd_adapter,
        backbone_variant=args.student_backbone,
        camera_aware_depth=args.camera_aware_depth,
        bev_norm=args.bev_norm,
        attribute_head=args.attribute_head,
        temporal_bev=args.temporal_bev,
        temporal_stereo_depth=args.temporal_stereo_depth,
        temporal_stereo_dynamic_confidence=args.temporal_stereo_dynamic_confidence,
        temporal_pre_encoder=args.temporal_pre_encoder,
        temporal_delta_limit=args.temporal_delta_limit,
        history_prediction=(args.hop_hm_weight + args.hop_reg_weight > 0),
        class_aware_motion=args.class_aware_motion_head,
        temporal_spatial_fusion=args.temporal_spatial_fusion,
        azimuth_multibin=args.azimuth_multibin,
        class_aware_size=args.class_aware_size_head,
        class_aware_center=args.class_aware_center_head,
        center_refiner_active_classes=args.center_refiner_active_classes,
        image_yaw_flip=args.image_yaw_flip_head,
        image_radial_center=args.image_radial_center_head,
        image_radial_center_max_residual=args.image_radial_center_max_residual,
        image_object_geometry=args.image_object_geometry_head,
        temporal_correlation_motion=args.temporal_correlation_motion,
        object_velocity_refiner=args.object_velocity_refiner,
        residual_azimuth_multibin=args.residual_azimuth_multibin,
        residual_yaw_vector=args.residual_yaw_vector,
        residual_yaw_active_classes=args.residual_yaw_active_classes,
        class_attribute_velocity_calibrator=args.class_attribute_velocity_calibrator,
        velocity_calibrator_motorcycle=args.velocity_calibrator_motorcycle,
        velocity_calibrator_active_classes=args.velocity_calibrator_active_classes,
        velocity_calibrator_bypass_attribute_gate=args.velocity_calibrator_bypass_attribute_gate,
        velocity_calibrator_max_residual=args.velocity_calibrator_max_residual,
        velocity_calibrator_initial_scales=args.velocity_calibrator_initial_scales,
        privileged_depth_teacher=getattr(args, "privileged_depth_teacher", False),
        camera_privileged_memory_adapter=getattr(
            args, "camera_privileged_memory_adapter", False,
        ),
        privileged_memory_slots=getattr(args, "privileged_memory_slots", 32),
        privileged_memory_latent_dim=getattr(
            args, "privileged_memory_latent_dim", 32,
        ),
    )


def aligned_student_teacher_outputs(
    teacher: OfficialStudent,
    heatmap_target: torch.Tensor,
    img: torch.Tensor,
    k: torch.Tensor,
    r: torch.Tensor,
    t: torch.Tensor,
    temporal_inputs: Optional[Sequence[torch.Tensor]] = None,
    return_auxiliary: bool = False,
    privileged_depth: Optional[torch.Tensor] = None,
    bypass_privileged_path: bool = False,
):
    """Run an in-domain teacher on exactly the student's selected camera input.

    Regression heads in ``OfficialStudent`` are class-conditioned.  Distilling the
    raw ten-channel base head would therefore teach a different box from the one
    deployed at inference.  This helper applies the teacher's class-aware center,
    motion and size residuals using the ground-truth class at supervised centres.
    """
    if temporal_inputs is None:
        outputs = teacher.forward_outputs(
            img, k, r, t, privileged_depth=privileged_depth,
            bypass_privileged_path=bypass_privileged_path,
        )
    else:
        outputs = teacher.forward_outputs(
            img, k, r, t, *temporal_inputs,
            privileged_depth=privileged_depth,
            bypass_privileged_path=bypass_privileged_path,
        )
    heatmap, regression, depth, bev, attributes, class_motion = outputs[:6]
    offset = 6
    class_size = None
    if teacher.azimuth_bin_head is not None or teacher.class_size_head is not None:
        _azimuth_bins, class_size = outputs[offset:offset + 2]
        offset += 2
    class_center = None
    if teacher.class_center_head is not None:
        class_center = outputs[offset]

    class_indices = heatmap_target.argmax(dim=1)
    regression = apply_class_center_residual(regression, class_center, class_indices)
    regression = apply_class_motion_residual(regression, class_motion, class_indices)
    regression = apply_class_size_residual(regression, class_size, class_indices)
    if return_auxiliary:
        return heatmap, regression, depth, bev, attributes
    return heatmap, regression, depth, bev


def sample_bev_object_relations(
    feature: torch.Tensor,
    keypoints_xy: torch.Tensor,
    object_mask: torch.Tensor,
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    spatial_axes: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample object-interior BEV features and form channel-free relations.

    ``spatial_axes`` describes the tensor's last two dimensions. The project
    student stores BEV as H=x, W=y (``xy``), whereas CenterPoint follows the
    conventional H=y, W=x layout (``yx``). Returning a 9x9 cosine-relation
    matrix removes the need to align the two networks' channel dimensions.
    Objects whose nine points leave either BEV range are excluded rather than
    silently sampling grid padding.
    """
    if feature.ndim != 4:
        raise ValueError(f"feature must be BCHW, got {tuple(feature.shape)}")
    if keypoints_xy.ndim != 4 or keypoints_xy.shape[-1] != 2:
        raise ValueError(
            "keypoints_xy must be [batch, objects, points, 2], got "
            f"{tuple(keypoints_xy.shape)}"
        )
    if object_mask.shape != keypoints_xy.shape[:2]:
        raise ValueError(
            f"object_mask {tuple(object_mask.shape)} does not match "
            f"keypoints {tuple(keypoints_xy.shape[:2])}"
        )
    if spatial_axes not in {"xy", "yx"}:
        raise ValueError(f"spatial_axes must be 'xy' or 'yx', got {spatial_axes}")
    if feature.shape[0] != keypoints_xy.shape[0]:
        raise ValueError("feature and keypoint batch sizes differ")

    feature = feature.float()
    keypoints_xy = keypoints_xy.to(device=feature.device, dtype=torch.float32)
    object_mask = object_mask.to(device=feature.device).bool()
    x = keypoints_xy[..., 0]
    y = keypoints_xy[..., 1]
    x_min, x_max = (float(x_range[0]), float(x_range[1]))
    y_min, y_max = (float(y_range[0]), float(y_range[1]))
    if not x_max > x_min or not y_max > y_min:
        raise ValueError("invalid physical BEV range")
    keypoint_valid = (
        torch.isfinite(keypoints_xy).all(dim=-1) &
        (x >= x_min) & (x < x_max) &
        (y >= y_min) & (y < y_max)
    )
    valid_objects = object_mask & keypoint_valid.all(dim=-1)

    norm_x = 2.0 * (x - x_min) / (x_max - x_min) - 1.0
    norm_y = 2.0 * (y - y_min) / (y_max - y_min) - 1.0
    if spatial_axes == "yx":
        # grid_sample coordinates are always (W, H): W=x and H=y.
        grid = torch.stack((norm_x, norm_y), dim=-1)
    else:
        # Student tensor H=x, W=y, so grid_sample's (W, H) is (y, x).
        grid = torch.stack((norm_y, norm_x), dim=-1)
    batch, objects, points, _ = grid.shape
    sampled = F.grid_sample(
        feature,
        grid.reshape(batch, objects * points, 1, 2),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )
    sampled = sampled.squeeze(-1).reshape(
        batch, feature.shape[1], objects, points,
    ).permute(0, 2, 3, 1)
    sampled = F.normalize(sampled, dim=-1, eps=1e-6)
    relations = torch.einsum("bmpc,bmqc->bmpq", sampled, sampled)
    relations = relations * valid_objects[..., None, None].to(relations.dtype)
    return relations, valid_objects


def sample_bev_object_features(
    feature: torch.Tensor,
    keypoints_xy: torch.Tensor,
    object_mask: torch.Tensor,
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    spatial_axes: str = "xy",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample normalized features at the nine audited object keypoints."""
    if feature.ndim != 4:
        raise ValueError(f"feature must be BCHW, got {tuple(feature.shape)}")
    if keypoints_xy.ndim != 4 or keypoints_xy.shape[-1] != 2:
        raise ValueError("keypoints_xy must be [batch, objects, points, 2]")
    if object_mask.shape != keypoints_xy.shape[:2]:
        raise ValueError("object mask and keypoint object dimensions differ")
    if spatial_axes not in {"xy", "yx"}:
        raise ValueError("spatial_axes must be xy or yx")
    feature = feature.float()
    keypoints_xy = keypoints_xy.to(feature.device, dtype=torch.float32)
    object_mask = object_mask.to(feature.device).bool()
    x, y = keypoints_xy[..., 0], keypoints_xy[..., 1]
    x_min, x_max = map(float, x_range)
    y_min, y_max = map(float, y_range)
    valid_points = (
        torch.isfinite(keypoints_xy).all(dim=-1) &
        (x >= x_min) & (x < x_max) &
        (y >= y_min) & (y < y_max)
    )
    valid_objects = object_mask & valid_points.all(dim=-1)
    norm_x = 2.0 * (x - x_min) / (x_max - x_min) - 1.0
    norm_y = 2.0 * (y - y_min) / (y_max - y_min) - 1.0
    grid = (
        torch.stack((norm_x, norm_y), dim=-1)
        if spatial_axes == "yx" else
        torch.stack((norm_y, norm_x), dim=-1)
    )
    batch, objects, points, _ = grid.shape
    sampled = F.grid_sample(
        feature, grid.reshape(batch, objects * points, 1, 2),
        mode="bilinear", padding_mode="zeros", align_corners=False,
    )
    sampled = sampled.squeeze(-1).reshape(
        batch, feature.shape[1], objects, points,
    ).permute(0, 2, 3, 1)
    sampled = F.normalize(sampled, dim=-1, eps=1e-6)
    sampled = sampled * valid_objects[..., None, None].to(sampled.dtype)
    return sampled, valid_objects


def masked_object_feature_loss(
    student_features: torch.Tensor,
    teacher_features: torch.Tensor,
    valid_objects: torch.Tensor,
) -> torch.Tensor:
    """Direct same-domain feature alignment at object-interior keypoints."""
    if student_features.shape != teacher_features.shape:
        raise ValueError("student and teacher keypoint feature shapes differ")
    if valid_objects.shape != student_features.shape[:2]:
        raise ValueError("valid object mask does not match sampled features")
    raw = F.smooth_l1_loss(
        student_features.float(), teacher_features.detach().float(),
        reduction="none", beta=0.25,
    )
    weights = valid_objects[..., None, None].to(raw.dtype)
    denominator = weights.sum().clamp(min=1.0) * raw.shape[-2] * raw.shape[-1]
    return (raw * weights).sum() / denominator


def instance_normalized_foreground_feature_loss(
    student_feature: torch.Tensor,
    teacher_feature: torch.Tensor,
    regression_target: torch.Tensor,
    regression_mask: torch.Tensor,
    bev_resolution: float,
    scales: Sequence[int] = (1, 2, 4),
) -> torch.Tensor:
    """Match foreground geometry with equal weight per GT instance."""
    if student_feature.shape != teacher_feature.shape:
        raise ValueError("student and teacher foreground features must align")
    if regression_mask.shape != regression_target[:, :1].shape:
        raise ValueError("regression mask does not align with target")
    if float(bev_resolution) <= 0:
        raise ValueError("bev_resolution must be positive")
    target = regression_target.detach().float()
    active = regression_mask[:, 0].detach() > 0.5
    object_losses = []
    for scale in tuple(int(value) for value in scales):
        if scale <= 0:
            raise ValueError("feature scales must be positive")
        if scale == 1:
            student_scaled, teacher_scaled = student_feature, teacher_feature
        else:
            size = (
                max(1, student_feature.shape[-2] // scale),
                max(1, student_feature.shape[-1] // scale),
            )
            student_scaled = F.adaptive_avg_pool2d(student_feature, size)
            teacher_scaled = F.adaptive_avg_pool2d(teacher_feature, size)
        difference = F.smooth_l1_loss(
            F.normalize(student_scaled.float(), dim=1),
            F.normalize(teacher_scaled.detach().float(), dim=1),
            reduction="none", beta=0.25,
        ).mean(dim=1)
        for batch_index in range(target.shape[0]):
            for centre in active[batch_index].nonzero(as_tuple=False):
                ix, iy = int(centre[0]), int(centre[1])
                values = target[batch_index, :, ix, iy]
                centre_x = (float(ix) + float(values[0])) / scale
                centre_y = (float(iy) + float(values[1])) / scale
                radius_x = max(
                    0.5 * float(torch.exp(values[4]).clamp(max=100.0)) /
                    float(bev_resolution) / scale, 0.75,
                )
                radius_y = max(
                    0.5 * float(torch.exp(values[3]).clamp(max=100.0)) /
                    float(bev_resolution) / scale, 0.75,
                )
                sine, cosine = float(values[6]), float(values[7])
                norm = max((sine * sine + cosine * cosine) ** 0.5, 1e-6)
                sine, cosine = sine / norm, cosine / norm
                support = int(math.ceil(2.5 * max(radius_x, radius_y)))
                x0, x1 = max(0, int(centre_x) - support), min(
                    difference.shape[-2], int(centre_x) + support + 1,
                )
                y0, y1 = max(0, int(centre_y) - support), min(
                    difference.shape[-1], int(centre_y) + support + 1,
                )
                if x0 >= x1 or y0 >= y1:
                    continue
                gx = torch.arange(x0, x1, device=difference.device).float()
                gy = torch.arange(y0, y1, device=difference.device).float()
                dx, dy = gx[:, None] - centre_x, gy[None, :] - centre_y
                local_x = cosine * dx + sine * dy
                local_y = -sine * dx + cosine * dy
                squared = (local_x / radius_x) ** 2 + (local_y / radius_y) ** 2
                weight = torch.exp(-0.5 * squared) * (squared <= 6.25)
                object_losses.append(
                    (difference[batch_index, x0:x1, y0:y1] * weight).sum() /
                    weight.sum().clamp(min=1e-6)
                )
    if not object_losses:
        return student_feature.sum() * 0.0
    return torch.stack(object_losses).mean()


def sparse_local_relative_depth_loss(
    depth_logits: torch.Tensor,
    depth_target: torch.Tensor,
    offsets: Sequence[int] = (1, 2, 4, 8),
) -> torch.Tensor:
    """Preserve local LiDAR depth ordering in addition to per-pixel CE.

    Sparse LiDAR projections often provide several points along the same scan
    line.  Matching their relative bin displacement transfers local geometry
    without asking the image student to reproduce LiDAR-only BEV responses.
    """
    if depth_logits.ndim != 4 or depth_target.shape != depth_logits.shape[:1] + depth_logits.shape[2:]:
        raise ValueError("depth target must match the BHW dimensions of logits")
    valid_offsets = tuple(dict.fromkeys(int(value) for value in offsets))
    if not valid_offsets or any(value <= 0 for value in valid_offsets):
        raise ValueError("relative depth offsets must be positive")
    probabilities = F.softmax(depth_logits.float(), dim=1)
    bins = torch.arange(
        depth_logits.shape[1], device=depth_logits.device, dtype=torch.float32,
    ).reshape(1, -1, 1, 1)
    expectation = (probabilities * bins).sum(dim=1)
    target = depth_target.to(depth_logits.device).float()
    valid = depth_target.to(depth_logits.device) >= 0
    total = expectation.new_tensor(0.0)
    count = expectation.new_tensor(0.0)
    normalizer = max(float(depth_logits.shape[1] - 1), 1.0)
    for offset in valid_offsets:
        if offset < expectation.shape[-1]:
            pair_valid = valid[:, :, :-offset] & valid[:, :, offset:]
            if pair_valid.any():
                pred_delta = (
                    expectation[:, :, offset:] - expectation[:, :, :-offset]
                ) / normalizer
                target_delta = (
                    target[:, :, offset:] - target[:, :, :-offset]
                ) / normalizer
                raw = F.smooth_l1_loss(pred_delta, target_delta, reduction="none", beta=0.1)
                total = total + raw[pair_valid].sum()
                count = count + pair_valid.sum()
        if offset < expectation.shape[-2]:
            pair_valid = valid[:, :-offset, :] & valid[:, offset:, :]
            if pair_valid.any():
                pred_delta = (
                    expectation[:, offset:, :] - expectation[:, :-offset, :]
                ) / normalizer
                target_delta = (
                    target[:, offset:, :] - target[:, :-offset, :]
                ) / normalizer
                raw = F.smooth_l1_loss(pred_delta, target_delta, reduction="none", beta=0.1)
                total = total + raw[pair_valid].sum()
                count = count + pair_valid.sum()
    return total / count.clamp(min=1.0)


def masked_object_relation_loss(
    student_relations: torch.Tensor,
    teacher_relations: torch.Tensor,
    valid_objects: torch.Tensor,
) -> torch.Tensor:
    """Smooth-L1 relation KD averaged over valid objects, not padding."""
    if student_relations.shape != teacher_relations.shape:
        raise ValueError(
            f"relation shapes differ: {tuple(student_relations.shape)} vs "
            f"{tuple(teacher_relations.shape)}"
        )
    if valid_objects.shape != student_relations.shape[:2]:
        raise ValueError("valid-object mask shape does not match relation tensors")
    per_element = F.smooth_l1_loss(
        student_relations.float(), teacher_relations.detach().float(),
        reduction="none", beta=0.5,
    )
    weights = valid_objects[..., None, None].to(per_element.dtype)
    denominator = (
        weights.sum().clamp(min=1.0) *
        per_element.shape[-1] * per_element.shape[-2]
    )
    return (per_element * weights).sum() / denominator


class CenterPointRelationTeacher(nn.Module):
    """Frozen LiDAR teacher that exposes only per-object relation matrices."""

    def __init__(self, config_path: str, checkpoint_path: str):
        super().__init__()
        config_path = str(Path(config_path).resolve())
        checkpoint_path = str(Path(checkpoint_path).resolve())
        if not Path(config_path).is_file():
            raise FileNotFoundError(f"CenterPoint config not found: {config_path}")
        if not Path(checkpoint_path).is_file():
            raise FileNotFoundError(f"CenterPoint checkpoint not found: {checkpoint_path}")
        mmdet3d_root = Path(os.getenv(
            "MMDET3D_ROOT",
            "/root/autodl-tmp/BEVnext/third_party/mmdetection3d",
        )).resolve()
        if not mmdet3d_root.is_dir():
            raise FileNotFoundError(f"MMDetection3D source tree not found: {mmdet3d_root}")
        if str(mmdet3d_root) not in sys.path:
            sys.path.insert(0, str(mmdet3d_root))
        try:
            from mmengine.config import Config
            from mmengine.runner import load_checkpoint
            from mmdet3d.registry import MODELS
            from mmdet3d.utils import register_all_modules
        except ImportError as exc:
            raise RuntimeError(
                "CenterPoint relation KD requires the configured MMDetection3D runtime"
            ) from exc
        register_all_modules(init_default_scope=True)
        cfg = Config.fromfile(config_path)
        self.detector = MODELS.build(cfg.model)
        load_checkpoint(self.detector, checkpoint_path, map_location="cpu", strict=True)
        self.detector.eval()
        for parameter in self.detector.parameters():
            parameter.requires_grad_(False)
        point_cloud_range = list(cfg.get(
            "point_cloud_range", [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
        ))
        self.x_range = (float(point_cloud_range[0]), float(point_cloud_range[3]))
        self.y_range = (float(point_cloud_range[1]), float(point_cloud_range[4]))

    def to(self, *args, **kwargs):
        module = super().to(*args, **kwargs)
        # MMEngine's BaseDataPreprocessor tracks its device in a non-parameter
        # attribute, so moving the enclosing detector is not sufficient.
        detector_device = next(self.detector.parameters()).device
        self.detector.data_preprocessor.to(detector_device)
        return module

    @staticmethod
    def _extract_bev_feature(extracted) -> torch.Tensor:
        point_features = extracted[1] if isinstance(extracted, tuple) else extracted
        if isinstance(point_features, (list, tuple)):
            point_features = point_features[0]
        if not torch.is_tensor(point_features) or point_features.ndim != 4:
            raise RuntimeError(
                "unexpected CenterPoint extract_feat output; expected BCHW point feature"
            )
        return point_features

    @torch.no_grad()
    def forward(
        self,
        padded_points: torch.Tensor,
        point_counts: torch.Tensor,
        lidar_keypoints_xy: torch.Tensor,
        object_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        relations = []
        valid_masks = []
        # Process one point cloud at a time to bound memory while the student's
        # computation graph remains live for back-propagation.
        with torch.cuda.amp.autocast(enabled=False):
            for batch_index in range(padded_points.shape[0]):
                count = int(point_counts[batch_index].item())
                if count <= 0:
                    raise RuntimeError("empty LiDAR frame in relation KD batch")
                points = padded_points[batch_index, :count].float()
                if self.detector.data_preprocessor.device != points.device:
                    raise RuntimeError(
                        "CenterPoint data preprocessor/device mismatch: "
                        f"{self.detector.data_preprocessor.device} vs {points.device}"
                    )
                prepared = self.detector.data_preprocessor(
                    {"inputs": {"points": [points]}}, training=False,
                )
                extracted = self.detector.extract_feat(prepared["inputs"], [{}])
                feature = self._extract_bev_feature(extracted)
                relation, valid = sample_bev_object_relations(
                    feature,
                    lidar_keypoints_xy[batch_index:batch_index + 1],
                    object_mask[batch_index:batch_index + 1],
                    self.x_range,
                    self.y_range,
                    spatial_axes="yx",
                )
                relations.append(relation.squeeze(0))
                valid_masks.append(valid.squeeze(0))
        return torch.stack(relations, dim=0).detach(), torch.stack(valid_masks, dim=0)


def make_transforms(mod):
    transform_teacher = T.Compose([
        T.ToTensor(),
        T.Resize((mod.TCH_H, mod.TCH_W), antialias=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    transform_student = T.Compose([
        T.ToTensor(),
        T.Resize((mod.IMG_H, mod.IMG_W), antialias=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform_teacher, transform_student


def make_loaders(mod, args):
    tt, ts = make_transforms(mod)
    is_teacher = args.cmd in {"train-teacher", "train-fused-teacher"}
    has_kd_teacher = args.cmd == "train-student" and (
        args.kd_hm_weight + args.kd_reg_weight + args.kd_feat_weight +
        args.kd_depth_weight + float(getattr(args, "kd_relation_weight", 0.0)) +
        float(getattr(args, "kd_keypoint_feature_weight", 0.0)) +
        float(getattr(args, "kd_attr_weight", 0.0)) > 0
    )
    # An aligned in-domain teacher consumes the exact same selected camera and
    # history tensors as the student.  Loading all six teacher images here would
    # add substantial I/O and preprocessing without changing the loss.
    teacher_type = getattr(args, "teacher_type", "legacy")
    need_teacher_inputs = is_teacher or (
        has_kd_teacher and teacher_type not in {
            "aligned_student", "project_privileged",
            "centerpoint_relation", "camera_ta_relation",
        }
    )
    need_student_inputs = args.cmd == "train-student"
    need_teacher_depth = is_teacher and args.depth_weight > 0
    need_student_depth = args.cmd == "train-student" and (
        args.depth_weight > 0 or float(getattr(args, "relative_depth_weight", 0.0)) > 0 or
        bool(getattr(args, "privileged_depth_teacher", False)) or
        (has_kd_teacher and teacher_type == "project_privileged")
    )
    need_temporal_inputs = args.cmd == "train-student" and getattr(args, "temporal_bev", False)
    need_history_targets = args.cmd == "train-student" and (
        float(getattr(args, "hop_hm_weight", 0.0)) + float(getattr(args, "hop_reg_weight", 0.0)) > 0
    )
    need_lidar_relation_inputs = (
        args.cmd == "train-student" and
        (
            (
                teacher_type in {"centerpoint_relation", "camera_ta_relation"} and
                (
                    float(getattr(args, "kd_relation_weight", 0.0)) +
                    float(getattr(args, "kd_keypoint_feature_weight", 0.0))
                ) > 0
            ) or
            (
                teacher_type == "project_privileged" and
                bool(getattr(args, "kd_project_v45_instance_residual", False))
            )
        )
    )
    target_scope = resolve_target_scope(args)
    train_view_mode = getattr(args, "student_train_views", "front") if need_student_inputs else "front"
    val_view_mode = getattr(args, "student_val_views", "front") if need_student_inputs else "front"
    student_camera = getattr(args, "student_camera", "CAM_FRONT")
    print(
        f"[DATA] teacher_inputs={need_teacher_inputs} student_inputs={need_student_inputs} "
        f"teacher_depth={need_teacher_depth} student_depth={need_student_depth} target_scope={target_scope} "
        f"temporal={need_temporal_inputs} history_targets={need_history_targets} "
        f"lidar_relation={need_lidar_relation_inputs} "
        f"train_views={train_view_mode} val_views={val_view_mode}"
    )
    train_set = NuScenesOfficialDetectionDataset(
        mod, args.data_root, args.version, "train", tt, ts,
        max_samples=args.train_max_samples, build_depth=False,
        need_teacher_inputs=need_teacher_inputs, need_student_inputs=need_student_inputs,
        need_teacher_depth=need_teacher_depth, need_student_depth=need_student_depth,
        need_temporal_inputs=need_temporal_inputs,
        need_history_targets=need_history_targets,
        target_scope=target_scope,
        student_view_mode=train_view_mode,
        student_camera=student_camera,
        pseudo_labels=getattr(args, "pseudo_labels", "") if args.cmd == "train-student" else "",
        pseudo_score_thr=getattr(args, "pseudo_score_thr", 0.2),
        pseudo_class_score_thrs=getattr(args, "pseudo_class_score_thrs", None),
        subset_mode=getattr(args, "train_subset_mode", "random"),
        subset_seed=args.seed,
        need_lidar_relation_inputs=need_lidar_relation_inputs,
        lidar_relation_keypoints_only=bool(
            getattr(args, "kd_project_v45_instance_residual", False)
        ),
        lidar_relation_max_points=getattr(args, "lidar_relation_max_points", 35000),
        lidar_relation_max_objects=getattr(args, "lidar_relation_max_objects", 128),
        lidar_relation_cache=getattr(args, "lidar_relation_cache", ""),
    )
    val_set = NuScenesOfficialDetectionDataset(
        mod, args.data_root, args.version, "val", tt, ts,
        max_samples=args.val_max_samples, build_depth=False,
        need_teacher_inputs=need_teacher_inputs, need_student_inputs=need_student_inputs,
        need_teacher_depth=need_teacher_depth, need_student_depth=need_student_depth,
        need_temporal_inputs=need_temporal_inputs,
        need_history_targets=need_history_targets,
        target_scope=target_scope,
        student_view_mode=val_view_mode,
        student_camera=student_camera,
        pseudo_labels="",
        subset_mode=getattr(args, "val_subset_mode", "first"),
        subset_seed=args.seed,
        need_lidar_relation_inputs=need_lidar_relation_inputs,
        lidar_relation_keypoints_only=bool(
            getattr(args, "kd_project_v45_instance_residual", False)
        ),
        lidar_relation_max_points=getattr(args, "lidar_relation_max_points", 35000),
        lidar_relation_max_objects=getattr(args, "lidar_relation_max_objects", 128),
        lidar_relation_cache=getattr(args, "lidar_relation_cache", ""),
    )
    train_sampler = None
    train_shuffle = True
    if train_view_mode == "balanced":
        target_view_classes = getattr(args, "velocity_target_view_classes", None)
        if target_view_classes:
            train_sampler = TargetClassViewSampler(
                train_set, target_view_classes, seed=args.seed,
                epoch_size=getattr(args, "velocity_target_view_epoch_size", 0),
                state_balanced_classes=getattr(
                    args, "velocity_target_state_balanced_classes", None,
                ) or (),
            )
        elif getattr(args, "class_balanced_sampling", False):
            train_sampler = ClassBalancedRandomViewSampler(
                train_set, seed=args.seed,
                max_oversample=getattr(args, "class_balance_cap", 3.0),
            )
        else:
            train_sampler = BalancedRandomViewSampler(train_set, seed=args.seed)
        train_shuffle = False
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=train_shuffle, sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True, persistent_workers=args.num_workers > 0, drop_last=True
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        pin_memory=True, persistent_workers=args.num_workers > 0, drop_last=False
    )
    return train_loader, val_loader


def configure_bev(mod, args):
    if getattr(args, "depth_bins", 0) > 0:
        mod.NUM_DEPTH_BINS = int(args.depth_bins)
    if getattr(args, "teacher_img_w", 0) > 0:
        mod.TCH_W = int(args.teacher_img_w)
    if getattr(args, "teacher_img_h", 0) > 0:
        mod.TCH_H = int(args.teacher_img_h)
    if getattr(args, "student_img_w", 0) > 0:
        mod.IMG_W = int(args.student_img_w)
    if getattr(args, "student_img_h", 0) > 0:
        mod.IMG_H = int(args.student_img_h)
    if mod.IMG_W % 8 != 0 or mod.IMG_H % 8 != 0:
        raise ValueError(f"student image size must be divisible by 8, got {mod.IMG_W}x{mod.IMG_H}")
    mod.FEAT_W = int(mod.IMG_W // 8)
    mod.FEAT_H = int(mod.IMG_H // 8)
    mod.BEV_X_MIN = float(args.bev_x_min)
    mod.BEV_X_MAX = float(args.bev_x_max)
    mod.BEV_Y_MIN = float(args.bev_y_min)
    mod.BEV_Y_MAX = float(args.bev_y_max)
    mod.BEV_RES = float(args.bev_res)
    mod.BEV_H = int(round((mod.BEV_X_MAX - mod.BEV_X_MIN) / mod.BEV_RES))
    mod.BEV_W = int(round((mod.BEV_Y_MAX - mod.BEV_Y_MIN) / mod.BEV_RES))
    print(
        f"[BEV] x=[{mod.BEV_X_MIN},{mod.BEV_X_MAX}] "
        f"y=[{mod.BEV_Y_MIN},{mod.BEV_Y_MAX}] res={mod.BEV_RES} "
        f"grid=({mod.BEV_H},{mod.BEV_W}) teacher_img=({mod.TCH_W}x{mod.TCH_H}) "
        f"student_img=({mod.IMG_W}x{mod.IMG_H}) depth_bins={mod.NUM_DEPTH_BINS}"
    )


def apply_student_exp_preset(args):
    if not hasattr(args, "exp") or args.exp == "custom":
        return

    presets = {
        # GT only, no teacher loss.
        "A0": dict(kd_hm_weight=0.0, kd_reg_weight=0.0, kd_feat_weight=0.0, kd_depth_weight=0.0),
        # Detection-output KD: class heatmap + 3D box regression.
        "A1": dict(kd_hm_weight=0.5, kd_reg_weight=0.2, kd_feat_weight=0.0, kd_depth_weight=0.0),
        # BEV feature KD only. Keep the weight conservative because feature KD is a regularizer.
        "A2": dict(kd_hm_weight=0.0, kd_reg_weight=0.0, kd_feat_weight=0.05, kd_depth_weight=0.0),
        # Front-camera depth distribution KD only.
        "A3": dict(kd_hm_weight=0.0, kd_reg_weight=0.0, kd_feat_weight=0.0, kd_depth_weight=0.03),
        # Full KD: output KD is primary; feature/depth KD are weaker auxiliary regularizers.
        "A4": dict(kd_hm_weight=0.5, kd_reg_weight=0.2, kd_feat_weight=0.05, kd_depth_weight=0.03),
    }
    for key, value in presets[args.exp].items():
        setattr(args, key, value)


def teacher_forward_loss(model, batch, args, device):
    imgs6, ks6, r6, t6 = [x.to(device, non_blocking=True) for x in batch[1:5]]
    hm_gt, reg_gt, reg_mask, vel_mask, depth6 = [x.to(device, non_blocking=True) for x in batch[9:14]]
    hm, reg, depth_all = model(imgs6, ks6, r6, t6)
    if not torch.isfinite(hm).all() or not torch.isfinite(reg).all() or not torch.isfinite(depth_all).all():
        raise FloatingPointError("teacher produced non-finite heatmap, regression, or depth output")
    loss_hm = gaussian_focal_loss(torch.sigmoid(hm), hm_gt)
    loss_reg = masked_reg_l1(reg, reg_gt, reg_mask, vel_mask)
    loss_depth = hm.new_tensor(0.0)
    if args.depth_weight > 0:
        dep_sum = hm.new_tensor(0.0)
        for v in range(depth_all.shape[1]):
            dep_sum = dep_sum + safe_depth_ce_loss(depth_all[:, v], depth6[:, v])
        loss_depth = dep_sum / depth_all.shape[1]
    loss = args.hm_weight * loss_hm + args.reg_weight * loss_reg + args.depth_weight * loss_depth
    return loss, {"hm": loss_hm, "reg": loss_reg, "depth": loss_depth, "det_loss": loss.detach()}


def pseudo_reg_channel_scales(args, reference: torch.Tensor):
    """Return broadcastable NDS-oriented pseudo regression group scales."""
    values = (
        [float(getattr(args, "pseudo_reg_center_scale", 1.0))] * 3
        + [float(getattr(args, "pseudo_reg_size_scale", 1.0))] * 3
        + [float(getattr(args, "pseudo_reg_yaw_scale", 1.0))] * 2
        + [float(getattr(args, "pseudo_reg_velocity_scale", 1.0))] * 2
    )
    return reference.new_tensor(values).view(1, 10, 1, 1)


def kd_reg_channel_scales(args, reference: torch.Tensor):
    """Return NDS-oriented channel scales for the aligned in-domain teacher regression loss."""
    values = (
        [float(getattr(args, "kd_reg_center_scale", 1.0))] * 3
        + [float(getattr(args, "kd_reg_size_scale", 1.0))] * 3
        + [float(getattr(args, "kd_reg_yaw_scale", 1.0))] * 2
        + [float(getattr(args, "kd_reg_velocity_scale", 1.0))] * 2
    )
    return reference.new_tensor(values).view(1, 10, 1, 1)


def supervised_reg_channel_scales(args, reference: torch.Tensor):
    """Emphasize NDS-sensitive size, yaw, and velocity channels without changing outputs."""
    values = (
        [1.0, 1.0, float(getattr(args, "reg_z_scale", 1.0))]
        + [float(getattr(args, "reg_size_scale", 1.0))] * 3
        + [float(getattr(args, "reg_yaw_scale", 1.0))] * 2
        + [float(getattr(args, "reg_velocity_scale", 1.0))] * 2
    )
    return reference.new_tensor(values).view(1, 10, 1, 1)


def velocity_target_as_displacement(target: torch.Tensor, delta_t: torch.Tensor):
    """Convert vx/vy channels in m/s to adjacent-frame displacement in metres."""
    converted = target.clone()
    intervals = delta_t.to(device=target.device, dtype=target.dtype).reshape(-1, 1, 1, 1)
    converted[:, 8:10] = converted[:, 8:10] * intervals.clamp(min=1e-3)
    return converted


def apply_class_motion_residual(regression: torch.Tensor, class_motion: Optional[torch.Tensor],
                                class_indices: torch.Tensor):
    """Select a class-specific [sin, cos, vx, vy] residual at every BEV cell."""
    if class_motion is None:
        return regression
    if class_motion.ndim != 5 or class_motion.shape[2] != 4:
        raise ValueError(f"expected class motion [B,C,4,H,W], got {tuple(class_motion.shape)}")
    if class_indices.shape != regression.shape[:1] + regression.shape[2:]:
        raise ValueError(
            f"class index shape {tuple(class_indices.shape)} does not match regression "
            f"{tuple(regression.shape)}"
        )
    gather_index = class_indices.long().clamp(0, class_motion.shape[1] - 1)
    gather_index = gather_index[:, None, None].expand(-1, 1, 4, -1, -1)
    selected = class_motion.gather(1, gather_index).squeeze(1)
    refined = regression.clone()
    refined[:, 6:10] = refined[:, 6:10] + selected
    return refined


def apply_class_size_residual(regression: torch.Tensor, class_size: Optional[torch.Tensor],
                              class_indices: torch.Tensor):
    """Select a class-specific log-[w,l,h] correction at every BEV cell."""
    if class_size is None:
        return regression
    if class_size.ndim != 5 or class_size.shape[2] != 3:
        raise ValueError(f"expected class size [B,C,3,H,W], got {tuple(class_size.shape)}")
    if class_indices.shape != regression.shape[:1] + regression.shape[2:]:
        raise ValueError(
            f"class index shape {tuple(class_indices.shape)} does not match regression "
            f"{tuple(regression.shape)}"
        )
    gather_index = class_indices.long().clamp(0, class_size.shape[1] - 1)
    gather_index = gather_index[:, None, None].expand(-1, 1, 3, -1, -1)
    selected = class_size.gather(1, gather_index).squeeze(1)
    refined = regression.clone()
    refined[:, 3:6] = refined[:, 3:6] + selected
    return refined


def apply_class_center_residual(regression: torch.Tensor, class_center: Optional[torch.Tensor],
                                class_indices: torch.Tensor):
    """Select a class-specific [dx, dy, z] correction at every BEV cell."""
    if class_center is None:
        return regression
    if class_center.ndim != 5 or class_center.shape[2] != 3:
        raise ValueError(f"expected class center [B,C,3,H,W], got {tuple(class_center.shape)}")
    gather_index = class_indices.long().clamp(0, class_center.shape[1] - 1)
    gather_index = gather_index[:, None, None].expand(-1, 1, 3, -1, -1)
    selected = class_center.gather(1, gather_index).squeeze(1)
    refined = regression.clone()
    refined[:, 0:3] = refined[:, 0:3] + selected
    return refined


def wrap_angle_tensor(angle: torch.Tensor):
    return torch.remainder(angle + math.pi, 2.0 * math.pi) - math.pi


def azimuth_multibin_targets(target: torch.Tensor, mod):
    """Encode global ego yaw as radial-relative two-bin residuals.

    Bin centres are 0 and pi.  The two regression channels retain the existing
    sin/cos shape but now describe the residual from the selected bin centre.
    """
    if target.ndim != 4 or target.shape[1] < 8:
        raise ValueError(f"expected regression [B,>=8,H,W], got {tuple(target.shape)}")
    _batch, _channels, height, width = target.shape
    grid_x = torch.arange(height, device=target.device, dtype=target.dtype).view(1, height, 1)
    grid_y = torch.arange(width, device=target.device, dtype=target.dtype).view(1, 1, width)
    center_x = float(mod.BEV_X_MIN) + (grid_x + target[:, 0]) * float(mod.BEV_RES)
    center_y = float(mod.BEV_Y_MIN) + (grid_y + target[:, 1]) * float(mod.BEV_RES)
    radial_angle = torch.atan2(center_y, center_x)
    global_yaw = torch.atan2(target[:, 6], target[:, 7])
    relative_yaw = wrap_angle_tensor(global_yaw - radial_angle)
    bin_targets = (relative_yaw.abs() >= (math.pi / 2.0)).long()
    residual = wrap_angle_tensor(relative_yaw - bin_targets.to(target.dtype) * math.pi)
    converted = target.clone()
    converted[:, 6] = torch.sin(residual)
    converted[:, 7] = torch.cos(residual)
    return converted, bin_targets


def azimuth_bin_classification_loss(bin_logits: Optional[torch.Tensor], bin_targets: torch.Tensor,
                                    heatmap_target: torch.Tensor, reg_mask: torch.Tensor,
                                    class_weights: Optional[Sequence[float]] = None):
    if bin_logits is None:
        return heatmap_target.new_tensor(0.0)
    if bin_logits.ndim != 5 or bin_logits.shape[2] != 2:
        raise ValueError(f"expected azimuth bins [B,C,2,H,W], got {tuple(bin_logits.shape)}")
    class_indices = heatmap_target.argmax(dim=1)
    gather_index = class_indices[:, None, None].expand(-1, 1, 2, -1, -1)
    selected_logits = bin_logits.gather(1, gather_index).squeeze(1).float()
    raw = F.cross_entropy(selected_logits, bin_targets.long(), reduction="none")
    if class_weights is None:
        weights_by_class = raw.new_ones(len(DETECTION_NAMES))
    else:
        if len(class_weights) != len(DETECTION_NAMES):
            raise ValueError(f"expected {len(DETECTION_NAMES)} azimuth class weights")
        weights_by_class = raw.new_tensor(class_weights)
    weights = weights_by_class[class_indices] * reg_mask[:, 0].float()
    weights[class_indices == NAME_TO_ID["traffic_cone"]] = 0.0
    return (raw * weights).sum() / weights.sum().clamp(min=1.0)


def residual_azimuth_multibin_loss(head_output: Optional[torch.Tensor],
                                   refined_prediction: torch.Tensor,
                                   target: torch.Tensor, heatmap_target: torch.Tensor,
                                   reg_mask: torch.Tensor,
                                   class_weights: Optional[Sequence[float]] = None,
                                   num_bins: int = 8):
    """Classify the frozen-base yaw error and regress its within-bin residual."""
    if head_output is None:
        return refined_prediction.new_tensor(0.0)
    expected_channels = 3 * int(num_bins) + 2
    if head_output.ndim != 5 or head_output.shape[2] != expected_channels:
        raise ValueError(
            f"expected residual MultiBin [B,C,{expected_channels},H,W], "
            f"got {tuple(head_output.shape)}"
        )
    class_indices = heatmap_target.argmax(dim=1)
    gather_index = class_indices[:, None, None].expand(
        -1, 1, expected_channels, -1, -1,
    )
    selected = head_output.gather(1, gather_index).squeeze(1).float()
    logits = selected[:, :num_bins]
    residual_sin = selected[:, num_bins:2 * num_bins]
    residual_cos_offset = selected[:, 2 * num_bins:3 * num_bins]
    applied_vector_residual = selected[:, 3 * num_bins:3 * num_bins + 2]

    # Remove the predicted correction to recover the unchanged S1 yaw used as the
    # reference. The target assignment itself must not backpropagate through S1.
    base_yaw_vector = (
        refined_prediction[:, 6:8].float() - applied_vector_residual
    ).detach()
    base_yaw = torch.atan2(base_yaw_vector[:, 0], base_yaw_vector[:, 1])
    target_yaw = torch.atan2(target[:, 6].float(), target[:, 7].float())
    target_correction = wrap_angle_tensor(target_yaw - base_yaw)
    centres = target_correction.new_tensor([
        0.0, math.pi / 4.0, math.pi / 2.0, 3.0 * math.pi / 4.0,
        math.pi, -3.0 * math.pi / 4.0, -math.pi / 2.0, -math.pi / 4.0,
    ])
    angular_distance = wrap_angle_tensor(
        target_correction.unsqueeze(1) - centres.view(1, num_bins, 1, 1)
    ).abs()
    bin_targets = angular_distance.argmin(dim=1)
    within_bin = wrap_angle_tensor(target_correction - centres[bin_targets])
    selected_bin = bin_targets.unsqueeze(1)
    predicted_sin = residual_sin.gather(1, selected_bin).squeeze(1)
    predicted_cos_offset = residual_cos_offset.gather(1, selected_bin).squeeze(1)
    classification_raw = F.cross_entropy(logits, bin_targets, reduction="none")
    residual_raw = (
        F.smooth_l1_loss(
            predicted_sin, torch.sin(within_bin), reduction="none", beta=0.25,
        ) +
        F.smooth_l1_loss(
            predicted_cos_offset, torch.cos(within_bin) - 1.0,
            reduction="none", beta=0.25,
        )
    )
    if class_weights is None:
        weights_by_class = classification_raw.new_ones(len(DETECTION_NAMES))
    else:
        if len(class_weights) != len(DETECTION_NAMES):
            raise ValueError(f"expected {len(DETECTION_NAMES)} residual azimuth class weights")
        weights_by_class = classification_raw.new_tensor(class_weights)
    weights = weights_by_class[class_indices] * reg_mask[:, 0].float()
    weights[class_indices == NAME_TO_ID["traffic_cone"]] = 0.0
    denominator = weights.sum().clamp(min=1.0)
    return (
        ((classification_raw + 2.0 * residual_raw) * weights).sum() / denominator
    )


def residual_yaw_vector_loss(head_output: Optional[torch.Tensor],
                             refined_prediction: torch.Tensor,
                             target: torch.Tensor, heatmap_target: torch.Tensor,
                             reg_mask: torch.Tensor,
                             class_weights: Optional[Sequence[float]] = None):
    """Supervise a smooth correction vector around the frozen S1 yaw."""
    if head_output is None:
        return refined_prediction.new_tensor(0.0)
    if head_output.ndim != 5 or head_output.shape[2] != 4:
        raise ValueError(f"expected residual yaw vector [B,C,4,H,W], got {tuple(head_output.shape)}")
    class_indices = heatmap_target.argmax(dim=1)
    selected = head_output.gather(
        1, class_indices[:, None, None].expand(-1, 1, 4, -1, -1),
    ).squeeze(1).float()
    raw_sin, raw_cos_offset = selected[:, 0], selected[:, 1]
    applied_vector_residual = selected[:, 2:4]
    base_vector = (refined_prediction[:, 6:8].float() - applied_vector_residual).detach()
    base_yaw = torch.atan2(base_vector[:, 0], base_vector[:, 1])
    target_yaw = torch.atan2(target[:, 6].float(), target[:, 7].float())
    target_correction = wrap_angle_tensor(target_yaw - base_yaw)
    correction = torch.atan2(raw_sin, 1.0 + raw_cos_offset)
    vector_raw = (
        F.smooth_l1_loss(raw_sin, torch.sin(target_correction), reduction="none", beta=0.25) +
        F.smooth_l1_loss(
            raw_cos_offset, torch.cos(target_correction) - 1.0,
            reduction="none", beta=0.25,
        )
    )
    angular_raw = 1.0 - torch.cos(correction - target_correction)
    if class_weights is None:
        weights_by_class = vector_raw.new_ones(len(DETECTION_NAMES))
    else:
        if len(class_weights) != len(DETECTION_NAMES):
            raise ValueError(f"expected {len(DETECTION_NAMES)} residual yaw class weights")
        weights_by_class = vector_raw.new_tensor(class_weights)
    weights = weights_by_class[class_indices] * reg_mask[:, 0].float()
    weights[class_indices == NAME_TO_ID["traffic_cone"]] = 0.0
    denominator = weights.sum().clamp(min=1.0)
    return ((vector_raw + angular_raw) * weights).sum() / denominator


def class_size_metric_losses(prediction: torch.Tensor, target: torch.Tensor,
                             heatmap_target: torch.Tensor, reg_mask: torch.Tensor,
                             class_weights: Optional[Sequence[float]] = None):
    """Class-normalized size residual and centre/orientation-aligned 3D IoU losses."""
    class_indices = heatmap_target.argmax(dim=1)
    means = prediction.new_tensor([
        CLASS_SIZE_MEANS[name] for name in DETECTION_NAMES
    ])[class_indices].permute(0, 3, 1, 2)
    pred_size = torch.exp(prediction[:, 3:6].float().clamp(-8.0, 8.0))
    target_size = torch.exp(target[:, 3:6].float().clamp(-8.0, 8.0))
    normalized_raw = F.smooth_l1_loss(
        pred_size / means.clamp(min=1e-3),
        target_size / means.clamp(min=1e-3),
        reduction="none", beta=0.25,
    ).mean(dim=1)
    intersection = torch.minimum(pred_size, target_size).prod(dim=1)
    union = pred_size.prod(dim=1) + target_size.prod(dim=1) - intersection
    ase_raw = 1.0 - intersection / union.clamp(min=1e-6)
    if class_weights is None:
        weights_by_class = ase_raw.new_ones(len(DETECTION_NAMES))
    else:
        if len(class_weights) != len(DETECTION_NAMES):
            raise ValueError(f"expected {len(DETECTION_NAMES)} size class weights")
        weights_by_class = ase_raw.new_tensor(class_weights)
    weights = weights_by_class[class_indices] * reg_mask[:, 0].float()
    denominator = weights.sum().clamp(min=1.0)
    return (
        (normalized_raw * weights).sum() / denominator,
        (ase_raw * weights).sum() / denominator,
    )


def decomposed_velocity_loss(prediction: torch.Tensor, target: torch.Tensor,
                             heatmap_target: torch.Tensor, reg_mask: torch.Tensor,
                             vel_mask: torch.Tensor,
                             class_weights: Optional[Sequence[float]] = None,
                             velocity_intervals: Optional[torch.Tensor] = None,
                             moving_threshold: float = 0.2,
                             moving_boost: float = 1.0):
    """Separate speed magnitude, motion direction, and static-object supervision."""
    class_indices = heatmap_target.argmax(dim=1)
    if class_weights is None:
        weights_by_class = prediction.new_ones(len(DETECTION_NAMES))
    else:
        if len(class_weights) != len(DETECTION_NAMES):
            raise ValueError(f"expected {len(DETECTION_NAMES)} velocity class weights")
        weights_by_class = prediction.new_tensor(class_weights)
    weights = weights_by_class[class_indices] * reg_mask[:, 0].float() * vel_mask[:, 0].float()
    weights[class_indices == NAME_TO_ID["traffic_cone"]] = 0.0
    weights[class_indices == NAME_TO_ID["barrier"]] = 0.0
    pred_velocity = prediction[:, 8:10].float()
    target_velocity = target[:, 8:10].float()
    if velocity_intervals is not None:
        intervals = velocity_intervals.to(prediction).reshape(-1, 1, 1, 1).clamp(min=1e-3)
        pred_velocity = pred_velocity / intervals
        target_velocity = target_velocity / intervals
    pred_speed = torch.linalg.vector_norm(pred_velocity, dim=1)
    target_speed = torch.linalg.vector_norm(target_velocity, dim=1)
    moving = target_speed >= float(moving_threshold)
    weights = weights * torch.where(
        moving, weights.new_tensor(float(moving_boost)), weights.new_tensor(1.0),
    )
    speed_raw = F.smooth_l1_loss(pred_speed, target_speed, reduction="none", beta=0.5)
    pred_direction = F.normalize(pred_velocity, dim=1, eps=1e-6)
    target_direction = F.normalize(target_velocity, dim=1, eps=1e-6)
    direction_raw = 1.0 - (pred_direction * target_direction).sum(dim=1).clamp(-1.0, 1.0)
    moving_weights = weights * moving.float()
    static_weights = weights * (~moving).float()
    speed_loss = (speed_raw * weights).sum() / weights.sum().clamp(min=1.0)
    direction_loss = (
        (direction_raw * moving_weights).sum() / moving_weights.sum().clamp(min=1.0)
    )
    static_loss = (
        (pred_speed * static_weights).sum() / static_weights.sum().clamp(min=1.0)
    )
    return speed_loss + direction_loss + 0.5 * static_loss


def yaw_velocity_consistency_loss(prediction: torch.Tensor, target: torch.Tensor,
                                  heatmap_target: torch.Tensor, reg_mask: torch.Tensor,
                                  vel_mask: torch.Tensor,
                                  class_weights: Optional[Sequence[float]] = None,
                                  velocity_intervals: Optional[torch.Tensor] = None,
                                  moving_threshold: float = 1.0,
                                  alignment_threshold: float = 0.5):
    """Align predicted heading and travel direction only for valid forward motion.

    nuScenes pedestrian orientation and static-object velocity do not obey this relation,
    and vehicles can occasionally reverse. The target-derived mask therefore restricts the
    auxiliary loss to moving vehicle/cycle instances whose annotated heading and velocity
    already agree. This changes training only and adds no deployed parameters.
    """
    class_indices = heatmap_target.argmax(dim=1)
    if class_weights is None:
        weights_by_class = prediction.new_ones(len(DETECTION_NAMES))
    else:
        if len(class_weights) != len(DETECTION_NAMES):
            raise ValueError(f"expected {len(DETECTION_NAMES)} motion class weights")
        weights_by_class = prediction.new_tensor(class_weights)
    weights = (
        weights_by_class[class_indices] * reg_mask[:, 0].float() * vel_mask[:, 0].float()
    )
    supported = weights.new_zeros(len(DETECTION_NAMES))
    for name in (
        "car", "truck", "construction_vehicle", "bus", "trailer",
        "motorcycle", "bicycle",
    ):
        supported[NAME_TO_ID[name]] = 1.0
    weights = weights * supported[class_indices]

    predicted_velocity = prediction[:, 8:10].float()
    target_velocity = target[:, 8:10].float()
    if velocity_intervals is not None:
        intervals = velocity_intervals.to(prediction).reshape(-1, 1, 1, 1).clamp(min=1e-3)
        predicted_velocity = predicted_velocity / intervals
        target_velocity = target_velocity / intervals
    target_speed = torch.linalg.vector_norm(target_velocity, dim=1)
    target_travel = F.normalize(target_velocity, dim=1, eps=1e-6)
    target_heading = F.normalize(
        torch.stack((target[:, 7], target[:, 6]), dim=1).float(), dim=1, eps=1e-6,
    )
    target_alignment = (target_heading * target_travel).sum(dim=1)
    valid_relation = (
        (target_speed >= float(moving_threshold)) &
        (target_alignment >= float(alignment_threshold))
    )

    predicted_heading = F.normalize(
        torch.stack((prediction[:, 7], prediction[:, 6]), dim=1).float(), dim=1, eps=1e-6,
    )
    predicted_travel = F.normalize(predicted_velocity, dim=1, eps=1e-6)
    raw = 1.0 - (predicted_heading * predicted_travel).sum(dim=1).clamp(-1.0, 1.0)
    relation_weights = weights * valid_relation.float()
    return (raw * relation_weights).sum() / relation_weights.sum().clamp(min=1.0)


def image_yaw_flip_classification_loss(
    image_logits: Optional[torch.Tensor],
    base_prediction: torch.Tensor,
    target: torch.Tensor,
    heatmap_target: torch.Tensor,
    reg_mask: torch.Tensor,
    k: torch.Tensor,
    r_cam_to_ego: torch.Tensor,
    t_cam_to_ego: torch.Tensor,
    mod,
    class_weights: Optional[Sequence[float]] = None,
):
    """Supervise a binary front/back decision at projected GT image centres."""
    if image_logits is None:
        return base_prediction.new_tensor(0.0)
    if class_weights is None:
        weights_by_class = base_prediction.new_ones(len(DETECTION_NAMES))
    else:
        if len(class_weights) != len(DETECTION_NAMES):
            raise ValueError(f"expected {len(DETECTION_NAMES)} yaw flip class weights")
        weights_by_class = base_prediction.new_tensor(class_weights)

    positive = (reg_mask[:, 0] > 0.5).nonzero(as_tuple=False)
    if positive.numel() == 0:
        return base_prediction.new_tensor(0.0)
    batch_indices, ix, iy = positive.unbind(dim=1)
    classes = heatmap_target.argmax(dim=1)[batch_indices, ix, iy]
    weights = weights_by_class[classes]

    gt_dx = target[batch_indices, 0, ix, iy].float()
    gt_dy = target[batch_indices, 1, ix, iy].float()
    gt_z = target[batch_indices, 2, ix, iy].float()
    x_ego = float(mod.BEV_X_MIN) + (ix.float() + gt_dx) * float(mod.BEV_RES)
    y_ego = float(mod.BEV_Y_MIN) + (iy.float() + gt_dy) * float(mod.BEV_RES)
    points_ego = torch.stack((x_ego, y_ego, gt_z), dim=1)
    camera_points = torch.bmm(
        r_cam_to_ego[batch_indices].float().transpose(1, 2),
        (points_ego - t_cam_to_ego[batch_indices].float()).unsqueeze(-1),
    ).squeeze(-1)
    depth = camera_points[:, 2]
    safe_depth = depth.clamp(min=1e-4)
    u = (
        k[batch_indices, 0, 0].float() * camera_points[:, 0] / safe_depth +
        k[batch_indices, 0, 2].float()
    )
    v = (
        k[batch_indices, 1, 1].float() * camera_points[:, 1] / safe_depth +
        k[batch_indices, 1, 2].float()
    )
    feat_h, feat_w = image_logits.shape[-2:]
    feat_x = torch.floor(u * float(feat_w) / float(mod.IMG_W)).long()
    feat_y = torch.floor(v * float(feat_h) / float(mod.IMG_H)).long()
    visible = (
        (depth > 1e-3) & (feat_x >= 0) & (feat_x < feat_w) &
        (feat_y >= 0) & (feat_y < feat_h) & (weights > 0)
    )
    if not visible.any():
        return base_prediction.new_tensor(0.0)

    batch_indices = batch_indices[visible]
    ix, iy, classes = ix[visible], iy[visible], classes[visible]
    feat_x, feat_y, weights = feat_x[visible], feat_y[visible], weights[visible]
    selected_logits = image_logits[batch_indices, classes, feat_y, feat_x].float()
    base_yaw = torch.atan2(
        base_prediction[batch_indices, 6, ix, iy].detach().float(),
        base_prediction[batch_indices, 7, ix, iy].detach().float(),
    )
    target_yaw = torch.atan2(
        target[batch_indices, 6, ix, iy].float(),
        target[batch_indices, 7, ix, iy].float(),
    )
    delta = wrap_angle_tensor(target_yaw - base_yaw).abs()
    flip_target = (delta > (math.pi / 2.0)).float()
    raw = F.binary_cross_entropy_with_logits(
        selected_logits, flip_target, reduction="none",
    )
    return (raw * weights).sum() / weights.sum().clamp(min=1.0)


def image_radial_center_regression_loss(
    image_residuals: Optional[torch.Tensor],
    base_prediction: torch.Tensor,
    target: torch.Tensor,
    heatmap_target: torch.Tensor,
    reg_mask: torch.Tensor,
    k: torch.Tensor,
    r_cam_to_ego: torch.Tensor,
    t_cam_to_ego: torch.Tensor,
    mod,
    class_weights: Optional[Sequence[float]] = None,
    max_residual: float = 2.0,
):
    """Regress GT centre error projected onto the camera-to-object ground ray."""
    if image_residuals is None:
        return base_prediction.new_tensor(0.0)
    if class_weights is None:
        weights_by_class = base_prediction.new_ones(len(DETECTION_NAMES))
    else:
        if len(class_weights) != len(DETECTION_NAMES):
            raise ValueError(f"expected {len(DETECTION_NAMES)} radial center class weights")
        weights_by_class = base_prediction.new_tensor(class_weights)
    positive = (reg_mask[:, 0] > 0.5).nonzero(as_tuple=False)
    if positive.numel() == 0:
        return base_prediction.new_tensor(0.0)
    batch_indices, ix, iy = positive.unbind(dim=1)
    classes = heatmap_target.argmax(dim=1)[batch_indices, ix, iy]
    weights = weights_by_class[classes]

    target_x = float(mod.BEV_X_MIN) + (
        ix.float() + target[batch_indices, 0, ix, iy].float()
    ) * float(mod.BEV_RES)
    target_y = float(mod.BEV_Y_MIN) + (
        iy.float() + target[batch_indices, 1, ix, iy].float()
    ) * float(mod.BEV_RES)
    target_z = target[batch_indices, 2, ix, iy].float()
    base_x = float(mod.BEV_X_MIN) + (
        ix.float() + base_prediction[batch_indices, 0, ix, iy].detach().float()
    ) * float(mod.BEV_RES)
    base_y = float(mod.BEV_Y_MIN) + (
        iy.float() + base_prediction[batch_indices, 1, ix, iy].detach().float()
    ) * float(mod.BEV_RES)
    target_points = torch.stack((target_x, target_y, target_z), dim=1)
    camera_points = torch.bmm(
        r_cam_to_ego[batch_indices].float().transpose(1, 2),
        (target_points - t_cam_to_ego[batch_indices].float()).unsqueeze(-1),
    ).squeeze(-1)
    depth = camera_points[:, 2]
    safe_depth = depth.clamp(min=1e-4)
    u = (
        k[batch_indices, 0, 0].float() * camera_points[:, 0] / safe_depth +
        k[batch_indices, 0, 2].float()
    )
    v = (
        k[batch_indices, 1, 1].float() * camera_points[:, 1] / safe_depth +
        k[batch_indices, 1, 2].float()
    )
    feat_h, feat_w = image_residuals.shape[-2:]
    feat_x = torch.floor(u * float(feat_w) / float(mod.IMG_W)).long()
    feat_y = torch.floor(v * float(feat_h) / float(mod.IMG_H)).long()

    base_xy = torch.stack((base_x, base_y), dim=1)
    target_xy = torch.stack((target_x, target_y), dim=1)
    ray = base_xy - t_cam_to_ego[batch_indices, :2].float()
    ray_norm = torch.linalg.vector_norm(ray, dim=1)
    visible = (
        (depth > 1e-3) & (feat_x >= 0) & (feat_x < feat_w) &
        (feat_y >= 0) & (feat_y < feat_h) & (ray_norm > 1e-3) & (weights > 0)
    )
    if not visible.any():
        return base_prediction.new_tensor(0.0)
    batch_indices = batch_indices[visible]
    classes = classes[visible]
    feat_x, feat_y, weights = feat_x[visible], feat_y[visible], weights[visible]
    ray_unit = ray[visible] / ray_norm[visible, None]
    target_residual = (
        (target_xy[visible] - base_xy[visible]) * ray_unit
    ).sum(dim=1).clamp(-float(max_residual), float(max_residual))
    prediction = image_residuals[
        batch_indices, classes, feat_y, feat_x
    ].float()
    raw = F.smooth_l1_loss(
        prediction, target_residual, reduction="none", beta=0.25,
    )
    return (raw * weights).sum() / weights.sum().clamp(min=1.0)


def class_balanced_motion_losses(prediction: torch.Tensor, target: torch.Tensor,
                                 heatmap_target: torch.Tensor, reg_mask: torch.Tensor,
                                 vel_mask: torch.Tensor, class_weights: Optional[Sequence[float]],
                                 velocity_intervals: Optional[torch.Tensor] = None,
                                 moving_boost: float = 1.0):
    """Directly optimize the class-averaged angular and velocity errors used by NDS."""
    class_indices = heatmap_target.argmax(dim=1)
    if class_weights is None:
        weights_by_class = prediction.new_ones(len(DETECTION_NAMES))
    else:
        if len(class_weights) != len(DETECTION_NAMES):
            raise ValueError(f"expected {len(DETECTION_NAMES)} motion class weights")
        weights_by_class = prediction.new_tensor(class_weights)
    per_cell_weight = weights_by_class[class_indices] * reg_mask[:, 0].float()

    # Traffic cones do not have a nuScenes orientation TP metric.
    yaw_weight = per_cell_weight.clone()
    yaw_weight[class_indices == NAME_TO_ID["traffic_cone"]] = 0.0
    pred_yaw = F.normalize(prediction[:, 6:8].float(), dim=1, eps=1e-6)
    target_yaw = F.normalize(target[:, 6:8].float(), dim=1, eps=1e-6)
    yaw_error = 1.0 - (pred_yaw * target_yaw).sum(dim=1).clamp(-1.0, 1.0)
    yaw_loss = (yaw_error * yaw_weight).sum() / yaw_weight.sum().clamp(min=1.0)

    velocity_weight = per_cell_weight * vel_mask[:, 0].float()
    velocity_weight[class_indices == NAME_TO_ID["traffic_cone"]] = 0.0
    velocity_weight[class_indices == NAME_TO_ID["barrier"]] = 0.0
    pred_velocity = prediction[:, 8:10].float()
    target_velocity = target[:, 8:10].float()
    if velocity_intervals is not None:
        intervals = velocity_intervals.to(prediction).reshape(-1, 1, 1, 1).clamp(min=1e-3)
        pred_velocity = pred_velocity / intervals
        target_velocity = target_velocity / intervals
    target_speed = torch.linalg.vector_norm(target_velocity, dim=1)
    velocity_weight = velocity_weight * torch.where(
        target_speed >= 0.2,
        velocity_weight.new_tensor(float(moving_boost)),
        velocity_weight.new_tensor(1.0),
    )
    velocity_error = torch.linalg.vector_norm(pred_velocity - target_velocity, dim=1)
    velocity_loss = (
        velocity_error.clamp(max=10.0) * velocity_weight
    ).sum() / velocity_weight.sum().clamp(min=1.0)
    return yaw_loss, velocity_loss


def batch_has_pseudo_targets(batch, pseudo_offset: int, relation_field_count: int = 0) -> bool:
    """Distinguish optional pseudo targets from trailing relation-cache fields.

    Dataset tuples append pseudo targets before the four relation tensors.  The
    validation dataset intentionally has no train-set pseudo labels, so a bare
    length check would otherwise reinterpret those relation tensors as pseudo
    heatmaps.
    """
    return len(batch) >= int(pseudo_offset) + 4 + int(relation_field_count)


def student_forward_loss(
    student, teacher, batch, args, device, kd_scale: float = 1.0,
    camera_teacher=None,
):
    img1, k1, r1, t1 = [x.to(device, non_blocking=True) for x in batch[5:9]]
    hm_gt, reg_gt, reg_mask, vel_mask, _depth6, depth1 = [x.to(device, non_blocking=True) for x in batch[9:15]]
    bev_fov_mask = batch[15].to(device, non_blocking=True) if len(batch) > 15 else torch.ones_like(reg_mask)
    student_cam_indices = batch[16].to(device, non_blocking=True).long() if len(batch) > 16 else None
    history_enabled = (
        float(getattr(args, "hop_hm_weight", 0.0)) + float(getattr(args, "hop_reg_weight", 0.0)) > 0
    )
    history_hm_s = history_reg_s = None
    if getattr(args, "temporal_bev", False):
        prev_img, prev_k, prev_r, prev_t, prev_to_current_r, prev_to_current_t, prev_valid = [
            tensor.to(device, non_blocking=True) for tensor in batch[18:25]
        ]
        student_outputs = student.forward_outputs(
            img1, k1, r1, t1, prev_img, prev_k, prev_r, prev_t,
            prev_to_current_r, prev_to_current_t, prev_valid,
            return_history=history_enabled,
            privileged_depth=(
                depth1 if student.privileged_depth_teacher_enabled else None
            ),
        )
        hm_s, reg_s, depth_s, bev_s, attr_s, class_motion_s = student_outputs[:6]
        has_v12_heads = student.azimuth_bin_head is not None or student.class_size_head is not None
        if has_v12_heads:
            azimuth_bins_s, class_size_s = student_outputs[6:8]
            history_offset = 8
        else:
            azimuth_bins_s = class_size_s = None
            history_offset = 6
        if student.class_center_head is not None:
            class_center_s = student_outputs[history_offset]
            history_offset += 1
        else:
            class_center_s = None
        if student.image_yaw_flip_head is not None:
            image_yaw_flip_s = student_outputs[history_offset]
            history_offset += 1
        else:
            image_yaw_flip_s = None
        if student.image_radial_center_head is not None:
            image_radial_center_s = student_outputs[history_offset]
            history_offset += 1
        else:
            image_radial_center_s = None
        if student.image_object_geometry_head is not None:
            image_yaw_flip_s = student_outputs[history_offset]
            image_radial_center_s = student_outputs[history_offset + 1]
            history_offset += 2
        if (
            student.residual_azimuth_multibin_head is not None or
            student.residual_yaw_vector_head is not None
        ):
            residual_azimuth_s = student_outputs[history_offset]
            history_offset += 1
        else:
            residual_azimuth_s = None
        if history_enabled:
            history_hm_s, history_reg_s = student_outputs[history_offset:history_offset + 2]
        velocity_intervals = batch[25].to(device, non_blocking=True)
    else:
        student_outputs = student.forward_outputs(
            img1, k1, r1, t1,
            privileged_depth=(
                depth1 if student.privileged_depth_teacher_enabled else None
            ),
        )
        hm_s, reg_s, depth_s, bev_s, attr_s, class_motion_s = student_outputs[:6]
        if student.azimuth_bin_head is not None or student.class_size_head is not None:
            azimuth_bins_s, class_size_s = student_outputs[6:8]
            residual_offset = 8
        else:
            azimuth_bins_s = class_size_s = None
            residual_offset = 6
        if student.class_center_head is not None:
            class_center_s = student_outputs[residual_offset]
            residual_offset += 1
        else:
            class_center_s = None
        if student.image_yaw_flip_head is not None:
            image_yaw_flip_s = student_outputs[residual_offset]
            residual_offset += 1
        else:
            image_yaw_flip_s = None
        if student.image_radial_center_head is not None:
            image_radial_center_s = student_outputs[residual_offset]
            residual_offset += 1
        else:
            image_radial_center_s = None
        if student.image_object_geometry_head is not None:
            image_yaw_flip_s = student_outputs[residual_offset]
            image_radial_center_s = student_outputs[residual_offset + 1]
            residual_offset += 2
        residual_azimuth_s = (
            student_outputs[residual_offset]
            if (
                student.residual_azimuth_multibin_head is not None or
                student.residual_yaw_vector_head is not None
            ) else None
        )
        velocity_intervals = None
    hm_positive_class_weights = getattr(args, "hm_positive_class_weights", None)
    if hm_positive_class_weights is not None:
        hm_positive_class_weights = hm_s.new_tensor(hm_positive_class_weights)
    loss_hm = gaussian_focal_loss(
        torch.sigmoid(hm_s), hm_gt, pos_class_weights=hm_positive_class_weights,
    )
    supervised_reg_gt = reg_gt
    if getattr(args, "velocity_displacement_target", False):
        if velocity_intervals is None:
            raise ValueError("velocity_displacement_target requires temporal inputs")
        supervised_reg_gt = velocity_target_as_displacement(reg_gt, velocity_intervals)
    azimuth_bin_targets = None
    if getattr(args, "azimuth_multibin", False):
        supervised_reg_gt, azimuth_bin_targets = azimuth_multibin_targets(supervised_reg_gt, student.mod)
    supervised_class_indices = hm_gt.argmax(dim=1)
    supervised_reg_prediction = apply_class_center_residual(
        reg_s, class_center_s, supervised_class_indices,
    )
    supervised_reg_prediction = apply_class_motion_residual(
        supervised_reg_prediction, class_motion_s, supervised_class_indices,
    )
    supervised_reg_prediction = apply_class_size_residual(
        supervised_reg_prediction, class_size_s, supervised_class_indices,
    )
    loss_reg = masked_reg_l1(
        supervised_reg_prediction, supervised_reg_gt, reg_mask, vel_mask,
        channel_scales=supervised_reg_channel_scales(args, supervised_reg_prediction),
    )
    center_error = masked_mean(
        torch.abs(supervised_reg_prediction[:, :3] - supervised_reg_gt[:, :3]),
        reg_mask,
    )
    yaw_metric_loss, velocity_metric_loss = class_balanced_motion_losses(
        supervised_reg_prediction, supervised_reg_gt, hm_gt, reg_mask, vel_mask,
        getattr(args, "motion_class_weights", None),
        velocity_intervals if getattr(args, "velocity_displacement_target", False) else None,
        moving_boost=float(getattr(args, "velocity_moving_boost", 1.0)),
    )
    azimuth_bin_loss = hm_s.new_tensor(0.0)
    if azimuth_bin_targets is not None:
        azimuth_bin_loss = azimuth_bin_classification_loss(
            azimuth_bins_s, azimuth_bin_targets, hm_gt, reg_mask,
            getattr(args, "motion_class_weights", None),
        )
    if residual_azimuth_s is not None:
        if student.residual_yaw_vector_head is not None:
            azimuth_bin_loss = residual_yaw_vector_loss(
                residual_azimuth_s, supervised_reg_prediction, supervised_reg_gt,
                hm_gt, reg_mask, getattr(args, "motion_class_weights", None),
            )
        else:
            azimuth_bin_loss = residual_azimuth_multibin_loss(
                residual_azimuth_s, supervised_reg_prediction, supervised_reg_gt,
                hm_gt, reg_mask, getattr(args, "motion_class_weights", None),
            )
    size_prior_loss = hm_s.new_tensor(0.0)
    ase_iou_loss = hm_s.new_tensor(0.0)
    if (
        float(getattr(args, "size_prior_weight", 0.0)) > 0 or
        float(getattr(args, "ase_iou_weight", 0.0)) > 0
    ):
        size_prior_loss, ase_iou_loss = class_size_metric_losses(
            supervised_reg_prediction, supervised_reg_gt, hm_gt, reg_mask,
            getattr(args, "size_class_weights", None),
        )
    velocity_decomposed_loss = hm_s.new_tensor(0.0)
    if float(getattr(args, "velocity_decomposed_weight", 0.0)) > 0:
        velocity_decomposed_loss = decomposed_velocity_loss(
            supervised_reg_prediction, supervised_reg_gt, hm_gt, reg_mask, vel_mask,
            getattr(args, "motion_class_weights", None),
            velocity_intervals if getattr(args, "velocity_displacement_target", False) else None,
            moving_boost=float(getattr(args, "velocity_moving_boost", 1.0)),
        )
    yaw_velocity_consistency = hm_s.new_tensor(0.0)
    if float(getattr(args, "yaw_velocity_consistency_weight", 0.0)) > 0:
        yaw_velocity_consistency = yaw_velocity_consistency_loss(
            supervised_reg_prediction, supervised_reg_gt, hm_gt, reg_mask, vel_mask,
            getattr(args, "motion_class_weights", None),
            velocity_intervals if getattr(args, "velocity_displacement_target", False) else None,
        )
    image_yaw_flip_loss = image_yaw_flip_classification_loss(
        image_yaw_flip_s, supervised_reg_prediction, supervised_reg_gt,
        hm_gt, reg_mask, k1, r1, t1, student.mod,
        getattr(args, "yaw_flip_class_weights", None),
    )
    image_radial_center_loss = image_radial_center_regression_loss(
        image_radial_center_s, supervised_reg_prediction, supervised_reg_gt,
        hm_gt, reg_mask, k1, r1, t1, student.mod,
        getattr(args, "radial_center_class_weights", None),
        max_residual=float(getattr(args, "image_radial_center_max_residual", 2.0)),
    )
    loss_depth = safe_depth_ce_loss(depth_s, depth1) if args.depth_weight > 0 else hm_s.new_tensor(0.0)
    relative_depth_loss = (
        sparse_local_relative_depth_loss(
            depth_s, depth1,
            offsets=getattr(args, "relative_depth_offsets", (1, 2, 4, 8)),
        )
        if float(getattr(args, "relative_depth_weight", 0.0)) > 0 else
        hm_s.new_tensor(0.0)
    )
    loss_attr = hm_s.new_tensor(0.0)
    if attr_s is not None and float(getattr(args, "attr_weight", 0.0)) > 0:
        attr_target = batch[17].to(device, non_blocking=True).long()
        valid_attr = attr_target >= 0
        if valid_attr.any():
            attr_vectors = attr_s.float().permute(0, 2, 3, 1)[valid_attr]
            loss_attr = F.cross_entropy(attr_vectors, attr_target[valid_attr])
    det_loss = (
        args.hm_weight * loss_hm + args.reg_weight * loss_reg + args.depth_weight * loss_depth +
        float(getattr(args, "attr_weight", 0.0)) * loss_attr +
        float(getattr(args, "yaw_cosine_weight", 0.0)) * yaw_metric_loss +
        float(getattr(args, "velocity_metric_weight", 0.0)) * velocity_metric_loss +
        float(getattr(args, "azimuth_bin_weight", 0.0)) * azimuth_bin_loss +
        float(getattr(args, "size_prior_weight", 0.0)) * size_prior_loss +
        float(getattr(args, "ase_iou_weight", 0.0)) * ase_iou_loss +
        float(getattr(args, "velocity_decomposed_weight", 0.0)) * velocity_decomposed_loss +
        float(getattr(args, "yaw_velocity_consistency_weight", 0.0)) * yaw_velocity_consistency +
        float(getattr(args, "image_yaw_flip_weight", 0.0)) * image_yaw_flip_loss +
        float(getattr(args, "image_radial_center_weight", 0.0)) * image_radial_center_loss +
        float(getattr(args, "relative_depth_weight", 0.0)) * relative_depth_loss
    )

    history_hm_loss = hm_s.new_tensor(0.0)
    history_reg_loss = hm_s.new_tensor(0.0)
    if history_enabled:
        history_hm_gt, history_reg_gt, history_reg_mask, history_vel_mask = [
            tensor.to(device, non_blocking=True) for tensor in batch[26:30]
        ]
        valid_history = prev_valid > 0.5
        if valid_history.any():
            history_hm_loss = gaussian_focal_loss(
                torch.sigmoid(history_hm_s[valid_history]), history_hm_gt[valid_history],
                pos_class_weights=hm_positive_class_weights,
            )
            history_reg_target = history_reg_gt[valid_history]
            if getattr(args, "velocity_displacement_target", False):
                history_reg_target = velocity_target_as_displacement(
                    history_reg_target, velocity_intervals[valid_history],
                )
            history_reg_loss = masked_reg_l1(
                history_reg_s[valid_history], history_reg_target,
                history_reg_mask[valid_history], history_vel_mask[valid_history],
                channel_scales=supervised_reg_channel_scales(args, history_reg_s),
            )

    kd_hm = hm_s.new_tensor(0.0)
    kd_reg = hm_s.new_tensor(0.0)
    kd_feat = hm_s.new_tensor(0.0)
    kd_depth = hm_s.new_tensor(0.0)
    kd_relation = hm_s.new_tensor(0.0)
    kd_keypoint_feature = hm_s.new_tensor(0.0)
    kd_attr = hm_s.new_tensor(0.0)
    pseudo_hm_loss = hm_s.new_tensor(0.0)
    pseudo_reg_loss = hm_s.new_tensor(0.0)
    pseudo_yaw_refiner_loss = hm_s.new_tensor(0.0)
    assistant_outputs = None
    camera_reference_outputs = None
    project_hm_accept = hm_s.new_tensor(0.0)
    project_reg_accept = hm_s.new_tensor(0.0)
    project_depth_accept = hm_s.new_tensor(0.0)
    project_attr_accept = hm_s.new_tensor(0.0)
    if (
        teacher is not None and
        args.teacher_type in {
            "aligned_student", "project_privileged", "camera_ta_relation",
        } and
        (
            args.kd_hm_weight + args.kd_reg_weight + args.kd_feat_weight +
            args.kd_depth_weight + float(getattr(args, "kd_attr_weight", 0.0)) +
            float(getattr(args, "kd_keypoint_feature_weight", 0.0)) +
            (
                float(getattr(args, "kd_relation_weight", 0.0))
                if (
                    args.teacher_type == "camera_ta_relation" or
                    getattr(args, "kd_project_instance_relation", False)
                ) else 0.0
            ) > 0
        )
    ):
        temporal_inputs = None
        if getattr(args, "temporal_bev", False):
            temporal_inputs = (
                prev_img, prev_k, prev_r, prev_t,
                prev_to_current_r, prev_to_current_t, prev_valid,
            )
        with torch.no_grad():
            assistant_outputs = aligned_student_teacher_outputs(
                teacher, hm_gt, img1, k1, r1, t1, temporal_inputs,
                return_auxiliary=True,
                privileged_depth=(
                    depth1 if args.teacher_type == "project_privileged" else None
                ),
            )
            if (
                args.teacher_type == "project_privileged" and
                getattr(args, "kd_project_quality_gate", False)
            ):
                reference_teacher = camera_teacher if camera_teacher is not None else teacher
                camera_reference_outputs = aligned_student_teacher_outputs(
                    reference_teacher, hm_gt, img1, k1, r1, t1, temporal_inputs,
                    return_auxiliary=True,
                    privileged_depth=None,
                    bypass_privileged_path=(camera_teacher is None),
                )
    if (
        teacher is not None and
        args.teacher_type in {"centerpoint_relation", "camera_ta_relation"} and
        float(getattr(args, "kd_relation_weight", 0.0)) > 0
    ):
        teacher_input, teacher_input_mask, relation_keypoints, relation_object_mask = [
            tensor.to(device, non_blocking=True) for tensor in batch[-4:]
        ]
        if getattr(args, "lidar_relation_cache", ""):
            teacher_relations = teacher_input.float()
            teacher_valid = teacher_input_mask.bool()
        else:
            with torch.no_grad():
                teacher_relations, teacher_valid = teacher(
                    teacher_input,
                    teacher_input_mask,
                    relation_keypoints[..., 2:4],
                    relation_object_mask,
                )
        student_relations, student_valid = sample_bev_object_relations(
            bev_s,
            relation_keypoints[..., :2],
            relation_object_mask,
            (float(student.mod.BEV_X_MIN), float(student.mod.BEV_X_MAX)),
            (float(student.mod.BEV_Y_MIN), float(student.mod.BEV_Y_MAX)),
            spatial_axes="xy",
        )
        relation_target = teacher_relations
        relation_valid = teacher_valid & student_valid
        if args.teacher_type == "camera_ta_relation":
            if assistant_outputs is None:
                raise RuntimeError("camera teaching assistant outputs were not computed")
            assistant_relations, assistant_valid = sample_bev_object_relations(
                assistant_outputs[3],
                relation_keypoints[..., :2],
                relation_object_mask,
                (float(student.mod.BEV_X_MIN), float(student.mod.BEV_X_MAX)),
                (float(student.mod.BEV_Y_MIN), float(student.mod.BEV_Y_MAX)),
                spatial_axes="xy",
            )
            relation_valid = relation_valid & assistant_valid
            relation_target = bridge_relation_target(
                assistant_relations, teacher_relations,
                getattr(args, "kd_relation_residual_alpha", 0.25),
            )
        kd_relation = masked_object_relation_loss(
            student_relations, relation_target, relation_valid,
        )

        if float(getattr(args, "kd_keypoint_feature_weight", 0.0)) > 0:
            if assistant_outputs is None:
                raise RuntimeError("keypoint feature KD requires camera assistant outputs")
            student_keypoint_features, student_keypoint_valid = sample_bev_object_features(
                bev_s, relation_keypoints[..., :2], relation_object_mask,
                (float(student.mod.BEV_X_MIN), float(student.mod.BEV_X_MAX)),
                (float(student.mod.BEV_Y_MIN), float(student.mod.BEV_Y_MAX)),
                spatial_axes="xy",
            )
            assistant_keypoint_features, assistant_keypoint_valid = sample_bev_object_features(
                assistant_outputs[3], relation_keypoints[..., :2], relation_object_mask,
                (float(student.mod.BEV_X_MIN), float(student.mod.BEV_X_MAX)),
                (float(student.mod.BEV_Y_MIN), float(student.mod.BEV_Y_MAX)),
                spatial_axes="xy",
            )
            kd_keypoint_feature = masked_object_feature_loss(
                student_keypoint_features, assistant_keypoint_features,
                student_keypoint_valid & assistant_keypoint_valid,
            )

    if teacher is not None and (
        args.kd_hm_weight + args.kd_reg_weight +
        args.kd_feat_weight + args.kd_depth_weight +
        float(getattr(args, "kd_attr_weight", 0.0)) +
        (
            float(getattr(args, "kd_relation_weight", 0.0))
            if getattr(args, "kd_project_instance_relation", False) else 0.0
        )
    ) > 0:
        with torch.no_grad():
            if args.teacher_type in {
                "aligned_student", "project_privileged", "camera_ta_relation",
            }:
                if assistant_outputs is None:
                    raise RuntimeError("aligned camera teacher outputs were not computed")
                hm_t, reg_t, depth_t, bev_t, attr_t = assistant_outputs
            else:
                imgs6, ks6, r6m, t6m = [
                    x.to(device, non_blocking=True) for x in batch[1:5]
                ]
                hm_t, reg_t, depth_t, bev_t = teacher(
                    imgs6, ks6, r6m, t6m, return_bev=True,
                )
                attr_t = None
        project_hm_accepted = None
        project_alpha = float(getattr(args, "kd_project_residual_alpha", 0.5))
        project_margin = float(getattr(args, "kd_project_quality_margin", 0.0))
        project_hybrid = bool(getattr(args, "kd_project_hybrid_routing", False))
        project_independent = bool(
            getattr(args, "kd_project_independent_camera_ta", False)
        )
        project_v45 = bool(
            getattr(args, "kd_project_v45_instance_residual", False)
        )
        if camera_reference_outputs is not None:
            hm_reference, _reg_reference, _depth_reference, bev_reference, _attr_reference = (
                camera_reference_outputs
            )
            if project_v45:
                # Classification remains entirely on the proven V31 assistant.
                hm_t = hm_reference.detach()
                project_hm_accepted = torch.zeros_like(hm_t, dtype=torch.bool)
                project_hm_accept = hm_t.new_tensor(0.0)
            else:
                hm_t, project_hm_accepted = quality_gated_logit_target(
                    hm_t.detach(), hm_reference.detach(), hm_gt,
                    alpha=project_alpha, margin=project_margin,
                )
                project_hm_accept = project_hm_accepted.float().mean()
            if project_hybrid and not project_independent:
                # V38 showed that the mature camera assistant's BEV features
                # carry useful size/attribute information.  Do not replace
                # them with the weaker privileged refinement.
                bev_t = bev_reference.detach()
            else:
                spatial_accept = project_hm_accepted.any(dim=1, keepdim=True).float()
                if project_independent:
                    bev_t = privileged_residual_bridge_target(
                        bev_t, bev_reference, project_alpha,
                    )
                else:
                    bev_t = bev_reference.detach() + project_alpha * (
                        bev_t.detach() - bev_reference.detach()
                    ) * spatial_accept
        teacher_prob = torch.sigmoid(hm_t.detach() / max(float(args.kd_temp), 1e-6))
        teacher_conf = teacher_prob.amax(dim=1, keepdim=True)
        gt_conf = hm_gt.amax(dim=1, keepdim=True)
        teacher_foreground = (teacher_conf >= args.kd_conf_thr).float()
        gt_foreground = (gt_conf >= args.kd_gt_thr).float()
        if getattr(args, "kd_full_object_foreground", False):
            extent_foreground = build_oriented_foreground_extent_mask(
                batch[10], batch[11], float(student.mod.BEV_RES),
                ellipse_scale=float(getattr(args, "kd_foreground_ellipse_scale", 1.0)),
            ).to(device=device, non_blocking=True)
            gt_foreground = torch.maximum(gt_foreground, extent_foreground)
        false_positive_mask = teacher_foreground * (1.0 - gt_foreground)
        fg_mask = ((teacher_foreground > 0) | (gt_foreground > 0)).float()
        fg_mask = dilate_mask(fg_mask, args.kd_mask_dilate)
        false_positive_mask = dilate_mask(false_positive_mask, args.kd_mask_dilate)
        fov_mask = bev_fov_mask.float().clamp(0.0, 1.0)
        if args.kd_hm_mask == "foreground":
            hm_weight = fov_mask * fg_mask
        else:
            hm_weight = fov_mask * (1.0 + args.kd_pos_boost * fg_mask)
        kd_hm_raw = F.binary_cross_entropy_with_logits(
            hm_s.float() / max(float(args.kd_temp), 1e-6),
            teacher_prob,
            reduction="none",
        ) * (float(args.kd_temp) ** 2)
        kd_hm = masked_mean(kd_hm_raw, hm_weight)

        reg_kd_mask = reg_mask * fov_mask
        if (
            args.teacher_type in {
                "aligned_student", "project_privileged", "camera_ta_relation",
            } and
            float(args.kd_reg_conf_thr) > 0
        ):
            teacher_gt_conf = teacher_prob.gather(
                1, supervised_class_indices.unsqueeze(1),
            )
            reg_kd_mask = reg_kd_mask * (
                teacher_gt_conf >= float(args.kd_reg_conf_thr)
            ).float()
        kd_reg_target = reg_t.detach()
        supervised_kd_target = reg_gt
        if getattr(args, "velocity_displacement_target", False):
            kd_reg_target = velocity_target_as_displacement(kd_reg_target, velocity_intervals)
            supervised_kd_target = velocity_target_as_displacement(
                supervised_kd_target, velocity_intervals,
            )
        if camera_reference_outputs is not None:
            reference_reg_target = camera_reference_outputs[1].detach()
            if getattr(args, "velocity_displacement_target", False):
                reference_reg_target = velocity_target_as_displacement(
                    reference_reg_target, velocity_intervals,
                )
            if project_hybrid:
                kd_reg_target, project_reg_accepted = hybrid_quality_gated_regression_target(
                    kd_reg_target, reference_reg_target, supervised_kd_target, reg_kd_mask,
                    privileged_channels=(0, 1, 2),
                    alpha=project_alpha, margin=project_margin,
                )
            else:
                kd_reg_target, project_reg_accepted = quality_gated_regression_target(
                    kd_reg_target, reference_reg_target, supervised_kd_target, reg_kd_mask,
                    alpha=project_alpha, margin=project_margin,
                )
            active_channels = (
                kd_reg_channel_scales(args, kd_reg_target) > 0
            ).expand_as(project_reg_accepted)
            if project_hybrid:
                privileged_channel_mask = torch.zeros_like(active_channels)
                privileged_channel_mask[:, :3] = True
                active_channels = active_channels & privileged_channel_mask
            active_centres = reg_kd_mask.bool().expand_as(project_reg_accepted) & active_channels
            project_reg_accept = (
                (project_reg_accepted & active_centres).float().sum() /
                active_centres.float().sum().clamp(min=1.0)
            )
        kd_reg_prediction = (
            supervised_reg_prediction
            if args.teacher_type in {
                "aligned_student", "project_privileged", "camera_ta_relation",
            } else reg_s
        )
        kd_reg = masked_reg_l1(
            kd_reg_prediction, kd_reg_target, reg_kd_mask, vel_mask,
            channel_scales=kd_reg_channel_scales(args, kd_reg_prediction),
        )

        if project_v45:
            if camera_teacher is None:
                raise RuntimeError("V45 instance residual requires independent V31 teacher")
            relation_keypoints = batch[-2].to(device, non_blocking=True)
            relation_object_mask = batch[-1].to(device, non_blocking=True).bool()
            residual_memory = bool(
                getattr(args, "kd_project_residual_memory", False)
            )
            student_low = (
                getattr(student, "_last_memory_residual", None)
                if residual_memory else
                getattr(student, "_last_low_level_bev", None)
            )
            privileged_low = getattr(
                teacher,
                "_last_bev_ctx" if residual_memory else "_last_low_level_bev",
                None,
            )
            camera_low = getattr(
                camera_teacher,
                "_last_bev_ctx" if residual_memory else "_last_low_level_bev",
                None,
            )
            if any(value is None for value in (student_low, privileged_low, camera_low)):
                raise RuntimeError("V45 low-level BEV seam was not populated")
            if residual_memory:
                low_target = project_alpha * (
                    privileged_low.detach() - camera_low.detach()
                )
            else:
                low_target = privileged_residual_bridge_target(
                    privileged_low, camera_low, project_alpha,
                )
            student_points, student_point_valid = sample_bev_object_features(
                student_low, relation_keypoints[..., :2], relation_object_mask,
                (float(student.mod.BEV_X_MIN), float(student.mod.BEV_X_MAX)),
                (float(student.mod.BEV_Y_MIN), float(student.mod.BEV_Y_MAX)),
                spatial_axes="xy",
            )
            teacher_points, teacher_point_valid = sample_bev_object_features(
                low_target, relation_keypoints[..., :2], relation_object_mask,
                (float(student.mod.BEV_X_MIN), float(student.mod.BEV_X_MAX)),
                (float(student.mod.BEV_Y_MIN), float(student.mod.BEV_Y_MAX)),
                spatial_axes="xy",
            )
            keypoint_loss = masked_object_feature_loss(
                student_points, teacher_points,
                student_point_valid & teacher_point_valid,
            )
            foreground_loss = instance_normalized_foreground_feature_loss(
                student_low, low_target, reg_gt, reg_mask,
                float(student.mod.BEV_RES),
                scales=getattr(args, "kd_feature_scales", (1, 2, 4)),
            )
            kd_feat = 0.5 * keypoint_loss + 0.5 * foreground_loss
            if (
                getattr(args, "kd_project_instance_relation", False) and
                float(getattr(args, "kd_relation_weight", 0.0)) > 0
            ):
                student_context = getattr(student, "_last_bev_ctx", None)
                privileged_context = getattr(teacher, "_last_bev_ctx", None)
                camera_context = getattr(camera_teacher, "_last_bev_ctx", None)
                if any(value is None for value in (
                    student_context, privileged_context, camera_context,
                )):
                    raise RuntimeError("V58 instance relation context seam is missing")
                student_relations, student_relation_valid = sample_bev_object_relations(
                    student_context, relation_keypoints[..., :2], relation_object_mask,
                    (float(student.mod.BEV_X_MIN), float(student.mod.BEV_X_MAX)),
                    (float(student.mod.BEV_Y_MIN), float(student.mod.BEV_Y_MAX)),
                    spatial_axes="xy",
                )
                privileged_relations, privileged_relation_valid = sample_bev_object_relations(
                    privileged_context, relation_keypoints[..., :2], relation_object_mask,
                    (float(student.mod.BEV_X_MIN), float(student.mod.BEV_X_MAX)),
                    (float(student.mod.BEV_Y_MIN), float(student.mod.BEV_Y_MAX)),
                    spatial_axes="xy",
                )
                camera_relations, camera_relation_valid = sample_bev_object_relations(
                    camera_context, relation_keypoints[..., :2], relation_object_mask,
                    (float(student.mod.BEV_X_MIN), float(student.mod.BEV_X_MAX)),
                    (float(student.mod.BEV_Y_MIN), float(student.mod.BEV_Y_MAX)),
                    spatial_axes="xy",
                )
                relation_target = bridge_relation_target(
                    camera_relations, privileged_relations, project_alpha,
                )
                relation_valid = (
                    student_relation_valid & privileged_relation_valid &
                    camera_relation_valid
                )
                kd_relation = masked_object_relation_loss(
                    student_relations, relation_target, relation_valid,
                )
        elif getattr(args, "kd_multiscale_region_feature", False):
            kd_feat = multiscale_region_feature_kd(
                bev_s, bev_t, fov_mask, fg_mask, false_positive_mask,
                scales=getattr(args, "kd_feature_scales", (1, 2, 4)),
                foreground_weight=getattr(args, "kd_feature_fg_weight", 1.0),
                false_positive_weight=getattr(args, "kd_feature_fp_weight", 0.5),
                background_weight=getattr(args, "kd_feature_bg_weight", 0.05),
                attention_strength=getattr(args, "kd_feature_attention_strength", 1.0),
            )
        else:
            if args.kd_feature_mask == "foreground":
                feat_mask = fov_mask * fg_mask
            else:
                feat_mask = fov_mask
            bev_s_norm = F.normalize(bev_s.float(), dim=1)
            bev_t_norm = F.normalize(bev_t.detach().float(), dim=1)
            kd_feat = masked_mean(
                F.smooth_l1_loss(
                    bev_s_norm, bev_t_norm, reduction="none", beta=0.5,
                ),
                feat_mask,
            )
        if args.kd_depth_weight > 0:
            if args.teacher_type in {
                "aligned_student", "project_privileged", "camera_ta_relation",
            }:
                selected_teacher_depth = depth_t
            else:
                if student_cam_indices is None:
                    default_idx = getattr(student.mod, "CAM_CHANNELS").index(
                        getattr(student.mod, "STUDENT_CAM")
                    )
                    student_cam_indices = torch.full(
                        (depth_t.shape[0],), default_idx,
                        device=depth_t.device, dtype=torch.long,
                    )
                batch_indices = torch.arange(depth_t.shape[0], device=depth_t.device)
                selected_teacher_depth = depth_t[batch_indices, student_cam_indices]
            teacher_depth = F.interpolate(
                selected_teacher_depth.float().detach(),
                size=depth_s.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            if camera_reference_outputs is not None:
                reference_depth = F.interpolate(
                    camera_reference_outputs[2].float().detach(),
                    size=depth_s.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                if project_hybrid and not project_independent:
                    teacher_depth = reference_depth
                elif project_independent:
                    # The independent 29.5-NDS camera assistant supplies the stable
                    # visual posterior. Transfer the privileged teacher's dense
                    # completion residual instead of discarding it as V43 did.
                    teacher_depth = privileged_residual_bridge_target(
                        teacher_depth, reference_depth, project_alpha,
                    )
                else:
                    teacher_depth, project_depth_accepted = quality_gated_categorical_target(
                        teacher_depth, reference_depth, depth1.long(),
                        alpha=project_alpha, margin=project_margin,
                    )
                    valid_depth = (depth1 >= 0).unsqueeze(1)
                    project_depth_accept = (
                        (project_depth_accepted & valid_depth).float().sum() /
                        valid_depth.float().sum().clamp(min=1.0)
                    )
            residual_memory = bool(
                getattr(args, "kd_project_residual_memory", False)
            )
            if residual_memory:
                student_depth_residual = getattr(
                    student, "_last_camera_depth_residual", None,
                )
                if student_depth_residual is None:
                    raise RuntimeError("camera depth residual adapter output is missing")
                privileged_depth_residual = project_alpha * (
                    F.interpolate(
                        depth_t.float().detach(), size=depth_s.shape[-2:],
                        mode="bilinear", align_corners=False,
                    ) - F.interpolate(
                        camera_reference_outputs[2].float().detach(),
                        size=depth_s.shape[-2:], mode="bilinear",
                        align_corners=False,
                    )
                )
                # Depth logits are invariant to a shared per-pixel offset. Remove
                # that null direction before supervising the deployable residual.
                student_depth_residual = (
                    student_depth_residual.float() -
                    student_depth_residual.float().mean(dim=1, keepdim=True)
                )
                privileged_depth_residual = (
                    privileged_depth_residual -
                    privileged_depth_residual.mean(dim=1, keepdim=True)
                )
                kd_depth = F.smooth_l1_loss(
                    student_depth_residual, privileged_depth_residual,
                    reduction="none", beta=0.5,
                ).mean(dim=1)
            else:
                # Average KL over pixels; batchmean would sum the full HxW map.
                kd_depth = F.kl_div(
                    F.log_softmax(depth_s.float(), dim=1),
                    F.softmax(teacher_depth, dim=1),
                    reduction="none",
                ).sum(dim=1)
            if project_v45:
                # Transfer dense completion only near real LiDAR anchors. This
                # avoids full-image imitation while exposing local dense geometry.
                depth_mask = dilate_mask(
                    (depth1 != -1).float().unsqueeze(1), 3,
                ).squeeze(1)
            elif args.kd_depth_valid_only:
                depth_mask = (depth1 != -1).float()
            else:
                depth_conf = F.softmax(teacher_depth, dim=1).amax(dim=1)
                depth_mask = (depth_conf >= args.kd_depth_conf_thr).float()
            kd_depth = masked_mean(kd_depth, depth_mask)

        if (
            float(getattr(args, "kd_attr_weight", 0.0)) > 0 and
            attr_s is not None and attr_t is not None
        ):
            attr_target = batch[17].to(device, non_blocking=True).long()
            valid_attr = attr_target >= 0
            if valid_attr.any():
                temperature = max(float(getattr(args, "kd_attr_temp", 1.0)), 1e-6)
                student_attr = attr_s.float().permute(0, 2, 3, 1)[valid_attr]
                teacher_attr_target = attr_t.detach()
                if camera_reference_outputs is not None:
                    if project_hybrid:
                        teacher_attr_target = camera_reference_outputs[4].detach()
                    else:
                        teacher_attr_target, project_attr_accepted = quality_gated_attribute_target(
                            teacher_attr_target, camera_reference_outputs[4].detach(), attr_target,
                            alpha=project_alpha, margin=project_margin,
                        )
                        valid_attr_map = valid_attr.unsqueeze(1)
                        project_attr_accept = (
                            (project_attr_accepted & valid_attr_map).float().sum() /
                            valid_attr_map.float().sum().clamp(min=1.0)
                        )
                teacher_attr = teacher_attr_target.float().permute(0, 2, 3, 1)[valid_attr]
                kd_attr = F.kl_div(
                    F.log_softmax(student_attr / temperature, dim=1),
                    F.softmax(teacher_attr / temperature, dim=1),
                    reduction="batchmean",
                ) * (temperature ** 2)

    pseudo_offset = 30 if history_enabled else 26
    relation_field_count = 4 if (
        (
            args.teacher_type in {"centerpoint_relation", "camera_ta_relation"} and
            (
                float(getattr(args, "kd_relation_weight", 0.0)) +
                float(getattr(args, "kd_keypoint_feature_weight", 0.0))
            ) > 0
        ) or bool(getattr(args, "kd_project_v45_instance_residual", False))
    ) else 0
    if batch_has_pseudo_targets(batch, pseudo_offset, relation_field_count) and (
        args.pseudo_hm_weight > 0 or args.pseudo_reg_weight > 0 or
        float(getattr(args, "pseudo_yaw_refiner_weight", 0.0)) > 0
    ):
        pseudo_hm, pseudo_reg, pseudo_reg_mask, pseudo_score = [
            tensor.to(device, non_blocking=True) for tensor in batch[pseudo_offset:pseudo_offset + 4]
        ]
        pseudo_positive_mask = (pseudo_hm > 0.01).float()
        pseudo_hm_raw = F.binary_cross_entropy_with_logits(
            hm_s.float(), pseudo_hm.float().clamp(0.0, 1.0), reduction="none",
        )
        pseudo_class_loss_weights = getattr(args, "pseudo_class_loss_weights", None)
        if pseudo_class_loss_weights is not None:
            class_weights = pseudo_hm_raw.new_tensor(pseudo_class_loss_weights).view(1, -1, 1, 1)
            pseudo_positive_mask = pseudo_positive_mask * class_weights
        pseudo_hm_loss = masked_mean(pseudo_hm_raw, pseudo_positive_mask)

        reg_weight = pseudo_reg_mask.float() * pseudo_score.float().clamp(0.0, 1.0)
        pseudo_reg_class_weights = getattr(args, "pseudo_reg_class_weights", None)
        if pseudo_reg_class_weights is not None:
            if len(pseudo_reg_class_weights) != len(DETECTION_NAMES):
                raise ValueError(
                    f"expected {len(DETECTION_NAMES)} pseudo regression class weights"
                )
            pseudo_class_indices = pseudo_hm.argmax(dim=1)
            class_weights = reg_weight.new_tensor(pseudo_reg_class_weights)
            reg_weight = reg_weight * class_weights[pseudo_class_indices].unsqueeze(1)
        reg_weight10 = reg_weight.expand_as(reg_s) * pseudo_reg_channel_scales(args, reg_s)
        pseudo_reg_target = pseudo_reg.float()
        if getattr(args, "velocity_displacement_target", False):
            pseudo_reg_target = velocity_target_as_displacement(pseudo_reg_target, velocity_intervals)
        pseudo_reg_raw = F.smooth_l1_loss(
            reg_s.float(), pseudo_reg_target, reduction="none", beta=1.0,
        )
        pseudo_reg_loss = (
            torch.nan_to_num(pseudo_reg_raw, nan=0.0, posinf=0.0, neginf=0.0) * reg_weight10
        ).sum() / reg_weight10.sum().clamp(min=1.0)

        # A strong image teacher is especially reliable for orientation, but the
        # S1 detector applies yaw through a class-conditioned residual head.  A
        # loss on the shared base reg tensor therefore supervises the wrong
        # deployed output.  Select the final prediction with pseudo classes and
        # train the zero-initialized residual head directly at teacher centres.
        if (
            student.residual_yaw_vector_head is not None and
            float(getattr(args, "pseudo_yaw_refiner_weight", 0.0)) > 0
        ):
            pseudo_class_indices = pseudo_hm.argmax(dim=1)
            pseudo_reg_prediction = apply_class_center_residual(
                reg_s, class_center_s, pseudo_class_indices,
            )
            pseudo_reg_prediction = apply_class_motion_residual(
                pseudo_reg_prediction, class_motion_s, pseudo_class_indices,
            )
            pseudo_reg_prediction = apply_class_size_residual(
                pseudo_reg_prediction, class_size_s, pseudo_class_indices,
            )
            pseudo_yaw_refiner_loss = residual_yaw_vector_loss(
                residual_azimuth_s, pseudo_reg_prediction, pseudo_reg_target,
                pseudo_hm, reg_weight,
                getattr(args, "pseudo_yaw_class_weights", None),
            )

    weighted_kd_branches = [
        (
            "camera_hm", args.kd_hm_weight * kd_hm,
            getattr(args, "kd_share_camera_hm", 0.0),
        ),
        (
            "camera_reg", args.kd_reg_weight * kd_reg,
            getattr(args, "kd_share_camera_reg", 0.0),
        ),
        (
            "camera_feature", args.kd_feat_weight * kd_feat,
            getattr(args, "kd_share_camera_feature", 0.0),
        ),
        (
            "camera_depth", args.kd_depth_weight * kd_depth,
            getattr(args, "kd_share_camera_depth", 0.0),
        ),
        (
            "camera_attr", float(getattr(args, "kd_attr_weight", 0.0)) * kd_attr,
            getattr(args, "kd_share_camera_attr", 0.0),
        ),
        (
            "lidar_relation",
            float(getattr(args, "kd_relation_weight", 0.0)) * kd_relation,
            getattr(args, "kd_share_lidar_relation", 0.0),
        ),
        (
            "camera_keypoint_feature",
            float(getattr(args, "kd_keypoint_feature_weight", 0.0)) * kd_keypoint_feature,
            getattr(args, "kd_share_camera_keypoint_feature", 0.0),
        ),
        (
            "lidar_hm", args.pseudo_hm_weight * pseudo_hm_loss,
            getattr(args, "kd_share_lidar_hm", 0.0),
        ),
        (
            "lidar_reg", args.pseudo_reg_weight * pseudo_reg_loss,
            getattr(args, "kd_share_lidar_reg", 0.0),
        ),
        (
            "lidar_yaw",
            float(getattr(args, "pseudo_yaw_refiner_weight", 0.0)) *
            pseudo_yaw_refiner_loss,
            getattr(args, "kd_share_lidar_yaw", 0.0),
        ),
    ]
    kd_branch_scales: Dict[str, torch.Tensor] = {}
    natural_anchor_diagnostics: Dict[str, float] = {}
    if getattr(args, "kd_gradient_balance", False) and torch.is_grad_enabled():
        kd_gradient_anchor = bev_s
        if getattr(args, "kd_gradient_anchor_image_feature", False):
            kd_gradient_anchor = getattr(student, "_last_shared_image_feature", None)
            if kd_gradient_anchor is None:
                raise RuntimeError("shared image-feature gradient anchor was not populated")
        if getattr(args, "kd_per_branch_mema", False):
            bev_branches = []
            natural_losses: Dict[str, torch.Tensor] = {}
            natural_anchors: Dict[str, Optional[torch.Tensor]] = {}
            natural_shares: Dict[str, float] = {}
            for name, branch_loss, share in weighted_kd_branches:
                if name in ("camera_feature", "camera_depth") and float(share) > 0.0:
                    natural_losses[name] = branch_loss
                    natural_shares[name] = float(share)
                    if name == "camera_feature":
                        natural_anchors[name] = getattr(
                            student, "_last_low_level_bev", None,
                        )
                    else:
                        natural_anchors[name] = depth_s if depth_s is not None else None
                else:
                    bev_branches.append((name, branch_loss, share))
            natural_scaled, natural_anchor_diagnostics = per_branch_mema_gate(
                natural_losses, natural_anchors, det_loss,
                decay=float(getattr(args, "kd_mema_decay", 0.90)),
                floor=float(getattr(args, "kd_mema_floor", 0.15)),
                project_conflicts=bool(getattr(
                    args, "kd_natural_anchor_project_conflicts", False,
                )),
                max_gradient_ratio=float(getattr(
                    args, "kd_natural_anchor_max_gradient_ratio", 0.0,
                )),
            )
            weighted_kd_branches = list(bev_branches) + [
                (name, natural_scaled[name], natural_shares[name])
                for name in natural_losses
            ]
        kd_loss, kd_branch_scales = gradient_balanced_kd_sum(
            weighted_kd_branches, kd_gradient_anchor,
            minimum=getattr(args, "kd_gradient_balance_min", 0.1),
            maximum=getattr(args, "kd_gradient_balance_max", 10.0),
            exact=getattr(args, "kd_gradient_balance_exact", False),
            reference_loss=det_loss,
            skip_negative_cosine=getattr(
                args, "kd_gradient_skip_negative_cosine", False,
            ),
            protected_reference_loss=loss_hm,
            project_negative_protected_cosine=getattr(
                args, "kd_gradient_project_heatmap_conflicts", False,
            ),
        )
        for key, value in natural_anchor_diagnostics.items():
            kd_branch_scales[key] = hm_s.new_tensor(value)
    else:
        kd_loss = sum(
            (loss for _name, loss, _share in weighted_kd_branches),
            hm_s.new_tensor(0.0),
        )
    history_loss = (
        float(getattr(args, "hop_hm_weight", 0.0)) * history_hm_loss +
        float(getattr(args, "hop_reg_weight", 0.0)) * history_reg_loss
    )
    kd_balance = adaptive_kd_balance_scale(
        det_loss, kd_loss,
        getattr(
            args, "_current_kd_balance_target",
            getattr(args, "kd_balance_target", 0.0),
        ),
        getattr(args, "kd_balance_min", 0.5),
        getattr(args, "kd_balance_max", 2.5),
    )
    loss = det_loss + float(kd_scale) * (kd_balance * kd_loss + history_loss)
    parts = {
        "hm": loss_hm, "reg": loss_reg, "depth": loss_depth, "attr": loss_attr,
        "center_error": center_error, "det_loss": det_loss,
        "yaw_metric": yaw_metric_loss, "velocity_metric": velocity_metric_loss,
        "azimuth_bin": azimuth_bin_loss, "size_prior": size_prior_loss,
        "ase_iou": ase_iou_loss, "velocity_decomposed": velocity_decomposed_loss,
        "yaw_velocity_consistency": yaw_velocity_consistency,
        "image_yaw_flip": image_yaw_flip_loss,
        "image_radial_center": image_radial_center_loss,
        "kd_hm": kd_hm, "kd_reg": kd_reg, "kd_feat": kd_feat, "kd_depth": kd_depth,
        "kd_attr": kd_attr, "kd_relation": kd_relation,
        "kd_keypoint_feature": kd_keypoint_feature,
        "relative_depth": relative_depth_loss,
        "pseudo_hm": pseudo_hm_loss, "pseudo_reg": pseudo_reg_loss,
        "pseudo_yaw_refiner": pseudo_yaw_refiner_loss,
        "history_hm": history_hm_loss, "history_reg": history_reg_loss,
        "history_loss": history_loss.detach(),
        "kd_loss": kd_loss.detach(), "kd_scale": hm_s.new_tensor(float(kd_scale)),
        "kd_balance": kd_balance.detach(),
        "kd_project_hm_accept": project_hm_accept.detach(),
        "kd_project_reg_accept": project_reg_accept.detach(),
        "kd_project_depth_accept": project_depth_accept.detach(),
        "kd_project_attr_accept": project_attr_accept.detach(),
    }
    for branch_name in (
        "camera_hm", "camera_reg", "camera_feature", "camera_depth", "camera_attr",
    ):
        parts[f"kd_grad_scale_{branch_name}"] = kd_branch_scales.get(
            branch_name, hm_s.new_tensor(0.0),
        ).detach()
    for branch_name in ("camera_feature", "camera_depth"):
        for diagnostic in ("cosine", "mema_scale", "projected", "gradient_ratio"):
            key = f"{branch_name}_{diagnostic}"
            if key in kd_branch_scales:
                parts[f"natural_anchor_{key}"] = kd_branch_scales[key].detach()
    return loss, parts


def csv_init(path: Path, fieldnames: Sequence[str]):
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()


def csv_append(path: Path, row: Dict, fieldnames: Sequence[str]):
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow({k: row.get(k, "") for k in fieldnames})


def save_training_checkpoint(path: Path, model, optimizer, scheduler, scaler, epoch: int,
                             best_val: float, best_metric_name: str, args, name: str):
    torch.save({
        "epoch": int(epoch),
        "name": name,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "best_val": float(best_val),
        "best_metric_name": best_metric_name,
        "args": vars(args),
    }, path)


def restore_training_checkpoint(path: str, model, optimizer=None, scheduler=None, scaler=None):
    if not path:
        return 1, float("inf")
    state = torch.load(path, map_location="cpu")
    load_model_state_compatible(model, state, strict=True)
    start_epoch = 1
    best_val = float("inf")
    if isinstance(state, dict) and "model_state_dict" in state:
        if optimizer is not None and "optimizer_state_dict" in state:
            optimizer.load_state_dict(state["optimizer_state_dict"])
        if scheduler is not None and "scheduler_state_dict" in state:
            scheduler.load_state_dict(state["scheduler_state_dict"])
        if scaler is not None and state.get("scaler_state_dict") is not None:
            scaler.load_state_dict(state["scaler_state_dict"])
        start_epoch = int(state.get("epoch", 0)) + 1
        best_val = float(state.get("best_val", best_val))
    return start_epoch, best_val


def train_loop(
    mod, args, model, train_loader, val_loader, teacher=None, name="model",
    camera_teacher=None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.to(device)
    if teacher is not None:
        teacher.to(device).eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
    if camera_teacher is not None:
        camera_teacher.to(device).eval()
        for p in camera_teacher.parameters():
            p.requires_grad_(False)

    activation_stats: Dict[str, Dict[str, float]] = {}
    activation_hooks = []
    if bool(getattr(args, "debug_nonfinite_activations", False)):
        debug_modules = (
            "backbone", "depth_head", "lss3d", "temporal_pre_encoder",
            "temporal_fusion", "bev_encoder", "det_head", "attr_head",
            "class_motion_head", "class_size_head",
        )

        def _activation_hook(name):
            def hook(_module, _inputs, output):
                tensors = []

                def collect(value):
                    if torch.is_tensor(value):
                        tensors.append(value.detach())
                    elif isinstance(value, (tuple, list)):
                        for item in value:
                            collect(item)
                    elif isinstance(value, dict):
                        for item in value.values():
                            collect(item)

                collect(output)
                if not tensors:
                    return
                finite = all(bool(torch.isfinite(tensor).all().item()) for tensor in tensors)
                finite_values = [
                    tensor.float()[torch.isfinite(tensor)] for tensor in tensors
                    if tensor.numel()
                ]
                finite_values = [value for value in finite_values if value.numel()]
                activation_stats[name] = {
                    "finite": float(finite),
                    "abs_max": max(
                        (float(value.abs().max().item()) for value in finite_values),
                        default=float("nan"),
                    ),
                    "tensor_count": float(len(tensors)),
                }

            return hook

        for module_name in debug_modules:
            module = getattr(model, module_name, None)
            if module is not None:
                activation_hooks.append(
                    module.register_forward_hook(_activation_hook(module_name))
                )
        print(f"[ACTIVATION_DEBUG] registered={list(debug_modules)}")

    motion_lr_multiplier = float(getattr(args, "motion_head_lr_multiplier", 1.0))
    backbone_lr_multiplier = float(getattr(args, "backbone_lr_multiplier", 1.0))
    motion_head = getattr(model, "class_motion_head", None)
    temporal_spatial_fusion = getattr(model, "temporal_spatial_fusion", None)
    temporal_correlation_motion_head = getattr(model, "temporal_correlation_motion_head", None)
    object_velocity_refiner = getattr(model, "object_velocity_refiner", None)
    class_attribute_velocity_calibrator = getattr(
        model, "class_attribute_velocity_calibrator", None,
    )
    residual_azimuth_multibin_head = getattr(model, "residual_azimuth_multibin_head", None)
    residual_yaw_vector_head = getattr(model, "residual_yaw_vector_head", None)
    azimuth_bin_head = getattr(model, "azimuth_bin_head", None)
    class_size_head = getattr(model, "class_size_head", None)
    class_center_head = getattr(model, "class_center_head", None)
    image_yaw_flip_head = getattr(model, "image_yaw_flip_head", None)
    image_radial_center_head = getattr(model, "image_radial_center_head", None)
    image_object_geometry_head = getattr(model, "image_object_geometry_head", None)
    upgrade_modules = [
        module for module in (
            motion_head, temporal_spatial_fusion, temporal_correlation_motion_head,
            object_velocity_refiner, residual_azimuth_multibin_head,
            residual_yaw_vector_head,
            class_attribute_velocity_calibrator, azimuth_bin_head, class_size_head,
            class_center_head,
            image_yaw_flip_head,
            image_radial_center_head,
            image_object_geometry_head,
        ) if module is not None
    ]
    use_motion_group = bool(upgrade_modules) and abs(motion_lr_multiplier - 1.0) > 1e-9
    use_backbone_group = (
        getattr(model, "backbone", None) is not None and
        abs(backbone_lr_multiplier - 1.0) > 1e-9
    )
    if use_motion_group or use_backbone_group:
        motion_parameters = [
            parameter for module in upgrade_modules for parameter in module.parameters()
            if parameter.requires_grad
        ] if use_motion_group else []
        motion_parameter_ids = {id(parameter) for parameter in motion_parameters}
        backbone_parameters = [
            parameter for parameter in model.backbone.parameters()
            if parameter.requires_grad and id(parameter) not in motion_parameter_ids
        ] if use_backbone_group else []
        backbone_parameter_ids = {id(parameter) for parameter in backbone_parameters}
        base_parameters = [
            parameter for parameter in model.parameters()
            if parameter.requires_grad and
            id(parameter) not in motion_parameter_ids and
            id(parameter) not in backbone_parameter_ids
        ]
        parameter_groups = []
        if base_parameters:
            parameter_groups.append({"params": base_parameters, "lr": args.lr})
        if backbone_parameters:
            parameter_groups.append({
                "params": backbone_parameters,
                "lr": args.lr * backbone_lr_multiplier,
            })
        if motion_parameters:
            parameter_groups.append({"params": motion_parameters, "lr": args.lr * motion_lr_multiplier})
        optimizer = torch.optim.AdamW(
            parameter_groups,
            weight_decay=args.weight_decay,
        )
        minimum_factor = 0.05
        schedule_epochs = int(getattr(args, "schedule_total_epochs", 0) or args.epochs)
        cosine_factor = lambda epoch: minimum_factor + (1.0 - minimum_factor) * (
            1.0 + math.cos(math.pi * min(epoch, schedule_epochs) / max(1, schedule_epochs))
        ) / 2.0
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=[cosine_factor] * len(parameter_groups),
        )
        print(
            f"[OPTIMIZER] base_lr={args.lr:g} "
            f"backbone_lr={args.lr * backbone_lr_multiplier:g} "
            f"motion_lr={args.lr * motion_lr_multiplier:g} "
            f"groups={len(parameter_groups)}"
        )
    else:
        optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=args.lr, weight_decay=args.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, int(getattr(args, "schedule_total_epochs", 0) or args.epochs)),
            eta_min=args.lr * 0.05,
        )
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")

    resume_path = getattr(args, "resume", "")
    if name == "student" and getattr(args, "resume_student", ""):
        resume_path = args.resume_student
    ckpt_epoch_offset = 0
    ckpt_best_val = float("inf")
    if resume_path:
        start_epoch, ckpt_best_val = restore_training_checkpoint(resume_path, model, optimizer, scheduler, scaler)
        repaired_bn = sanitize_batchnorm_running_stats(model)
        if repaired_bn:
            print(f"[BN_SANITIZE] repaired {repaired_bn} non-finite running-stat values after resume")
        ckpt_epoch_offset = max(0, start_epoch - 1)
        print(f"[RESUME] {name} from {resume_path}; start_epoch={start_epoch} best_val={ckpt_best_val:.6f}")
        if bool(getattr(args, "freeze_privileged_memory_after_resume", False)):
            adapter = getattr(model, "privileged_residual_adapter", None)
            if adapter is None:
                raise ValueError("privileged residual adapter is missing after resume")
            for parameter in adapter.parameters():
                parameter.requires_grad_(False)
                parameter.grad = None
            print(
                "[FREEZE_AFTER_RESUME] learned geometry adapter frozen; "
                "pure-GT detector fine-tuning continues"
            )

    fields = [
        "epoch", "lr", "train_loss", "train_det_loss", "train_hm", "train_reg", "train_depth",
        "train_attr", "train_center_error", "val_loss", "val_det_loss", "val_hm",
        "val_reg", "val_depth", "val_attr", "val_center_error",
        "best_metric", "sec",
    ]
    if name == "student":
        fields += [
            "kd_scale", "kd_balance_target", "train_kd_balance", "val_kd_balance",
            "train_kd_loss", "val_kd_loss",
            "train_kd_hm", "train_kd_reg", "train_kd_feat", "train_kd_depth",
            "val_kd_hm", "val_kd_reg", "val_kd_feat", "val_kd_depth",
            "train_kd_attr", "val_kd_attr",
            "train_kd_relation", "val_kd_relation",
            "train_kd_keypoint_feature", "val_kd_keypoint_feature",
            "train_kd_project_hm_accept", "val_kd_project_hm_accept",
            "train_kd_project_reg_accept", "val_kd_project_reg_accept",
            "train_kd_project_depth_accept", "val_kd_project_depth_accept",
            "train_kd_project_attr_accept", "val_kd_project_attr_accept",
            "train_kd_grad_scale_camera_hm", "val_kd_grad_scale_camera_hm",
            "train_kd_grad_scale_camera_reg", "val_kd_grad_scale_camera_reg",
            "train_kd_grad_scale_camera_feature", "val_kd_grad_scale_camera_feature",
            "train_kd_grad_scale_camera_depth", "val_kd_grad_scale_camera_depth",
            "train_kd_grad_scale_camera_attr", "val_kd_grad_scale_camera_attr",
            "train_natural_anchor_camera_feature_cosine",
            "val_natural_anchor_camera_feature_cosine",
            "train_natural_anchor_camera_feature_mema_scale",
            "val_natural_anchor_camera_feature_mema_scale",
            "train_natural_anchor_camera_feature_projected",
            "val_natural_anchor_camera_feature_projected",
            "train_natural_anchor_camera_feature_gradient_ratio",
            "val_natural_anchor_camera_feature_gradient_ratio",
            "train_natural_anchor_camera_depth_cosine",
            "val_natural_anchor_camera_depth_cosine",
            "train_natural_anchor_camera_depth_mema_scale",
            "val_natural_anchor_camera_depth_mema_scale",
            "train_natural_anchor_camera_depth_projected",
            "val_natural_anchor_camera_depth_projected",
            "train_natural_anchor_camera_depth_gradient_ratio",
            "val_natural_anchor_camera_depth_gradient_ratio",
            "train_relative_depth", "val_relative_depth",
            "train_pseudo_hm", "train_pseudo_reg", "train_pseudo_yaw_refiner",
            "val_pseudo_hm", "val_pseudo_reg", "val_pseudo_yaw_refiner",
            "train_history_loss", "train_history_hm", "train_history_reg",
            "val_history_loss", "val_history_hm", "val_history_reg",
            "train_yaw_metric", "train_velocity_metric",
            "val_yaw_metric", "val_velocity_metric",
            "train_azimuth_bin", "val_azimuth_bin",
            "train_size_prior", "val_size_prior",
            "train_ase_iou", "val_ase_iou",
            "train_velocity_decomposed", "val_velocity_decomposed",
            "train_yaw_velocity_consistency", "val_yaw_velocity_consistency",
            "train_image_yaw_flip", "val_image_yaw_flip",
            "train_image_radial_center", "val_image_radial_center",
        ]
    csv_path = out_dir / f"{name}_train.csv"
    csv_init(csv_path, fields)

    existing_rows = []
    if csv_path.exists():
        with csv_path.open("r", newline="", encoding="utf-8") as f:
            existing_rows = list(csv.DictReader(f))
    epoch_offset = max(len(existing_rows), ckpt_epoch_offset)
    select_by = getattr(args, "select_best_by", "auto")
    if select_by == "auto":
        select_by = "det_loss" if name == "student" else "loss"
    print(f"[BEST] {name} checkpoint selection uses val_{select_by}")

    best_val = ckpt_best_val
    for r in existing_rows:
        try:
            key = "val_det_loss" if select_by == "det_loss" else "val_loss"
            best_val = min(best_val, float(r.get(key, r.get("val_loss", "inf"))))
        except ValueError:
            pass
    if epoch_offset:
        print(f"[RESUME] {name} csv has {epoch_offset} completed epochs; appending from epoch {epoch_offset + 1}.")
    bad_epochs = 0

    for local_epoch in range(1, args.epochs + 1):
        epoch = epoch_offset + local_epoch
        t0 = time.time()
        update_audit_names = tuple(getattr(args, "assert_module_updates", ()) or ())
        update_snapshots = snapshot_named_trainable_modules(model, update_audit_names)
        model.train()
        if bool(getattr(args, "freeze_base_for_motion", False)):
            # Keep all frozen BatchNorm statistics and deterministic base outputs exact.
            model.eval()
            model.class_motion_head.train()
        if bool(getattr(args, "freeze_base_for_velocity_refiner", False)):
            # V14-A must not change heatmaps, yaw, size, attributes, or normalization
            # statistics. Only the new displacement refiner is optimized.
            model.eval()
            model.object_velocity_refiner.train()
        if bool(getattr(args, "freeze_base_for_velocity_calibrator", False)):
            # V15-B trains only the identity-initialized class/attribute velocity
            # calibrator. Detection, size, yaw, attributes, and BN statistics stay fixed.
            model.eval()
            model.class_attribute_velocity_calibrator.train()
        if bool(getattr(args, "freeze_base_for_yaw_refiner", False)):
            # V14-B changes only the residual MultiBin head. This protects S1 heatmaps,
            # velocity, size, attributes, and all normalization statistics.
            model.eval()
            if model.residual_azimuth_multibin_head is not None:
                model.residual_azimuth_multibin_head.train()
            if model.residual_yaw_vector_head is not None:
                model.residual_yaw_vector_head.train()
        if bool(getattr(args, "freeze_base_for_temporal_stereo", False)):
            # V19-T0 learns only the identity-initialized stereo evidence gate.
            # The S1 detector and every normalization buffer remain fixed.
            model.eval()
            model.temporal_stereo_depth_matcher.train()
        if bool(getattr(args, "freeze_base_for_center_refiner", False)):
            model.eval()
            model.class_center_head.train()
        if bool(getattr(args, "freeze_base_for_depth_distillation", False)):
            model.eval()
            model.depth_head.train()
        if bool(getattr(args, "freeze_base_for_image_yaw_flip", False)):
            model.eval()
            model.image_yaw_flip_head.train()
        if bool(getattr(args, "freeze_base_for_image_radial_center", False)):
            model.eval()
            model.image_radial_center_head.train()
        if bool(getattr(args, "freeze_base_for_image_object_geometry", False)):
            model.eval()
            model.image_object_geometry_head.train()
        if bool(getattr(args, "freeze_base_for_privileged_teacher", False)):
            # Preserve every proven detector parameter and normalization buffer.
            # Only the two custom, identity-initialized privileged modules learn.
            model.eval()
            model.privileged_depth_completion.train()
            model.privileged_bev_refiner.train()
        if bool(getattr(args, "freeze_base_for_privileged_memory_adapter", False)):
            model.eval()
            model.privileged_residual_adapter.train()
        optimizer.zero_grad(set_to_none=True)
        sums: Dict[str, float] = {}
        n = 0
        pending_grad = 0
        nonfinite_batches = 0
        kd_scale = kd_scale_for_epoch(args, epoch) if name == "student" else 1.0
        kd_balance_target = (
            kd_balance_target_for_epoch(args, epoch) if name == "student" else 0.0
        )
        if name == "student":
            args._current_kd_balance_target = kd_balance_target
        # Samplers own their RNG epoch and increment it from __iter__.  A fresh
        # process otherwise resumes at sampler epoch zero, silently repeating
        # the camera/order sequence from epoch one.  Pin it to the absolute
        # training epoch before every DataLoader iteration.
        if hasattr(train_loader.sampler, "epoch"):
            train_loader.sampler.epoch = epoch - 1
            print(
                f"[SAMPLER_EPOCH] {name} training_epoch={epoch} "
                f"sampler_epoch={train_loader.sampler.epoch}"
            )
        bar = tqdm(train_loader, desc=f"Train {name} ep{epoch:02d}", dynamic_ncols=True, file=sys.stdout)
        for it, batch in enumerate(bar, 1):
            with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
                if name == "teacher":
                    loss, parts = teacher_forward_loss(model, batch, args, device)
                else:
                    loss, parts = student_forward_loss(
                        model, teacher, batch, args, device, kd_scale=kd_scale,
                        camera_teacher=camera_teacher,
                    )
            if not torch.isfinite(loss):
                optimizer.zero_grad(set_to_none=True)
                pending_grad = 0
                nonfinite_batches += 1
                diagnostics = {"loss": float(loss.detach().float().item())}
                diagnostics.update({
                    key: float(value.detach().float().item())
                    for key, value in parts.items()
                    if torch.is_tensor(value) and value.numel() == 1
                })
                if nonfinite_batches <= 5 or nonfinite_batches in {10, 20, 30}:
                    print(
                        f"[NONFINITE_TRAIN] name={name} epoch={epoch} it={it} "
                        f"count={nonfinite_batches} diagnostics={diagnostics}",
                        flush=True,
                    )
                    if activation_stats:
                        print(
                            f"[NONFINITE_ACTIVATIONS] name={name} epoch={epoch} "
                            f"it={it} stats={activation_stats}",
                            flush=True,
                        )
                limit = int(getattr(args, "max_nonfinite_batches", 0))
                if nonfinite_batches > limit:
                    raise FloatingPointError(
                        f"{name} epoch {epoch} exceeded non-finite batch limit "
                        f"{limit} at iteration {it}; diagnostics={diagnostics}"
                    )
                continue
            scaler.scale(loss / args.grad_accum).backward()
            pending_grad += 1
            if it % args.grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                pending_grad = 0
            sums["loss"] = sums.get("loss", 0.0) + float(loss.item())
            for k, v in parts.items():
                sums[k] = sums.get(k, 0.0) + float(v.item())
            n += 1
            bar.set_postfix(loss=sums["loss"] / max(1, n), hm=sums.get("hm", 0.0) / max(1, n), reg=sums.get("reg", 0.0) / max(1, n))
            if getattr(args, "save_every_iters", 0) > 0 and it % args.save_every_iters == 0:
                save_training_checkpoint(
                    out_dir / f"{name}_iter_last.pth",
                    model, optimizer, scheduler, scaler, epoch, best_val, select_by, args, name,
                )

        if pending_grad > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        if update_snapshots:
            update_deltas = assert_named_modules_updated(
                model, update_snapshots, context=f"{name} epoch {epoch}",
            )
            print(
                "[MODULE_UPDATE_AUDIT] " + " ".join(
                    f"{module_name}={delta:.9g}"
                    for module_name, delta in update_deltas.items()
                ),
                flush=True,
            )

        if nonfinite_batches:
            print(
                f"[NONFINITE_TRAIN_SUMMARY] name={name} epoch={epoch} "
                f"count={nonfinite_batches} finite_batches={n}",
                flush=True,
            )

        repaired_bn = sanitize_batchnorm_running_stats(model)
        if repaired_bn:
            print(f"[BN_SANITIZE] repaired {repaired_bn} non-finite running-stat values before validation")
        assert_finite_model_parameters(model, context=f"{name} epoch {epoch}")
        val = validate(
            args, model, val_loader, teacher, name, device, kd_scale=1.0,
            camera_teacher=camera_teacher,
        )
        scheduler.step()
        sec = time.time() - t0
        train_avg = {k: v / max(1, n) for k, v in sums.items()}
        val_det_loss = val.get(
            "det_loss",
            args.hm_weight * val.get("hm", 0.0) + args.reg_weight * val.get("reg", 0.0) +
            args.depth_weight * val.get("depth", 0.0) +
            float(getattr(args, "attr_weight", 0.0)) * val.get("attr", 0.0),
        )
        row = {
            "epoch": epoch, "lr": optimizer.param_groups[0]["lr"], "sec": sec,
            "kd_scale": kd_scale, "kd_balance_target": kd_balance_target,
            "train_loss": train_avg.get("loss", 0.0), "train_det_loss": train_avg.get("det_loss", 0.0),
            "train_hm": train_avg.get("hm", 0.0),
            "train_reg": train_avg.get("reg", 0.0), "train_depth": train_avg.get("depth", 0.0),
            "train_attr": train_avg.get("attr", 0.0),
            "train_center_error": train_avg.get("center_error", 0.0),
            "val_loss": val.get("loss", 0.0), "val_det_loss": val_det_loss, "val_hm": val.get("hm", 0.0),
            "val_reg": val.get("reg", 0.0), "val_depth": val.get("depth", 0.0),
            "val_attr": val.get("attr", 0.0),
            "val_center_error": val.get("center_error", 0.0),
            "train_kd_balance": train_avg.get("kd_balance", 1.0),
            "val_kd_balance": val.get("kd_balance", 1.0),
            "train_kd_loss": train_avg.get("kd_loss", 0.0), "val_kd_loss": val.get("kd_loss", 0.0),
            "train_kd_hm": train_avg.get("kd_hm", 0.0), "train_kd_reg": train_avg.get("kd_reg", 0.0),
            "train_kd_feat": train_avg.get("kd_feat", 0.0), "train_kd_depth": train_avg.get("kd_depth", 0.0),
            "val_kd_hm": val.get("kd_hm", 0.0), "val_kd_reg": val.get("kd_reg", 0.0),
            "val_kd_feat": val.get("kd_feat", 0.0), "val_kd_depth": val.get("kd_depth", 0.0),
            "train_kd_attr": train_avg.get("kd_attr", 0.0),
            "val_kd_attr": val.get("kd_attr", 0.0),
            "train_kd_relation": train_avg.get("kd_relation", 0.0),
            "val_kd_relation": val.get("kd_relation", 0.0),
            "train_kd_keypoint_feature": train_avg.get("kd_keypoint_feature", 0.0),
            "val_kd_keypoint_feature": val.get("kd_keypoint_feature", 0.0),
            "train_kd_project_hm_accept": train_avg.get("kd_project_hm_accept", 0.0),
            "val_kd_project_hm_accept": val.get("kd_project_hm_accept", 0.0),
            "train_kd_project_reg_accept": train_avg.get("kd_project_reg_accept", 0.0),
            "val_kd_project_reg_accept": val.get("kd_project_reg_accept", 0.0),
            "train_kd_project_depth_accept": train_avg.get("kd_project_depth_accept", 0.0),
            "val_kd_project_depth_accept": val.get("kd_project_depth_accept", 0.0),
            "train_kd_project_attr_accept": train_avg.get("kd_project_attr_accept", 0.0),
            "val_kd_project_attr_accept": val.get("kd_project_attr_accept", 0.0),
            "train_kd_grad_scale_camera_hm": train_avg.get("kd_grad_scale_camera_hm", 0.0),
            "val_kd_grad_scale_camera_hm": val.get("kd_grad_scale_camera_hm", 0.0),
            "train_kd_grad_scale_camera_reg": train_avg.get("kd_grad_scale_camera_reg", 0.0),
            "val_kd_grad_scale_camera_reg": val.get("kd_grad_scale_camera_reg", 0.0),
            "train_kd_grad_scale_camera_feature": train_avg.get("kd_grad_scale_camera_feature", 0.0),
            "val_kd_grad_scale_camera_feature": val.get("kd_grad_scale_camera_feature", 0.0),
            "train_kd_grad_scale_camera_depth": train_avg.get("kd_grad_scale_camera_depth", 0.0),
            "val_kd_grad_scale_camera_depth": val.get("kd_grad_scale_camera_depth", 0.0),
            "train_kd_grad_scale_camera_attr": train_avg.get("kd_grad_scale_camera_attr", 0.0),
            "val_kd_grad_scale_camera_attr": val.get("kd_grad_scale_camera_attr", 0.0),
            **{
                f"train_natural_anchor_{branch}_{diagnostic}": train_avg.get(
                    f"natural_anchor_{branch}_{diagnostic}", 0.0,
                )
                for branch in ("camera_feature", "camera_depth")
                for diagnostic in ("cosine", "mema_scale", "projected", "gradient_ratio")
            },
            **{
                f"val_natural_anchor_{branch}_{diagnostic}": val.get(
                    f"natural_anchor_{branch}_{diagnostic}", 0.0,
                )
                for branch in ("camera_feature", "camera_depth")
                for diagnostic in ("cosine", "mema_scale", "projected", "gradient_ratio")
            },
            "train_relative_depth": train_avg.get("relative_depth", 0.0),
            "val_relative_depth": val.get("relative_depth", 0.0),
            "train_pseudo_hm": train_avg.get("pseudo_hm", 0.0),
            "train_pseudo_reg": train_avg.get("pseudo_reg", 0.0),
            "train_pseudo_yaw_refiner": train_avg.get("pseudo_yaw_refiner", 0.0),
            "val_pseudo_hm": val.get("pseudo_hm", 0.0),
            "val_pseudo_reg": val.get("pseudo_reg", 0.0),
            "val_pseudo_yaw_refiner": val.get("pseudo_yaw_refiner", 0.0),
            "train_history_loss": train_avg.get("history_loss", 0.0),
            "train_history_hm": train_avg.get("history_hm", 0.0),
            "train_history_reg": train_avg.get("history_reg", 0.0),
            "val_history_loss": val.get("history_loss", 0.0),
            "val_history_hm": val.get("history_hm", 0.0),
            "val_history_reg": val.get("history_reg", 0.0),
            "train_yaw_metric": train_avg.get("yaw_metric", 0.0),
            "train_velocity_metric": train_avg.get("velocity_metric", 0.0),
            "val_yaw_metric": val.get("yaw_metric", 0.0),
            "val_velocity_metric": val.get("velocity_metric", 0.0),
            "train_azimuth_bin": train_avg.get("azimuth_bin", 0.0),
            "val_azimuth_bin": val.get("azimuth_bin", 0.0),
            "train_size_prior": train_avg.get("size_prior", 0.0),
            "val_size_prior": val.get("size_prior", 0.0),
            "train_ase_iou": train_avg.get("ase_iou", 0.0),
            "val_ase_iou": val.get("ase_iou", 0.0),
            "train_velocity_decomposed": train_avg.get("velocity_decomposed", 0.0),
            "val_velocity_decomposed": val.get("velocity_decomposed", 0.0),
            "train_yaw_velocity_consistency": train_avg.get("yaw_velocity_consistency", 0.0),
            "val_yaw_velocity_consistency": val.get("yaw_velocity_consistency", 0.0),
            "train_image_yaw_flip": train_avg.get("image_yaw_flip", 0.0),
            "val_image_yaw_flip": val.get("image_yaw_flip", 0.0),
            "train_image_radial_center": train_avg.get("image_radial_center", 0.0),
            "val_image_radial_center": val.get("image_radial_center", 0.0),
        }
        metric = val_det_loss if select_by == "det_loss" else val.get("loss", 0.0)
        row["best_metric"] = metric
        csv_append(csv_path, row, fields)

        improved = metric < best_val
        if improved:
            best_val = metric
            bad_epochs = 0
            save_training_checkpoint(out_dir / f"{name}_best.pth", model, optimizer, scheduler, scaler, epoch, best_val, select_by, args, name)
        else:
            bad_epochs += 1
        last_path = out_dir / f"{name}_last.pth"
        save_training_checkpoint(last_path, model, optimizer, scheduler, scaler, epoch, best_val, select_by, args, name)
        if getattr(args, "save_every_epoch", False):
            save_training_checkpoint(
                out_dir / f"{name}_epoch{epoch:02d}.pth",
                model, optimizer, scheduler, scaler, epoch, best_val, select_by, args, name,
            )
        print(f"[ep{epoch:02d}] train_loss={row['train_loss']:.4f} val_loss={row['val_loss']:.4f} best_val={best_val:.4f}")
        patience = int(getattr(args, "early_stopping_patience", 0))
        min_epochs = int(getattr(args, "min_epochs", 1))
        if patience > 0 and local_epoch >= min_epochs and bad_epochs >= patience:
            print(
                f"[EARLY_STOP] no val_{select_by} improvement for {bad_epochs} epochs "
                f"after local_epoch={local_epoch}"
            )
            break

    for hook in activation_hooks:
        hook.remove()


@torch.no_grad()
def validate(
    args, model, loader, teacher, name, device, kd_scale: float = 1.0,
    camera_teacher=None,
):
    model.eval()
    sums: Dict[str, float] = {}
    n = 0
    nonfinite_batches = 0
    for it, batch in enumerate(
        tqdm(loader, desc=f"Val {name}", dynamic_ncols=True, file=sys.stdout), 1,
    ):
        with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
            if name == "teacher":
                loss, parts = teacher_forward_loss(model, batch, args, device)
            else:
                loss, parts = student_forward_loss(
                    model, teacher, batch, args, device, kd_scale=kd_scale,
                    camera_teacher=camera_teacher,
                )
        if not torch.isfinite(loss):
            nonfinite_batches += 1
            diagnostics = {"loss": float(loss.detach().float().item())}
            diagnostics.update({
                key: float(value.detach().float().item())
                for key, value in parts.items()
                if torch.is_tensor(value) and value.numel() == 1
            })
            print(
                f"[NONFINITE_VAL] name={name} it={it} count={nonfinite_batches} "
                f"diagnostics={diagnostics}",
                flush=True,
            )
            limit = int(getattr(args, "max_nonfinite_batches", 0))
            if nonfinite_batches > limit:
                raise FloatingPointError(
                    f"validation for {name} exceeded non-finite batch limit "
                    f"{limit} at iteration {it}; diagnostics={diagnostics}"
                )
            continue
        sums["loss"] = sums.get("loss", 0.0) + float(loss.item())
        for k, v in parts.items():
            sums[k] = sums.get(k, 0.0) + float(v.item())
        n += 1
    if n == 0:
        raise FloatingPointError(f"validation for {name} produced no finite batches")
    return {k: v / max(1, n) for k, v in sums.items()}


def peak_nms(hm: torch.Tensor, kernel=3):
    pad = (kernel - 1) // 2
    hmax = F.max_pool2d(hm, kernel, stride=1, padding=pad)
    return hm * (hmax == hm).float()


def cv_rotated_box(box):
    cx, cy, z, w, l, h, yaw, vx, vy, score, cls_id = box[:11]
    return ((float(cx), float(cy)), (float(max(w, 1e-3)), float(max(l, 1e-3))), float(yaw * 180.0 / math.pi))


def rotated_iou(box1, box2):
    r1, r2 = cv_rotated_box(box1), cv_rotated_box(box2)
    area1 = max(float(box1[3] * box1[4]), 1e-6)
    area2 = max(float(box2[3] * box2[4]), 1e-6)
    try:
        _typ, pts = cv2.rotatedRectangleIntersection(r1, r2)
    except Exception:
        return 0.0
    if pts is None:
        return 0.0
    inter = abs(cv2.contourArea(cv2.convexHull(pts, returnPoints=True)))
    return float(inter / max(area1 + area2 - inter, 1e-6))


def nms_classwise_reference(boxes, iou_thr):
    kept = []
    for cls_id in range(len(DETECTION_NAMES)):
        cls_boxes = [b for b in boxes if int(b[10]) == cls_id]
        cls_boxes = sorted(cls_boxes, key=lambda x: x[9], reverse=True)
        while cls_boxes:
            b = cls_boxes.pop(0)
            kept.append(b)
            cls_boxes = [c for c in cls_boxes if rotated_iou(b, c) < iou_thr]
    return kept


def _rotated_box_aabb(box):
    points = cv2.boxPoints(cv_rotated_box(box))
    return (
        float(points[:, 0].min()), float(points[:, 1].min()),
        float(points[:, 0].max()), float(points[:, 1].max()),
    )


def _aabb_grid_cells(aabb, cell_size=5.0):
    min_x, min_y, max_x, max_y = aabb
    gx0, gy0 = math.floor(min_x / cell_size), math.floor(min_y / cell_size)
    gx1, gy1 = math.floor(max_x / cell_size), math.floor(max_y / cell_size)
    for gx in range(gx0, gx1 + 1):
        for gy in range(gy0, gy1 + 1):
            yield gx, gy


def nms_classwise(boxes, iou_thr):
    """Exact greedy rotated NMS with a spatial index for candidate pruning."""
    kept = []
    for cls_id in range(len(DETECTION_NAMES)):
        cls_boxes = sorted(
            (
                box for box in boxes
                if int(box[10]) == cls_id and all(math.isfinite(float(value)) for value in box[:10])
            ),
            key=lambda box: box[9], reverse=True,
        )
        class_kept = []
        grid = {}
        for box in cls_boxes:
            cells = tuple(_aabb_grid_cells(_rotated_box_aabb(box)))
            possible = set()
            for cell in cells:
                possible.update(grid.get(cell, ()))
            if any(rotated_iou(class_kept[index], box) >= iou_thr for index in sorted(possible)):
                continue
            kept_index = len(class_kept)
            class_kept.append(box)
            for cell in cells:
                grid.setdefault(cell, []).append(kept_index)
        kept.extend(class_kept)
    return kept


def _circular_weighted_mean(angles, weights, reference):
    """Average nearby yaw hypotheses without crossing the +/-pi branch cut."""
    aligned = [float(reference) + wrap_pi(float(angle) - float(reference)) for angle in angles]
    sin_sum = sum(float(weight) * math.sin(angle) for angle, weight in zip(aligned, weights))
    cos_sum = sum(float(weight) * math.cos(angle) for angle, weight in zip(aligned, weights))
    if abs(sin_sum) + abs(cos_sum) <= 1e-9:
        return float(reference)
    return wrap_pi(math.atan2(sin_sum, cos_sum))


def weighted_box_fusion_classwise(
    boxes, iou_thr, score_power=2.0, geometry="center", yaw_gate=math.pi / 3.0,
    score_mode="max",
):
    """Greedily fuse corroborating cross-view boxes while preserving AP scores.

    Per-view NMS has already run before this function.  The optional camera id at
    tuple index 12 lets one contribution per view vote for an object.  The maximum
    score, class and learned attribute stay unchanged; only geometric estimates
    supported by at least two views are averaged.
    """
    if geometry not in {"center", "size", "yaw", "full"}:
        raise ValueError(f"unsupported weighted fusion geometry={geometry}")
    if score_mode not in {"max", "noisy_or"}:
        raise ValueError(f"unsupported weighted fusion score_mode={score_mode}")
    fused_boxes = []
    for cls_id in range(len(DETECTION_NAMES)):
        remaining = sorted(
            (
                tuple(box) for box in boxes
                if int(box[10]) == cls_id and
                all(math.isfinite(float(value)) for value in box[:10])
            ),
            key=lambda box: box[9], reverse=True,
        )
        while remaining:
            seed = remaining[0]
            cluster = [box for box in remaining if rotated_iou(seed, box) >= iou_thr]
            # Tuples do not preserve identity after some callers reconstruct them;
            # remove by value/count using explicit membership instead.
            pending = list(cluster)
            survivors = []
            for box in remaining:
                try:
                    index = pending.index(box)
                except ValueError:
                    survivors.append(box)
                else:
                    pending.pop(index)
            remaining = survivors

            # Keep only the strongest proposal from each camera.  Older callers
            # without a camera id treat every proposal as an independent view.
            by_view = {}
            for index, box in enumerate(cluster):
                view_id = int(box[12]) if len(box) > 12 else index
                if view_id not in by_view or box[9] > by_view[view_id][9]:
                    by_view[view_id] = box
            voters = list(by_view.values())
            if len(voters) < 2:
                fused_boxes.append(seed)
                continue

            weights = [max(float(box[9]), 1e-4) ** float(score_power) for box in voters]
            denominator = max(sum(weights), 1e-9)
            values = list(seed)
            for channel in (0, 1, 2):
                values[channel] = sum(
                    weight * float(box[channel]) for weight, box in zip(weights, voters)
                ) / denominator
            if geometry in {"size", "full"}:
                for channel in (3, 4, 5):
                    values[channel] = math.exp(sum(
                        weight * math.log(max(float(box[channel]), 1e-4))
                        for weight, box in zip(weights, voters)
                    ) / denominator)
            if geometry in {"yaw", "full"}:
                yaw_voters = [
                    (box, weight) for box, weight in zip(voters, weights)
                    if abs(wrap_pi(float(box[6]) - float(seed[6]))) <= float(yaw_gate)
                ]
                if len(yaw_voters) >= 2:
                    values[6] = _circular_weighted_mean(
                        [box[6] for box, _weight in yaw_voters],
                        [weight for _box, weight in yaw_voters],
                        seed[6],
                    )
            # Velocity is intentionally not fused.  V17's causal class-aware
            # calibration is better validated than single-frame cross-view averaging.
            if score_mode == "noisy_or":
                values[9] = 1.0 - math.prod(
                    1.0 - min(max(float(box[9]), 0.0), 1.0) for box in voters
                )
            else:
                values[9] = float(seed[9])
            fused_boxes.append(tuple(values))
    return fused_boxes


def image_yaw_flip_probability(
    logits: Optional[torch.Tensor], batch_index: int, class_index: int,
    center_ego: Sequence[float], k: Optional[torch.Tensor],
    r_cam_to_ego: Optional[torch.Tensor], t_cam_to_ego: Optional[torch.Tensor], mod,
) -> Optional[float]:
    if logits is None or k is None or r_cam_to_ego is None or t_cam_to_ego is None:
        return None
    point = torch.tensor(center_ego, dtype=torch.float32)
    camera = r_cam_to_ego[batch_index].float().transpose(0, 1).matmul(
        point - t_cam_to_ego[batch_index].float()
    )
    depth = float(camera[2])
    if depth <= 1e-3:
        return None
    u = float(k[batch_index, 0, 0]) * float(camera[0]) / depth + float(k[batch_index, 0, 2])
    v = float(k[batch_index, 1, 1]) * float(camera[1]) / depth + float(k[batch_index, 1, 2])
    feat_h, feat_w = logits.shape[-2:]
    feat_x = int(math.floor(u * feat_w / float(mod.IMG_W)))
    feat_y = int(math.floor(v * feat_h / float(mod.IMG_H)))
    if not (0 <= feat_x < feat_w and 0 <= feat_y < feat_h):
        return None
    return float(torch.sigmoid(logits[batch_index, class_index, feat_y, feat_x].float()))


def image_radial_center_residual(
    residuals: Optional[torch.Tensor], batch_index: int, class_index: int,
    center_ego: Sequence[float], k: Optional[torch.Tensor],
    r_cam_to_ego: Optional[torch.Tensor], t_cam_to_ego: Optional[torch.Tensor], mod,
) -> Optional[float]:
    if residuals is None or k is None or r_cam_to_ego is None or t_cam_to_ego is None:
        return None
    point = torch.tensor(center_ego, dtype=torch.float32)
    camera = r_cam_to_ego[batch_index].float().transpose(0, 1).matmul(
        point - t_cam_to_ego[batch_index].float()
    )
    depth = float(camera[2])
    if depth <= 1e-3:
        return None
    u = float(k[batch_index, 0, 0]) * float(camera[0]) / depth + float(k[batch_index, 0, 2])
    v = float(k[batch_index, 1, 1]) * float(camera[1]) / depth + float(k[batch_index, 1, 2])
    feat_h, feat_w = residuals.shape[-2:]
    feat_x = int(math.floor(u * feat_w / float(mod.IMG_W)))
    feat_y = int(math.floor(v * feat_h / float(mod.IMG_H)))
    if not (0 <= feat_x < feat_w and 0 <= feat_y < feat_h):
        return None
    return float(residuals[batch_index, class_index, feat_y, feat_x])


def depth_distribution_refined_center(
    center_ego: Sequence[float], depth_logits: Optional[torch.Tensor],
    batch_index: int, k: Optional[torch.Tensor],
    r_cam_to_ego: Optional[torch.Tensor], t_cam_to_ego: Optional[torch.Tensor], mod,
    alpha: float = 0.0, statistic: str = "mean", window_radius: int = 1,
    max_shift: float = 2.0,
) -> Tuple[float, float]:
    """Blend a detected centre toward the image depth distribution on its camera ray.

    The detector already computes per-pixel depth logits for lift/splat.  Reusing
    those logits adds no parameters or image encoder work.  A small image-space
    window is averaged to reduce one-pixel noise; the resulting camera-depth
    proposal is converted back to ego coordinates and clipped before blending.
    """
    cx, cy, cz = (float(value) for value in center_ego)
    if (
        depth_logits is None or k is None or r_cam_to_ego is None or
        t_cam_to_ego is None or float(alpha) <= 0.0
    ):
        return cx, cy
    if statistic not in {"mean", "mode"}:
        raise ValueError(f"unsupported depth statistic: {statistic}")
    point = torch.tensor((cx, cy, cz), dtype=torch.float32)
    rotation = r_cam_to_ego[batch_index].float()
    translation = t_cam_to_ego[batch_index].float()
    camera = rotation.transpose(0, 1).matmul(point - translation)
    current_depth = float(camera[2])
    if current_depth <= 1e-3:
        return cx, cy
    u = (
        float(k[batch_index, 0, 0]) * float(camera[0]) / current_depth +
        float(k[batch_index, 0, 2])
    )
    v = (
        float(k[batch_index, 1, 1]) * float(camera[1]) / current_depth +
        float(k[batch_index, 1, 2])
    )
    _batch, depth_bins, feat_h, feat_w = depth_logits.shape
    feat_x = int(math.floor(u * feat_w / float(mod.IMG_W)))
    feat_y = int(math.floor(v * feat_h / float(mod.IMG_H)))
    if not (0 <= feat_x < feat_w and 0 <= feat_y < feat_h):
        return cx, cy
    radius = max(0, int(window_radius))
    x0, x1 = max(0, feat_x - radius), min(feat_w, feat_x + radius + 1)
    y0, y1 = max(0, feat_y - radius), min(feat_h, feat_y + radius + 1)
    patch = depth_logits[batch_index, :, y0:y1, x0:x1].float()
    probabilities = torch.softmax(patch, dim=0).mean(dim=(1, 2))
    edges = torch.linspace(
        float(mod.DEPTH_MIN), float(mod.DEPTH_MAX),
        steps=int(depth_bins) + 1, dtype=torch.float32,
    )
    centres = 0.5 * (edges[:-1] + edges[1:])
    if statistic == "mode":
        proposed_depth = float(centres[int(probabilities.argmax().item())])
    else:
        proposed_depth = float((probabilities * centres).sum())
    if not math.isfinite(proposed_depth) or proposed_depth <= 1e-3:
        return cx, cy
    proposed_camera = camera * (proposed_depth / current_depth)
    proposed_ego = rotation.matmul(proposed_camera) + translation
    delta_x = float(proposed_ego[0]) - cx
    delta_y = float(proposed_ego[1]) - cy
    delta_norm = math.hypot(delta_x, delta_y)
    limit = max(0.0, float(max_shift))
    if limit > 0.0 and delta_norm > limit:
        scale = limit / max(delta_norm, 1e-6)
        delta_x *= scale
        delta_y *= scale
    return cx + float(alpha) * delta_x, cy + float(alpha) * delta_y


def decode_official_boxes(hm_logits, reg, mod, score_thr, topk, nms_iou, attr_logits=None,
                          velocity_intervals=None, class_score_thrs=None, class_motion=None,
                          azimuth_bins=None, class_size=None, class_center=None,
                          image_yaw_flip_logits=None, k=None, r_cam_to_ego=None,
                          t_cam_to_ego=None, image_yaw_flip_threshold=0.5,
                          image_radial_center=None,
                          depth_logits=None, depth_distribution_alpha=0.0,
                          depth_distribution_statistic="mean",
                          depth_distribution_window_radius=1,
                          depth_distribution_max_shift=2.0,
                          attribute_velocity_gate: bool = False):
    if class_score_thrs is not None and len(class_score_thrs) != len(DETECTION_NAMES):
        raise ValueError(
            f"class_score_thrs has {len(class_score_thrs)} values, expected {len(DETECTION_NAMES)}"
        )
    scores = peak_nms(torch.sigmoid(hm_logits.float()))
    b, c, h, w = scores.shape
    decoded = []
    for bi in range(b):
        flat = scores[bi].reshape(-1)
        vals, inds = torch.topk(flat, k=min(topk, flat.numel()))
        boxes = []
        for score, ind in zip(vals.tolist(), inds.tolist()):
            cls_id = int(ind // (h * w))
            threshold = float(class_score_thrs[cls_id]) if class_score_thrs is not None else score_thr
            if not math.isfinite(float(score)) or score < threshold:
                continue
            rem = int(ind % (h * w))
            ix, iy = int(rem // w), int(rem % w)
            values = [float(reg[bi, j, ix, iy].item()) for j in range(10)]
            if class_motion is not None:
                motion_residual = class_motion[bi, cls_id, :, ix, iy]
                for channel, residual in zip(range(6, 10), motion_residual.tolist()):
                    values[channel] += float(residual)
            if class_size is not None:
                size_residual = class_size[bi, cls_id, :, ix, iy]
                for channel, residual in zip(range(3, 6), size_residual.tolist()):
                    values[channel] += float(residual)
            if class_center is not None:
                center_residual = class_center[bi, cls_id, :, ix, iy]
                for channel, residual in zip(range(0, 3), center_residual.tolist()):
                    values[channel] += float(residual)
            if not all(math.isfinite(value) for value in values):
                continue
            dx, dy, z = values[:3]
            size_logs = values[3:6]
            if any(abs(value) > 20.0 for value in size_logs):
                continue
            bw, bl, bh = [math.exp(value) for value in size_logs]
            if min(bw, bl, bh) < 0.05 or max(bw, bl, bh) > 50.0:
                continue
            cx = mod.BEV_X_MIN + (ix + dx) * mod.BEV_RES
            cy = mod.BEV_Y_MIN + (iy + dy) * mod.BEV_RES
            radial_residual = image_radial_center_residual(
                image_radial_center, bi, cls_id, (cx, cy, z),
                k, r_cam_to_ego, t_cam_to_ego, mod,
            )
            if radial_residual is not None:
                camera_x = float(t_cam_to_ego[bi, 0])
                camera_y = float(t_cam_to_ego[bi, 1])
                ray_x, ray_y = cx - camera_x, cy - camera_y
                ray_norm = math.hypot(ray_x, ray_y)
                if ray_norm > 1e-3:
                    cx += radial_residual * ray_x / ray_norm
                    cy += radial_residual * ray_y / ray_norm
            cx, cy = depth_distribution_refined_center(
                (cx, cy, z), depth_logits, bi, k, r_cam_to_ego,
                t_cam_to_ego, mod,
                alpha=float(depth_distribution_alpha),
                statistic=str(depth_distribution_statistic),
                window_radius=int(depth_distribution_window_radius),
                max_shift=float(depth_distribution_max_shift),
            )
            yaw_residual = math.atan2(values[6], values[7])
            if azimuth_bins is not None:
                bin_logits = azimuth_bins[bi, cls_id, :, ix, iy]
                bin_index = int(bin_logits.argmax().item())
                yaw = wrap_pi(yaw_residual + bin_index * math.pi + math.atan2(cy, cx))
            else:
                yaw = yaw_residual
            flip_probability = image_yaw_flip_probability(
                image_yaw_flip_logits, bi, cls_id, (cx, cy, z),
                k, r_cam_to_ego, t_cam_to_ego, mod,
            )
            if (
                flip_probability is not None and
                flip_probability > float(image_yaw_flip_threshold)
            ):
                yaw = wrap_pi(yaw + math.pi)
            vx, vy = values[8:10]
            if velocity_intervals is not None:
                interval = max(1e-3, float(velocity_intervals[bi]))
                vx, vy = vx / interval, vy / interval
            attr_id = -1
            if attr_logits is not None:
                det_name = DETECTION_NAMES[cls_id]
                valid_ids = [ATTRIBUTE_TO_ID[name] for name in VALID_ATTRIBUTES_BY_DETECTION[det_name]]
                if valid_ids:
                    attr_values = attr_logits[bi, valid_ids, ix, iy].float()
                    attr_id = valid_ids[int(attr_values.argmax().item())]
                    if attribute_velocity_gate:
                        probabilities = torch.softmax(attr_values, dim=0)
                        moving_names = set(MOVING_ATTRIBUTES_BY_DETECTION[det_name])
                        moving_probability = sum(
                            float(probabilities[index].item())
                            for index, attr_index in enumerate(valid_ids)
                            if ATTRIBUTE_NAMES[attr_index] in moving_names
                        )
                        vx *= moving_probability
                        vy *= moving_probability
            boxes.append((cx, cy, z, bw, bl, bh, yaw, vx, vy, float(score), cls_id, attr_id))
        decoded.append(nms_classwise(boxes, nms_iou))
    return decoded


def attribute_for(det_name: str, vx: float, vy: float):
    speed = math.sqrt(vx * vx + vy * vy)
    default_attribute = {
        "car": "vehicle.parked",
        "pedestrian": "pedestrian.moving",
        "trailer": "vehicle.parked",
        "truck": "vehicle.parked",
        "bus": "vehicle.moving",
        "motorcycle": "cycle.without_rider",
        "construction_vehicle": "vehicle.parked",
        "bicycle": "cycle.without_rider",
        "barrier": "",
        "traffic_cone": "",
    }
    if speed > 0.2:
        if det_name in {"car", "construction_vehicle", "bus", "truck", "trailer"}:
            return "vehicle.moving"
        if det_name in {"bicycle", "motorcycle"}:
            return "cycle.with_rider"
        return default_attribute.get(det_name, "")
    if det_name == "pedestrian":
        return "pedestrian.standing"
    if det_name == "bus":
        return "vehicle.stopped"
    return default_attribute.get(det_name, "")


def transform_box_between_ego_frames(nusc, sample_token, box, source_cam: str, target_cam: str = "CAM_FRONT"):
    if source_cam == target_cam:
        return box
    sample = nusc.get("sample", sample_token)
    source_sd = nusc.get("sample_data", sample["data"][source_cam])
    target_sd = nusc.get("sample_data", sample["data"][target_cam])
    source_pose = nusc.get("ego_pose", source_sd["ego_pose_token"])
    target_pose = nusc.get("ego_pose", target_sd["ego_pose_token"])
    source_t = np.array(source_pose["translation"], dtype=np.float64)
    target_t = np.array(target_pose["translation"], dtype=np.float64)
    source_r = PyQuaternion(source_pose["rotation"]).rotation_matrix
    target_r = PyQuaternion(target_pose["rotation"]).rotation_matrix
    source_yaw = yaw_from_quat(source_pose["rotation"])
    target_yaw = yaw_from_quat(target_pose["rotation"])

    cx, cy, z, w, l, h, yaw, vx, vy, score, cls_id = box[:11]
    extra = tuple(box[11:])
    center_global = source_r.dot(np.array([cx, cy, z], dtype=np.float64)) + source_t
    center_target = target_r.T.dot(center_global - target_t)
    velocity_global = source_r.dot(np.array([vx, vy, 0.0], dtype=np.float64))
    velocity_target = target_r.T.dot(velocity_global)
    yaw_target = wrap_pi(source_yaw + yaw - target_yaw)
    return (
        float(center_target[0]), float(center_target[1]), float(center_target[2]),
        w, l, h, yaw_target, float(velocity_target[0]), float(velocity_target[1]), score, cls_id,
        *extra,
    )


def ego_box_to_nusc_result(nusc, sample_token, box, reference_cam: str = "CAM_FRONT"):
    sample = nusc.get("sample", sample_token)
    sd = nusc.get("sample_data", sample["data"][reference_cam])
    ego_pose = nusc.get("ego_pose", sd["ego_pose_token"])
    ego_t = np.array(ego_pose["translation"], dtype=np.float64)
    ego_R = PyQuaternion(ego_pose["rotation"]).rotation_matrix
    ego_yaw = yaw_from_quat(ego_pose["rotation"])

    cx, cy, z, w, l, h, yaw, vx, vy, score, cls_id = box[:11]
    pos_global = ego_R.dot(np.array([cx, cy, z], dtype=np.float64)) + ego_t
    vel_global = ego_R.dot(np.array([vx, vy, 0.0], dtype=np.float64))
    rot_global = PyQuaternion(axis=[0, 0, 1], angle=ego_yaw + yaw)
    det_name = DETECTION_NAMES[int(cls_id)]
    attr_id = int(box[11]) if len(box) > 11 else -1
    attribute_name = (
        ATTRIBUTE_NAMES[attr_id] if 0 <= attr_id < len(ATTRIBUTE_NAMES)
        else attribute_for(det_name, float(vel_global[0]), float(vel_global[1]))
    )
    return {
        "sample_token": sample_token,
        "translation": [float(pos_global[0]), float(pos_global[1]), float(pos_global[2])],
        "size": [float(w), float(l), float(h)],
        "rotation": [float(x) for x in rot_global.elements],
        "velocity": [float(vel_global[0]), float(vel_global[1])],
        "detection_name": det_name,
        "detection_score": float(score),
        "attribute_name": attribute_name,
    }


def all_split_sample_tokens(nusc, mod, split: str) -> List[str]:
    split_scenes = set(mod.create_splits_scenes()[split])
    tokens: List[str] = []
    for scene in nusc.scene:
        if scene["name"] not in split_scenes:
            continue
        token = scene["first_sample_token"]
        while token:
            sample = nusc.get("sample", token)
            tokens.append(sample["token"])
            token = sample["next"]
    return tokens


def global_eval_box_visible_in_camera(nusc, sample_token: str, eval_box, cam_name: str) -> bool:
    sample = nusc.get("sample", sample_token)
    sd = nusc.get("sample_data", sample["data"][cam_name])
    cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
    pose = nusc.get("ego_pose", sd["ego_pose_token"])
    box = Box(
        center=eval_box.translation,
        size=eval_box.size,
        orientation=PyQuaternion(eval_box.rotation),
    )
    box.translate(-np.asarray(pose["translation"], dtype=np.float64))
    box.rotate(PyQuaternion(pose["rotation"]).inverse)
    box.translate(-np.asarray(cs["translation"], dtype=np.float64))
    box.rotate(PyQuaternion(cs["rotation"]).inverse)
    intrinsic = np.asarray(cs["camera_intrinsic"], dtype=np.float64)
    imsize = (int(sd["width"]), int(sd["height"]))
    return bool(box_in_image(box, intrinsic, imsize, vis_level=BoxVisibility.ANY))


def filter_eval_boxes_to_camera_fov(nusc, eval_boxes, cam_name: str, verbose: bool = True):
    before = 0
    after = 0
    for sample_token in eval_boxes.sample_tokens:
        boxes = eval_boxes.boxes[sample_token]
        before += len(boxes)
        kept = [
            box for box in boxes
            if global_eval_box_visible_in_camera(nusc, sample_token, box, cam_name)
        ]
        eval_boxes.boxes[sample_token] = kept
        after += len(kept)
    if verbose:
        print(f"[FRONT_FOV] {cam_name}: kept {after}/{before} boxes")
    return eval_boxes


def point_in_front_sector(x: float, y: float, opening_angle_deg: float = 90.0) -> bool:
    """Return whether an ego-frame point lies inside a forward-facing angular sector."""
    if not math.isfinite(x) or not math.isfinite(y) or opening_angle_deg <= 0:
        return False
    half_angle = math.radians(opening_angle_deg / 2.0)
    return abs(math.atan2(y, x)) <= half_angle


def filter_eval_boxes_to_front_sector(
    nusc, eval_boxes, opening_angle_deg: float = 90.0, verbose: bool = True,
):
    """Filter global nuScenes boxes to a vehicle-forward sector at CAM_FRONT time."""
    before = 0
    after = 0
    for sample_token in eval_boxes.sample_tokens:
        sample = nusc.get("sample", sample_token)
        sd = nusc.get("sample_data", sample["data"]["CAM_FRONT"])
        pose = nusc.get("ego_pose", sd["ego_pose_token"])
        ego_t = np.asarray(pose["translation"], dtype=np.float64)
        ego_r = PyQuaternion(pose["rotation"]).rotation_matrix
        kept = []
        boxes = eval_boxes.boxes[sample_token]
        before += len(boxes)
        for box in boxes:
            center_global = np.asarray(box.translation, dtype=np.float64)
            center_ego = ego_r.T.dot(center_global - ego_t)
            if point_in_front_sector(float(center_ego[0]), float(center_ego[1]), opening_angle_deg):
                kept.append(box)
        eval_boxes.boxes[sample_token] = kept
        after += len(kept)
    if verbose:
        print(f"[FRONT_SECTOR] angle={opening_angle_deg:g}: kept {after}/{before} boxes")
    return eval_boxes


def filter_eval_boxes_to_tokens(eval_boxes, keep_tokens: Sequence[str], verbose: bool = True):
    keep = set(keep_tokens)
    before_tokens = len(eval_boxes.sample_tokens)
    eval_boxes.boxes = {
        token: boxes for token, boxes in eval_boxes.boxes.items() if token in keep
    }
    if verbose:
        print(f"[PROXY_SUBSET] kept {len(eval_boxes.sample_tokens)}/{before_tokens} sample tokens")
    return eval_boxes


@torch.no_grad()
def export_and_eval(mod, args, model, dataset):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    nusc = dataset.nusc
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / "results_nusc.json"
    all_tokens = all_split_sample_tokens(nusc, mod, args.eval_set)
    expected_camera_passes = (
        len(dataset.student_camera_indices)
        if args.model == "student" and args.student_eval_views == "all"
        else 1
    )
    pending_boxes: Dict[str, List[Tuple]] = {}
    pending_camera_passes: Dict[str, int] = {}
    emitted_tokens = set()
    prediction_count = 0
    merge_seconds = 0.0
    json_write_seconds = 0.0
    first_result = True

    def merge_boxes(boxes: List[Tuple]) -> List[Tuple]:
        if getattr(args, "merge_mode", "nms") == "weighted":
            merged = weighted_box_fusion_classwise(
                boxes, args.merge_nms_iou,
                score_power=args.merge_fusion_score_power,
                geometry=args.merge_fusion_geometry,
                yaw_gate=math.radians(args.merge_fusion_yaw_gate_deg),
                score_mode=args.merge_fusion_score_mode,
            )
        else:
            merged = nms_classwise(boxes, args.merge_nms_iou)
        return sorted(merged, key=lambda box: box[9], reverse=True)[:args.max_boxes_per_sample]

    export_start = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)

    with result_path.open("w", encoding="utf-8") as result_file:
        result_file.write(
            '{"meta":{"use_camera":true,"use_lidar":false,"use_radar":false,'
            '"use_map":false,"use_external":false},"results":{'
        )

        def emit_result(token: str, boxes: List[Tuple]):
            nonlocal first_result, prediction_count, merge_seconds, json_write_seconds
            if token in emitted_tokens:
                raise RuntimeError(f"duplicate evaluation token: {token}")
            merge_start = time.perf_counter()
            merged = merge_boxes(boxes)
            merge_seconds += time.perf_counter() - merge_start
            encoded = [ego_box_to_nusc_result(nusc, token, box, "CAM_FRONT") for box in merged]
            write_start = time.perf_counter()
            if not first_result:
                result_file.write(",")
            json.dump(token, result_file)
            result_file.write(":")
            json.dump(encoded, result_file, separators=(",", ":"))
            json_write_seconds += time.perf_counter() - write_start
            first_result = False
            emitted_tokens.add(token)
            prediction_count += len(encoded)

        for batch in tqdm(loader, desc="Export official JSON", dynamic_ncols=True, file=sys.stdout):
            tokens = list(batch[0])
            velocity_intervals = None
            if args.model in {"teacher", "fused_teacher"}:
                imgs6, ks6, r6, t6 = [x.to(device, non_blocking=True) for x in batch[1:5]]
                hm, reg, _depth = model(imgs6, ks6, r6, t6)
                attr_logits = None
            else:
                img1, k1, r1, t1 = [x.to(device, non_blocking=True) for x in batch[5:9]]
                if getattr(args, "temporal_bev", False):
                    temporal = [tensor.to(device, non_blocking=True) for tensor in batch[18:25]]
                    student_outputs = model.forward_outputs(
                        img1, k1, r1, t1, *temporal,
                        privileged_depth=(
                            batch[14].to(device, non_blocking=True)
                            if model.privileged_depth_teacher_enabled else None
                        ),
                    )
                    if getattr(args, "velocity_displacement_target", False):
                        velocity_intervals = batch[25].tolist()
                else:
                    student_outputs = model.forward_outputs(
                        img1, k1, r1, t1,
                        privileged_depth=(
                            batch[14].to(device, non_blocking=True)
                            if model.privileged_depth_teacher_enabled else None
                        ),
                    )
                hm, reg, _depth, _bev, attr_logits, class_motion = student_outputs[:6]
                if model.azimuth_bin_head is not None or model.class_size_head is not None:
                    azimuth_bins, class_size = student_outputs[6:8]
                    output_offset = 8
                else:
                    azimuth_bins = class_size = None
                    output_offset = 6
                if model.class_center_head is not None:
                    class_center = student_outputs[output_offset]
                    output_offset += 1
                else:
                    class_center = None
                if model.image_yaw_flip_head is not None:
                    image_yaw_flip_logits = student_outputs[output_offset]
                    output_offset += 1
                else:
                    image_yaw_flip_logits = None
                if model.image_radial_center_head is not None:
                    image_radial_center = student_outputs[output_offset]
                    output_offset += 1
                else:
                    image_radial_center = None
                if model.image_object_geometry_head is not None:
                    image_yaw_flip_logits = student_outputs[output_offset]
                    image_radial_center = student_outputs[output_offset + 1]
                    output_offset += 2
                if args.student_fov_only:
                    bev_fov_mask = batch[15].to(device, non_blocking=True) if len(batch) > 15 else torch.ones_like(hm[:, :1])
                    hm = hm.masked_fill(bev_fov_mask <= 0, -30.0)
            boxes_batch = decode_official_boxes(
                hm.cpu(), reg.cpu(), mod, args.score_thr, args.topk, args.nms_iou,
                attr_logits.cpu() if attr_logits is not None else None,
                velocity_intervals=velocity_intervals,
                class_score_thrs=getattr(args, "class_score_thrs", None),
                class_motion=class_motion.cpu() if args.model == "student" and class_motion is not None else None,
                azimuth_bins=azimuth_bins.cpu() if args.model == "student" and azimuth_bins is not None else None,
                class_size=class_size.cpu() if args.model == "student" and class_size is not None else None,
                class_center=(
                    class_center.cpu()
                    if args.model == "student" and class_center is not None else None
                ),
                image_yaw_flip_logits=(
                    image_yaw_flip_logits.cpu()
                    if args.model == "student" and image_yaw_flip_logits is not None else None
                ),
                k=k1.cpu() if args.model == "student" else None,
                r_cam_to_ego=r1.cpu() if args.model == "student" else None,
                t_cam_to_ego=t1.cpu() if args.model == "student" else None,
                image_yaw_flip_threshold=float(
                    getattr(args, "image_yaw_flip_threshold", 0.5)
                ),
                image_radial_center=(
                    image_radial_center.cpu()
                    if args.model == "student" and image_radial_center is not None else None
                ),
                depth_logits=(
                    _depth.cpu() if args.model == "student" else None
                ),
                depth_distribution_alpha=float(
                    getattr(args, "depth_distribution_alpha", 0.0)
                ),
                depth_distribution_statistic=getattr(
                    args, "depth_distribution_statistic", "mean"
                ),
                depth_distribution_window_radius=int(
                    getattr(args, "depth_distribution_window_radius", 1)
                ),
                depth_distribution_max_shift=float(
                    getattr(args, "depth_distribution_max_shift", 2.0)
                ),
                attribute_velocity_gate=(
                    bool(getattr(args, "attribute_velocity_gate", False)) and
                    getattr(model, "class_attribute_velocity_calibrator", None) is None
                ),
            )
            if args.model == "student" and len(batch) > 16:
                camera_indices = batch[16].tolist()
            else:
                camera_indices = [mod.CAM_CHANNELS.index("CAM_FRONT")] * len(tokens)
            for token, boxes, camera_idx in zip(tokens, boxes_batch, camera_indices):
                source_cam = mod.CAM_CHANNELS[int(camera_idx)]
                canonical_boxes = [
                    transform_box_between_ego_frames(
                        nusc, token, tuple(box) + (int(camera_idx),), source_cam, "CAM_FRONT",
                    )
                    for box in boxes
                ]
                pending_boxes.setdefault(token, []).extend(canonical_boxes)
                pending_camera_passes[token] = pending_camera_passes.get(token, 0) + 1
                if pending_camera_passes[token] == expected_camera_passes:
                    emit_result(token, pending_boxes.pop(token))
                    pending_camera_passes.pop(token)
                elif pending_camera_passes[token] > expected_camera_passes:
                    raise RuntimeError(f"too many camera passes for token: {token}")

        if pending_boxes:
            raise RuntimeError(
                f"incomplete camera groups for {len(pending_boxes)} evaluation tokens"
            )
        for token in all_tokens:
            if token not in emitted_tokens:
                emit_result(token, [])
        result_file.write("}}")

    if device.type == "cuda":
        torch.cuda.synchronize()
    inference_end = time.perf_counter()
    peak_allocated_mib = (
        torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        if device.type == "cuda" else None
    )
    peak_reserved_mib = (
        torch.cuda.max_memory_reserved(device) / (1024 ** 2)
        if device.type == "cuda" else None
    )

    write_end = time.perf_counter()
    print(f"[OK] wrote {result_path}")

    sample_count = len(dataset.sample_tokens)
    camera_pass_count = len(dataset)
    timing = {
        "scope": (
            "export pipeline only: image read/transform, model forward, per-view decode, "
            "cross-view merge and JSON write; official metric computation excluded"
        ),
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "scene_sample_count": sample_count,
        "camera_pass_count": camera_pass_count,
        "image_read_forward_decode_seconds": max(
            0.0, inference_end - export_start - merge_seconds - json_write_seconds,
        ),
        "cross_view_merge_seconds": merge_seconds,
        "json_write_seconds": json_write_seconds,
        "export_total_seconds": write_end - export_start,
        "export_ms_per_scene_sample": (
            1000.0 * (write_end - export_start) / max(sample_count, 1)
        ),
        "image_read_forward_decode_ms_per_camera_pass": (
            1000.0 * (inference_end - export_start) / max(camera_pass_count, 1)
        ),
        "peak_cuda_memory_allocated_mib": peak_allocated_mib,
        "peak_cuda_memory_reserved_mib": peak_reserved_mib,
    }

    protocol_path = out_dir / "evaluation_protocol.json"
    with protocol_path.open("w", encoding="utf-8") as f:
        json.dump({
            "model": args.model,
            "student_eval_views": getattr(args, "student_eval_views", None),
            "student_camera": getattr(args, "student_camera", None),
            "student_eval_cameras": getattr(args, "student_eval_cameras", None),
            "student_fov_only": getattr(args, "student_fov_only", None),
            "metric_scope": args.metric_scope,
            "subset_manifest": getattr(args, "subset_manifest", ""),
            "evaluated_sample_count": len(dataset.sample_tokens),
            "timing": timing,
            "score_threshold": args.score_thr,
            "per_view_nms_iou": args.nms_iou,
            "merge_nms_iou": args.merge_nms_iou,
            "merge_nms_backend": (
                "cross_view_weighted_fusion" if getattr(args, "merge_mode", "nms") == "weighted"
                else "exact_spatial_index"
            ),
            "merge_mode": getattr(args, "merge_mode", "nms"),
            "merge_fusion_geometry": getattr(args, "merge_fusion_geometry", "center"),
            "merge_fusion_score_power": getattr(args, "merge_fusion_score_power", 2.0),
            "merge_fusion_yaw_gate_deg": getattr(args, "merge_fusion_yaw_gate_deg", 60.0),
            "merge_fusion_score_mode": getattr(args, "merge_fusion_score_mode", "max"),
            "attribute_policy": "learned_v1" if getattr(args, "attribute_head", False) else "mmdet3d_velocity_default_v1",
            "max_boxes_per_sample": args.max_boxes_per_sample,
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "student_backbone": getattr(args, "student_backbone", None),
            "camera_aware_depth": getattr(args, "camera_aware_depth", None),
            "bev_norm": getattr(args, "bev_norm", None),
            "depth_bins": getattr(mod, "NUM_DEPTH_BINS", None),
            "attribute_head": getattr(args, "attribute_head", False),
            "temporal_bev": getattr(args, "temporal_bev", False),
            "temporal_pre_encoder": getattr(args, "temporal_pre_encoder", False),
            "temporal_delta_limit": getattr(args, "temporal_delta_limit", 0.0),
            "temporal_stereo_depth": getattr(args, "temporal_stereo_depth", False),
            "temporal_stereo_dynamic_confidence": getattr(
                args, "temporal_stereo_dynamic_confidence", False,
            ),
            "residual_yaw_vector": getattr(args, "residual_yaw_vector", False),
            "residual_yaw_active_classes": getattr(
                args, "residual_yaw_active_classes", None,
            ),
            "image_yaw_flip_head": getattr(args, "image_yaw_flip_head", False),
            "image_yaw_flip_threshold": getattr(
                args, "image_yaw_flip_threshold", 0.5,
            ),
            "image_radial_center_head": getattr(
                args, "image_radial_center_head", False,
            ),
            "image_object_geometry_head": getattr(
                args, "image_object_geometry_head", False,
            ),
            "image_radial_center_max_residual": getattr(
                args, "image_radial_center_max_residual", 2.0,
            ),
            "depth_distribution_alpha": getattr(
                args, "depth_distribution_alpha", 0.0,
            ),
            "depth_distribution_statistic": getattr(
                args, "depth_distribution_statistic", "mean",
            ),
            "depth_distribution_window_radius": getattr(
                args, "depth_distribution_window_radius", 1,
            ),
            "depth_distribution_max_shift": getattr(
                args, "depth_distribution_max_shift", 2.0,
            ),
            "class_aware_center_head": getattr(
                args, "class_aware_center_head", False,
            ),
            "center_refiner_active_classes": getattr(
                args, "center_refiner_active_classes", None,
            ),
            "class_attribute_velocity_calibrator": getattr(
                args, "class_attribute_velocity_calibrator", False,
            ),
            "velocity_calibrator_motorcycle": getattr(
                args, "velocity_calibrator_motorcycle", True,
            ),
            "velocity_calibrator_active_classes": getattr(
                args, "velocity_calibrator_active_classes", None,
            ),
            "velocity_calibrator_bypass_attribute_gate": getattr(
                args, "velocity_calibrator_bypass_attribute_gate", False,
            ),
            "velocity_calibrator_max_residual": getattr(
                args, "velocity_calibrator_max_residual", 0.75,
            ),
            "velocity_calibrator_initial_scales": getattr(
                args, "velocity_calibrator_initial_scales", None,
            ),
        }, f, indent=2)

    if args.skip_eval:
        return
    if prediction_count == 0:
        summary_path = out_dir / "metrics_summary.json"
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump({
                "mean_ap": 0.0,
                "nd_score": 0.0,
                "eval_time": 0.0,
                "empty_predictions": True,
                "prediction_count": 0,
            }, f, indent=2)
        print("[EMPTY_PREDICTIONS] no boxes survived export; recorded NDS=0 and mAP=0")
        if args.delete_result_json_after_eval:
            result_path.unlink()
            print(f"[CLEANUP] deleted regenerable {result_path}")
        return
    cfg = config_factory("detection_cvpr_2019")
    nusc_eval = NuScenesEval(nusc, config=cfg, result_path=str(result_path), eval_set=args.eval_set, output_dir=str(out_dir), verbose=True)
    if args.metric_scope == "front_fov":
        if args.model != "student" or args.student_eval_views != "front":
            raise ValueError("front_fov metric requires a front-view student evaluation")
        nusc_eval.gt_boxes = filter_eval_boxes_to_camera_fov(
            nusc, nusc_eval.gt_boxes, args.student_camera, verbose=True,
        )
        nusc_eval.pred_boxes = filter_eval_boxes_to_camera_fov(
            nusc, nusc_eval.pred_boxes, args.student_camera, verbose=True,
        )
    elif args.metric_scope == "front_90":
        if args.model != "student" or args.student_eval_views != "front":
            raise ValueError("front_90 metric requires a front-view student evaluation")
        nusc_eval.gt_boxes = filter_eval_boxes_to_front_sector(
            nusc, nusc_eval.gt_boxes, opening_angle_deg=90.0, verbose=True,
        )
        nusc_eval.pred_boxes = filter_eval_boxes_to_front_sector(
            nusc, nusc_eval.pred_boxes, opening_angle_deg=90.0, verbose=True,
        )
    elif args.metric_scope == "proxy_subset":
        nusc_eval.gt_boxes = filter_eval_boxes_to_tokens(
            nusc_eval.gt_boxes, dataset.sample_tokens, verbose=True,
        )
        nusc_eval.pred_boxes = filter_eval_boxes_to_tokens(
            nusc_eval.pred_boxes, dataset.sample_tokens, verbose=True,
        )
        nusc_eval.sample_tokens = nusc_eval.gt_boxes.sample_tokens
    nusc_eval.main(plot_examples=0, render_curves=False)
    summary_path = out_dir / "metrics_summary.json"
    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8") as f:
            summary = json.load(f)
        print("[NDS]", summary.get("nd_score"))
        print("[mAP]", summary.get("mean_ap"))
    if args.delete_result_json_after_eval:
        result_path.unlink()
        print(f"[CLEANUP] deleted regenerable {result_path}")


def build_argparser():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add_common(p):
        p.add_argument("--data-root", default=os.getenv("DATA_ROOT", str(THIS_DIR.parent / "datasets" / "nuscenes")))
        p.add_argument("--version", default=os.getenv("NUSC_VERSION", "v1.0-trainval"))
        p.add_argument("--out-dir", required=True)
        p.add_argument("--batch-size", type=int, default=2)
        p.add_argument("--num-workers", type=int, default=4)
        p.add_argument("--amp", dest="amp", action="store_true", default=True)
        p.add_argument("--no-amp", dest="amp", action="store_false")
        p.add_argument(
            "--assert-module-updates", nargs="*", default=(),
            help=(
                "Named modules that must change during every training epoch. "
                "Used to fail fast on silent AMP/optimizer update loss."
            ),
        )
        p.add_argument("--bev-x-min", type=float, default=-50.0)
        p.add_argument("--bev-x-max", type=float, default=50.0)
        p.add_argument("--bev-y-min", type=float, default=-50.0)
        p.add_argument("--bev-y-max", type=float, default=50.0)
        p.add_argument("--bev-res", type=float, default=0.5)
        p.add_argument("--teacher-img-w", type=int, default=int(os.getenv("TEACHER_IMG_W", "0")))
        p.add_argument("--teacher-img-h", type=int, default=int(os.getenv("TEACHER_IMG_H", "0")))
        p.add_argument("--student-img-w", type=int, default=int(os.getenv("STUDENT_IMG_W", "0")))
        p.add_argument("--student-img-h", type=int, default=int(os.getenv("STUDENT_IMG_H", "0")))
        p.add_argument("--depth-bins", type=int, default=0)
        p.add_argument(
            "--student-backbone",
            choices=["stride8", "multiscale", "efficientnet_b2"],
            default="stride8",
        )
        p.add_argument("--camera-aware-depth", action="store_true")
        p.add_argument(
            "--privileged-depth-teacher", action="store_true",
            help=(
                "Enable the project-native selected-view LiDAR sparse-depth teacher. "
                "This is a training/evaluation teacher architecture, not deployable student input."
            ),
        )
        p.add_argument(
            "--camera-privileged-memory-adapter", action="store_true",
            help=(
                "Attach a deployable camera-only residual memory path for learning "
                "privileged depth and BEV increments; it never consumes LiDAR."
            ),
        )
        p.add_argument("--privileged-memory-slots", type=int, default=32)
        p.add_argument("--privileged-memory-latent-dim", type=int, default=32)
        p.add_argument("--bev-norm", choices=["batch", "group"], default="batch")
        p.add_argument("--attribute-head", action="store_true")
        p.add_argument("--temporal-bev", action="store_true")
        p.add_argument(
            "--temporal-stereo-depth", action="store_true",
            help="Add causal previous-frame geometric matching evidence to depth logits.",
        )
        p.add_argument(
            "--temporal-stereo-dynamic-confidence", action="store_true",
            help=(
                "Restrict temporal stereo with monocular uncertainty and learn a "
                "pixel-wise confidence from matching diagnostics."
            ),
        )
        p.add_argument("--temporal-pre-encoder", action="store_true")
        p.add_argument(
            "--temporal-delta-limit", type=float, default=0.0,
            help=(
                "Symmetric bound for the temporal BEV residual before fusion. "
                "0 preserves the historical unbounded implementation."
            ),
        )
        p.add_argument("--temporal-spatial-fusion", action="store_true")
        p.add_argument("--velocity-displacement-target", action="store_true")
        p.add_argument("--class-aware-motion-head", action="store_true")
        p.add_argument("--temporal-correlation-motion", action="store_true")
        p.add_argument("--object-velocity-refiner", action="store_true")
        p.add_argument("--residual-azimuth-multibin", action="store_true")
        p.add_argument("--residual-yaw-vector", action="store_true")
        p.add_argument(
            "--residual-yaw-active-classes", nargs="+", choices=DETECTION_NAMES,
            default=None,
            help="Classes changed by the smooth residual yaw head.",
        )
        p.add_argument("--class-attribute-velocity-calibrator", action="store_true")
        p.add_argument(
            "--velocity-calibrator-no-motorcycle",
            dest="velocity_calibrator_motorcycle", action="store_false", default=True,
            help="Keep the proven S1 motorcycle velocity instead of calibrating it.",
        )
        p.add_argument(
            "--velocity-calibrator-active-classes", nargs="+", choices=DETECTION_NAMES,
            default=None,
            help="Override the V15 calibrator classes, e.g. bus motorcycle.",
        )
        p.add_argument(
            "--velocity-calibrator-bypass-attribute-gate", action="store_true",
            help="Do not suppress selected calibrator classes with frozen attribute logits.",
        )
        p.add_argument(
            "--velocity-calibrator-max-residual", type=float, default=0.75,
            help="Maximum per-axis displacement residual in metres.",
        )
        p.add_argument(
            "--velocity-calibrator-initial-scales", nargs=len(DETECTION_NAMES),
            type=float, default=None,
            metavar=tuple(name.upper() for name in DETECTION_NAMES),
            help="Identity-preserving class scale initialization override.",
        )
        p.add_argument("--azimuth-multibin", action="store_true")
        p.add_argument("--class-aware-size-head", action="store_true")
        p.add_argument("--class-aware-center-head", action="store_true")
        p.add_argument(
            "--image-yaw-flip-head", action="store_true",
            help="Use stride-8 image appearance to classify front/back pi flips.",
        )
        p.add_argument(
            "--image-yaw-flip-threshold", type=float, default=0.5,
            help="Inference probability threshold for applying a pi yaw flip.",
        )
        p.add_argument("--image-radial-center-head", action="store_true")
        p.add_argument(
            "--image-object-geometry-head", action="store_true",
            help="Joint dilated object-region yaw-flip and radial-depth refinement head.",
        )
        p.add_argument(
            "--image-radial-center-max-residual", type=float, default=2.0,
        )
        p.add_argument(
            "--depth-distribution-alpha", type=float, default=0.0,
            help="Blend factor for parameter-free centre refinement from depth logits.",
        )
        p.add_argument(
            "--depth-distribution-statistic", choices=["mean", "mode"],
            default="mean",
        )
        p.add_argument(
            "--depth-distribution-window-radius", type=int, default=1,
            help="Feature-map radius used to average object-centre depth probabilities.",
        )
        p.add_argument(
            "--depth-distribution-max-shift", type=float, default=2.0,
            help="Maximum ego-ground-plane centre correction in metres before blending.",
        )
        p.add_argument(
            "--center-refiner-active-classes", nargs="+", choices=DETECTION_NAMES,
            default=None,
            help="Classes changed by the bounded class-aware center residual head.",
        )
        p.add_argument("--attribute-velocity-gate", action="store_true")
        p.add_argument("--target-scope", choices=["auto", "full", "student_fov"], default="auto")
        p.add_argument("--seed", type=int, default=int(os.getenv("SEED", "0")))
        p.add_argument("--student-camera", choices=[
            "CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
            "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT",
        ], default="CAM_FRONT")
        p.add_argument(
            "--student-eval-cameras", nargs="+", choices=[
                "CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
                "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT",
            ], default=None,
            help=(
                "Optional camera subset for --student-eval-views all. The same "
                "single-view student weights are reused for every selected camera."
            ),
        )
        p.add_argument("--fast-runtime", dest="fast_runtime", action="store_true", default=True)
        p.add_argument("--reference-runtime", dest="fast_runtime", action="store_false")
        p.add_argument("--splat-depth-chunk", type=int, default=4)
        p.add_argument("--compact-backbone", dest="compact_backbone", action="store_true", default=True)
        p.add_argument("--legacy-full-backbone", dest="compact_backbone", action="store_false")
        p.add_argument("--merge-mode", choices=["nms", "weighted"], default="nms")
        p.add_argument(
            "--merge-fusion-geometry", choices=["center", "size", "yaw", "full"],
            default="center",
        )
        p.add_argument("--merge-fusion-score-power", type=float, default=2.0)
        p.add_argument("--merge-fusion-yaw-gate-deg", type=float, default=60.0)
        p.add_argument(
            "--merge-fusion-score-mode", choices=["max", "noisy_or"], default="max",
        )

    def add_train(p):
        add_common(p)
        p.add_argument("--epochs", type=int, default=12)
        p.add_argument(
            "--schedule-total-epochs", type=int, default=0,
            help=(
                "Fixed scheduler horizon. Set this to the original total epoch budget "
                "when --epochs is only the number of additional resume epochs."
            ),
        )
        p.add_argument("--grad-accum", type=int, default=2)
        p.add_argument("--lr", type=float, default=2e-4)
        p.add_argument("--weight-decay", type=float, default=1e-4)
        p.add_argument("--grad-clip", type=float, default=5.0)
        p.add_argument("--hm-weight", type=float, default=1.0)
        p.add_argument("--reg-weight", type=float, default=1.0)
        p.add_argument(
            "--reg-z-scale", type=float, default=1.0,
            help="Relative GT regression weight for height z; x/y remain 1.",
        )
        p.add_argument("--reg-size-scale", type=float, default=1.0)
        p.add_argument("--reg-yaw-scale", type=float, default=1.0)
        p.add_argument("--reg-velocity-scale", type=float, default=1.0)
        p.add_argument("--yaw-cosine-weight", type=float, default=0.0)
        p.add_argument("--velocity-metric-weight", type=float, default=0.0)
        p.add_argument("--azimuth-bin-weight", type=float, default=0.0)
        p.add_argument("--size-prior-weight", type=float, default=0.0)
        p.add_argument("--ase-iou-weight", type=float, default=0.0)
        p.add_argument("--velocity-decomposed-weight", type=float, default=0.0)
        p.add_argument("--yaw-velocity-consistency-weight", type=float, default=0.0)
        p.add_argument("--image-yaw-flip-weight", type=float, default=0.0)
        p.add_argument("--image-radial-center-weight", type=float, default=0.0)
        p.add_argument(
            "--yaw-flip-class-weights", nargs=len(DETECTION_NAMES), type=float,
            default=None, metavar=tuple(name.upper() for name in DETECTION_NAMES),
        )
        p.add_argument(
            "--radial-center-class-weights", nargs=len(DETECTION_NAMES), type=float,
            default=None, metavar=tuple(name.upper() for name in DETECTION_NAMES),
        )
        p.add_argument("--motion-head-lr-multiplier", type=float, default=1.0)
        p.add_argument("--backbone-lr-multiplier", type=float, default=1.0)
        p.add_argument(
            "--motion-class-weights", nargs=len(DETECTION_NAMES), type=float, default=None,
            metavar=tuple(name.upper() for name in DETECTION_NAMES),
            help="Per-class weights for direct yaw and velocity metric losses.",
        )
        p.add_argument(
            "--size-class-weights", nargs=len(DETECTION_NAMES), type=float, default=None,
            metavar=tuple(name.upper() for name in DETECTION_NAMES),
            help="Per-class weights for normalized size and aligned-IoU losses.",
        )
        p.add_argument("--depth-weight", type=float, default=0.2)
        p.add_argument("--attr-weight", type=float, default=0.0)
        p.add_argument("--train-max-samples", type=int, default=0)
        p.add_argument("--val-max-samples", type=int, default=0)
        p.add_argument("--train-subset-mode", choices=["first", "random"], default="random")
        p.add_argument("--val-subset-mode", choices=["first", "random"], default="first")
        p.add_argument("--resume", default="")
        p.add_argument("--save-every-iters", type=int, default=0)
        p.add_argument("--save-every-epoch", action="store_true")
        p.add_argument(
            "--max-nonfinite-batches", type=int, default=0,
            help=(
                "Maximum non-finite batches allowed per train or validation epoch. "
                "Every occurrence is logged; exceeding the limit fails the run."
            ),
        )
        p.add_argument(
            "--debug-nonfinite-activations", action="store_true",
            help="Record selected module output statistics for non-finite diagnosis.",
        )
        p.add_argument("--select-best-by", choices=["auto", "loss", "det_loss"], default="auto")
        p.add_argument("--early-stopping-patience", type=int, default=0)
        p.add_argument("--min-epochs", type=int, default=1)
        p.add_argument("--class-balanced-sampling", action="store_true")
        p.add_argument("--class-balance-cap", type=float, default=3.0)
        p.add_argument(
            "--velocity-target-view-classes", nargs="+", choices=DETECTION_NAMES,
            default=None,
            help="Uniformly sample visible camera views of the selected velocity classes.",
        )
        p.add_argument(
            "--velocity-target-view-epoch-size", type=int, default=0,
            help="Number of target-class camera views per epoch; 0 keeps train split length.",
        )
        p.add_argument(
            "--velocity-target-state-balanced-classes", nargs="+", choices=DETECTION_NAMES,
            default=None,
            help="Balance static and moving visible-view strata for selected target classes.",
        )
        p.add_argument(
            "--velocity-moving-boost", type=float, default=1.0,
            help="Extra metric-loss weight for GT objects moving at least 0.2 m/s.",
        )
        p.add_argument("--freeze-base-for-motion", action="store_true")
        p.add_argument("--freeze-base-for-velocity-refiner", action="store_true")
        p.add_argument("--freeze-base-for-yaw-refiner", action="store_true")
        p.add_argument("--freeze-base-for-center-refiner", action="store_true")
        p.add_argument(
            "--freeze-base-for-depth-distillation", action="store_true",
            help=(
                "Freeze the proven V31 detector and optimize only its camera depth "
                "head from supervised and privileged residual geometry losses."
            ),
        )
        p.add_argument("--freeze-base-for-image-yaw-flip", action="store_true")
        p.add_argument("--freeze-base-for-image-radial-center", action="store_true")
        p.add_argument("--freeze-base-for-image-object-geometry", action="store_true")
        p.add_argument("--freeze-base-for-velocity-calibrator", action="store_true")
        p.add_argument("--freeze-base-for-temporal-stereo", action="store_true")
        p.add_argument(
            "--freeze-base-for-privileged-teacher", action="store_true",
            help=(
                "Freeze the proven camera detector and train only the custom sparse-depth "
                "completion and BEV refinement teacher modules."
            ),
        )
        p.add_argument(
            "--freeze-base-for-privileged-memory-adapter", action="store_true",
            help=(
                "Freeze the audited V31 network and train only the deployable "
                "camera privileged residual memory adapter."
            ),
        )
        p.add_argument(
            "--velocity-calibrator-scale-only", action="store_true",
            help="With a frozen V15 calibrator, train only per-class scale/rotation controls.",
        )

    p_teacher = sub.add_parser("train-teacher")
    add_train(p_teacher)

    p_fused_teacher = sub.add_parser("train-fused-teacher")
    add_train(p_fused_teacher)
    p_fused_teacher.add_argument("--student-init", required=True)
    p_fused_teacher.add_argument("--partial-student-init", action="store_true")
    p_fused_teacher.add_argument("--no-pretrained-backbone", action="store_true")
    p_fused_teacher.add_argument("--freeze-image-encoder", action="store_true")

    p_student = sub.add_parser("train-student")
    add_train(p_student)
    p_student.add_argument("--teacher-ckpt", default="")
    p_student.add_argument(
        "--camera-teacher-ckpt", default="",
        help=(
            "Independent strong camera teaching-assistant checkpoint used as the "
            "visual reference for project-native privileged residual KD."
        ),
    )
    p_student.add_argument(
        "--teacher-type",
        choices=[
            "legacy", "fused_student", "aligned_student",
            "project_privileged", "centerpoint_relation", "camera_ta_relation",
        ],
        default="legacy",
        help=(
            "aligned_student loads a project-trained student checkpoint as the "
            "teacher and feeds it the exact same selected camera/history inputs; "
            "project_privileged loads the fully project-native selected-view "
            "LiDAR sparse-depth teacher; "
            "centerpoint_relation uses a frozen LiDAR CenterPoint only during training; "
            "camera_ta_relation combines the aligned camera assistant with a cached "
            "LiDAR relation residual."
        ),
    )
    p_student.add_argument("--resume-student", default="")
    p_student.add_argument("--student-init", default="")
    p_student.add_argument("--partial-student-init", action="store_true")
    p_student.add_argument("--no-pretrained-backbone", action="store_true")
    p_student.add_argument(
        "--freeze-backbone-stages", type=int, default=0,
        help="Freeze this many early EfficientNet feature stages, including BN statistics.",
    )
    p_student.add_argument("--exp", choices=["A0", "A1", "A2", "A3", "A4", "custom"], default="custom")
    p_student.add_argument("--kd-hm-weight", type=float, default=0.5)
    p_student.add_argument("--kd-reg-weight", type=float, default=0.2)
    p_student.add_argument("--kd-reg-center-scale", type=float, default=1.0)
    p_student.add_argument("--kd-reg-size-scale", type=float, default=1.0)
    p_student.add_argument("--kd-reg-yaw-scale", type=float, default=1.0)
    p_student.add_argument("--kd-reg-velocity-scale", type=float, default=1.0)
    p_student.add_argument("--kd-feat-weight", type=float, default=0.2)
    p_student.add_argument("--kd-depth-weight", type=float, default=0.0)
    p_student.add_argument("--kd-attr-weight", type=float, default=0.0)
    p_student.add_argument("--kd-attr-temp", type=float, default=1.0)
    p_student.add_argument(
        "--kd-project-quality-gate", action="store_true",
        help=(
            "For the project-native privileged teacher, run the same frozen teacher "
            "with and without its LiDAR path and transfer only privileged responses "
            "that are more accurate than the camera-only reference under ground truth."
        ),
    )
    p_student.add_argument(
        "--kd-project-residual-alpha", type=float, default=0.5,
        help="Fraction of each accepted LiDAR-on minus LiDAR-off teacher residual to transfer.",
    )
    p_student.add_argument(
        "--kd-project-quality-margin", type=float, default=0.0,
        help="Required supervised error improvement before accepting a privileged response.",
    )
    p_student.add_argument(
        "--kd-project-hybrid-routing", action="store_true",
        help=(
            "Keep the LiDAR-off camera teacher for BEV features, depth, attributes, "
            "and non-centre box channels; add quality-gated privileged residuals only "
            "to class heatmaps and centre xyz."
        ),
    )
    p_student.add_argument(
        "--kd-project-independent-camera-ta", action="store_true",
        help=(
            "Use --camera-teacher-ckpt as an independent camera assistant. Keep "
            "its stable visual outputs, add quality-gated privileged heatmap/centre "
            "residuals, and retain privileged BEV/depth residuals."
        ),
    )
    p_student.add_argument(
        "--kd-project-v45-instance-residual", action="store_true",
        help=(
            "V45: keep classification on the independent V31 assistant and distil "
            "only privileged dense-depth, low-level nine-point/instance foreground, "
            "and quality-gated centre residuals."
        ),
    )
    p_student.add_argument(
        "--kd-project-residual-memory", action="store_true",
        help=(
            "Supervise the deployable camera adapter itself with LiDAR-on minus "
            "independent-V31 depth and instance-BEV residuals instead of forcing "
            "the shared V31 representation to imitate the privileged teacher."
        ),
    )
    p_student.add_argument(
        "--kd-project-instance-relation", action="store_true",
        help=(
            "Use V37-style nine-keypoint object relations, with the target formed "
            "from the independent V31 relation plus the custom privileged residual."
        ),
    )
    p_student.add_argument(
        "--kd-relation-weight", type=float, default=0.0,
        help="Object-interior 9x9 BEV relation distillation weight for CenterPoint.",
    )
    p_student.add_argument(
        "--kd-relation-residual-alpha", type=float, default=0.25,
        help="Fraction of the LiDAR-minus-camera-assistant relation residual to transfer.",
    )
    p_student.add_argument(
        "--kd-keypoint-feature-weight", type=float, default=0.0,
        help="Direct same-camera feature KD at the nine object-interior keypoints.",
    )
    p_student.add_argument(
        "--centerpoint-config",
        default="/root/v36_centerpoint_teacher/v36_centerpoint_keyframe_config.py",
        help="MMDetection3D config used to construct the frozen LiDAR teacher.",
    )
    p_student.add_argument("--lidar-relation-max-points", type=int, default=35000)
    p_student.add_argument("--lidar-relation-max-objects", type=int, default=128)
    p_student.add_argument(
        "--lidar-relation-cache", default="",
        help="Optional audited SQLite cache of frozen CenterPoint object relations.",
    )
    p_student.add_argument("--kd-temp", type=float, default=2.0)
    p_student.add_argument("--kd-conf-thr", type=float, default=0.10)
    p_student.add_argument("--kd-gt-thr", type=float, default=0.05)
    p_student.add_argument("--kd-pos-boost", type=float, default=4.0)
    p_student.add_argument("--kd-mask-dilate", type=int, default=5)
    p_student.add_argument(
        "--kd-hm-mask", choices=["fov", "foreground"], default="fov",
        help="Restrict dense heatmap KD to the selected view's foreground support when requested.",
    )
    p_student.add_argument(
        "--kd-reg-conf-thr", type=float, default=0.0,
        help="For aligned_student regression KD, require this teacher confidence at GT centres.",
    )
    p_student.add_argument("--kd-feature-mask", choices=["fov", "foreground"], default="foreground")
    p_student.add_argument("--kd-multiscale-region-feature", action="store_true")
    p_student.add_argument("--kd-feature-scales", nargs="+", type=int, default=[1, 2, 4])
    p_student.add_argument("--kd-feature-fg-weight", type=float, default=1.0)
    p_student.add_argument("--kd-feature-fp-weight", type=float, default=0.5)
    p_student.add_argument("--kd-feature-bg-weight", type=float, default=0.05)
    p_student.add_argument("--kd-feature-attention-strength", type=float, default=1.0)
    p_student.add_argument(
        "--kd-full-object-foreground", action="store_true",
        help="Use oriented full-object extents instead of centre-only heatmap support for region KD.",
    )
    p_student.add_argument("--kd-foreground-ellipse-scale", type=float, default=1.0)
    p_student.add_argument("--kd-depth-conf-thr", type=float, default=0.35)
    p_student.add_argument("--kd-depth-valid-only", dest="kd_depth_valid_only", action="store_true", default=True)
    p_student.add_argument("--kd-depth-any-pixel", dest="kd_depth_valid_only", action="store_false")
    p_student.add_argument("--kd-warmup-epochs", type=int, default=2)
    p_student.add_argument(
        "--kd-early-decay-start-epoch", type=int, default=0,
        help="First global epoch of linear early-KD decay; 0 disables it.",
    )
    p_student.add_argument(
        "--kd-early-decay-end-epoch", type=int, default=0,
        help="Last global epoch with non-zero KD before pure GT fine-tuning.",
    )
    p_student.add_argument(
        "--freeze-privileged-memory-after-resume", action="store_true",
        help=(
            "After restoring a staged checkpoint, freeze the deployable geometry "
            "adapter while retaining its learned residual during pure-GT training."
        ),
    )
    p_student.add_argument(
        "--kd-balance-target", type=float, default=0.0,
        help="Adaptive target ratio of weighted KD loss to detection loss; 0 disables balancing.",
    )
    p_student.add_argument(
        "--kd-balance-target-start", type=float, default=-1.0,
        help="Optional early-training total KD/detection budget; negative uses --kd-balance-target.",
    )
    p_student.add_argument(
        "--kd-balance-target-middle", type=float, default=-1.0,
        help="Optional midpoint KD/detection budget; negative uses --kd-balance-target.",
    )
    p_student.add_argument(
        "--kd-balance-target-end", type=float, default=-1.0,
        help="Optional final-epoch KD/detection budget; negative uses --kd-balance-target.",
    )
    p_student.add_argument("--kd-balance-min", type=float, default=0.5)
    p_student.add_argument("--kd-balance-max", type=float, default=2.5)
    p_student.add_argument(
        "--kd-gradient-balance", action="store_true",
        help=(
            "Rescale active KD branches from their gradient norms on the shared "
            "student BEV so predeclared branch shares, rather than raw scalar "
            "loss magnitudes, determine their influence."
        ),
    )
    p_student.add_argument("--kd-gradient-balance-min", type=float, default=0.1)
    p_student.add_argument("--kd-gradient-balance-max", type=float, default=10.0)
    p_student.add_argument(
        "--kd-gradient-balance-exact", action="store_true",
        help=(
            "Apply the requested BEV-gradient shares without min/max clipping. "
            "The outer adaptive KD budget still bounds total KD influence."
        ),
    )
    p_student.add_argument(
        "--kd-gradient-skip-negative-cosine", action="store_true",
        help=(
            "Measure every KD branch against the supervised detection gradient at "
            "the shared BEV; skip branches with negative cosine before norm balancing."
        ),
    )
    p_student.add_argument(
        "--kd-gradient-project-heatmap-conflicts", action="store_true",
        help=(
            "Project any KD gradient component opposing the supervised class "
            "heatmap gradient while retaining its orthogonal geometry signal."
        ),
    )
    p_student.add_argument(
        "--kd-gradient-anchor-image-feature", action="store_true",
        help=(
            "Measure every KD branch at the shared current-image backbone feature. "
            "Unlike the final BEV anchor, this has a gradient path from depth, "
            "low-level BEV, response, and centre losses alike."
        ),
    )
    p_student.add_argument(
        "--kd-per-branch-mema", action="store_true",
        help=(
            "Control feature/depth KD at their natural attachment tensors "
            "instead of pretending they reach the final BEV anchor."
        ),
    )
    p_student.add_argument("--kd-mema-decay", type=float, default=0.90)
    p_student.add_argument("--kd-mema-floor", type=float, default=0.15)
    p_student.add_argument(
        "--kd-natural-anchor-project-conflicts", action="store_true",
        help=(
            "Apply value-preserving PCGrad to conflicting feature/depth KD "
            "gradients at each branch's natural attachment tensor."
        ),
    )
    p_student.add_argument(
        "--kd-natural-anchor-max-gradient-ratio", type=float, default=0.0,
        help=(
            "Optional per-branch cap relative to the supervised detection "
            "gradient norm at the same attachment tensor; 0 disables it."
        ),
    )
    p_student.add_argument("--kd-share-camera-hm", type=float, default=0.10)
    p_student.add_argument("--kd-share-camera-reg", type=float, default=0.10)
    p_student.add_argument("--kd-share-camera-feature", type=float, default=0.25)
    p_student.add_argument("--kd-share-camera-depth", type=float, default=0.0)
    p_student.add_argument("--kd-share-camera-attr", type=float, default=0.05)
    p_student.add_argument("--kd-share-camera-keypoint-feature", type=float, default=0.10)
    p_student.add_argument("--kd-share-lidar-relation", type=float, default=0.05)
    p_student.add_argument("--kd-share-lidar-hm", type=float, default=0.15)
    p_student.add_argument("--kd-share-lidar-reg", type=float, default=0.30)
    p_student.add_argument("--kd-share-lidar-yaw", type=float, default=0.0)
    p_student.add_argument(
        "--relative-depth-weight", type=float, default=0.0,
        help="Training-only local relative-depth geometry loss on sparse LiDAR projections.",
    )
    p_student.add_argument(
        "--relative-depth-offsets", nargs="+", type=int, default=[1, 2, 4, 8],
    )
    p_student.add_argument(
        "--hop-hm-weight", type=float, default=0.0,
        help="Training-only short-term historical heatmap prediction loss weight.",
    )
    p_student.add_argument(
        "--hop-reg-weight", type=float, default=0.0,
        help="Training-only short-term historical box regression loss weight.",
    )
    p_student.add_argument("--pseudo-labels", default="")
    p_student.add_argument("--pseudo-score-thr", type=float, default=0.2)
    p_student.add_argument(
        "--pseudo-class-score-thrs", nargs=len(DETECTION_NAMES), type=float, default=None,
        metavar=tuple(name.upper() for name in DETECTION_NAMES),
        help="Per-class PGD score thresholds in DETECTION_NAMES order; overrides --pseudo-score-thr.",
    )
    p_student.add_argument("--pseudo-hm-weight", type=float, default=0.0)
    p_student.add_argument(
        "--pseudo-class-loss-weights", nargs=len(DETECTION_NAMES), type=float, default=None,
        metavar=tuple(name.upper() for name in DETECTION_NAMES),
        help="Per-class weights for foreground-only pseudo heatmap distillation.",
    )
    p_student.add_argument("--pseudo-reg-weight", type=float, default=0.0)
    p_student.add_argument(
        "--pseudo-reg-class-weights", nargs=len(DETECTION_NAMES), type=float,
        default=None, metavar=tuple(name.upper() for name in DETECTION_NAMES),
        help=(
            "Per-class weights for teacher box regression. This permits a "
            "LiDAR teacher to supervise classes where its geometry is reliable "
            "without importing known weak long-tail predictions."
        ),
    )
    p_student.add_argument("--pseudo-reg-center-scale", type=float, default=1.0)
    p_student.add_argument("--pseudo-reg-size-scale", type=float, default=1.0)
    p_student.add_argument("--pseudo-reg-yaw-scale", type=float, default=1.0)
    p_student.add_argument("--pseudo-reg-velocity-scale", type=float, default=1.0)
    p_student.add_argument(
        "--pseudo-yaw-refiner-weight", type=float, default=0.0,
        help=(
            "Distill cached teacher yaw directly into the deployed residual-yaw "
            "head at pseudo centres; requires --residual-yaw-vector."
        ),
    )
    p_student.add_argument(
        "--pseudo-yaw-class-weights", nargs=len(DETECTION_NAMES), type=float,
        default=None, metavar=tuple(name.upper() for name in DETECTION_NAMES),
        help="Per-class weights for teacher-guided residual yaw refinement.",
    )
    p_student.add_argument(
        "--hm-positive-class-weights", nargs=len(DETECTION_NAMES), type=float, default=None,
        metavar=tuple(name.upper() for name in DETECTION_NAMES),
        help="Positive-center weights for the supervised Gaussian focal heatmap loss.",
    )
    p_student.add_argument("--student-train-views", choices=["front", "balanced"], default="front")
    p_student.add_argument("--student-val-views", choices=["front", "all"], default="front")

    p_eval = sub.add_parser("eval")
    add_common(p_eval)
    p_eval.add_argument("--model", choices=["teacher", "student", "fused_teacher"], required=True)
    p_eval.add_argument("--ckpt", required=True)
    p_eval.add_argument("--eval-set", default="val")
    p_eval.add_argument("--max-samples", type=int, default=0)
    p_eval.add_argument(
        "--subset-manifest", default="",
        help="JSON token manifest for a deterministic proxy subset; mutually exclusive with --max-samples.",
    )
    p_eval.add_argument("--score-thr", type=float, default=0.05)
    p_eval.add_argument(
        "--class-score-thrs", nargs=len(DETECTION_NAMES), type=float, default=None,
        metavar=tuple(name.upper() for name in DETECTION_NAMES),
        help="Per-class decoding thresholds in DETECTION_NAMES order; overrides --score-thr.",
    )
    p_eval.add_argument("--topk", type=int, default=300)
    p_eval.add_argument("--nms-iou", type=float, default=0.2)
    p_eval.add_argument("--merge-nms-iou", type=float, default=0.1)
    p_eval.add_argument("--max-boxes-per-sample", type=int, default=500)
    p_eval.add_argument("--student-eval-views", choices=["front", "all"], default="front")
    p_eval.add_argument("--student-fov-only", dest="student_fov_only", action="store_true", default=True)
    p_eval.add_argument("--no-student-fov-only", dest="student_fov_only", action="store_false")
    p_eval.add_argument(
        "--metric-scope", choices=["official_360", "front_fov", "front_90", "proxy_subset"], default="official_360",
    )
    p_eval.add_argument("--delete-result-json-after-eval", action="store_true")
    p_eval.add_argument("--skip-eval", action="store_true")
    return ap


def main():
    args = build_argparser().parse_args()
    if float(getattr(args, "reg_z_scale", 1.0)) < 0:
        raise ValueError("--reg-z-scale must be non-negative")
    if not 0.0 <= float(getattr(args, "kd_reg_conf_thr", 0.0)) <= 1.0:
        raise ValueError("--kd-reg-conf-thr must be in [0, 1]")
    if not 0.0 <= float(getattr(args, "kd_relation_residual_alpha", 0.25)) <= 1.0:
        raise ValueError("--kd-relation-residual-alpha must be in [0, 1]")
    if not 0.0 <= float(getattr(args, "kd_project_residual_alpha", 0.5)) <= 1.0:
        raise ValueError("--kd-project-residual-alpha must be in [0, 1]")
    if float(getattr(args, "kd_project_quality_margin", 0.0)) < 0.0:
        raise ValueError("--kd-project-quality-margin must be non-negative")
    if (
        getattr(args, "kd_project_quality_gate", False) and
        getattr(args, "teacher_type", "legacy") != "project_privileged"
    ):
        raise ValueError("--kd-project-quality-gate requires --teacher-type project_privileged")
    if (
        getattr(args, "kd_project_hybrid_routing", False) and
        not getattr(args, "kd_project_quality_gate", False)
    ):
        raise ValueError("--kd-project-hybrid-routing requires --kd-project-quality-gate")
    if getattr(args, "kd_project_independent_camera_ta", False):
        if not getattr(args, "kd_project_quality_gate", False):
            raise ValueError(
                "--kd-project-independent-camera-ta requires --kd-project-quality-gate"
            )
        if not getattr(args, "camera_teacher_ckpt", ""):
            raise ValueError(
                "--kd-project-independent-camera-ta requires --camera-teacher-ckpt"
            )
    if getattr(args, "kd_project_v45_instance_residual", False):
        if not getattr(args, "kd_project_independent_camera_ta", False):
            raise ValueError(
                "--kd-project-v45-instance-residual requires "
                "--kd-project-independent-camera-ta"
            )
    if getattr(args, "kd_project_residual_memory", False):
        if not getattr(args, "kd_project_v45_instance_residual", False):
            raise ValueError(
                "--kd-project-residual-memory requires "
                "--kd-project-v45-instance-residual"
            )
        if not getattr(args, "camera_privileged_memory_adapter", False):
            raise ValueError(
                "--kd-project-residual-memory requires "
                "--camera-privileged-memory-adapter"
            )
    if getattr(args, "kd_project_instance_relation", False):
        if not getattr(args, "kd_project_v45_instance_residual", False):
            raise ValueError(
                "--kd-project-instance-relation requires "
                "--kd-project-v45-instance-residual"
            )
        if float(getattr(args, "kd_relation_weight", 0.0)) <= 0.0:
            raise ValueError(
                "--kd-project-instance-relation requires --kd-relation-weight > 0"
            )
    if (
        getattr(args, "kd_gradient_balance_exact", False) and
        not getattr(args, "kd_gradient_balance", False)
    ):
        raise ValueError("--kd-gradient-balance-exact requires --kd-gradient-balance")
    if (
        getattr(args, "kd_gradient_project_heatmap_conflicts", False) and
        not getattr(args, "kd_gradient_balance", False)
    ):
        raise ValueError(
            "--kd-gradient-project-heatmap-conflicts requires --kd-gradient-balance"
        )
    if (
        getattr(args, "kd_gradient_anchor_image_feature", False) and
        not getattr(args, "kd_gradient_balance", False)
    ):
        raise ValueError(
            "--kd-gradient-anchor-image-feature requires --kd-gradient-balance"
        )
    if (
        getattr(args, "kd_per_branch_mema", False) and
        not getattr(args, "kd_gradient_balance", False)
    ):
        raise ValueError("--kd-per-branch-mema requires --kd-gradient-balance")
    if (
        getattr(args, "kd_per_branch_mema", False) and
        getattr(args, "kd_gradient_anchor_image_feature", False)
    ):
        raise ValueError(
            "--kd-per-branch-mema is mutually exclusive with "
            "--kd-gradient-anchor-image-feature"
        )
    if (
        getattr(args, "kd_natural_anchor_project_conflicts", False) and
        not getattr(args, "kd_per_branch_mema", False)
    ):
        raise ValueError(
            "--kd-natural-anchor-project-conflicts requires --kd-per-branch-mema"
        )
    if float(getattr(args, "kd_natural_anchor_max_gradient_ratio", 0.0)) < 0.0:
        raise ValueError("--kd-natural-anchor-max-gradient-ratio must be non-negative")
    if (
        getattr(args, "kd_gradient_skip_negative_cosine", False) and
        not getattr(args, "kd_gradient_balance", False)
    ):
        raise ValueError(
            "--kd-gradient-skip-negative-cosine requires --kd-gradient-balance"
        )
    if float(getattr(args, "kd_foreground_ellipse_scale", 1.0)) <= 0.0:
        raise ValueError("--kd-foreground-ellipse-scale must be positive")
    if float(getattr(args, "relative_depth_weight", 0.0)) < 0.0:
        raise ValueError("--relative-depth-weight must be non-negative")
    if (
        float(getattr(args, "kd_keypoint_feature_weight", 0.0)) > 0.0 and
        float(getattr(args, "kd_relation_weight", 0.0)) <= 0.0
    ):
        raise ValueError("--kd-keypoint-feature-weight currently requires --kd-relation-weight > 0")
    if any(int(value) <= 0 for value in getattr(args, "relative_depth_offsets", (1, 2, 4, 8))):
        raise ValueError("--relative-depth-offsets must be positive")
    if float(getattr(args, "kd_balance_target", 0.0)) < 0.0:
        raise ValueError("--kd-balance-target must be non-negative")
    for schedule_name in (
        "kd_balance_target_start", "kd_balance_target_middle", "kd_balance_target_end",
    ):
        schedule_value = float(getattr(args, schedule_name, -1.0))
        if schedule_value < -1.0:
            raise ValueError(f"--{schedule_name.replace('_', '-')} must be -1 or non-negative")
    if getattr(args, "kd_gradient_balance", False):
        grad_min = float(getattr(args, "kd_gradient_balance_min", 0.1))
        grad_max = float(getattr(args, "kd_gradient_balance_max", 10.0))
        if grad_min <= 0.0 or grad_max < grad_min:
            raise ValueError("invalid --kd-gradient-balance min/max bounds")
        share_names = (
            "kd_share_camera_hm", "kd_share_camera_reg",
            "kd_share_camera_feature", "kd_share_camera_depth",
            "kd_share_camera_attr", "kd_share_camera_keypoint_feature",
            "kd_share_lidar_relation",
            "kd_share_lidar_hm", "kd_share_lidar_reg",
            "kd_share_lidar_yaw",
        )
        shares = [float(getattr(args, name, 0.0)) for name in share_names]
        if any(value < 0.0 for value in shares) or sum(shares) <= 0.0:
            raise ValueError("gradient-balanced KD shares must be non-negative and sum positive")
    if (
        getattr(args, "cmd", "") == "train-student" and
        getattr(args, "teacher_type", "legacy") in {
            "aligned_student", "project_privileged",
            "centerpoint_relation", "camera_ta_relation",
        } and
        resolve_target_scope(args) != "student_fov"
    ):
        raise ValueError(
            "aligned camera and CenterPoint relation KD require "
            "--target-scope student_fov"
        )
    if getattr(args, "temporal_pre_encoder", False) and not getattr(args, "temporal_bev", False):
        raise ValueError("--temporal-pre-encoder requires --temporal-bev")
    if getattr(args, "temporal_stereo_depth", False) and not getattr(args, "temporal_bev", False):
        raise ValueError("--temporal-stereo-depth requires --temporal-bev")
    if (
        getattr(args, "temporal_stereo_dynamic_confidence", False) and
        not getattr(args, "temporal_stereo_depth", False)
    ):
        raise ValueError(
            "--temporal-stereo-dynamic-confidence requires --temporal-stereo-depth"
        )
    if getattr(args, "temporal_spatial_fusion", False) and not getattr(args, "temporal_bev", False):
        raise ValueError("--temporal-spatial-fusion requires --temporal-bev")
    if getattr(args, "velocity_displacement_target", False) and not getattr(args, "temporal_bev", False):
        raise ValueError("--velocity-displacement-target requires --temporal-bev")
    if getattr(args, "temporal_correlation_motion", False) and not getattr(args, "temporal_bev", False):
        raise ValueError("--temporal-correlation-motion requires --temporal-bev")
    if getattr(args, "temporal_correlation_motion", False) and not getattr(args, "class_aware_motion_head", False):
        raise ValueError("--temporal-correlation-motion requires --class-aware-motion-head")
    if getattr(args, "object_velocity_refiner", False) and not getattr(args, "temporal_bev", False):
        raise ValueError("--object-velocity-refiner requires --temporal-bev")
    if getattr(args, "object_velocity_refiner", False) and not getattr(args, "class_aware_motion_head", False):
        raise ValueError("--object-velocity-refiner requires --class-aware-motion-head")
    if getattr(args, "freeze_base_for_velocity_refiner", False) and not getattr(args, "object_velocity_refiner", False):
        raise ValueError("--freeze-base-for-velocity-refiner requires --object-velocity-refiner")
    if getattr(args, "class_attribute_velocity_calibrator", False) and not getattr(args, "class_aware_motion_head", False):
        raise ValueError("--class-attribute-velocity-calibrator requires --class-aware-motion-head")
    if getattr(args, "class_attribute_velocity_calibrator", False) and not getattr(args, "attribute_head", False):
        raise ValueError("--class-attribute-velocity-calibrator requires --attribute-head")
    if getattr(args, "freeze_base_for_velocity_calibrator", False) and not getattr(args, "class_attribute_velocity_calibrator", False):
        raise ValueError(
            "--freeze-base-for-velocity-calibrator requires "
            "--class-attribute-velocity-calibrator"
        )
    if getattr(args, "velocity_calibrator_scale_only", False) and not getattr(args, "freeze_base_for_velocity_calibrator", False):
        raise ValueError(
            "--velocity-calibrator-scale-only requires "
            "--freeze-base-for-velocity-calibrator"
        )
    if getattr(args, "velocity_calibrator_bypass_attribute_gate", False) and not getattr(
        args, "class_attribute_velocity_calibrator", False,
    ):
        raise ValueError(
            "--velocity-calibrator-bypass-attribute-gate requires "
            "--class-attribute-velocity-calibrator"
        )
    if float(getattr(args, "velocity_calibrator_max_residual", 0.75)) <= 0:
        raise ValueError("--velocity-calibrator-max-residual must be positive")
    if getattr(args, "velocity_target_view_classes", None) and getattr(
        args, "student_train_views", "front",
    ) != "balanced":
        raise ValueError("--velocity-target-view-classes requires --student-train-views balanced")
    state_balanced_classes = set(
        getattr(args, "velocity_target_state_balanced_classes", None) or ()
    )
    target_view_classes = set(getattr(args, "velocity_target_view_classes", None) or ())
    if not state_balanced_classes.issubset(target_view_classes):
        raise ValueError(
            "--velocity-target-state-balanced-classes must be a subset of "
            "--velocity-target-view-classes"
        )
    if float(getattr(args, "velocity_moving_boost", 1.0)) <= 0:
        raise ValueError("--velocity-moving-boost must be positive")
    if float(getattr(args, "azimuth_bin_weight", 0.0)) > 0 and not (
        getattr(args, "azimuth_multibin", False) or
        getattr(args, "residual_azimuth_multibin", False) or
        getattr(args, "residual_yaw_vector", False)
    ):
        raise ValueError(
            "--azimuth-bin-weight requires --azimuth-multibin or "
            "--residual-azimuth-multibin"
        )
    if getattr(args, "residual_azimuth_multibin", False) and not getattr(args, "class_aware_motion_head", False):
        raise ValueError("--residual-azimuth-multibin requires --class-aware-motion-head")
    if getattr(args, "residual_yaw_vector", False) and not getattr(args, "class_aware_motion_head", False):
        raise ValueError("--residual-yaw-vector requires --class-aware-motion-head")
    if getattr(args, "residual_yaw_vector", False) and getattr(args, "residual_azimuth_multibin", False):
        raise ValueError("--residual-yaw-vector and --residual-azimuth-multibin are mutually exclusive")
    if getattr(args, "freeze_base_for_yaw_refiner", False) and not (
        getattr(args, "residual_azimuth_multibin", False) or
        getattr(args, "residual_yaw_vector", False)
    ):
        raise ValueError(
            "--freeze-base-for-yaw-refiner requires a residual yaw head"
        )
    if getattr(args, "freeze_base_for_center_refiner", False) and not getattr(
        args, "class_aware_center_head", False,
    ):
        raise ValueError(
            "--freeze-base-for-center-refiner requires --class-aware-center-head"
        )
    if float(getattr(args, "image_yaw_flip_weight", 0.0)) > 0 and not (
        getattr(args, "image_yaw_flip_head", False) or
        getattr(args, "image_object_geometry_head", False)
    ):
        raise ValueError(
            "--image-yaw-flip-weight requires --image-yaw-flip-head or "
            "--image-object-geometry-head"
        )
    if getattr(args, "freeze_base_for_image_yaw_flip", False) and not getattr(
        args, "image_yaw_flip_head", False,
    ):
        raise ValueError(
            "--freeze-base-for-image-yaw-flip requires --image-yaw-flip-head"
        )
    if not 0.0 <= float(getattr(args, "image_yaw_flip_threshold", 0.5)) <= 1.0:
        raise ValueError("--image-yaw-flip-threshold must be in [0, 1]")
    if float(getattr(args, "image_radial_center_max_residual", 2.0)) <= 0:
        raise ValueError("--image-radial-center-max-residual must be positive")
    if not 0.0 <= float(getattr(args, "depth_distribution_alpha", 0.0)) <= 1.0:
        raise ValueError("--depth-distribution-alpha must be in [0, 1]")
    if int(getattr(args, "depth_distribution_window_radius", 1)) < 0:
        raise ValueError("--depth-distribution-window-radius must be non-negative")
    if float(getattr(args, "depth_distribution_max_shift", 2.0)) <= 0:
        raise ValueError("--depth-distribution-max-shift must be positive")
    if float(getattr(args, "image_radial_center_weight", 0.0)) > 0 and not (
        getattr(args, "image_radial_center_head", False) or
        getattr(args, "image_object_geometry_head", False)
    ):
        raise ValueError(
            "--image-radial-center-weight requires --image-radial-center-head or "
            "--image-object-geometry-head"
        )
    if getattr(args, "freeze_base_for_image_radial_center", False) and not getattr(
        args, "image_radial_center_head", False,
    ):
        raise ValueError(
            "--freeze-base-for-image-radial-center requires "
            "--image-radial-center-head"
        )
    if getattr(args, "freeze_base_for_image_object_geometry", False) and not getattr(
        args, "image_object_geometry_head", False,
    ):
        raise ValueError(
            "--freeze-base-for-image-object-geometry requires "
            "--image-object-geometry-head"
        )
    if getattr(args, "image_object_geometry_head", False) and (
        getattr(args, "image_yaw_flip_head", False) or
        getattr(args, "image_radial_center_head", False)
    ):
        raise ValueError(
            "--image-object-geometry-head is mutually exclusive with the V24/V25 heads"
        )
    if getattr(args, "freeze_base_for_temporal_stereo", False) and not getattr(
        args, "temporal_stereo_depth", False,
    ):
        raise ValueError(
            "--freeze-base-for-temporal-stereo requires --temporal-stereo-depth"
        )
    if getattr(args, "freeze_base_for_privileged_teacher", False) and not getattr(
        args, "privileged_depth_teacher", False,
    ):
        raise ValueError(
            "--freeze-base-for-privileged-teacher requires --privileged-depth-teacher"
        )
    if getattr(args, "freeze_base_for_privileged_memory_adapter", False) and not getattr(
        args, "camera_privileged_memory_adapter", False,
    ):
        raise ValueError(
            "--freeze-base-for-privileged-memory-adapter requires "
            "--camera-privileged-memory-adapter"
        )
    decay_start = int(getattr(args, "kd_early_decay_start_epoch", 0))
    decay_end = int(getattr(args, "kd_early_decay_end_epoch", 0))
    if (decay_start == 0) != (decay_end == 0) or (decay_start and decay_end < decay_start):
        raise ValueError("early KD decay requires 0 < start <= end")
    if getattr(args, "freeze_privileged_memory_after_resume", False):
        if not getattr(args, "resume_student", ""):
            raise ValueError("--freeze-privileged-memory-after-resume requires --resume-student")
        if not getattr(args, "camera_privileged_memory_adapter", False):
            raise ValueError(
                "--freeze-privileged-memory-after-resume requires "
                "--camera-privileged-memory-adapter"
            )
    if getattr(args, "attribute_velocity_gate", False) and not getattr(args, "attribute_head", False):
        raise ValueError("--attribute-velocity-gate requires --attribute-head")
    if getattr(args, "azimuth_multibin", False) and any(
        float(getattr(args, name, 0.0)) > 0
        for name in ("kd_reg_weight", "pseudo_reg_weight", "hop_reg_weight")
    ):
        raise ValueError(
            "--azimuth-multibin currently requires KD/pseudo/HoP regression weights to be zero"
        )
    if (
        float(getattr(args, "hop_hm_weight", 0.0)) + float(getattr(args, "hop_reg_weight", 0.0)) > 0
        and not getattr(args, "temporal_bev", False)
    ):
        raise ValueError("HoP auxiliary losses require --temporal-bev")
    torch.backends.cudnn.benchmark = True
    cv2.setNumThreads(0)
    set_global_seed(args.seed)
    mod = load_base_module()
    set_global_seed(args.seed)
    mod.DATA_ROOT = args.data_root
    mod.NUSC_VERSION = args.version
    configure_bev(mod, args)
    configure_fast_runtime(mod, args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[ENV] cmd={args.cmd} data_root={args.data_root} version={args.version} device={device}")

    if args.cmd == "train-teacher":
        train_loader, val_loader = make_loaders(mod, args)
        model = OfficialTeacher(mod)
        train_loop(mod, args, model, train_loader, val_loader, teacher=None, name="teacher")
        return

    if args.cmd == "train-fused-teacher":
        train_loader, val_loader = make_loaders(mod, args)
        model = SixViewFusedTeacher(
            mod,
            pretrained_backbone=not args.no_pretrained_backbone,
            use_kd_adapter=False,
            backbone_variant=args.student_backbone,
            camera_aware_depth=args.camera_aware_depth,
            bev_norm=args.bev_norm,
            attribute_head=args.attribute_head,
            temporal_bev=False,
        )
        compact_student_backbone(model, enabled=args.compact_backbone)
        instantiate_student_lazy(mod, model, device)
        init_state = torch.load(args.student_init, map_location="cpu")
        if args.partial_student_init:
            load_model_state_partial(model, init_state)
        else:
            load_model_state_compatible(model, init_state, strict=True)
        print(f"[INIT] fused teacher from single-view student: {args.student_init}")
        if args.freeze_image_encoder:
            model.freeze_image_encoder()
            trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
            print(f"[FREEZE] fused teacher image backbone and depth head frozen; trainable_params={trainable}")
        train_loop(mod, args, model, train_loader, val_loader, teacher=None, name="teacher")
        return

    if args.cmd == "train-student":
        apply_student_exp_preset(args)
        if args.teacher_type == "legacy" and args.student_train_views == "balanced" and (
            args.kd_hm_weight + args.kd_reg_weight + args.kd_feat_weight +
            args.kd_depth_weight + float(getattr(args, "kd_relation_weight", 0.0)) +
            float(getattr(args, "kd_keypoint_feature_weight", 0.0)) > 0
        ):
            print(
                "[WARN] balanced-view training with the legacy dense BEV teacher uses slightly different "
                "camera timestamps; prefer A0 or cached global pseudo labels for protocol experiments."
            )
        print(
            f"[EXP] {args.exp} kd_hm={args.kd_hm_weight} kd_reg={args.kd_reg_weight} "
            f"kd_feat={args.kd_feat_weight} kd_depth={args.kd_depth_weight} "
            f"kd_attr={getattr(args, 'kd_attr_weight', 0.0)} "
            f"kd_relation={getattr(args, 'kd_relation_weight', 0.0)} "
            f"kd_keypoint_feature={getattr(args, 'kd_keypoint_feature_weight', 0.0)} "
            f"relative_depth={getattr(args, 'relative_depth_weight', 0.0)} "
            f"pseudo_hm={args.pseudo_hm_weight} pseudo_reg={args.pseudo_reg_weight} "
            f"pseudo_yaw_refiner={args.pseudo_yaw_refiner_weight} "
            f"hop_hm={args.hop_hm_weight} hop_reg={args.hop_reg_weight}"
        )
        if (
            args.pseudo_hm_weight + args.pseudo_reg_weight +
            args.pseudo_yaw_refiner_weight > 0 and not args.pseudo_labels
        ):
            raise ValueError("--pseudo-labels is required when a pseudo-label loss weight is enabled")
        if args.pseudo_yaw_refiner_weight > 0 and not args.residual_yaw_vector:
            raise ValueError("--pseudo-yaw-refiner-weight requires --residual-yaw-vector")
        train_loader, val_loader = make_loaders(mod, args)
        teacher = None
        camera_teacher = None
        if (
            args.kd_hm_weight + args.kd_reg_weight + args.kd_feat_weight +
            args.kd_depth_weight + float(getattr(args, "kd_attr_weight", 0.0)) +
            float(getattr(args, "kd_relation_weight", 0.0)) +
            float(getattr(args, "kd_keypoint_feature_weight", 0.0)) > 0
        ):
            if not args.teacher_ckpt:
                raise ValueError("--teacher-ckpt is required when any legacy KD weight is enabled")
            if args.teacher_type == "fused_student":
                teacher = SixViewFusedTeacher(
                    mod, pretrained_backbone=False, use_kd_adapter=False,
                    backbone_variant=args.student_backbone,
                    camera_aware_depth=args.camera_aware_depth,
                    bev_norm=args.bev_norm,
                    attribute_head=args.attribute_head,
                    temporal_bev=False,
                    temporal_pre_encoder=False,
                )
                compact_student_backbone(teacher, enabled=args.compact_backbone)
                instantiate_student_lazy(mod, teacher, device)
            elif args.teacher_type in {"aligned_student", "project_privileged"}:
                teacher_args = args
                if args.teacher_type == "project_privileged":
                    teacher_args = argparse.Namespace(**vars(args))
                    teacher_args.privileged_depth_teacher = True
                    teacher_args.camera_privileged_memory_adapter = False
                teacher = build_official_student_from_args(
                    mod, teacher_args, pretrained_backbone=False, use_kd_adapter=False,
                )
                compact_student_backbone(teacher, enabled=args.compact_backbone)
                instantiate_student_lazy(mod, teacher, device)
                if args.teacher_type == "project_privileged":
                    print(
                        "[PROJECT_TEACHER] project-native selected-view LiDAR "
                        "sparse-depth teacher; no public teacher architecture or weights"
                    )
            elif args.teacher_type == "camera_ta_relation":
                if float(getattr(args, "kd_relation_weight", 0.0)) <= 0:
                    raise ValueError(
                        "camera_ta_relation requires --kd-relation-weight > 0"
                    )
                if not args.lidar_relation_cache:
                    raise ValueError(
                        "camera_ta_relation requires --lidar-relation-cache"
                    )
                if not Path(args.lidar_relation_cache).is_file():
                    raise FileNotFoundError(
                        f"LiDAR relation cache not found: {args.lidar_relation_cache}"
                    )
                teacher = build_official_student_from_args(
                    mod, args, pretrained_backbone=False, use_kd_adapter=False,
                )
                compact_student_backbone(teacher, enabled=args.compact_backbone)
                instantiate_student_lazy(mod, teacher, device)
                print(
                    "[CAMERA_TA] aligned camera assistant plus cached LiDAR "
                    f"relation residual from {args.lidar_relation_cache}"
                )
            elif args.teacher_type == "centerpoint_relation":
                if any(
                    float(value) > 0 for value in (
                        args.kd_hm_weight, args.kd_reg_weight,
                        args.kd_feat_weight, args.kd_depth_weight,
                        getattr(args, "kd_attr_weight", 0.0),
                    )
                ):
                    raise ValueError(
                        "centerpoint_relation currently supports only "
                        "--kd-relation-weight; set dense KD weights to zero"
                    )
                if args.lidar_relation_cache:
                    if not Path(args.lidar_relation_cache).is_file():
                        raise FileNotFoundError(
                            f"LiDAR relation cache not found: {args.lidar_relation_cache}"
                        )
                    teacher = nn.Identity()
                    print(f"[RELATION_CACHE] training from {args.lidar_relation_cache}")
                else:
                    teacher = CenterPointRelationTeacher(
                        args.centerpoint_config, args.teacher_ckpt,
                    )
            else:
                teacher = OfficialTeacher(mod)
            if args.teacher_type != "centerpoint_relation":
                load_model_state_compatible(
                    teacher, torch.load(args.teacher_ckpt, map_location="cpu"), strict=True,
                )
            if getattr(args, "kd_project_independent_camera_ta", False):
                camera_teacher_args = argparse.Namespace(**vars(args))
                camera_teacher_args.privileged_depth_teacher = False
                camera_teacher_args.camera_privileged_memory_adapter = False
                camera_teacher = build_official_student_from_args(
                    mod, camera_teacher_args,
                    pretrained_backbone=False, use_kd_adapter=False,
                )
                compact_student_backbone(
                    camera_teacher, enabled=args.compact_backbone,
                )
                instantiate_student_lazy(mod, camera_teacher, device)
                load_model_state_compatible(
                    camera_teacher,
                    torch.load(args.camera_teacher_ckpt, map_location="cpu"),
                    strict=True,
                )
                print(
                    "[INDEPENDENT_CAMERA_TA] loaded strong camera-only assistant "
                    f"from {args.camera_teacher_ckpt}; privileged teacher bypass is disabled"
                )
        else:
            print("[EXP] no KD weights enabled; teacher checkpoint will not be loaded for this student run.")
        set_global_seed(args.seed)
        student = build_official_student_from_args(
            mod, args,
            pretrained_backbone=not args.no_pretrained_backbone,
            use_kd_adapter=(
                args.kd_feat_weight > 0 and
                args.teacher_type not in {
                    "aligned_student", "project_privileged", "camera_ta_relation",
                }
            ),
        )
        compact_student_backbone(student, enabled=args.compact_backbone)
        instantiate_student_lazy(mod, student, device)
        if args.student_init:
            init_path = Path(args.student_init)
            if init_path.exists():
                init_state = torch.load(init_path, map_location="cpu")
                if args.partial_student_init:
                    load_model_state_partial(student, init_state)
                    print(f"[INIT] partially loaded student init: {init_path}")
                else:
                    load_model_state_compatible(student, init_state, strict=True)
                    print(f"[INIT] loaded common student init: {init_path}")
            else:
                init_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "model_state_dict": student.state_dict(),
                    "seed": args.seed,
                    "student_img": [mod.IMG_W, mod.IMG_H],
                    "bev_grid": [mod.BEV_H, mod.BEV_W],
                }, init_path)
                print(f"[INIT] saved common student init: {init_path}")
        if args.freeze_backbone_stages:
            freeze_method = getattr(student.backbone, "freeze_stages", None)
            if freeze_method is None:
                raise ValueError(
                    "--freeze-backbone-stages requires a backbone with freeze_stages support"
                )
            freeze_method(args.freeze_backbone_stages)
            backbone_trainable = sum(
                parameter.numel() for parameter in student.backbone.parameters()
                if parameter.requires_grad
            )
            print(
                f"[FREEZE] backbone early stages={args.freeze_backbone_stages} "
                f"trainable_backbone_params={backbone_trainable}"
            )
        if args.freeze_base_for_motion:
            if student.class_motion_head is None:
                raise ValueError("--freeze-base-for-motion requires --class-aware-motion-head")
            for parameter in student.parameters():
                parameter.requires_grad_(False)
            for parameter in student.class_motion_head.parameters():
                parameter.requires_grad_(True)
            if student.temporal_correlation_motion_head is not None:
                for parameter in student.temporal_correlation_motion_head.parameters():
                    parameter.requires_grad_(True)
            trainable = sum(parameter.numel() for parameter in student.parameters() if parameter.requires_grad)
            print(f"[FREEZE] base detector frozen; motion-only trainable_params={trainable}")
        if args.freeze_base_for_velocity_refiner:
            if student.object_velocity_refiner is None:
                raise ValueError(
                    "--freeze-base-for-velocity-refiner requires --object-velocity-refiner"
                )
            for parameter in student.parameters():
                parameter.requires_grad_(False)
            for parameter in student.object_velocity_refiner.parameters():
                parameter.requires_grad_(True)
            trainable = sum(parameter.numel() for parameter in student.parameters() if parameter.requires_grad)
            print(
                "[FREEZE] S1 detector frozen; object velocity refiner only "
                f"trainable_params={trainable}"
            )
        if args.freeze_base_for_velocity_calibrator:
            if student.class_attribute_velocity_calibrator is None:
                raise ValueError(
                    "--freeze-base-for-velocity-calibrator requires "
                    "--class-attribute-velocity-calibrator"
                )
            for parameter in student.parameters():
                parameter.requires_grad_(False)
            for parameter in student.class_attribute_velocity_calibrator.parameters():
                parameter.requires_grad_(True)
            if args.velocity_calibrator_scale_only:
                for parameter in student.class_attribute_velocity_calibrator.local_residual.parameters():
                    parameter.requires_grad_(False)
            trainable = sum(
                parameter.numel() for parameter in student.parameters()
                if parameter.requires_grad
            )
            print(
                "[FREEZE] S1 detector frozen; class/attribute velocity calibrator only "
                f"trainable_params={trainable} "
                f"scale_only={bool(args.velocity_calibrator_scale_only)}"
            )
        if args.freeze_base_for_yaw_refiner:
            yaw_refiner = (
                student.residual_yaw_vector_head or
                student.residual_azimuth_multibin_head
            )
            if yaw_refiner is None:
                raise ValueError(
                    "--freeze-base-for-yaw-refiner requires a residual yaw head"
                )
            for parameter in student.parameters():
                parameter.requires_grad_(False)
            for parameter in yaw_refiner.parameters():
                parameter.requires_grad_(True)
            trainable = sum(parameter.numel() for parameter in student.parameters() if parameter.requires_grad)
            print(
                "[FREEZE] S1 detector frozen; residual azimuth MultiBin only "
                f"trainable_params={trainable}"
            )
        if args.freeze_base_for_center_refiner:
            if student.class_center_head is None:
                raise ValueError(
                    "--freeze-base-for-center-refiner requires --class-aware-center-head"
                )
            for parameter in student.parameters():
                parameter.requires_grad_(False)
            for parameter in student.class_center_head.parameters():
                parameter.requires_grad_(True)
            trainable = sum(
                parameter.numel() for parameter in student.parameters()
                if parameter.requires_grad
            )
            print(
                "[FREEZE] S1 detector frozen; class-aware center residual only "
                f"trainable_params={trainable}"
            )
        if args.freeze_base_for_depth_distillation:
            for parameter in student.parameters():
                parameter.requires_grad_(False)
            for parameter in student.depth_head.parameters():
                parameter.requires_grad_(True)
            trainable = sum(
                parameter.numel() for parameter in student.parameters()
                if parameter.requires_grad
            )
            print(
                "[FREEZE] V31 detector frozen; camera depth head only "
                f"trainable_params={trainable}"
            )
        if args.freeze_base_for_image_yaw_flip:
            if student.image_yaw_flip_head is None:
                raise ValueError(
                    "--freeze-base-for-image-yaw-flip requires --image-yaw-flip-head"
                )
            for parameter in student.parameters():
                parameter.requires_grad_(False)
            for parameter in student.image_yaw_flip_head.parameters():
                parameter.requires_grad_(True)
            trainable = sum(
                parameter.numel() for parameter in student.parameters()
                if parameter.requires_grad
            )
            print(
                "[FREEZE] S1 detector frozen; image yaw flip classifier only "
                f"trainable_params={trainable}"
            )
        if args.freeze_base_for_image_radial_center:
            if student.image_radial_center_head is None:
                raise ValueError(
                    "--freeze-base-for-image-radial-center requires "
                    "--image-radial-center-head"
                )
            for parameter in student.parameters():
                parameter.requires_grad_(False)
            for parameter in student.image_radial_center_head.parameters():
                parameter.requires_grad_(True)
            trainable = sum(
                parameter.numel() for parameter in student.parameters()
                if parameter.requires_grad
            )
            print(
                "[FREEZE] S1 detector frozen; image radial center head only "
                f"trainable_params={trainable}"
            )
        if args.freeze_base_for_image_object_geometry:
            if student.image_object_geometry_head is None:
                raise ValueError(
                    "--freeze-base-for-image-object-geometry requires "
                    "--image-object-geometry-head"
                )
            for parameter in student.parameters():
                parameter.requires_grad_(False)
            for parameter in student.image_object_geometry_head.parameters():
                parameter.requires_grad_(True)
            trainable = sum(
                parameter.numel() for parameter in student.parameters()
                if parameter.requires_grad
            )
            print(
                "[FREEZE] S1 detector frozen; joint image object geometry head only "
                f"trainable_params={trainable}"
            )
        if args.freeze_base_for_temporal_stereo:
            if student.temporal_stereo_depth_matcher is None:
                raise ValueError(
                    "--freeze-base-for-temporal-stereo requires "
                    "--temporal-stereo-depth"
                )
            for parameter in student.parameters():
                parameter.requires_grad_(False)
            for parameter in student.temporal_stereo_depth_matcher.parameters():
                parameter.requires_grad_(True)
            trainable = sum(
                parameter.numel() for parameter in student.parameters()
                if parameter.requires_grad
            )
            print(
                "[FREEZE] S1 detector frozen; temporal stereo matcher only "
                f"trainable_params={trainable}"
            )
        if args.freeze_base_for_privileged_teacher:
            if (
                student.privileged_depth_completion is None or
                student.privileged_bev_refiner is None
            ):
                raise ValueError(
                    "--freeze-base-for-privileged-teacher requires "
                    "--privileged-depth-teacher"
                )
            for parameter in student.parameters():
                parameter.requires_grad_(False)
            for module in (
                student.privileged_depth_completion,
                student.privileged_bev_refiner,
            ):
                for parameter in module.parameters():
                    parameter.requires_grad_(True)
            trainable = sum(
                parameter.numel() for parameter in student.parameters()
                if parameter.requires_grad
            )
            print(
                "[FREEZE] proven camera detector frozen; project-native privileged "
                f"teacher modules only trainable_params={trainable}"
            )
        if args.freeze_base_for_privileged_memory_adapter:
            if student.privileged_residual_adapter is None:
                raise ValueError(
                    "--freeze-base-for-privileged-memory-adapter requires "
                    "--camera-privileged-memory-adapter"
                )
            for parameter in student.parameters():
                parameter.requires_grad_(False)
            for parameter in student.privileged_residual_adapter.parameters():
                parameter.requires_grad_(True)
            trainable = sum(
                parameter.numel() for parameter in student.parameters()
                if parameter.requires_grad
            )
            print(
                "[FREEZE] audited V31 frozen; camera privileged residual memory "
                f"only trainable_params={trainable}"
            )
        if args.resume_student:
            print(f"[RESUME] will restore student in train_loop: {args.resume_student}")
        train_loop(
            mod, args, student, train_loader, val_loader,
            teacher=teacher, name="student", camera_teacher=camera_teacher,
        )
        return

    if args.cmd == "eval":
        if args.student_eval_cameras and args.student_eval_views != "all":
            raise ValueError(
                "--student-eval-cameras requires --student-eval-views all"
            )
        if args.subset_manifest and args.max_samples > 0:
            raise ValueError("--subset-manifest and --max-samples are mutually exclusive")
        if args.max_samples > 0 and not args.skip_eval and args.metric_scope != "proxy_subset":
            raise ValueError(
                "--max-samples requires --skip-eval or --metric-scope proxy_subset; official NDS needs the full split."
            )
        if args.metric_scope == "proxy_subset" and args.max_samples <= 0 and not args.subset_manifest:
            raise ValueError("--metric-scope proxy_subset requires --max-samples or --subset-manifest")
        if args.subset_manifest and args.metric_scope != "proxy_subset":
            raise ValueError("--subset-manifest requires --metric-scope proxy_subset")
        tt, ts = make_transforms(mod)
        dataset = NuScenesOfficialDetectionDataset(
            mod, args.data_root, args.version, args.eval_set, tt, ts,
            max_samples=args.max_samples, build_depth=False,
            need_teacher_inputs=args.model in {"teacher", "fused_teacher"},
            need_student_inputs=args.model == "student",
            need_teacher_depth=False,
            need_student_depth=(
                args.model == "student" and args.privileged_depth_teacher
            ),
            need_temporal_inputs=args.model == "student" and args.temporal_bev,
            build_detection_targets=False,
            target_scope=resolve_target_scope(args),
            student_view_mode=args.student_eval_views if args.model == "student" else "front",
            student_camera=args.student_camera,
            subset_manifest=args.subset_manifest,
            student_cameras=args.student_eval_cameras,
        )
        if args.model == "teacher":
            model = OfficialTeacher(mod)
        elif args.model == "fused_teacher":
            model = SixViewFusedTeacher(
                mod, pretrained_backbone=False, use_kd_adapter=False,
                backbone_variant=args.student_backbone,
                camera_aware_depth=args.camera_aware_depth,
                bev_norm=args.bev_norm,
                attribute_head=args.attribute_head,
                temporal_bev=False,
                temporal_pre_encoder=False,
            )
            compact_student_backbone(model, enabled=args.compact_backbone)
            instantiate_student_lazy(mod, model, device)
        else:
            model = OfficialStudent(
                mod, pretrained_backbone=False, use_kd_adapter=False,
                backbone_variant=args.student_backbone,
                camera_aware_depth=args.camera_aware_depth,
                bev_norm=args.bev_norm,
                attribute_head=args.attribute_head,
                temporal_bev=args.temporal_bev,
                temporal_stereo_depth=args.temporal_stereo_depth,
                temporal_stereo_dynamic_confidence=args.temporal_stereo_dynamic_confidence,
                temporal_pre_encoder=args.temporal_pre_encoder,
                temporal_delta_limit=args.temporal_delta_limit,
                class_aware_motion=args.class_aware_motion_head,
                temporal_spatial_fusion=args.temporal_spatial_fusion,
                azimuth_multibin=args.azimuth_multibin,
                class_aware_size=args.class_aware_size_head,
                class_aware_center=args.class_aware_center_head,
                center_refiner_active_classes=args.center_refiner_active_classes,
                image_yaw_flip=args.image_yaw_flip_head,
                image_radial_center=args.image_radial_center_head,
                image_radial_center_max_residual=args.image_radial_center_max_residual,
                image_object_geometry=args.image_object_geometry_head,
                temporal_correlation_motion=args.temporal_correlation_motion,
                object_velocity_refiner=args.object_velocity_refiner,
                residual_azimuth_multibin=args.residual_azimuth_multibin,
                residual_yaw_vector=args.residual_yaw_vector,
                residual_yaw_active_classes=args.residual_yaw_active_classes,
                class_attribute_velocity_calibrator=args.class_attribute_velocity_calibrator,
                velocity_calibrator_motorcycle=args.velocity_calibrator_motorcycle,
                velocity_calibrator_active_classes=args.velocity_calibrator_active_classes,
                velocity_calibrator_bypass_attribute_gate=args.velocity_calibrator_bypass_attribute_gate,
                velocity_calibrator_max_residual=args.velocity_calibrator_max_residual,
                velocity_calibrator_initial_scales=args.velocity_calibrator_initial_scales,
                privileged_depth_teacher=args.privileged_depth_teacher,
                camera_privileged_memory_adapter=args.camera_privileged_memory_adapter,
                privileged_memory_slots=args.privileged_memory_slots,
                privileged_memory_latent_dim=args.privileged_memory_latent_dim,
            )
            compact_student_backbone(model, enabled=args.compact_backbone)
            instantiate_student_lazy(mod, model, device)
        repaired_bn = prepare_model_for_eval(model, torch.load(args.ckpt, map_location="cpu"))
        if repaired_bn:
            print(f"[BN_SANITIZE] repaired {repaired_bn} non-finite running-stat values for evaluation")
        export_and_eval(mod, args, model, dataset)


if __name__ == "__main__":
    main()
