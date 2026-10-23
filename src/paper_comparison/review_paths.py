"""Portable locations for the unchanged paper comparison implementation."""
import os
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ.get('ACTIVECAM_OUTPUT_ROOT', BASE / 'runs/dual_comparison'))
CACHE = Path(os.environ.get('ACTIVECAM_CACHE_ROOT', BASE / 'runs/fullval_camera_cache_fp32_batch1_formal'))
CHECKPOINT = Path(os.environ.get('ACTIVECAM_CHECKPOINT', BASE / 'checkpoints/v31_student_inference_state_fp32.pth'))
DATA_ROOT = Path(os.environ.get('ACTIVECAM_DATA_ROOT', BASE / 'data/nuscenes'))
EXTERNAL = Path(os.environ.get('ACTIVECAM_EXTERNAL_ROOT', BASE / 'external'))
