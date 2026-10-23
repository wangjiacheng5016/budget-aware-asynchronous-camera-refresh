# Budget-aware asynchronous multi-camera refresh

Original research code for **Budget-Aware Multi-Camera 3D Object Detection via Asynchronous Refresh and Motion-Compensated Temporal Reuse**. This public repository provides the camera scheduler, sparse temporal memory, motion compensation and fusion, detector implementation, evaluation entry points, adapted tracker comparisons, and hardware benchmark scripts for manuscript review. Browsing, cloning and downloading this repository require no account or password.

## What is included

| Location | Purpose |
| --- | --- |
| `src/activecam_bev/activecam_core.py` | Budgeted refresh selection, detection propagation and temporal reuse |
| `src/activecam_bev/router_core.py`, `router_model.py` | Memory and routing policies |
| `src/legacy_v31/` | Detector/model definitions, native export and inference dependencies |
| `src/paper_comparison/` | Dual-frame shared-checkpoint replay, image ancestry checks, AB3DMOT/SimpleTrack adaptations |
| `src/activecam_bev/eval_*.py` | Fixed-view, routing, staleness, budget, robustness and history-protocol evaluations |
| `src/activecam_bev/scene_paired_bootstrap.py` | Paired scene-level uncertainty analysis |
| `benchmarks/jetson/` | Separate TensorRT timing and CPU scheduling/memory benchmarks |
| `tests/` | Small synthetic tests requiring no dataset or checkpoint |
| `provenance/source_hashes.json` | SHA-256 of original experiment sources and released versions |

Only code, configuration, tests and documentation are distributed. **nuScenes data, trained checkpoints, detector caches, experimental predictions and manuscript files are not included.** Data must be obtained under the [nuScenes access terms](https://www.nuscenes.org/nuscenes). This repository itself has no access restrictions; the dataset is a separate prerequisite.

The exact paper checkpoint is identified below but is not downloadable from this code-only release. Consequently, the synthetic tests and implementation can be inspected immediately; **exact numerical reproduction of the reported detector results requires that checkpoint and the licensed dataset**. Training utilities are retained because the model and decoder depend on them; their presence does not establish a complete, independently verified recipe for recreating the original trained weights.

## Installation

The primary implementation was executed on Linux with Python 3.12, PyTorch 2.3.0+cu121, torchvision 0.18.0+cu121 and NumPy 1.26.4. A CUDA GPU is needed for native detector cache generation. Cache replay uses CPU; its parallel mode uses Linux/POSIX `fork`. Loading the full nuScenes metadata and all detector caches requires substantial host RAM; a 2 GB instance is insufficient for this full-data path.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
export PYTHONPATH="$PWD/src:$PWD/src/activecam_bev:$PWD/src/legacy_v31:$PWD/src/paper_comparison${PYTHONPATH:+:$PYTHONPATH}"
python tools/fetch_baselines.py
```

Baseline fetching uses public upstream repositories at the exact commits in [THIRD_PARTY.md](THIRD_PARTY.md); it does not redistribute their source inside this repository. Both checkouts are required by the comparison's source manifest, including for the no-reuse and motion conditions. TensorRT and CUDA Python bindings are optional platform-specific dependencies used only by the Jetson scripts. MMEngine/MMDetection3D are optional teacher-training dependencies, unnecessary for the frozen student inference and mechanism comparison.

## Tests without research data

After setting `PYTHONPATH` as above:

```bash
python -m unittest discover -s tests -v
python -m compileall -q src tools benchmarks tests
```

These are synthetic regression tests for scheduling, geometry, temporal support and budget accounting. They do not reproduce the paper's detection metrics.

## Primary dual-frame mechanism comparison

See [the protocol](docs/EXPERIMENT_PROTOCOL.md) before running a comparison. All eleven conditions share the same frozen **7,191,090-parameter** detector. Each selected camera uses its current image and its previous same-camera keyframe. The experiment is an adapted mechanism comparison; it is not a retraining or reproduction of complete AB3DMOT, SimpleTrack, BEVDet4D or Fast-BEV systems.

Prepare the full nuScenes validation metadata and camera keyframes (6,019 samples / 150 scenes). Arrange the dataset as expected by the nuScenes devkit: `v1.0-trainval/`, `maps/`, and the referenced `samples/` camera images. Place the separately obtained checkpoint at a local path. Set all paths explicitly:

```bash
export ACTIVECAM_DATA_ROOT=/absolute/path/to/nuscenes
export ACTIVECAM_CHECKPOINT=/absolute/path/to/v31_student_inference_state_fp32.pth
export ACTIVECAM_CACHE_ROOT="$PWD/runs/fullval_camera_cache_fp32_batch1_formal"
export ACTIVECAM_OUTPUT_ROOT="$PWD/runs/dual_comparison"
# Optional: ACTIVECAM_EXTERNAL_ROOT, default "$PWD/external"
```

Expected checkpoint SHA-256:

```text
c50f8a1e3c35d9e211e666e1bfa2edd434c9d86e683ad3d0f527f8dfdb1d9043
```

Run the fixed sequence:

```bash
# Generates all six single-camera caches using the native dual-frame model,
# FP32, batch size one. This is the expensive GPU stage.
python tools/export_camera_cache.py

# Checks all cache hashes and raw-image ancestry, then 24 fixed GPU inference
# spot checks against the frozen numerical repeatability envelope.
python src/paper_comparison/prepare_dual_cache.py

# Exact serial/parallel equality on 80 frames before any formal scoring.
python tools/validate_replay.py

# K=3,4 x five mechanisms, and K=6 no reuse; official nuScenes detection scoring.
python tools/run_comparison.py
```

The commands refuse existing output directories rather than mixing new runs with old results. Use new cache/output roots for another run. A failure should be inspected before restarting. The primary comparison intentionally requires full validation metadata; `--smoke-frames` truncates replay after loading and verifying the full cache, and is not a nuScenes-mini mode.

Each formal condition writes `status.json`, `source_manifest.json`, `ancestry_trace.json`, official `metrics_summary.json`/`metrics_details.json`, and losslessly compressed predictions under `ACTIVECAM_OUTPUT_ROOT/mechanism_results/`. Detector cache construction reads all views offline; replay exposes only the scheduled current views and the documented historical observations to each method. Replay wall time excludes image I/O and detector inference and must not be reported as online end-to-end latency.

## Other evaluation and timing entry points

Use each script's `--help` for required paths and options. Examples:

```bash
python -m activecam_bev.eval_router --help
python -m activecam_bev.eval_controlled_staleness --help
python -m activecam_bev.eval_variable_budget --help
python -m activecam_bev.eval_robustness --help
python -m activecam_bev.eval_common_support_staleness --help
python -m activecam_bev.scene_paired_bootstrap --help
python src/activecam_bev/benchmark_activecam_end_to_end.py --help
```

For desktop end-to-end FP32 timing, supply `--checkpoint`, `--data-root`, `--legacy-src src/legacy_v31`, `--output-dir` and `--camera-count`; the paper timing setup uses 30 warmup and 200 measured frames, repeated three times. This path includes real image reading/preprocessing, transfer, dual-frame forward, decoding, propagation and fusion.

Jetson benchmark scripts require separately generated TensorRT engines and their input bundles; no engines or bundles are distributed. The temporal Jetson timing script uses previous=current placeholders and identity motion and measures the graph timing, not validation accuracy or a complete online pipeline. CPU scheduler/memory timing is a separate measurement. See the protocol for interpretation limits.

## Source provenance and reuse

The release copies the experiment implementations and changes only path/source-location plumbing where recorded in `provenance/source_hashes.json`. New `tools/` scripts provide portable execution of the existing export and comparison sequence. The original algorithmic code is not replaced with a simplified illustrative implementation. Historical environment defaults inside legacy training utilities are retained; always pass explicit paths.

External code is governed by its own upstream terms. No additional author reuse license has been selected for this review release; public access does not by itself grant a particular software license. See [THIRD_PARTY.md](THIRD_PARTY.md).
