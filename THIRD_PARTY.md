# External sources and dependencies

External repositories are fetched separately, not vendored in this release. Read and retain their upstream license and attribution files when using them.

| Project | Upstream | Frozen commit | Use |
| --- | --- | --- | --- |
| AB3DMOT | https://github.com/xinshuoweng/AB3DMOT | `61f3bd72574093e367916c757b4747ca445f978c` | Adapted association and Kalman prediction core |
| SimpleTrack | https://github.com/tusen-ai/SimpleTrack | `05c96bb7ed98fc179856f327544612a66c839b5e` | Adapted bounded-context tracking/prediction core |

The adapters are provided in `src/paper_comparison/mechanism_adapters.py`; their scope and departures from the original full systems are documented in `docs/EXPERIMENT_PROTOCOL.md`. No upstream pretrained weights are distributed or claimed to be reproduced.

Other dependencies include PyTorch/torchvision, NumPy, SciPy, OpenCV, Pillow, PyYAML, FilterPy, Numba, pyquaternion, tqdm and the official nuScenes devkit. They are installed from their own distributions and retain their respective licenses. Optional Jetson paths require NVIDIA CUDA/TensorRT, supplied separately under NVIDIA's terms. Optional teacher training imports MMEngine/MMDetection3D, supplied separately under their respective terms.

nuScenes images, metadata and annotations remain subject to the dataset's access and use terms and are not redistributed here.

The author-provided source is publicly accessible for manuscript review. No additional author software reuse license has been selected for this release; this statement does not replace or change any third-party license.
