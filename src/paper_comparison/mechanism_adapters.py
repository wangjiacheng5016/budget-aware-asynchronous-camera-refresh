"""Official tracker cores adapted to frozen camera detections and bounded history.

No learned model is added. AB3DMOT visual/log imports are omitted by AST;
its tracking class and mathematical modules execute unchanged.
"""
import ast
import copy
import sys
from pathlib import Path
from types import SimpleNamespace, ModuleType

import numpy as np
import yaml
from pyquaternion import Quaternion

from review_paths import EXTERNAL
ROOT = EXTERNAL
sys.path[:0] = [str(ROOT / 'tracker_deps'), str(ROOT / 'AB3DMOT'), str(ROOT / 'SimpleTrack')]

# The camera-axis rotation helper imports optional KITTI file-I/O utilities.
# This replay uses global boxes and never loads KITTI pose files.
rotation_path = ROOT / 'AB3DMOT/AB3DMOT_libs/kitti_oxts.py'
rotation_tree = ast.parse(rotation_path.read_text())
rotation_tree.body = [node for node in rotation_tree.body if not
    (isinstance(node, ast.ImportFrom) and node.module == 'xinshuo_io')]
rotation_module = ModuleType('AB3DMOT_libs.kitti_oxts')
rotation_module.__file__ = str(rotation_path)
sys.modules[rotation_module.__name__] = rotation_module
exec(compile(rotation_tree, str(rotation_path), 'exec'), rotation_module.__dict__)
# Old @jit relied on the object-mode fallback removed from modern Numba.
# Execute the same official rotation bodies in Python; numerical results agree.
for rotation_name in ('rotx', 'roty', 'rotz'):
    rotation_module.__dict__[rotation_name] = rotation_module.__dict__[rotation_name].py_func


def yaw(record):
    return Quaternion(record['rotation']).yaw_pitch_roll[0]


def ab_box(record):
    x, y, z = record['translation']
    w, l, h = record['size']
    # World X/Y/Z -> KITTI camera x/z/-y, gravity center -> bottom center.
    return [h, w, l, x, -z + h / 2, y, -yaw(record)]


def ab_record(box, score, velocity):
    h, w, l, x, camera_y, camera_z, angle = box
    return dict(translation=[float(x), float(camera_z), float(-camera_y + h / 2)],
                size=[float(w), float(l), float(h)], rotation=list(Quaternion(axis=[0, 0, 1], angle=-float(angle))),
                velocity=[float(velocity[0]), float(velocity[2])], detection_score=float(score))


def simple_box(record):
    x, y, z = record['translation']
    w, l, h = record['size']
    return [x, y, z, yaw(record), l, w, h, record['detection_score']]


def simple_record(box, velocity):
    return dict(translation=[float(box.x), float(box.y), float(box.z)],
                size=[float(box.w), float(box.l), float(box.h)],
                rotation=list(Quaternion(axis=[0, 0, 1], angle=float(box.o))),
                velocity=[float(velocity[0]), float(velocity[1])], detection_score=float(box.s))


def load_ab_class():
    path = ROOT / 'AB3DMOT/AB3DMOT_libs/model.py'
    tree = ast.parse(path.read_text())
    excluded = {'AB3DMOT_libs.vis', 'xinshuo_miscellaneous', 'xinshuo_io'}
    tree.body = [node for node in tree.body if not (isinstance(node, ast.ImportFrom) and node.module in excluded)]
    namespace = dict(print_log=lambda *args, **kwargs: None)
    exec(compile(tree, str(path), 'exec'), namespace)
    return namespace['AB3DMOT']


AB_CLASS = None
TRACKING_CLASSES = ('bicycle', 'bus', 'car', 'motorcycle', 'pedestrian', 'trailer', 'truck')


def ab_predict(observations):
    global AB_CLASS
    if AB_CLASS is None:
        AB_CLASS = load_ab_class()
    output = []
    config = SimpleNamespace(vis=False, dataset='nuScenes', det_name='megvii', ego_com=False, affi_pro=False)
    for name in TRACKING_CLASSES:
        tracker = AB_CLASS(config, name.title())
        # Immediate birth and 3-past-frame survival implement the common budget.
        # Keep official nuScenes association metric/threshold/algorithm per class.
        tracker.min_hits = 1
        tracker.max_age = len(observations) + 1
        for index, (timestamp, records) in enumerate(observations):
            if index:
                dt = timestamp - observations[index - 1][0]
                for track in tracker.trackers:
                    track.kf.F = track.kf.F.astype(float)
                    track.kf.F[:3, 7:10] = np.eye(3) * dt
            chosen = [r for r in records if r['detection_name'] == name]
            dets = np.asarray([ab_box(r) for r in chosen], dtype=float).reshape(-1, 7)
            info = np.zeros((len(chosen), 7))
            if chosen:
                info[:, -1] = [r['detection_score'] for r in chosen]
            results, _ = tracker.track(dict(dets=dets, info=info), index, 'bounded_window')
        by_id = {track.id: track for track in tracker.trackers}
        for result in results[0]:
            track = by_id[int(result[7])]
            record = ab_record(result[:7], result[-1], track.get_velocity().reshape(-1))
            record['detection_name'] = name
            output.append(record)
    return output


def simple_predict(observations):
    from mot_3d.mot import MOTModel
    from mot_3d.frame_data import FrameData
    config = yaml.safe_load((ROOT / 'SimpleTrack/configs/nu_configs/giou.yaml').read_text())
    config['running']['score_threshold'] = 0.05
    config['running']['max_age_since_update'] = len(observations) + 1
    config['running']['min_hits_to_birth'] = 0
    output = []
    for name in TRACKING_CLASSES:
        tracker = MOTModel(copy.deepcopy(config))
        for timestamp, records in observations:
            chosen = [r for r in records if r['detection_name'] == name]
            data = FrameData([simple_box(r) for r in chosen], np.eye(4), timestamp,
                             pc=None, det_types=[name] * len(chosen), aux_info={'is_key_frame': True})
            results = tracker.frame_mot(data)
        by_id = {track.id: track for track in tracker.trackers}
        for box, track_id, state, _ in results:
            if not state.startswith('alive'):
                continue
            record = simple_record(box, by_id[track_id].motion_model.kf.x[7:9].reshape(-1))
            record['detection_name'] = name
            output.append(record)
    return output


def common_attribute(record):
    name = record['detection_name']
    speed = float(np.linalg.norm(record['velocity']))
    if name in ('car', 'truck', 'bus', 'trailer', 'construction_vehicle'):
        return 'vehicle.moving' if speed > 0.2 else ('vehicle.stopped' if name == 'bus' else 'vehicle.parked')
    if name in ('bicycle', 'motorcycle'):
        return 'cycle.with_rider' if speed > 0.2 else 'cycle.without_rider'
    if name == 'pedestrian':
        return 'pedestrian.moving' if speed > 0.2 else 'pedestrian.standing'
    return ''


def self_check():
    from AB3DMOT_libs.box import Box3D
    from mot_3d.data_protos import BBox
    record = dict(translation=[2., -3., 1.2], size=[1.7, 4.3, 1.8],
                  rotation=list(Quaternion(axis=[0, 0, 1], angle=0.7)),
                  velocity=[3., -2.], detection_name='car', detection_score=0.8)
    converted = ab_record(ab_box(record), 0.8, [3., 0., -2.])
    simple = simple_record(BBox.array2bbox(simple_box(record)), [3., -2.])
    for candidate in (converted, simple):
        for field in ('translation', 'size', 'velocity'):
            assert np.allclose(candidate[field], record[field], atol=1e-8), field
        assert abs(yaw(candidate) - yaw(record)) < 1e-8
    from nuscenes.utils.data_classes import Box
    canonical = Box(record['translation'], record['size'], Quaternion(record['rotation'])).corners().T
    native = Box3D.box2corners3d_camcoord(Box3D.array2bbox_raw(np.asarray(ab_box(record))))
    world = np.stack([native[:, 0], native[:, 2], -native[:, 1]], axis=1)
    assert max(min(np.linalg.norm(point - canonical, axis=1)) for point in world) < 1e-8
    observations = []
    for step in range(3):
        moving = copy.deepcopy(record)
        moving['translation'][0] += step
        observations.append((step * .5, [moving]))
    observations.append((1.5, []))
    checks = {}
    for method, function in [('ab3dmot', ab_predict), ('simpletrack', simple_predict)]:
        results = function(observations)
        assert len(results) == 1, (method, results)
        assert results[0]['translation'][0] > 4.5, (method, results)
        assert results[0]['velocity'][0] > 1.0, (method, results)
        checks[method] = results[0]
    return dict(passed=True, asymmetric_box_roundtrip=True, ab_corner_check=True,
                learned_motion_after_missing_frame=checks)


if __name__ == '__main__':
    import json
    result = self_check()
    (ROOT / 'mechanism_adapter_self_check.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
