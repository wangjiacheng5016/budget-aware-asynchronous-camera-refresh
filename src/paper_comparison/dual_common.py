"""Frozen dual-frame cache provenance and bounded raw-image ancestry."""
import hashlib
import json
from pathlib import Path

from review_paths import ROOT, BASE, CACHE, CHECKPOINT
CAMERAS = ('CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
           'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT')



def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def ancestry(nusc, tokens):
    output = {}
    for token in tokens:
        sample = nusc.get('sample', token)
        previous = nusc.get('sample', sample['prev']) if sample['prev'] else None
        if previous:
            assert previous['scene_token'] == sample['scene_token']
        output[token] = {}
        for camera in CAMERAS:
            sd = nusc.get('sample_data', sample['data'][camera])
            prior = nusc.get('sample_data', previous['data'][camera]) if previous else None
            if prior:
                assert prior['timestamp'] < sd['timestamp']
            output[token][camera] = dict(current_sd=sd['token'], timestamp=sd['timestamp'],
                previous_sd=prior['token'] if prior else None,
                previous_timestamp=prior['timestamp'] if prior else None)
    return output


def raw_context(provenance, target, observations):
    """Count unique raw sample_data ancestors, not cached prediction ages."""
    raw = {}
    for token, cameras in observations:
        for camera in cameras:
            source = provenance[token][camera]
            raw[source['current_sd']] = (camera, source['timestamp'])
            if source['previous_sd']:
                raw[source['previous_sd']] = (camera, source['previous_timestamp'])
    ages = [(provenance[target][camera]['timestamp'] - timestamp) / 1e6
            for camera, timestamp in raw.values()]
    assert all(age >= 0 for age in ages), 'Future raw image dependency'
    return dict(raw_images=len(raw), past_raw_images=sum(age > 0 for age in ages),
                max_raw_image_age_s=max(ages, default=0.0), future_raw_images=0)


def official_metrics(method, predictions, nusc, config_factory, NuScenesEval, output_dir):
    """Compact JSON avoids unnecessary disk expansion; official evaluator unchanged."""
    from result_io import write_result_json
    d = output_dir / 'evaluations' / method
    d.mkdir(parents=True, exist_ok=True)
    path = d / 'results_nusc.json'
    write_result_json(path, predictions)
    NuScenesEval(nusc, config=config_factory('detection_cvpr_2019'), result_path=str(path),
                 eval_set='val', output_dir=str(d), verbose=False).main(plot_examples=0, render_curves=False)
    m = json.loads((d / 'metrics_summary.json').read_text())
    return dict(method=method, NDS=m['nd_score'], mAP=m['mean_ap'],
                **{label: m['tp_errors'][key] for label, key in
                   [('mATE', 'trans_err'), ('mASE', 'scale_err'), ('mAOE', 'orient_err'),
                    ('mAVE', 'vel_err'), ('mAAE', 'attr_err')]})
