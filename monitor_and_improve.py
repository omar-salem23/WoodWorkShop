#!/usr/bin/env python3
"""Watch the CPU run; if its held-out test misses targets, train one val-selected improvement run."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / 'runs/woodwork_full'
THRESHOLDS = {'map50': .65, 'map50_95': .35, 'min_class_map50_95': .20,
              'background_false_positive_rate': .10}


def fails(metrics: dict) -> list[str]:
    test = metrics['test']
    reasons = []
    if test['map50'] < THRESHOLDS['map50']: reasons.append('test map50 below target')
    if test['map50_95'] < THRESHOLDS['map50_95']: reasons.append('test map50-95 below target')
    if min(test['per_class_map50_95'], default=0) < THRESHOLDS['min_class_map50_95']:
        reasons.append('a test class map50-95 below target')
    rate = test['background'].get('image_false_positive_rate')
    if rate is not None and rate > THRESHOLDS['background_false_positive_rate']:
        reasons.append('too many background images contain false detections')
    return reasons


def wait_for_pid(pid: int, output: Path, timeout: float) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            command = Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
        except FileNotFoundError:
            return
        if str(output) not in command:
            return
        time.sleep(45)
    raise TimeoutError(f'Training process {pid} is still active after the monitor timeout')


def prepare_test_images(data_yaml: Path, run: Path, weights: str) -> None:
    import yaml
    from ultralytics import YOLO
    config = yaml.safe_load(data_yaml.read_text())
    test_images = Path(config['path']) / config['test']
    labels = test_images.parent / 'labels'
    chosen: list[Path] = []
    manifest: list[dict] = []
    for class_id, class_name in enumerate(('pen', 'screwdriver', 'screw')):
        found = 0
        for image in sorted(test_images.iterdir()):
            if image.suffix.lower() not in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}:
                continue
            label = labels / f'{image.stem}.txt'
            if label.exists() and any(row.split() and row.split()[0] == str(class_id)
                                      for row in label.read_text().splitlines()):
                if image not in chosen:
                    chosen.append(image)
                    manifest.append({'image': image.name, 'sample_class': class_name})
                found += 1
                if found >= 2:
                    break
    backgrounds = 0
    for image in sorted(test_images.iterdir()):
        if image.suffix.lower() not in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}:
            continue
        label = labels / f'{image.stem}.txt'
        if not label.exists() or not label.read_text().strip():
            chosen.append(image)
            manifest.append({'image': image.name, 'sample_class': 'background'})
            backgrounds += 1
            if backgrounds == 4:
                break
    output = run / 'test_examples'
    output.mkdir(exist_ok=True)
    (output/'sample_manifest.json').write_text(json.dumps(manifest, indent=2))
    predictions = YOLO(weights).predict(source=[str(p) for p in chosen], imgsz=512,
        conf=.25, device='cpu', save=True, save_conf=True, project=str(run),
        name='test_examples/predictions', exist_ok=True, verbose=False)
    (output/'prediction_count.json').write_text(json.dumps({
        'images': len(predictions), 'images_with_detections': sum(len(r.boxes) > 0 for r in predictions),
        'output_images': str(run/'test_examples/predictions')}, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pid', type=int, required=True)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--data', type=Path, default=Path('/workspace/datasets/WoodWorkShop/prepared_with_backgrounds/data.yaml'))
    parser.add_argument('--wait-hours', type=float, default=72)
    args = parser.parse_args()
    output, data = args.output.resolve(), args.data.resolve()
    status_file = output / 'unattended_status.json'
    status = {'status': 'watching', 'initial_output': str(output), 'thresholds': THRESHOLDS}
    status_file.write_text(json.dumps(status, indent=2))
    wait_for_pid(args.pid, output, args.wait_hours * 3600)
    initial_file = output / 'metrics.json'
    if not initial_file.exists():
        status.update(status='training_failed', detail='CPU process ended without final metrics.json')
        status_file.write_text(json.dumps(status, indent=2)); return 2
    initial = json.loads(initial_file.read_text())
    reasons = fails(initial)
    chosen_output = output
    if reasons:
        status.update(status='automatic_retraining', reasons=reasons,
                      selected_from_validation=initial['weights'])
        status_file.write_text(json.dumps(status, indent=2))
        followup = ROOT / 'runs/woodwork_auto_retrain'
        if followup.exists():
            followup = ROOT / f"runs/woodwork_auto_retrain_{int(time.time())}"
        command = [sys.executable, '-u', str(ROOT/'train.py'), '--data', str(data),
                   '--model', initial['weights'], '--epochs', '40', '--retry-epochs', '60',
                   '--imgsz', '640', '--batch', '6', '--workers', '2', '--output', str(followup)]
        (followup.parent/'auto_retrain_command.json').write_text(json.dumps(command, indent=2))
        with (followup.parent/'auto_retrain.log').open('w') as log:
            result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode or not (followup/'metrics.json').exists():
            status.update(status='automatic_retraining_failed', returncode=result.returncode,
                          output=str(followup))
            status_file.write_text(json.dumps(status, indent=2)); return 3
        chosen_output = followup
    final = json.loads((chosen_output/'metrics.json').read_text())
    final_reasons = fails(final)
    status.update(status='preparing_test_examples', final_output=str(chosen_output))
    status_file.write_text(json.dumps(status, indent=2))
    try:
        prepare_test_images(data, chosen_output, final['weights'])
    except Exception as exc:
        status['test_examples_error'] = repr(exc)
    status.update(status='completed' if not final_reasons else 'completed_targets_not_met',
                  final_output=str(chosen_output), final_metrics=str(chosen_output/'metrics.json'),
                  reasons=final_reasons, weights=final['weights'])
    status_file.write_text(json.dumps(status, indent=2))
    return 0 if not final_reasons else 1

if __name__ == '__main__':
    raise SystemExit(main())
