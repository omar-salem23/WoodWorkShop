#!/usr/bin/env python3
"""Run the trained workshop detector on an image or a folder of images."""
from __future__ import annotations
import argparse
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path, help='Image file or directory of images')
    parser.add_argument('--model', type=Path, default=ROOT/'models/best.pt')
    parser.add_argument('--output', type=Path, default=ROOT/'runs/predictions')
    parser.add_argument('--conf', type=float, default=0.25)
    parser.add_argument('--device', default='cpu', help='cpu, 0, or auto')
    args = parser.parse_args()
    if not args.source.exists():
        raise FileNotFoundError(f'Input not found: {args.source}')
    if not args.model.is_file():
        raise FileNotFoundError(f'Model not found: {args.model}')
    model = YOLO(str(args.model))
    device = None if args.device == 'auto' else args.device
    results = model.predict(source=str(args.source), conf=args.conf,
        device=device, save=True, project=str(args.output), name='results', exist_ok=True)
    for result in results:
        counts = {}
        for box in result.boxes:
            class_id = int(box.cls.item())
            name = model.names[class_id]
            counts[name] = counts.get(name, 0) + 1
        print(f'{Path(result.path).name}: {counts or "no detections"}')
    print(f'Annotated images saved in: {args.output.resolve() / "results"}')


if __name__ == '__main__':
    main()
