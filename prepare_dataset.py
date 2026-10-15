#!/usr/bin/env python3
"""Prepare three-class YOLO data, with an auditable manifest and leakage checks."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
import shutil
from collections import Counter
from pathlib import Path
import yaml
from PIL import Image

IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
CLASS_MAP = {'pen': 0, 'pencil': 0, 'screwdriver': 1, 'screw': 2}
CLASS_NAMES = ['pen', 'screwdriver', 'screw']


def prepare(source_yaml: Path, output: Path, missing_policy: str = 'exclude') -> dict:
    source_yaml, output = source_yaml.resolve(), output.resolve()
    source = yaml.safe_load(source_yaml.read_text(encoding='utf-8'))
    names = source['names']
    if isinstance(names, dict):
        if sorted(int(k) for k in names) != list(range(len(names))):
            raise ValueError('Source class IDs must be contiguous from zero')
        names = [names[k] for k in sorted(names, key=int)]
    if not isinstance(names, list) or not names:
        raise ValueError('Expected class names in source YAML')
    remap = {i: CLASS_MAP[str(n).strip().lower()] for i, n in enumerate(names)}
    if set(remap.values()) != {0, 1, 2}:
        raise ValueError('Source classes must cover pen, screwdriver, and screw')
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Output must be empty: {output}')
    output.mkdir(parents=True, exist_ok=True)
    report = {'source': str(source_yaml), 'source_names': names, 'classes': CLASS_NAMES,
              'missing_policy': missing_policy, 'excluded': [], 'repairs': [], 'splits': {}}
    seen_pixels = {}
    # Retain held-out copies; exclude their duplicates from earlier learning stages.
    for split in ('test', 'valid', 'train'):
        image_dir, label_dir = source_yaml.parent/split/'images', source_yaml.parent/split/'labels'
        if not image_dir.is_dir() or not label_dir.is_dir():
            raise FileNotFoundError(f'Missing split directories: {split}')
        images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        if len({p.stem for p in images}) != len(images):
            raise ValueError(f'Ambiguous image stems: {split}')
        labels = {p.stem: p for p in label_dir.glob('*.txt')}
        if labels.keys() - {p.stem for p in images}:
            raise ValueError(f'Orphan labels in {split}')
        for folder in ('images', 'labels'):
            (output/split/folder).mkdir(parents=True)
        counts = Counter(); kept = backgrounds = 0
        for p in images:
            relative = str(p.relative_to(source_yaml.parent))
            label = labels.get(p.stem)
            if label is None and missing_policy == 'exclude':
                report['excluded'].append({'image': relative, 'reason': 'missing_label'})
                continue
            with Image.open(p) as im:
                rgb = im.convert('RGB')
                digest = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
            if digest in seen_pixels:
                report['excluded'].append({'image': relative, 'reason': 'duplicate_pixels', 'retained': seen_pixels[digest]})
                continue
            converted = []
            for line_no, line in enumerate(label.read_text().splitlines() if label else [], 1):
                if not line.strip():
                    continue
                fields = line.split()
                if len(fields) != 5:
                    raise ValueError(f'{label}:{line_no}: expected five fields')
                old_id = int(fields[0]); box = list(map(float, fields[1:]))
                if old_id not in remap or not all(math.isfinite(v) for v in box):
                    raise ValueError(f'{label}:{line_no}: invalid class or box')
                if box[2] == 0 or box[3] == 0:
                    report['repairs'].append({'image': relative, 'line': line_no, 'reason': 'removed_zero_area_box', 'original': line})
                    continue
                if any(v < 0 or v > 1 for v in box):
                    raise ValueError(f'{label}:{line_no}: invalid normalized box')
                xc,yc,w,h = box
                x1,y1,x2,y2 = max(0,xc-w/2),max(0,yc-h/2),min(1,xc+w/2),min(1,yc+h/2)
                if x2<=x1 or y2<=y1:
                    raise ValueError(f'{label}:{line_no}: box has no visible area')
                clipped=[(x1+x2)/2,(y1+y2)/2,x2-x1,y2-y1]
                if any(abs(a-b)>1e-8 for a,b in zip(box,clipped)):
                    report['repairs'].append({'image': relative,'line':line_no,'reason':'clipped_to_image','original':line})
                converted.append(str(remap[old_id])+' '+' '.join(f'{v:.10g}' for v in clipped))
                counts[remap[old_id]] += 1
            dest=output/split/'images'/p.name
            try:
                os.link(p,dest)
            except OSError:
                shutil.copy2(p,dest)
            # Preserve the user's background images without inventing txt files.
            if label is not None:
                (output/split/'labels'/(p.stem+'.txt')).write_text('\n'.join(converted)+('\n' if converted else ''))
            seen_pixels[digest]=relative; kept+=1; backgrounds+=not converted
        report['splits'][split]={'source_images':len(images),'images':kept,'backgrounds':backgrounds,'boxes':dict(counts)}
        if not kept or set(counts)!={0,1,2}:
            raise ValueError(f'{split}: retained data must contain all three classes')
    config={'path':str(output),'train':'train/images','val':'valid/images','test':'test/images','names':dict(enumerate(CLASS_NAMES))}
    (output/'data.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
    (output/'preparation_report.json').write_text(json.dumps(report,indent=2))
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source_yaml',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--missing-labels',choices=['exclude','background'],default='exclude',help='Use background only if confirmed that unlabelled images contain no target objects')
    args=parser.parse_args()
    report=prepare(args.source_yaml,args.output,args.missing_labels)
    print(json.dumps({'splits':report['splits'],'excluded':dict(Counter(x['reason'] for x in report['excluded'])),'repairs':len(report['repairs'])},indent=2))

if __name__=='__main__':
    main()
