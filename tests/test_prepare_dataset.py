import hashlib
import tempfile
import unittest
from pathlib import Path

import yaml
from PIL import Image
from prepare_dataset import prepare


class PreparationTests(unittest.TestCase):
    def test_backgrounds_mapping_repairs_and_deduplication(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'source'
            for i, split in enumerate(('train', 'valid', 'test')):
                for sub in ('images', 'labels'):
                    (root / split / sub).mkdir(parents=True)
                Image.new('RGB', (20, 20), (i * 60, 30, 70)).save(root / split / 'images/objects.png')
                Image.new('RGB', (20, 20), (i * 60, 200, 70)).save(root / split / 'images/background.png')
                text = ''.join(f'{c} 0.5 0.5 0.2 0.2\n' for c in range(4))
                if split == 'train':
                    text += '2 0 0.5 0 0.1\n'
                (root / split / 'labels/objects.txt').write_text(text)
            (root / 'train/images/duplicate.png').write_bytes((root / 'test/images/objects.png').read_bytes())
            (root / 'train/labels/duplicate.txt').write_bytes((root / 'test/labels/objects.txt').read_bytes())
            (root / 'data.yaml').write_text(yaml.safe_dump({'names': ['pen', 'pencil', 'screwdriver', 'screw']}))
            hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*') if p.is_file()}
            out = Path(tmp) / 'prepared'
            report = prepare(root / 'data.yaml', out, 'background')
            self.assertEqual(len(report['repairs']), 1)
            self.assertEqual(len(report['excluded']), 1)
            self.assertFalse((out / 'train/images/duplicate.png').exists())
            for split in ('train', 'valid', 'test'):
                self.assertTrue((out / split / 'images/background.png').exists())
                self.assertFalse((out / split / 'labels/background.txt').exists())
                rows = (out / split / 'labels/objects.txt').read_text().splitlines()
                self.assertEqual([int(row.split()[0]) for row in rows], [0, 0, 1, 2])
                self.assertEqual(report['splits'][split]['backgrounds'], 1)
            self.assertEqual(hashes, {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*') if p.is_file()})
            with self.assertRaises(FileExistsError):
                prepare(root / 'data.yaml', out, 'background')


if __name__ == '__main__':
    unittest.main()
