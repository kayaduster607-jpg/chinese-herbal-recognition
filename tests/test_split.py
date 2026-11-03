import tempfile
import unittest
from pathlib import Path
from train import split_dataset


class DatasetSplitTests(unittest.TestCase):
    def test_reproducible_disjoint_split_and_overwrite_protection(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / 'source'
            for label in ('herb_a', 'herb_b'):
                folder = source / label
                folder.mkdir(parents=True)
                for index in range(20):
                    (folder / f'{index:02}.jpg').write_bytes(b'fixture')
            first, second = root / 'first', root / 'second'
            split_dataset(source, first, 42)
            split_dataset(source, second, 42)
            for label in ('herb_a', 'herb_b'):
                names = []
                for split, expected in (('train', 14), ('val', 3), ('test', 3)):
                    a = {p.name for p in (first / split / label).iterdir()}
                    b = {p.name for p in (second / split / label).iterdir()}
                    self.assertEqual(a, b)
                    self.assertEqual(len(a), expected)
                    names.extend(a)
                self.assertEqual(len(set(names)), 20)
            with self.assertRaises(FileExistsError):
                split_dataset(source, first, 42)


if __name__ == '__main__':
    unittest.main()
