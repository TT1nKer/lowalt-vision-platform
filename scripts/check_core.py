"""Run the published CPU suite; report legacy tests requiring unshipped code.

This is intentionally narrower than unittest discover, which still exposes the
pre-existing missing-source errors. No inference-quality claims are made here.
"""
from pathlib import Path
import sys
import unittest

# These eight modules fail to import in the original public checkout cc4bfc3.
# Keep omissions explicit so they cannot be mistaken for passing tests.
UNSHIPPED = {
    'test_dlp_adapter', 'test_equipment_ontology', 'test_parking_e3_ingest',
    'test_parking_e3_trajectory_ingest', 'test_trajectory_features',
    'test_uav_data_schema', 'test_uav_physical_scale', 'test_uav_source_ingest',
}


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    suite = unittest.TestSuite()
    for path in sorted((root / 'tests').rglob('test_*.py')):
        if path.stem in UNSHIPPED:
            print(f'OUT OF PUBLIC CORE: {path.relative_to(root)} (unshipped uav_data/equipment_data source)', flush=True)
            continue
        module = '.'.join(path.relative_to(root).with_suffix('').parts)
        suite.addTests(unittest.defaultTestLoader.loadTestsFromName(module))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)


if __name__ == '__main__':
    main()
