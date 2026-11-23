"""Native sources round only through src/cpp/rounding.hpp.

Spatial ties (nearest voxel) round up and value ties round to even. Both rules
live in rounding.hpp so CPU, CUDA and every sampler resolve a tie identically;
this test rejects ad-hoc rounding elsewhere in the native sources.
"""
from pathlib import Path
import re
import unittest

NATIVE = Path(__file__).resolve().parents[1] / 'src' / 'cpp'
RULES = NATIVE / 'rounding.hpp'

# Library rounding functions and CUDA round-to-nearest conversions, plus the
# hand-written forms of both tie rules (ties-to-even parity checks and +0.5 floors).
FORBIDDEN = [
    re.compile(r'\b(?:std::)?(?:nearbyintf?|rintf?|lrintf?|llrintf?|lroundf?|llroundf?|roundf?|roundeven)\s*\('),
    re.compile(r'__(?:float|double)2u?(?:int|ll)_rn\b'),
    re.compile(r'==\s*0?\.5f?\s*&&'),
    re.compile(r'\bfloorf?\s*\([^;]*\+\s*0?\.5f?\s*\)'),
]


class RoundingRules(unittest.TestCase):
    def test_native_sources_use_shared_tie_rules(self):
        found = []
        for path in sorted(NATIVE.rglob('*')):
            if path == RULES or path.suffix not in {'.hpp', '.cpp', '.cu', '.cuh', '.h'}:
                continue
            if 'third_party' in path.parts:
                continue
            for number, line in enumerate(path.read_text(errors='replace').splitlines(), 1):
                code = line.split('//', 1)[0]
                if any(p.search(code) for p in FORBIDDEN):
                    found.append(f'{path.relative_to(NATIVE)}:{number}: {line.strip()}')
        self.assertEqual(found, [], 'round through vcr::nearest_index / nearest_rounds_up '
                                    '(spatial) or vcr::round_half_even (values):\n' + '\n'.join(found))


if __name__ == '__main__':
    unittest.main()
