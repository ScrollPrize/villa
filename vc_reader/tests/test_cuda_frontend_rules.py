"""Native sources avoid constructs CUDA's host front end (cudafe++) mis-translates.

CUDA 13.0's cudafe++ rewrites host code before the host compiler sees it. A
std::array type first named with a function-local constexpr bound, as in

    constexpr std::size_t limit = 256; std::array<std::uint32_t, limit> counts{};

is printed as `std::array<unsigned, unsigned long((256))>` wherever the same type
is named again, which the host compiler rejects. Bounds that depend on a template
parameter, or arrays of a template parameter type, are unaffected. CI builds no
CUDA, so this scan is what keeps the CUDA build compiling.
"""
from pathlib import Path
import re
import unittest

NATIVE = Path(__file__).resolve().parents[1] / 'src' / 'cpp'

# A function-local (indented) constexpr integer initialised with a literal.
LOCAL_CONSTANT = re.compile(r'^\s+(?:static\s+)?constexpr\s+[\w:]+\s+(\w+)\s*=\s*\d+[uUlL]*\s*;')
# std::array<concrete element, bound>: the element type is not a single template parameter.
ARRAY = re.compile(r'\bstd::array\s*<\s*([\w:]+(?:\s*<[^<>]*>)?)\s*,\s*(\w+)\s*>')
TEMPLATE_PARAMETER = re.compile(r'^[A-Z]\w*$')


class CudaFrontendRules(unittest.TestCase):
    def test_array_bounds_are_not_function_local_constants(self):
        found = []
        for path in sorted(NATIVE.rglob('*')):
            if path.suffix not in {'.hpp', '.cpp', '.cu', '.cuh', '.h'} or 'third_party' in path.parts:
                continue
            local = set()
            for number, line in enumerate(path.read_text(errors='replace').splitlines(), 1):
                code = line.split('//', 1)[0]
                if not code.startswith((' ', '\t')):
                    local.clear()  # back at namespace scope
                if match := LOCAL_CONSTANT.match(code):
                    local.add(match.group(1))
                for element, bound in ARRAY.findall(code):
                    if bound in local and not TEMPLATE_PARAMETER.match(element):
                        found.append(f'{path.relative_to(NATIVE)}:{number}: {line.strip()}')
        self.assertEqual(found, [], 'name std::array bounds with a namespace-scope constant or a literal, '
                                    'not a function-local constexpr (cudafe++ mis-prints the type):\n'
                                    + '\n'.join(found))


if __name__ == '__main__':
    unittest.main()
