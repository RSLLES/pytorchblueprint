# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from blueprint.utils.strings import format_number


def test_format_float():
    """Should maintain a constant number of output characters."""
    tests = {
        "2.37e-11": 2.3693e-11,
        "5.77e-05": 5.774e-5,
        "  0.0102": 1.02e-2,
        "    1.23": 1.2345,
        "     154": 154,
        "5.65e+03": 5648,
        "5.47e+09": 5.47e9,
        "8.85e+11": 8.85e11,
    }
    for y, x in tests.items():
        assert y == format_number(x)
