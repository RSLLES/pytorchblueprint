# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from blueprint import utils


def test_format_duration_keeps_the_two_largest_units():
    """A duration shows its two leading units, even when the second one is zero."""
    assert utils.format.format_duration(3 * 3600 + 42 * 60 + 7) == "3h42m"
    assert utils.format.format_duration(3600 + 5) == "1h0m"
    assert utils.format.format_duration(45.9) == "45s"
    assert utils.format.format_duration(0) == "0s"
