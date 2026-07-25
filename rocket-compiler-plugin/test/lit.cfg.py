"""Lit config for the Rocket compiler plugin's tests.

lit discovers a test suite by walking up from each test file looking for a
lit.cfg.py in an ancestor directory. iree-src has one at compiler/, runtime/,
etc., but this directory lives outside iree-src entirely (rocket-compiler-plugin
is a sibling repo merged into the mono-repo, loaded via IREE_CMAKE_PLUGIN_PATHS),
so none of iree-src's configs are ancestors of these test files. Without this,
`ctest -R rocket_conv_layout` fails with "did not discover any tests".
"""

# pylint: disable=undefined-variable

import lit.formats

config.name = "Rocket"
config.suffixes = [".mlir"]
config.test_format = lit.formats.ShTest(
    execute_external=True, force_execute_external=True
)
