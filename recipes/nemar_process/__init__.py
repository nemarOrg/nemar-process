"""NEMAR's processing package: the recipes and what they share.

``nemar_process.sj`` is the signalJourney validator pointed at the schema vendored in ``_vendor/``.
``nemar_process.common`` is the layer every recipe is built on (loading, naming, the resolved plan,
measurements, files, records), ``nemar_process.nemar_floor`` is the first recipe, and
``python -m nemar_process`` (installed in the image as ``nemar-recipe``) runs a recipe on a dataset.
Every signal-processing call is an eegprep function: a recipe holds no numerics (AGENTS.md, and
``tests/recipes/test_upstream_first.py`` enforces it).
"""
