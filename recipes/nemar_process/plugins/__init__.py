"""NEMAR's reviewed quality plugins (ADR 0134 decision 6).

Each plugin is a package here, ``nemar_process.plugins.<name with - as _>``, with a ``manifest.json``
and a ``run``. A plugin runs in NEMAR's pipeline only when a recipe pins it (``PLUGINS`` beside its step
order) and its source tree has the digest ``tests/recipes/plugin_digests.json`` records for that
version (``docs/plugins.md``). Two ship: ``nemar-quality`` (the share of brain components and the
ICLabel class view) and ``nemar-spectrum`` (the spectrum view); no recipe pins them yet.
"""
