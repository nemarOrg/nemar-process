"""Real plugins for the runner's tests (ADR 0134 Phase 5): modules with manifests, loaded and run by
``nemar_process.common.plugins`` exactly as NEMAR's own are, never stand-ins for its logic.

They live outside the recipes package, so they never run in NEMAR's pipeline: a test loads them with
``package="plugin_fixtures"``. ``fx-echo`` returns what the test hands it, to drive each output rule.
"""
