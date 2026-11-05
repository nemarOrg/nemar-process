"""The ``nemar-floor`` recipe as the command and the record builders see it."""

from __future__ import annotations

from nemar_process.common import qc
from nemar_process.common.recipe import Column, DatasetStep, Distribution, Recipe
from nemar_process.nemar_floor import outputs, steps

PERCENT_EDGES = tuple(range(0, 101, 10))
DB_EDGES = (-10, -5, 0, 5, 10, 15, 20, 30, 40)

COLUMNS = (
    Column("path", "text"),
    Column("datatype", "text"),
    Column("status", "text"),
    Column("code", "text"),
    Column("step", "text"),
    Column("n_channels", "int"),
    Column("n_signal_channels", "int"),
    Column("duration_s", "float", 6),
    Column("srate_hz", "float", 6),
    Column("plan_hash", "str"),
    Column("pct_good_channels", "float", 4),
    Column("pct_good_samples", "float", 4),
    Column("line_hz", "int"),
    Column("line_db_before", "float", 3),
    Column("line_db_after", "float", 3),
    Column("n_bad_channels", "int"),
    Column("n_bad_segments", "int"),
    Column("skipped_steps", "str"),
    Column("record_content_sha256", "str"),
)

DISTRIBUTIONS = (
    Distribution("percentGoodChannels", "pct_good_channels", PERCENT_EDGES),
    Distribution("percentGoodSamples", "pct_good_samples", PERCENT_EDGES),
    Distribution("lineNoiseBeforeDb", "line_db_before", DB_EDGES),
    Distribution("lineNoiseAfterDb", "line_db_after", DB_EDGES),
)


def dataset_steps() -> list[DatasetStep]:
    """The recipe's steps with their recipe-level parameters: a constant, never read from a recording.

    The load step is the BIDS route (a set goes to ``pop_loadset``) and the bad-channel step is the
    RANSAC branch (the other is ``clean_channels_nolocs``); each recording's record says which ran.
    """
    from nemar_process.common import io

    table = [
        ("load", io.BIDS_FUNCTION, steps.load_parameters()),
        ("signal_channels", steps.SELECT_FUNCTION, {"channel_types": steps.SIGNAL_CHANNEL_TYPES}),
        ("flatline", steps.FLATLINE_FUNCTION, steps.FLATLINE_CALL),
        ("drift", steps.DRIFT_FUNCTION, steps.DRIFT_CALL),
        ("line_noise_measure", steps.MEASURE_FUNCTION, qc.line_noise_parameters()),
        ("line_noise_50", steps.BAND_STOP_FUNCTION, steps.band_stop_parameters(50)),
        ("line_noise_60", steps.BAND_STOP_FUNCTION, steps.band_stop_parameters(60)),
        ("bad_channels", steps.RANSAC_FUNCTION, steps.ransac_parameters()),
        ("bad_windows", steps.WINDOWS_FUNCTION, steps.windows_parameters()),
        ("qc", steps.QC_FUNCTION, steps.QC_PARAMETERS),
    ]
    assert tuple(step_id for step_id, _, _ in table) == steps.STEP_ORDER
    return [
        DatasetStep(step_id, function, dict(parameters), steps.DESCRIPTIONS[step_id])
        for step_id, function, parameters in table
    ]


RECIPE = Recipe(
    info=outputs.INFO,
    columns=COLUMNS,
    distributions=DISTRIBUTIONS,
    dataset_steps=dataset_steps,
    run_recording=outputs.run_recording,
)
