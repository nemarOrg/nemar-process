from nemar_process.common.plugins import PluginResult

# Why fxShare may have no value: the floor said it did not measure it.
METRIC_SKIP_WORDS = frozenset({"fx_not_measured"})


def run(inputs, parameters):
    floor = inputs["nemar-floor"]
    why = floor.get("why")
    return PluginResult(
        metrics={
            "fx-metrics:fxShare": floor["share"],
            "fx-metrics:fxLevel": floor["level"] * parameters["scale"],
        },
        metrics_skipped={} if why is None else {"fx-metrics:fxShare": why},
    )
