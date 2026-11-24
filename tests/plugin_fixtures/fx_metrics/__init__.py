from nemar_process.common.plugins import PluginResult


def run(inputs, parameters):
    floor = inputs["nemar-floor"]
    return PluginResult(
        metrics={
            "fx-metrics:fxShare": floor["share"],
            "fx-metrics:fxLevel": floor["level"] * parameters["scale"],
        }
    )
