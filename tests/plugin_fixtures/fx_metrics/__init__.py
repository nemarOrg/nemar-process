from nemar_process.common.plugins import PluginResult


def run(inputs, parameters):
    floor = inputs["nemar-floor"]
    return PluginResult(
        metrics={"fxShare": floor["share"], "fxLevel": floor["level"] * parameters["scale"]}
    )
