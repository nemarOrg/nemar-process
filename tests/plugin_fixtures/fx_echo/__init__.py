from nemar_process.common.plugins import PluginResult


def run(inputs, parameters):
    return PluginResult(metrics={"fxEcho": inputs["nemar-floor"]["value"]})
