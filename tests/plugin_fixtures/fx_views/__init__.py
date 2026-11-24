from nemar_process.common.plugins import PluginResult, TableArray


def run(inputs, parameters):
    ica = inputs["nemar-ica"]
    return PluginResult(
        metrics={},
        views={
            "icclasses": {"roles": {"histogram": dict(ica["classes"])}},
            "spectrum": {
                "roles": {"values": TableArray(("freq_hz", "Fz"), ica["spectrum"])},
                "parameters": dict(ica["spectrum_parameters"]),
            },
        },
    )
