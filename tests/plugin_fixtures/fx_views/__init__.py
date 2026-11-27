from nemar_process.common.plugins import PluginResult, TableArray


def run(inputs, parameters):
    ica = inputs["nemar-ica"]
    return PluginResult(
        metrics={},
        views={
            "icclasses": {"roles": {"histogram": dict(ica["classes"])}},
            "spectrum": {
                "roles": {"values": TableArray(("freq_hz", "Fz"), ica["spectrum"])},
                # the input is read-only (a list arrives as a tuple); parameters are strict JSON
                "parameters": {
                    key: list(value) if isinstance(value, tuple) else value
                    for key, value in ica["spectrum_parameters"].items()
                },
            },
        },
    )
