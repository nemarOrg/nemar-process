"""A fixture plugin that returns one view of each generic kind from what its host hands it."""

import numpy as np
from nemar_process.common.plugins import PluginResult, TableArray


def run(inputs, parameters):
    ica = inputs["nemar-ica"]
    sensors = tuple(ica["sensors"])
    count = len(sensors)
    times = np.array([0.0, 0.004, 0.008])
    erp = np.column_stack([times, np.arange(3.0 * count).reshape(3, count) - 2.5])
    epochs = np.column_stack([np.array([1.0, 2.0]), np.full((2, count), 1.5)])
    return PluginResult(
        metrics={},
        views={
            "erp": {"roles": {"values": TableArray(("time_s", *sensors), erp)}},
            "epochs": {"roles": {"values": TableArray(("epoch", *sensors), epochs)}},
            "summary": {
                "roles": {"table": TableArray(("n_bad", "share"), np.array([[2.0, 25.0]]))}
            },
            "amplitude": {
                "roles": {
                    "histogram": {
                        "edges": [0, 10, 20],
                        "counts": [count - 1, 1],
                        "underflow": 0,
                        "overflow": 0,
                        "unit": "uV",
                    }
                }
            },
            "topo": {"roles": {"values": TableArray(("alpha", "beta"), np.array(ica["maps"]))}},
        },
    )
