import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from param_processor import Params_cls
from QoI import QoI_cls


class DemoQoI(QoI_cls):
    """Example QoI: returns energy and stress components."""

    def qoi(self, params_dict):
        E = params_dict["E"]
        A = params_dict["A"]
        # Toy computations
        energy = 0.5 * E * np.sum(A)
        stress = np.array([E * a for a in A])
        return {
            "energy": np.array([energy]),  # scalar as 1-element array for consistent shape
            "stress": stress,
        }


def main():
    class DemoParams(Params_cls):
        def __init__(self):
            self.params0 = {
                "E": 210.0,
                "nu": 0.30,                          # frozen by omission
                "A": np.array([0.2, 5.0, 7.0]),
            }

            self.active_specs_input = {
                "E":  {"lower": 100.0, "upper": 300.0, "scale": "linear", "select": None},
                "A0": {"param": "A", "select": np.array([1,0,0], bool), "lower": 0.0,  "upper": 1.0, "scale": "linear"},
                "A1": {"param": "A", "select": np.array([0,1,0], bool), "lower": 1e-3, "upper": 1e2, "scale": "log"},
            }

            super().__init__()

    params_handler = DemoParams()
    qoi = DemoQoI(params_handler)

    x = params_handler.pack(params_handler.base)
    q = qoi.compute_QoI(x)

    print("QoI dim:", qoi.qoi_dim)
    print("Flat QoI:", q)
    print("Layout:", qoi._layout)


if __name__ == "__main__":
    main()
