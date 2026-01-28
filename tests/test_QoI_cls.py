import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from surroptim.param_processor import params_cls
from surroptim.QoI import QoI_cls


def demo_qoi_func(params_dict):
    """Example QoI function: returns energy and stress components."""
    E = params_dict["E"]
    A = params_dict["A"]
    energy = 0.5 * E * np.sum(A)
    stress = np.array([E * a for a in A])
    return {
        "energy": np.array([energy]),
        "stress": stress,
    }


def main():
    init_params = {
        "E": 210.0,
        "nu": 0.30,
        "A": np.array([0.2, 5.0, 7.0]),
    }

    active_specs = {
        "E":  {"lower": 100.0, "upper": 300.0, "scale": "linear", "select": None},
        "A0": {"param": "A", "select": np.array([1,0,0], bool), "lower": 0.0,  "upper": 1.0, "scale": "linear"},
        "A1": {"param": "A", "select": np.array([0,1,0], bool), "lower": 1e-3, "upper": 1e2, "scale": "log"},
    }

    params_handler = params_cls(init_params=init_params, active_specs=active_specs)

    # Use the new constructor-style API: pass the QoI function to QoI_cls
    qoi = QoI_cls(params_handler, qoi_func=demo_qoi_func)

    x = params_handler.pack(params_handler.base)
    q = qoi.compute_QoI(x)

    print("QoI dim:", qoi.qoi_dim)
    print("Flat QoI:", q)
    print("Layout:", qoi._layout)


if __name__ == "__main__":
    main()
