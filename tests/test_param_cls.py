import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from surroptim.param_processor import params_cls


def make_demo_params():
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

    return params_cls(init_params=init_params, active_specs=active_specs)


if __name__ == "__main__":
    P = make_demo_params()
    params0 = P.init_params

    print("=== LAYOUT ===")
    print(P.describe_layout())

    # -------------------------
    # SINGLE roundtrip test
    # -------------------------
    print("\n=== SINGLE (tolerant checks) ===")
    x  = P.pack(params0)
    z  = P.physical_to_unit(x)
    g  = P.unit_to_gauss(z)
    z2 = P.gauss_to_unit(g)
    x2 = P.unit_to_physical(z2)
    params_rec = P.unpack(x2)

    abs_err = np.max(np.abs(x - x2))
    rel_err = np.max(np.abs((x - x2) / np.maximum(1.0, np.abs(x))))

    print("params0:", params0)
    print("x (physical)      :", x, "shape", x.shape)
    print("z (unit [-1,1])   :", z, "shape", z.shape)
    print("g (gauss)         :", g, "shape", g.shape)
    print("x2 (back physical):", x2)
    print("params_rec        :", params_rec)
    print("max abs err:", abs_err)
    print("max rel err:", rel_err)
    print("roundtrip x close? (rtol=1e-7, atol=1e-9):",
          np.allclose(x, x2, rtol=1e-7, atol=1e-9))

    # -------------------------
    # DIRECT physical <-> gauss
    # -------------------------
    print("\n=== DIRECT physical <-> gauss (tolerant checks) ===")
    g_dir  = P.physical_to_gauss(x)
    x_dir2 = P.gauss_to_physical(g_dir)

    abs_err_d = np.max(np.abs(x - x_dir2))
    rel_err_d = np.max(np.abs((x - x_dir2) / np.maximum(1.0, np.abs(x))))

    print("g_dir :", g_dir)
    print("x_dir2:", x_dir2)
    print("max abs err:", abs_err_d)
    print("max rel err:", rel_err_d)
    print("direct roundtrip close? (rtol=1e-7, atol=1e-9):",
          np.allclose(x, x_dir2, rtol=1e-7, atol=1e-9))

    # -------------------------
    # BATCH test
    # -------------------------
    print("\n=== BATCH (list[dict] -> arrays -> back) ===")
    batch = [
        params0,
        {**params0, "E": 250.0, "A": np.array([0.9, 10.0, 7.0])},
        {**params0, "E": 120.0, "A": np.array([0.1,  0.01, 7.0])},
    ]

    X  = P.pack(batch)                 # (M,N) physical
    Z  = P.physical_to_unit(X)         # (M,N) unit
    G  = P.unit_to_gauss(Z)            # (M,N) gauss
    Zb = P.gauss_to_unit(G)            # (M,N)
    Xb = P.unit_to_physical(Zb)        # (M,N)
    batch_rec = P.unpack(Xb)           # list[dict]

    abs_err_B = np.max(np.abs(X - Xb))
    rel_err_B = np.max(np.abs((X - Xb) / np.maximum(1.0, np.abs(X))))

    print("X shape:", X.shape, "Z shape:", Z.shape, "G shape:", G.shape)
    print("\nFirst 3 rows X:\n", X[:3])
    print("\nFirst 3 rows Z:\n", Z[:3])
    print("\nFirst 3 rows G:\n", G[:3])

    print("\nRecovered dicts:")
    for i, d in enumerate(batch_rec):
        print(f"[{i}] {d}")

    print("\nmax abs err:", abs_err_B)
    print("max rel err:", rel_err_B)
    print("Batch roundtrip X close? (rtol=1e-7, atol=1e-9):",
          np.allclose(X, Xb, rtol=1e-7, atol=1e-9))

    # -------------------------
    # Convenience conversions: unit->dict and gauss->dict
    # -------------------------
    print("\n=== unit_to_dict / gauss_to_dict (single + batch) ===")
    print("unit_to_dict(z):", P.unit_to_dict(z))
    print("gauss_to_dict(g):", P.gauss_to_dict(g))

    dicts_from_unit = P.unit_to_dict(Z)
    dicts_from_gauss = P.gauss_to_dict(G)
    print("unit_to_dict(Z)[0]:", dicts_from_unit[0])
    print("gauss_to_dict(G)[0]:", dicts_from_gauss[0])

    # -------------------------
    # 1000 random unit points -> dicts: show first 3
    # -------------------------
    print("\n=== 1000 RANDOM UNIT POINTS -> dicts: first 3 ===")
    rng = np.random.default_rng(123)
    Z1000 = rng.uniform(-1.0, 1.0, size=(1000, P.dim))
    dicts1000 = P.unit_to_dict(Z1000)
    print(dicts1000[0])
    print(dicts1000[1])
    print(dicts1000[2])
