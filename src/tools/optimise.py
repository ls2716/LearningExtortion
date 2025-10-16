from __future__ import annotations
import numpy as np
import cma
import os

from joblib import Parallel, delayed


def _init_x0(cfg, n_params: int, rng: np.random.Generator):
    x0 = cfg.get("x0", "zero")
    if isinstance(x0, str):
        if x0 == "zero":
            return np.zeros(n_params, dtype=np.float64)
        if x0 == "random":
            return rng.uniform(0.0, 1.0, size=n_params).astype(np.float64)
        if x0.startswith("file:"):
            path = x0.split("file:", 1)[1]
            return np.load(path)
        raise ValueError(f"Unknown x0 mode: {x0}")
    elif isinstance(x0, list):
        return np.array(x0, dtype=np.float64)
    else:
        raise ValueError("x0 must be 'zero'|'random'|'file:<path>'|list")


def run_cma(objective, extortion_cfg, cma_cfg, seed: int):
    # Prevent numpy and related libraries from parallelising
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    n_params = extortion_cfg["params"]["n_params"]
    rng = np.random.default_rng(seed)
    x0 = _init_x0(cma_cfg, n_params, rng)
    sigma0 = float(cma_cfg["sigma0"])

    if cma_cfg.get("pop_heuristic", True) and not cma_cfg.get("popsize"):
        popsize = 8 + int(3 * np.log(n_params))
    else:
        popsize = cma_cfg.get("popsize")

    options = {
        "bounds": tuple(cma_cfg["bounds"]) if "bounds" in cma_cfg else (0., 6.0),
        "seed": cma_cfg.get("seed", seed),
        "tolfun": cma_cfg.get("tolfun", 1e-6),
        "tolx": cma_cfg.get("tolx", 1e-6),
        "maxfevals": cma_cfg.get("maxfevals", None),
        "popsize": popsize,
        "verb_disp": cma_cfg.get("verb_disp", 1),
    }

    es = cma.CMAEvolutionStrategy(x0.tolist(), sigma0, options)
    n_jobs = cma_cfg.get("n_jobs", -1)  # Use all CPUs by default
    while not es.stop():
        X = es.ask()
        # Parallel evaluation of the objective function
        results = Parallel(n_jobs=n_jobs)(delayed(objective)(x) for x in X)
        es.tell(X, results)
        es.disp()
    es.result_pretty()
    return np.array(es.result.xbest), float(es.result.fbest)


