import os
import time
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd

import benders

@dataclass
class GenConfig:
    chi: float = 1.0
    alpha: float = 0.6
    delta: float = 1.0
    w_low: int = 0
    w_high: int = 100
    f_low: int = 1000
    f_high: int = 5000
    c_low: float = 1.0
    c_high: float = 100.0
    dense_w: bool = True  # if False, uses sparse random OD with density
    w_density: float = 0.25  # used only if dense_w=False


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def generate_instance_csv(
    out_dir: str,
    n_nodes: int,
    n_hubs: int,
    seed: int,
    cfg: GenConfig,
) -> None:
    """
    Generates one instance and writes:
      w.csv (i,j,w)
      f.csv (k,f)
      c1.csv (i,k,c1)
      c2.csv (k,m,c2)
      c3.csv (m,j,c3)
      scalars.csv (chi,alpha,delta)
    Labels are strings: "1", "2", ..., "n"
    Hubs are first n_hubs nodes: "1"..str(n_hubs)
    """
    assert 1 <= n_hubs <= n_nodes

    rng = np.random.default_rng(seed)
    _ensure_dir(out_dir)

    nodes = [str(i) for i in range(1, n_nodes + 1)]
    hubs = [str(i) for i in range(1, n_hubs + 1)]

    # scalars
    scalars = pd.DataFrame([{"chi": cfg.chi, "alpha": cfg.alpha, "delta": cfg.delta}])
    scalars.to_csv(os.path.join(out_dir, "scalars.csv"), index=False)

    # f(k)
    f_vals = rng.integers(cfg.f_low, cfg.f_high + 1, size=n_hubs)
    f_df = pd.DataFrame({"k": hubs, "f": f_vals})
    f_df.to_csv(os.path.join(out_dir, "f.csv"), index=False)

    # costs
    # c1(i,k)
    c1 = rng.uniform(cfg.c_low, cfg.c_high, size=(n_nodes, n_hubs))
    c1_df = pd.DataFrame(
        [(nodes[i], hubs[k], float(c1[i, k])) for i in range(n_nodes) for k in range(n_hubs)],
        columns=["i", "k", "c1"],
    )
    c1_df.to_csv(os.path.join(out_dir, "c1.csv"), index=False)

    # c2(k,m)
    c2 = rng.uniform(cfg.c_low, cfg.c_high, size=(n_hubs, n_hubs))
    c2_df = pd.DataFrame(
        [(hubs[k], hubs[m], float(c2[k, m])) for k in range(n_hubs) for m in range(n_hubs)],
        columns=["k", "m", "c2"],
    )
    c2_df.to_csv(os.path.join(out_dir, "c2.csv"), index=False)

    # c3(m,j)  (m is hub, j is node)
    c3 = rng.uniform(cfg.c_low, cfg.c_high, size=(n_hubs, n_nodes))
    c3_df = pd.DataFrame(
        [(hubs[m], nodes[j], float(c3[m, j])) for m in range(n_hubs) for j in range(n_nodes)],
        columns=["m", "j", "c3"],
    )
    c3_df.to_csv(os.path.join(out_dir, "c3.csv"), index=False)

    # w(i,j)
    if cfg.dense_w:
        w_mat = rng.integers(cfg.w_low, cfg.w_high + 1, size=(n_nodes, n_nodes))
        np.fill_diagonal(w_mat, 0)
        w_rows = [(nodes[i], nodes[j], int(w_mat[i, j])) for i in range(n_nodes) for j in range(n_nodes)]
    else:
        # sparse OD: choose a subset of pairs
        all_pairs = [(i, j) for i in range(n_nodes) for j in range(n_nodes) if i != j]
        rng.shuffle(all_pairs)
        m = max(1, int(cfg.w_density * len(all_pairs)))
        chosen = set(all_pairs[:m])
        w_rows = []
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i == j:
                    wij = 0
                elif (i, j) in chosen:
                    wij = int(rng.integers(cfg.w_low, cfg.w_high + 1))
                else:
                    wij = 0
                w_rows.append((nodes[i], nodes[j], wij))

    w_df = pd.DataFrame(w_rows, columns=["i", "j", "w"])
    w_df.to_csv(os.path.join(out_dir, "w.csv"), index=False)


def benchmark_suite(
    base_dir: str,
    sizes: List[int],
    hub_ratio: float,
    reps: int,
    start_seed: int,
    cfg: GenConfig,
    max_iterations: int,
    tolerance: float,
    core_point_value: float,
    add_user_cuts: bool,
    use_pareto: bool,
    mip_gap: float,
    time_limit: Optional[float],
    log_callback: bool,
) -> pd.DataFrame:
    """
    For each size n in sizes and repetition r:
      - generate instance under base_dir/instances/n{n}_r{r}
      - run outer-loop Benders
      - run callback Benders (if available)
    Return a DataFrame with per-run results.
    """
    _ensure_dir(base_dir)

    records: List[Dict] = []

    for n in sizes:
        h = max(1, int(round(hub_ratio * n)))
        h = min(h, n)

        for r in range(reps):
            seed = start_seed + 1000 * n + r
            inst_dir = os.path.join(base_dir, f"n{n}_h{h}_r{r}_seed{seed}")
            generate_instance_csv(inst_dir, n_nodes=n, n_hubs=h, seed=seed, cfg=cfg)

            inst = benders.load_instance_from_csv(inst_dir)

            # --- Outer-loop
            t0 = time.perf_counter()
            outer = benders.solve_hub_location_benders_outer_loop(
                inst,
                max_iterations=max_iterations,
                tolerance=tolerance,
                core_point_value=core_point_value,
                log_master=False,
                log_lp=False,
            )
            t_outer = time.perf_counter() - t0

            rec = {
                "n_nodes": n,
                "n_hubs": h,
                "rep": r,
                "seed": seed,
                "outer_time_sec": float(t_outer),
                "outer_iters": int(outer.iterations),
                "outer_ub": float(outer.best_upper_bound),
                "outer_lb": float(outer.last_lower_bound),
            }

            # --- Callback-based (if available in your environment)
            cb_time = np.nan
            cb_obj = np.nan
            cb_ok = False
            if hasattr(benders, "solve_hub_location_benders_callbacks"):
                try:
                    t1 = time.perf_counter()
                    cb = benders.solve_hub_location_benders_callbacks(
                        inst,
                        add_user_cuts=add_user_cuts,
                        core_point_value=core_point_value,
                        use_pareto=use_pareto,
                        mip_gap=mip_gap,
                        time_limit=time_limit,
                        log_output=log_callback,
                    )
                    cb_time = time.perf_counter() - t1
                    cb_obj = float(cb.objective)
                    cb_ok = True
                except Exception as e:
                    # keep NaNs, but log short info
                    print(f"[callback failed] n={n} r={r} -> {type(e).__name__}: {e}")

            rec.update({
                "callback_ok": cb_ok,
                "callback_time_sec": float(cb_time) if cb_ok else np.nan,
                "callback_obj": float(cb_obj) if cb_ok else np.nan,
            })

            records.append(rec)

            print(
                f"[done] n={n} h={h} r={r} | outer={t_outer:.3f}s it={outer.iterations} "
                f"| callback={'{:.3f}s'.format(cb_time) if cb_ok else 'N/A'}"
            )

    df = pd.DataFrame.from_records(records)
    return df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summary by size: mean/std times (outer, callback), and mean iterations for outer.
    """
    g = df.groupby(["n_nodes", "n_hubs"], as_index=False)
    out = g.agg(
        reps=("rep", "count"),
        outer_time_mean=("outer_time_sec", "mean"),
        outer_time_std=("outer_time_sec", "std"),
        outer_iters_mean=("outer_iters", "mean"),
        outer_iters_std=("outer_iters", "std"),
        callback_ok_count=("callback_ok", "sum"),
        callback_time_mean=("callback_time_sec", "mean"),
        callback_time_std=("callback_time_sec", "std"),
    )
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, default="instances_benchmark")
    parser.add_argument("--sizes", type=int, nargs="+", default=[10, 15, 20, 25])
    parser.add_argument("--hub_ratio", type=float, default=1.0, help="n_hubs = round(hub_ratio * n_nodes)")
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--start_seed", type=int, default=123)

    # generation config
    parser.add_argument("--chi", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=0.6)
    parser.add_argument("--delta", type=float, default=1.0)
    parser.add_argument("--dense_w", action="store_true", help="dense w matrix (recommended)")
    parser.add_argument("--sparse_w", action="store_true", help="sparse OD with density")
    parser.add_argument("--w_density", type=float, default=0.25)

    # algorithm config
    parser.add_argument("--max_iter", type=int, default=100)
    parser.add_argument("--tol", type=float, default=1e-1)
    parser.add_argument("--core", type=float, default=0.5)

    # callback config
    parser.add_argument("--add_user_cuts", action="store_true")
    parser.add_argument("--no_pareto", action="store_true")
    parser.add_argument("--mip_gap", type=float, default=0.0)
    parser.add_argument("--time_limit", type=float, default=None)
    parser.add_argument("--log_callback", action="store_true")

    args = parser.parse_args()

    cfg = GenConfig(
        chi=args.chi,
        alpha=args.alpha,
        delta=args.delta,
        dense_w=(not args.sparse_w),
        w_density=args.w_density,
    )

    df = benchmark_suite(
        base_dir=args.base_dir,
        sizes=args.sizes,
        hub_ratio=args.hub_ratio,
        reps=args.reps,
        start_seed=args.start_seed,
        cfg=cfg,
        max_iterations=args.max_iter,
        tolerance=args.tol,
        core_point_value=args.core,
        add_user_cuts=args.add_user_cuts,
        use_pareto=(not args.no_pareto),
        mip_gap=args.mip_gap,
        time_limit=args.time_limit,
        log_callback=args.log_callback,
    )

    summary = summarize(df)

    _ensure_dir(args.base_dir)
    df_path = os.path.join(args.base_dir, "summary.csv")
    sum_path = os.path.join(args.base_dir, "summary_by_size.csv")
    df.to_csv(df_path, index=False)
    summary.to_csv(sum_path, index=False)

    print("\n=== Per-run results saved to ===")
    print(df_path)
    print("\n=== Aggregated results saved to ===")
    print(sum_path)

    print("\n=== Aggregated table ===")
    with pd.option_context("display.max_rows", 200, "display.max_columns", 200):
        print(summary)


if __name__ == "__main__":
    main()