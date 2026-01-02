from __future__ import annotations

import os
import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd
from docplex.mp.model import Model

try:
    import cplex
    from cplex.callbacks import LazyConstraintCallback, UserCutCallback
    from docplex.mp.callbacks.cb_mixin import ConstraintCallbackMixin
    _HAS_CPLEX_CALLBACKS = True
except Exception:
    _HAS_CPLEX_CALLBACKS = False


Node = str
Hub = str
ODPair = Tuple[Node, Node]


@dataclass(frozen=True)
class HubLocationInstance:
    nodes: List[Node]
    hubs: List[Hub]
    od_pairs: List[ODPair]  # (i,j) pairs provided in w.csv
    w: Dict[ODPair, float]
    f: Dict[Hub, float]
    c1: Dict[Tuple[Node, Hub], float]
    c2: Dict[Tuple[Hub, Hub], float]
    c3: Dict[Tuple[Hub, Node], float]
    chi: float
    alpha: float
    delta: float

    def route_cost(self, i: Node, k: Hub, m: Hub, j: Node) -> float:
        # ct(i,k,m,j) = chi*c1(i,k) + alpha*c2(k,m) + delta*c3(m,j)
        return (
            self.chi * self.c1[(i, k)]
            + self.alpha * self.c2[(k, m)]
            + self.delta * self.c3[(m, j)]
        )


@dataclass
class ExactSolveResult:
    objective: float
    y: Dict[Tuple[Node, Hub], int]


@dataclass
class OuterLoopBendersResult:
    best_upper_bound: float
    last_lower_bound: float
    best_y: Optional[Dict[Tuple[Node, Hub], int]]
    iterations: int
    lower_bounds: List[float]
    upper_bounds: List[float]
    wall_time_sec: float


@dataclass
class CallbackBendersResult:
    objective: float
    y: Dict[Tuple[Node, Hub], int]
    wall_time_sec: float
    # These are not "iterations" in the outer-loop sense; they're event logs:
    best_bound_trace: List[Tuple[float, float]]   # (time_sec, best_bound)
    incumbent_trace: List[Tuple[float, float]]    # (time_sec, incumbent_obj)


# ---------------------------------------------------------
# IO
# ---------------------------------------------------------
def load_instance_from_csv(folder: str = ".") -> HubLocationInstance:
    w_df = pd.read_csv(os.path.join(folder, "w.csv"))          # i,j,w
    f_df = pd.read_csv(os.path.join(folder, "f.csv"))          # k,f
    c1_df = pd.read_csv(os.path.join(folder, "c1.csv"))        # i,k,c1
    c2_df = pd.read_csv(os.path.join(folder, "c2.csv"))        # k,m,c2
    c3_df = pd.read_csv(os.path.join(folder, "c3.csv"))        # m,j,c3
    s_df = pd.read_csv(os.path.join(folder, "scalars.csv"))    # chi,alpha,delta

    chi = float(s_df.loc[0, "chi"])
    alpha = float(s_df.loc[0, "alpha"])
    delta = float(s_df.loc[0, "delta"])

    nodes = sorted(set(w_df["i"]).union(set(w_df["j"])))
    hubs = sorted(set(f_df["k"]))

    w = {(row.i, row.j): float(row.w) for row in w_df.itertuples(index=False)}
    f = {row.k: float(row.f) for row in f_df.itertuples(index=False)}
    c1 = {(row.i, row.k): float(row.c1) for row in c1_df.itertuples(index=False)}
    c2 = {(row.k, row.m): float(row.c2) for row in c2_df.itertuples(index=False)}
    c3 = {(row.m, row.j): float(row.c3) for row in c3_df.itertuples(index=False)}

    od_pairs = sorted(w.keys())

    return HubLocationInstance(
        nodes=nodes,
        hubs=hubs,
        od_pairs=od_pairs,
        w=w,
        f=f,
        c1=c1,
        c2=c2,
        c3=c3,
        chi=chi,
        alpha=alpha,
        delta=delta,
    )


# ---------------------------------------------------------
# Exact MIP (optional reference)
# ---------------------------------------------------------
def solve_hub_location_mip(
    inst: HubLocationInstance,
    mip_gap: float = 0.0,
    time_limit: Optional[float] = None,
    log_output: bool = False,
) -> ExactSolveResult:
    mdl = Model("hub_location_mip")
    mdl.parameters.mip.tolerances.mipgap = mip_gap
    if time_limit is not None:
        mdl.parameters.timelimit = time_limit

    I, K, OD = inst.nodes, inst.hubs, inst.od_pairs

    y = {(i, k): mdl.binary_var(name=f"y_{i}_{k}") for i in I for k in K}
    x = {(i, j, k, m): mdl.continuous_var(lb=0, name=f"x_{i}_{j}_{k}_{m}")
         for (i, j) in OD for k in K for m in K}

    fixed = mdl.sum(inst.f[k] * y[(k, k)] for k in K)
    transport = mdl.sum(
        inst.w[(i, j)] * inst.route_cost(i, k, m, j) * x[(i, j, k, m)]
        for (i, j) in OD for k in K for m in K
    )
    mdl.minimize(fixed + transport)

    for i in I:
        mdl.add_constraint(mdl.sum(y[(i, k)] for k in K) == 1)
    for i in I:
        for k in K:
            mdl.add_constraint(y[(i, k)] <= y[(k, k)])
    for (i, j) in OD:
        for k in K:
            mdl.add_constraint(mdl.sum(x[(i, j, k, m)] for m in K) == y[(i, k)])
    for (i, j) in OD:
        for m in K:
            mdl.add_constraint(mdl.sum(x[(i, j, k, m)] for k in K) == y[(j, m)])

    sol = mdl.solve(log_output=log_output)
    if sol is None:
        raise RuntimeError("Exact MIP solve failed.")
    obj = float(sol.objective_value)
    y_sol = {(i, k): int(round(sol.get_value(y[(i, k)]))) for i in I for k in K}
    return ExactSolveResult(objective=obj, y=y_sol)


# ---------------------------------------------------------
# Outer-loop Benders
# ---------------------------------------------------------
def build_dual_subproblem(inst: HubLocationInstance):
    dual = Model("dual_subproblem")
    INF = dual.infinity

    I, K, OD = inst.nodes, inst.hubs, inst.od_pairs

    sigma = {(i, j, k): dual.continuous_var(lb=-INF, ub=INF, name=f"sigma_{i}_{j}_{k}")
             for (i, j) in OD for k in K}
    pi = {(i, j, m): dual.continuous_var(lb=-INF, ub=INF, name=f"pi_{i}_{j}_{m}")
          for (i, j) in OD for m in K}

    for (i, j) in OD:
        wij = inst.w[(i, j)]
        for k in K:
            for m in K:
                rhs = wij * inst.route_cost(i, k, m, j)
                dual.add_constraint(sigma[(i, j, k)] + pi[(i, j, m)] <= rhs)

    # for coefficient aggregation
    sigma_by_ik = {(i, k): [] for i in I for k in K}
    pi_by_jm = {(j, m): [] for j in I for m in K}
    for (i, j) in OD:
        for k in K:
            sigma_by_ik[(i, k)].append(sigma[(i, j, k)])
        for m in K:
            pi_by_jm[(j, m)].append(pi[(i, j, m)])

    S = {(i, k): dual.sum(sigma_by_ik[(i, k)]) for i in I for k in K}
    P = {(j, m): dual.sum(pi_by_jm[(j, m)]) for j in I for m in K}

    return dual, sigma, pi, sigma_by_ik, pi_by_jm, S, P


def solve_hub_location_benders_outer_loop(
    inst: HubLocationInstance,
    max_iterations: int = 100,
    tolerance: float = 1e-1,
    core_point_value: float = 0.5,
    log_master: bool = False,
    log_lp: bool = False,
) -> OuterLoopBendersResult:
    t0 = time.perf_counter()

    I, K, OD = inst.nodes, inst.hubs, inst.od_pairs

    # Master: minimize lb
    master = Model("benders_master_outer")
    y = {(i, k): master.binary_var(name=f"y_{i}_{k}") for i in I for k in K}
    lb = master.continuous_var(lb=-master.infinity, name="lb")

    for i in I:
        master.add_constraint(master.sum(y[(i, k)] for k in K) == 1)
    for i in I:
        for k in K:
            master.add_constraint(y[(i, k)] <= y[(k, k)])

    master.add_constraint(lb >= master.sum(inst.f[k] * y[(k, k)] for k in K))
    master.minimize(lb)

    dual, sigma_var, pi_var, sigma_by_ik, pi_by_jm, S, P = build_dual_subproblem(inst)
    core = {(i, k): core_point_value for i in I for k in K}

    best_ub = float("inf")
    best_y: Optional[Dict[Tuple[Node, Hub], int]] = None
    last_lb = -float("inf")

    lb_hist: List[float] = []
    ub_hist: List[float] = []

    # initial master solve
    msol = master.solve(log_output=log_master)
    if msol is None:
        raise RuntimeError("Outer-loop master initial solve failed.")
    y_hat = {(i, k): float(msol.get_value(y[(i, k)])) for i in I for k in K}
    last_lb = float(msol.get_value(lb))
    lb_hist.append(last_lb)
    ub_hist.append(float("nan"))

    for it in range(1, max_iterations + 1):
        fixed_const = sum(inst.f[k] * round(y_hat[(k, k)]) for k in K)

        # 1) Dual maximize at current y_hat (uses y as coefficients)
        dual_obj = dual.sum(y_hat[(i, k)] * S[(i, k)] for i in I for k in K) + \
                   dual.sum(y_hat[(j, m)] * P[(j, m)] for j in I for m in K)
        dual.maximize(dual_obj)

        dsol = dual.solve(log_output=log_lp)
        if dsol is None:
            raise RuntimeError(f"Outer-loop dual solve failed at iter={it}.")
        ub_candidate = float(dsol.objective_value) + fixed_const

        if ub_candidate < best_ub:
            best_ub = ub_candidate
            best_y = {(i, k): int(round(y_hat[(i, k)])) for i in I for k in K}

        # 2) Pareto enhancement on same optimal face (optional)
        face_eq = dual.add_constraint(dual_obj == float(dsol.objective_value))
        pareto_obj = dual.sum(core[(i, k)] * S[(i, k)] for i in I for k in K) + \
                     dual.sum(core[(j, m)] * P[(j, m)] for j in I for m in K)
        dual.maximize(pareto_obj)
        psol = dual.solve(log_output=log_lp)
        if psol is None:
            sol_used = dsol
        else:
            sol_used = psol
        dual.remove_constraint(face_eq)

        # 3) Build aggregated cut coefficients
        coeff = {(i, k): 0.0 for i in I for k in K}
        for i in I:
            for k in K:
                ssum = sum(sol_used.get_value(v) for v in sigma_by_ik[(i, k)])
                psum = sum(sol_used.get_value(v) for v in pi_by_jm[(i, k)])  # j=i, m=k
                coeff[(i, k)] = float(ssum + psum)

        cut_expr = master.sum(inst.f[k] * y[(k, k)] for k in K) + master.sum(
            coeff[(i, k)] * y[(i, k)] for i in I for k in K
        )
        master.add_constraint(lb >= cut_expr)

        # 4) Resolve master
        msol = master.solve(log_output=log_master)
        if msol is None:
            raise RuntimeError(f"Outer-loop master solve failed at iter={it}.")
        last_lb = float(msol.get_value(lb))
        y_hat = {(i, k): float(msol.get_value(y[(i, k)])) for i in I for k in K}

        lb_hist.append(last_lb)
        ub_hist.append(best_ub)

        gap = best_ub - last_lb
        open_hubs = [k for k in K if int(round(y_hat[(k, k)])) == 1]
        print(f"[outer] iter={it:03d} | LB={last_lb:.6f} | UB={best_ub:.6f} | gap={gap:.6f} | open={open_hubs}")

        if gap <= tolerance:
            return OuterLoopBendersResult(
                best_upper_bound=best_ub,
                last_lower_bound=last_lb,
                best_y=best_y,
                iterations=it,
                lower_bounds=lb_hist,
                upper_bounds=ub_hist,
                wall_time_sec=time.perf_counter() - t0,
            )

    return OuterLoopBendersResult(
        best_upper_bound=best_ub,
        last_lower_bound=last_lb,
        best_y=best_y,
        iterations=max_iterations,
        lower_bounds=lb_hist,
        upper_bounds=ub_hist,
        wall_time_sec=time.perf_counter() - t0,
    )


# ---------------------------------------------------------
# Plotting (academic defaults)
# ---------------------------------------------------------
def plot_outer_bounds(lb: List[float], ub: List[float], save_path: str = "bounds_outer.png") -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 12,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "legend.fontsize": 11,
        "figure.dpi": 120,
    })

    iters = list(range(len(lb)))
    ub_clean = [math.nan if (u is None or (isinstance(u, float) and (math.isinf(u) or math.isnan(u)))) else float(u)
                for u in ub]

    plt.figure()
    plt.plot(iters, lb, label="Lower bound")
    plt.plot(iters, ub_clean, label="Upper bound")
    plt.xlabel("Iteration")
    plt.ylabel("Objective value")
    plt.grid(True)
    plt.legend()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_callback_traces(
    best_bound_trace: List[Tuple[float, float]],
    incumbent_trace: List[Tuple[float, float]],
    save_path: str = "traces_callback.png",
) -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 12,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "legend.fontsize": 11,
        "figure.dpi": 120,
    })

    plt.figure()

    if best_bound_trace:
        t, v = zip(*best_bound_trace)
        plt.plot(list(t), list(v), label="Best bound")

    if incumbent_trace:
        t, v = zip(*incumbent_trace)
        plt.plot(list(t), list(v), label="Incumbent objective")

    plt.xlabel("Time (s)")
    plt.ylabel("Objective value")
    plt.grid(True)
    plt.legend()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


# ---------------------------------------------------------
# Callback-based Benders (Lazy + optional UserCut)
# ---------------------------------------------------------
if _HAS_CPLEX_CALLBACKS:
    class DualTransportSolver:
        def __init__(self, inst: HubLocationInstance, core_point_value: float = 0.5):
            self.inst = inst
            I, K, OD = inst.nodes, inst.hubs, inst.od_pairs

            dual = Model("dual_transport")
            INF = dual.infinity

            sigma = {(i, j, k): dual.continuous_var(lb=-INF, ub=INF, name=f"sigma_{i}_{j}_{k}")
                     for (i, j) in OD for k in K}
            pi = {(i, j, m): dual.continuous_var(lb=-INF, ub=INF, name=f"pi_{i}_{j}_{m}")
                  for (i, j) in OD for m in K}

            for (i, j) in OD:
                wij = inst.w[(i, j)]
                for k in K:
                    for m in K:
                        rhs = wij * inst.route_cost(i, k, m, j)
                        dual.add_constraint(sigma[(i, j, k)] + pi[(i, j, m)] <= rhs)

            sigma_by_ik = {(i, k): [] for i in I for k in K}
            pi_by_jm = {(j, m): [] for j in I for m in K}
            for (i, j) in OD:
                for k in K:
                    sigma_by_ik[(i, k)].append(sigma[(i, j, k)])
                for m in K:
                    pi_by_jm[(j, m)].append(pi[(i, j, m)])

            S = {(i, k): dual.sum(sigma_by_ik[(i, k)]) for i in I for k in K}
            P = {(j, m): dual.sum(pi_by_jm[(j, m)]) for j in I for m in K}

            core = {(i, k): core_point_value for i in I for k in K}

            self.dual = dual
            self.sigma_by_ik = sigma_by_ik
            self.pi_by_jm = pi_by_jm
            self.S = S
            self.P = P
            self.core = core

        def separate(self, y_val: Dict[Tuple[Node, Hub], float], use_pareto: bool = True) -> Tuple[float, Dict[Tuple[Node, Hub], float]]:
            inst = self.inst
            dual = self.dual
            I, K = inst.nodes, inst.hubs

            obj = dual.sum(y_val[(i, k)] * self.S[(i, k)] for i in I for k in K) + \
                  dual.sum(y_val[(j, m)] * self.P[(j, m)] for j in I for m in K)
            dual.maximize(obj)

            sol = dual.solve(log_output=False)
            if sol is None:
                raise RuntimeError("Dual subproblem solve failed.")
            q = float(sol.objective_value)

            if use_pareto:
                face_eq = dual.add_constraint(obj == q)
                pareto_obj = dual.sum(self.core[(i, k)] * self.S[(i, k)] for i in I for k in K) + \
                             dual.sum(self.core[(j, m)] * self.P[(j, m)] for j in I for m in K)
                dual.maximize(pareto_obj)
                sol2 = dual.solve(log_output=False)
                if sol2 is not None:
                    sol = sol2
                dual.remove_constraint(face_eq)

            coeff = {(i, k): 0.0 for i in I for k in K}
            for i in I:
                for k in K:
                    ssum = sum(sol.get_value(v) for v in self.sigma_by_ik[(i, k)])
                    psum = sum(sol.get_value(v) for v in self.pi_by_jm[(i, k)])  # j=i, m=k
                    coeff[(i, k)] = float(ssum + psum)

            return q, coeff


    class _BendersCutBase(ConstraintCallbackMixin):
        def __init__(self):
            super().__init__()
            self.inst: Optional[HubLocationInstance] = None
            self.y_vars: Optional[Dict[Tuple[Node, Hub], object]] = None
            self.theta_var: Optional[object] = None
            self.dual_solver: Optional[DualTransportSolver] = None
            self.eps: float = 1e-6
            self.use_pareto: bool = True

            # tracing
            self.t0: float = 0.0
            self.best_bound_trace: List[Tuple[float, float]] = []
            self.incumbent_trace: List[Tuple[float, float]] = []

        def _record_progress(self):
            # These CPLEX methods exist on MIP callbacks; guard just in case.
            now = time.perf_counter() - self.t0
            try:
                bb = float(self.get_best_objective_value())
                self.best_bound_trace.append((now, bb))
            except Exception:
                pass
            try:
                inc = float(self.get_objective_value())
                self.incumbent_trace.append((now, inc))
            except Exception:
                pass

        def _try_add_cut(self):
            self._record_progress()

            sol = self.make_solution_from_watched()

            assert self.inst is not None
            assert self.y_vars is not None
            assert self.theta_var is not None
            assert self.dual_solver is not None

            I, K = self.inst.nodes, self.inst.hubs

            y_val = {(i, k): float(sol.get_value(self.y_vars[(i, k)])) for i in I for k in K}
            theta_val = float(sol.get_value(self.theta_var))

            q, coeff = self.dual_solver.separate(y_val, use_pareto=self.use_pareto)

            if q > theta_val + self.eps:
                cut_ct = (self.theta_var >= self.model.sum(coeff[(i, k)] * self.y_vars[(i, k)] for i in I for k in K))
                cpx_lhs, cpx_sense, cpx_rhs = self.linear_ct_to_cplex(cut_ct)

                # IMPORTANT: CPLEX callback add() uses positional arguments
                try:
                    self.add(cpx_lhs, cpx_sense, cpx_rhs)
                except TypeError:
                    # Fallback for some variants
                    self.add(cut=cpx_lhs, sense=cpx_sense, rhs=cpx_rhs)

    class BendersLazyCallback(_BendersCutBase, LazyConstraintCallback):
        def __init__(self, env):
            LazyConstraintCallback.__init__(self, env)
            _BendersCutBase.__init__(self)

        def __call__(self):
            self._try_add_cut()


    class BendersUserCutCallback(_BendersCutBase, UserCutCallback):
        def __init__(self, env):
            UserCutCallback.__init__(self, env)
            _BendersCutBase.__init__(self)

        def __call__(self):
            self._try_add_cut()


def solve_hub_location_benders_callbacks(
    inst: HubLocationInstance,
    add_user_cuts: bool = True,
    core_point_value: float = 0.5,
    use_pareto: bool = True,
    mip_gap: float = 0.0,
    time_limit: Optional[float] = None,
    log_output: bool = True,
) -> CallbackBendersResult:
    if not _HAS_CPLEX_CALLBACKS:
        raise RuntimeError(
            "CPLEX callback API is not available in this environment. "
            "Install/configure IBM ILOG CPLEX with its Python API to use callback-based Benders."
        )

    t0 = time.perf_counter()
    I, K = inst.nodes, inst.hubs

    master = Model("benders_master_callback")
    master.parameters.mip.tolerances.mipgap = mip_gap
    if time_limit is not None:
        master.parameters.timelimit = time_limit

    # y binaries + theta for transport lower bound
    y = {(i, k): master.binary_var(name=f"y_{i}_{k}") for i in I for k in K}
    theta = master.continuous_var(lb=0.0, name="theta")

    master.minimize(master.sum(inst.f[k] * y[(k, k)] for k in K) + theta)

    for i in I:
        master.add_constraint(master.sum(y[(i, k)] for k in K) == 1)
    for i in I:
        for k in K:
            master.add_constraint(y[(i, k)] <= y[(k, k)])

    dual_solver = DualTransportSolver(inst, core_point_value=core_point_value)

    # Lazy cuts (integer incumbents)
    lazy_cb = master.register_callback(BendersLazyCallback)
    lazy_cb.inst = inst
    lazy_cb.y_vars = y
    lazy_cb.theta_var = theta
    lazy_cb.dual_solver = dual_solver
    lazy_cb.eps = 1e-6
    lazy_cb.use_pareto = use_pareto
    lazy_cb.t0 = t0
    lazy_cb.register_watched_vars(list(y.values()) + [theta])

    user_cb = None
    if add_user_cuts:
        user_cb = master.register_callback(BendersUserCutCallback)
        user_cb.inst = inst
        user_cb.y_vars = y
        user_cb.theta_var = theta
        user_cb.dual_solver = dual_solver
        user_cb.eps = 1e-6
        user_cb.use_pareto = use_pareto
        user_cb.t0 = t0
        user_cb.register_watched_vars(list(y.values()) + [theta])

    sol = master.solve(log_output=log_output)
    if sol is None:
        raise RuntimeError("Callback-based solve failed (no solution).")

    obj = float(sol.objective_value)
    y_sol = {(i, k): int(round(sol.get_value(y[(i, k)]))) for i in I for k in K}

    # merge traces (both callbacks may exist)
    best_bound_trace = list(lazy_cb.best_bound_trace)
    incumbent_trace = list(lazy_cb.incumbent_trace)
    if user_cb is not None:
        best_bound_trace += user_cb.best_bound_trace
        incumbent_trace += user_cb.incumbent_trace

    # sort by time
    best_bound_trace.sort(key=lambda p: p[0])
    incumbent_trace.sort(key=lambda p: p[0])

    return CallbackBendersResult(
        objective=obj,
        y=y_sol,
        wall_time_sec=time.perf_counter() - t0,
        best_bound_trace=best_bound_trace,
        incumbent_trace=incumbent_trace,
    )


# ---------------------------------------------------------
# Runner: execute both methods and compare times
# ---------------------------------------------------------
def run_comparison(folder: str = ".", plot: bool = True) -> None:
    inst = load_instance_from_csv(folder)

    print("\n=== Method A: Outer-loop Benders ===")
    outer = solve_hub_location_benders_outer_loop(
        inst,
        max_iterations=100,
        tolerance=1e-1,
        core_point_value=0.5,
        log_master=False,
        log_lp=False,
    )
    print(f"[outer] time_sec={outer.wall_time_sec:.3f} | iters={outer.iterations} | UB={outer.best_upper_bound:.6f} | LB={outer.last_lower_bound:.6f}")
    if plot:
        plot_outer_bounds(outer.lower_bounds, outer.upper_bounds, save_path="bounds_outer.png")

    if _HAS_CPLEX_CALLBACKS:
        print("\n=== Method B: Callback-based Benders (Lazy + optional UserCut) ===")
        cb = solve_hub_location_benders_callbacks(
            inst,
            add_user_cuts=True,
            core_point_value=0.5,
            use_pareto=True,
            mip_gap=0.0,
            time_limit=None,
            log_output=True,
        )
        print(f"[callback] time_sec={cb.wall_time_sec:.3f} | obj={cb.objective:.6f}")
        if plot:
            plot_callback_traces(cb.best_bound_trace, cb.incumbent_trace, save_path="traces_callback.png")
    else:
        print("\n=== Method B: Callback-based Benders ===")
        print("Skipped: CPLEX callback API not available in this Python environment.")


# if __name__ == "__main__":
#     run_comparison(folder=".", plot=True)