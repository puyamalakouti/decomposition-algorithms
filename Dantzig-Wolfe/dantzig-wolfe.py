from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

from docplex.mp.model import Model


# -----------------------------
# Data structures
# -----------------------------
@dataclass(frozen=True)
class MultiCommodityTransportInstance:
    products: List[str]
    origins: List[str]
    destinations: List[str]
    supply: Dict[Tuple[str, str], float]        # (product, origin) -> value
    demand: Dict[Tuple[str, str], float]        # (product, destination) -> value
    unit_cost: Dict[Tuple[str, str, str], float]  # (product, origin, destination) -> value
    arc_capacity: Dict[Tuple[str, str], float]  # (origin, destination) -> value


@dataclass
class FlowPlan:
    product: str
    flow: Dict[Tuple[str, str], float]          # (origin, destination) -> x
    true_cost: float                             # sum unit_cost * flow


# -----------------------------
# Example instance (embedded)
# -----------------------------
def build_example_instance() -> MultiCommodityTransportInstance:
    products = ["p1", "p2", "p3"]
    origins = ["i1", "i2", "i3"]
    destinations = ["j1", "j2", "j3", "j4", "j5", "j6", "j7"]

    supply = {
        ("p1", "i1"): 400, ("p1", "i2"): 700, ("p1", "i3"): 800,
        ("p2", "i1"): 800, ("p2", "i2"): 1600, ("p2", "i3"): 1800,
        ("p3", "i1"): 200, ("p3", "i2"): 300, ("p3", "i3"): 300,
    }

    demand = {
        ("p1", "j1"): 300, ("p1", "j2"): 300, ("p1", "j3"): 100, ("p1", "j4"): 75,  ("p1", "j5"): 650, ("p1", "j6"): 225, ("p1", "j7"): 250,
        ("p2", "j1"): 500, ("p2", "j2"): 750, ("p2", "j3"): 400, ("p2", "j4"): 250, ("p2", "j5"): 950, ("p2", "j6"): 850, ("p2", "j7"): 500,
        ("p3", "j1"): 100, ("p3", "j2"): 100, ("p3", "j3"): 0,   ("p3", "j4"): 50,  ("p3", "j5"): 200, ("p3", "j6"): 100, ("p3", "j7"): 250,
    }

    arc_capacity = {(i, j): 700.0 for i in origins for j in destinations}

    unit_cost: Dict[Tuple[str, str, str], float] = {}

    def add_cost(prod: str, org: str, row: List[float]):
        assert len(row) == len(destinations)
        for dest, val in zip(destinations, row):
            unit_cost[(prod, org, dest)] = float(val)

    add_cost("p1", "i1", [30, 10, 8, 10, 11, 71, 6])
    add_cost("p1", "i2", [22, 7, 10, 7, 21, 82, 13])
    add_cost("p1", "i3", [19, 11, 12, 10, 25, 83, 15])

    add_cost("p2", "i1", [39, 14, 11, 14, 16, 82, 8])
    add_cost("p2", "i2", [27, 9, 12, 9, 26, 95, 17])
    add_cost("p2", "i3", [24, 14, 17, 13, 28, 99, 20])

    add_cost("p3", "i1", [41, 15, 12, 16, 17, 86, 8])
    add_cost("p3", "i2", [29, 9, 13, 9, 28, 99, 18])
    add_cost("p3", "i3", [26, 14, 17, 13, 31, 104, 20])

    # balance check
    for p in products:
        ssum = sum(supply[(p, i)] for i in origins)
        dsum = sum(demand[(p, j)] for j in destinations)
        if abs(ssum - dsum) > 1e-6:
            raise ValueError(f"Unbalanced data for {p}: supply={ssum} demand={dsum}")

    return MultiCommodityTransportInstance(
        products=products,
        origins=origins,
        destinations=destinations,
        supply=supply,
        demand=demand,
        unit_cost=unit_cost,
        arc_capacity=arc_capacity,
    )


# -----------------------------
# Transportation subproblem
# -----------------------------
def solve_transportation_plan(
    origins: List[str],
    destinations: List[str],
    supply: Dict[str, float],
    demand: Dict[str, float],
    objective_coef: Dict[Tuple[str, str], float],
    log_output: bool = False,
) -> Tuple[float, Dict[Tuple[str, str], float]]:
    mdl = Model("transport_plan")

    x = {(i, j): mdl.continuous_var(lb=0.0, name=f"x_{i}_{j}") for i in origins for j in destinations}

    # objective
    mdl.minimize(mdl.sum(objective_coef[(i, j)] * x[(i, j)] for i in origins for j in destinations))

    # supply balance
    for i in origins:
        mdl.add_constraint(mdl.sum(x[(i, j)] for j in destinations) == float(supply[i]))

    # demand balance
    for j in destinations:
        mdl.add_constraint(mdl.sum(x[(i, j)] for i in origins) == float(demand[j]))

    sol = mdl.solve(log_output=log_output)
    if sol is None:
        raise RuntimeError("Transportation subproblem: no solution.")

    flows = {(i, j): float(sol.get_value(x[(i, j)])) for i in origins for j in destinations}
    return float(sol.objective_value), flows


def pricing_problem(
    inst: MultiCommodityTransportInstance,
    product: str,
    arc_multipliers: Dict[Tuple[str, str], float],
    convexity_multiplier: float,
    phase: int,
    log_output: bool = False,
) -> Tuple[float, Dict[Tuple[str, str], float], float]:
    """
    Returns:
      reduced_cost, flow_plan, true_cost

    Phase 1: min sum (-pi_ij) x_ij  - alpha_p
    Phase 2: min sum (c_ij - pi_ij) x_ij - alpha_p
    """
    I, J = inst.origins, inst.destinations

    s = {i: float(inst.supply[(product, i)]) for i in I}
    d = {j: float(inst.demand[(product, j)]) for j in J}

    if phase == 1:
        coef = {(i, j): -float(arc_multipliers[(i, j)]) for i in I for j in J}
    else:
        coef = {(i, j): float(inst.unit_cost[(product, i, j)]) - float(arc_multipliers[(i, j)]) for i in I for j in J}

    val, flows = solve_transportation_plan(I, J, s, d, coef, log_output=log_output)
    reduced_cost = float(val) - float(convexity_multiplier)

    true_cost = sum(float(inst.unit_cost[(product, i, j)]) * flows[(i, j)] for i in I for j in J)
    return reduced_cost, flows, float(true_cost)


# -----------------------------
# Restricted master (primal)
# -----------------------------
def solve_restricted_master_primal(
    inst: MultiCommodityTransportInstance,
    plans: Dict[str, List[FlowPlan]],
    phase: int,
    slack_fixed_zero: bool,
    log_output: bool = False,
) -> Tuple[float, Dict[Tuple[str, int], float], float]:
    mdl = Model("restricted_master_primal")

    # keep LP behavior stable
    mdl.parameters.preprocessing.presolve = 0
    mdl.parameters.lpmethod = 2  # dual simplex

    lam = {}
    for p in inst.products:
        for k in range(len(plans[p])):
            lam[(p, k)] = mdl.continuous_var(lb=0.0, name=f"lam_{p}_{k}")

    if phase == 1 and not slack_fixed_zero:
        slack = mdl.continuous_var(lb=0.0, name="slack")
    else:
        slack = None  # treated as 0

    # objective
    if phase == 1 and slack is not None:
        mdl.minimize(slack)
    else:
        mdl.minimize(mdl.sum(plans[p][k].true_cost * lam[(p, k)]
                             for p in inst.products for k in range(len(plans[p]))))

    # convexity
    for p in inst.products:
        mdl.add_constraint(mdl.sum(lam[(p, k)] for k in range(len(plans[p]))) == 1.0,
                           ctname=f"conv_{p}")

    # capacity
    for i in inst.origins:
        for j in inst.destinations:
            lhs = mdl.sum(plans[p][k].flow[(i, j)] * lam[(p, k)]
                          for p in inst.products for k in range(len(plans[p])))
            rhs = float(inst.arc_capacity[(i, j)]) + (slack if slack is not None else 0.0)
            mdl.add_constraint(lhs <= rhs, ctname=f"cap_{i}_{j}")

    sol = mdl.solve(log_output=log_output)
    if sol is None:
        raise RuntimeError("Restricted master (primal) infeasible or no solution.")

    lam_val = {(p, k): float(sol.get_value(lam[(p, k)])) for p in inst.products for k in range(len(plans[p]))}
    slack_val = float(sol.get_value(slack)) if slack is not None else 0.0
    return float(sol.objective_value), lam_val, slack_val


# -----------------------------
# Restricted master (dual) — explicit
# -----------------------------
def solve_restricted_master_dual(
    inst: MultiCommodityTransportInstance,
    plans: Dict[str, List[FlowPlan]],
    phase: int,
    slack_present: bool,
    log_output: bool = False,
) -> Tuple[Dict[Tuple[str, str], float], Dict[str, float], float]:
    """
    Dual variables:
      alpha_p free       (convexity equality)
      pi_ij <= 0         (capacity <= constraints in a minimization primal)

    Dual objective:
      max  sum_p alpha_p + sum_{i,j} cap_ij * pi_ij

    Dual constraints, one per column (p,k):
      phase 1: alpha_p + sum_{i,j} x_{p,k}(i,j)*pi_ij <= 0
      phase 2: alpha_p + sum_{i,j} x_{p,k}(i,j)*pi_ij <= cost_{p,k}

    If slack exists in primal (phase 1 before fixing to 0):
      slack has coefficient -1 in every capacity constraint and cost 1,
      -> constraint: sum_{i,j} pi_ij >= -1
    """
    mdl = Model("restricted_master_dual")

    mdl.parameters.preprocessing.presolve = 0
    mdl.parameters.lpmethod = 2  # dual simplex

    INF = mdl.infinity
    alpha = {p: mdl.continuous_var(lb=-INF, ub=INF, name=f"alpha_{p}") for p in inst.products}
    pi = {(i, j): mdl.continuous_var(lb=-INF, ub=0.0, name=f"pi_{i}_{j}") for i in inst.origins for j in inst.destinations}

    mdl.maximize(
        mdl.sum(alpha[p] for p in inst.products) +
        mdl.sum(float(inst.arc_capacity[(i, j)]) * pi[(i, j)] for i in inst.origins for j in inst.destinations)
    )

    for p in inst.products:
        for k, col in enumerate(plans[p]):
            lhs = alpha[p] + mdl.sum(col.flow[(i, j)] * pi[(i, j)] for i in inst.origins for j in inst.destinations)
            rhs = 0.0 if phase == 1 else float(col.true_cost)
            mdl.add_constraint(lhs <= rhs, ctname=f"dual_col_{p}_{k}")

    if slack_present:
        mdl.add_constraint(mdl.sum(pi[(i, j)] for i in inst.origins for j in inst.destinations) >= -1.0,
                           ctname="dual_slack_feas")

    sol = mdl.solve(log_output=log_output)
    if sol is None:
        raise RuntimeError("Restricted master (dual) infeasible or no solution.")

    pi_val = {(i, j): float(sol.get_value(pi[(i, j)])) for i in inst.origins for j in inst.destinations}
    alpha_val = {p: float(sol.get_value(alpha[p])) for p in inst.products}
    return pi_val, alpha_val, float(sol.objective_value)


# -----------------------------
# Column generation (two-phase)
# -----------------------------
def run_column_generation(
    inst: MultiCommodityTransportInstance,
    max_iter: int = 100,
    reduced_cost_threshold: float = -1e-4,
    slack_to_phase2_threshold: float = 1e-4,
    log_master: bool = False,
    log_pricing: bool = False,
) -> Dict[str, List[FlowPlan]]:
    plans: Dict[str, List[FlowPlan]] = {p: [] for p in inst.products}

    # initial multipliers = 0 => build one initial plan per product
    pi0 = {(i, j): 0.0 for i in inst.origins for j in inst.destinations}
    for p in inst.products:
        rc, flow, true_cost = pricing_problem(
            inst, p, arc_multipliers=pi0, convexity_multiplier=0.0, phase=2, log_output=log_pricing
        )
        plans[p].append(FlowPlan(product=p, flow=flow, true_cost=true_cost))

    phase = 1
    slack_fixed_zero = False

    t0 = time.perf_counter()

    for it in range(1, max_iter + 1):
        # primal solve
        mp_obj, lam_val, slack_val = solve_restricted_master_primal(
            inst, plans, phase=phase, slack_fixed_zero=slack_fixed_zero, log_output=log_master
        )

        # phase switching logic
        if phase == 1 and slack_val < slack_to_phase2_threshold:
            phase = 2
            slack_fixed_zero = True
            mp_obj, lam_val, slack_val = solve_restricted_master_primal(
                inst, plans, phase=phase, slack_fixed_zero=slack_fixed_zero, log_output=log_master
            )

        # dual solve (explicit)
        slack_present = (phase == 1 and not slack_fixed_zero)
        pi, alpha, dual_obj = solve_restricted_master_dual(
            inst, plans, phase=phase, slack_present=slack_present, log_output=log_master
        )

        # pricing
        added = 0
        for p in inst.products:
            rc, flow, true_cost = pricing_problem(
                inst, p, arc_multipliers=pi, convexity_multiplier=alpha[p], phase=phase, log_output=log_pricing
            )
            if rc < reduced_cost_threshold:
                plans[p].append(FlowPlan(product=p, flow=flow, true_cost=true_cost))
                added += 1

        col_counts = {p: len(plans[p]) for p in inst.products}
        print(
            f"iter={it:03d} | phase={phase} | master_obj={mp_obj:.6f} | dual_obj={dual_obj:.6f} "
            f"| slack={slack_val:.6f} | new_cols={added} | cols={col_counts}"
        )

        if added == 0 and phase == 2:
            break

    print(f"\nFinished in {time.perf_counter() - t0:.3f}s")
    return plans


# -----------------------------
# Recover final solution (flows)
# -----------------------------
def recover_final_solution(inst: MultiCommodityTransportInstance, plans: Dict[str, List[FlowPlan]], log_master: bool = False) -> None:
    mp_obj, lam_val, _ = solve_restricted_master_primal(inst, plans, phase=2, slack_fixed_zero=True, log_output=log_master)

    # x_hat(p,i,j) = sum_k flow_{p,k}(i,j) * lambda_{p,k}
    x_hat = {(p, i, j): 0.0 for p in inst.products for i in inst.origins for j in inst.destinations}
    for p in inst.products:
        for k, col in enumerate(plans[p]):
            lam = lam_val[(p, k)]
            for i in inst.origins:
                for j in inst.destinations:
                    x_hat[(p, i, j)] += col.flow[(i, j)] * lam

    total_cost = sum(x_hat[(p, i, j)] * float(inst.unit_cost[(p, i, j)])
                     for p in inst.products for i in inst.origins for j in inst.destinations)

    print("\n=== Final solution (phase 2, slack=0) ===")
    print(f"Objective (master): {mp_obj:.6f}")
    print(f"Total cost (reconstructed): {total_cost:.6f}")

    for p in inst.products:
        print(f"\nproduct={p}")
        for i in inst.origins:
            row = [x_hat[(p, i, j)] for j in inst.destinations]
            print(f"  {i}: " + "  ".join(f"{v:8.2f}" for v in row))


def main() -> None:
    inst = build_example_instance()
    plans = run_column_generation(
        inst,
        max_iter=100,
        reduced_cost_threshold=-1e-4,
        slack_to_phase2_threshold=1e-4,
        log_master=False,
        log_pricing=False,
    )
    recover_final_solution(inst, plans, log_master=False)


if __name__ == "__main__":
    main()


