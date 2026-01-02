from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from docplex.mp.model import Model


# -----------------------------
# Data structures
# -----------------------------
@dataclass
class FeatureSelectionSVMInstance:
    X: np.ndarray                 # (n, p)
    y: np.ndarray                 # (n,), in {-1,+1}
    sample_ids: List[str]
    feature_ids: List[str]
    C: float
    D: float
    u: np.ndarray                 # (p,)
    k1_idx: np.ndarray            # indices of K1 (nonzero weights in LP relaxation)
    z_hat: Optional[float]        # preprocessing Step 2 objective


@dataclass
class RestrictedMIPResult:
    objective: float
    w_plus: np.ndarray
    w_minus: np.ndarray
    gamma: float
    slack: np.ndarray
    ybin: np.ndarray


@dataclass
class SubgradientRunResult:
    best_lb: float
    best_ub: float
    best_mip: Optional[RestrictedMIPResult]
    lb_hist: List[float]
    ub_hist: List[float]
    zlam_hist: List[float]
    step_hist: List[float]
    scale_hist: List[float]       # "pi" scaling used in adaptive improvement steps
    lam_norm_hist: List[float]
    wall_time_sec: float


# -----------------------------
# Utilities
# -----------------------------
def ensure_pm1_labels(y_vals: np.ndarray) -> np.ndarray:
    yv = y_vals.astype(float)
    uniq = set(np.unique(yv).tolist())
    if uniq.issubset({-1.0, 1.0}):
        return yv.astype(int)
    if uniq.issubset({0.0, 1.0}):
        return np.where(yv > 0.5, 1, -1).astype(int)
    raise ValueError(f"Unsupported label set: {sorted(uniq)} (expected -1/+1 or 0/1).")


def objective_breakdown(C: float, D: float, w_plus: np.ndarray, w_minus: np.ndarray, slack: np.ndarray, ybin: np.ndarray) -> Dict[str, float]:
    l1 = float(np.sum(w_plus + w_minus))
    hinge = float(C * np.sum(slack))
    feat = float(D * np.sum(ybin))
    return {"L1": l1, "Slack": hinge, "FeaturePenalty": feat, "Total": l1 + hinge + feat}


def confusion_table(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    tab = np.zeros((2, 2), dtype=int)
    for yt, yp in zip(y_true, y_pred):
        r = 0 if yt == -1 else 1
        c = 0 if yp == -1 else 1
        tab[r, c] += 1
    return tab


def predict_labels(X: np.ndarray, w: np.ndarray, gamma: float, zero_as: int = 1) -> np.ndarray:
    dec = X @ w + float(gamma)
    y_pred = np.where(dec > 0, 1, -1).astype(int)
    if zero_as in (-1, 1):
        y_pred = np.where(dec == 0, int(zero_as), y_pred)
    return y_pred


# -----------------------------
# Plot bounds
# -----------------------------
def plot_bounds(lb_hist: List[float], ub_hist: List[float], out_png: str = "lr_bounds.png") -> None:
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.size"] = 11

    it = np.arange(1, len(lb_hist) + 1)
    plt.figure(figsize=(7.2, 4.2))
    plt.plot(it, lb_hist, label="Best LB (dual)")
    plt.plot(it, ub_hist, label="Best UB (feasible)")
    plt.xlabel("Iteration")
    plt.ylabel("Objective value")
    plt.title("Projected Subgradient Progress")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()
    print(f"[plot] saved: {out_png}")


# -----------------------------
# IO: read x/y
# -----------------------------
def read_xy_from_excel(xlsx_path: str) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    x_df = pd.read_excel(xlsx_path, sheet_name="x", index_col=0)
    sample_ids = [str(v).strip().replace(".0", "") for v in x_df.index.tolist()]
    feature_ids = [str(v).strip().replace(".0", "") for v in x_df.columns.tolist()]
    X = x_df.to_numpy(dtype=float)

    # y: NO HEADER, two columns [id, label]
    y_raw = pd.read_excel(xlsx_path, sheet_name="y", header=None)
    if y_raw.shape[1] < 2:
        raise ValueError("Sheet 'y' must have at least 2 columns: [id, label] with NO header.")
    y_raw = y_raw.iloc[:, :2].copy()
    y_raw.columns = ["id", "label"]
    y_raw["id"] = y_raw["id"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)

    y_map = dict(zip(y_raw["id"].tolist(), y_raw["label"].tolist()))
    missing = [sid for sid in sample_ids if sid not in y_map]
    if missing:
        raise ValueError(f"Missing labels for some samples (showing up to 10): {missing[:10]}")

    y_vals = np.array([y_map[sid] for sid in sample_ids], dtype=float)
    y_vals = ensure_pm1_labels(y_vals)
    return X, y_vals, sample_ids, feature_ids


# -----------------------------
# Step 1: LP relaxation -> w̄, K1 (nonzero weights)
# -----------------------------
def solve_lp_relaxation(X: np.ndarray, y: np.ndarray, C: float, log_output: bool = False) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, float]:
    n, p = X.shape
    I = range(n)
    K = range(p)

    mdl = Model("lp_relaxation")
    w_plus = [mdl.continuous_var(lb=0.0, name=f"wplus_{k}") for k in K]
    w_minus = [mdl.continuous_var(lb=0.0, name=f"wminus_{k}") for k in K]
    gamma = mdl.continuous_var(lb=-mdl.infinity, ub=mdl.infinity, name="gamma")
    slack = [mdl.continuous_var(lb=0.0, name=f"slack_{i}") for i in I]

    mdl.minimize(mdl.sum(w_plus[k] + w_minus[k] for k in K) + float(C) * mdl.sum(slack[i] for i in I))

    for i in I:
        expr = mdl.sum(float(X[i, k]) * (w_plus[k] - w_minus[k]) for k in K) + gamma
        if int(y[i]) == 1:
            mdl.add_constraint(expr >= 1.0 - slack[i])
        else:
            mdl.add_constraint(expr <= -1.0 + slack[i])

    sol = mdl.solve(log_output=log_output)
    if sol is None:
        raise RuntimeError("LP relaxation failed.")

    wplus_val = np.array([sol.get_value(v) for v in w_plus], dtype=float)
    wminus_val = np.array([sol.get_value(v) for v in w_minus], dtype=float)
    gamma_val = float(sol.get_value(gamma))
    slack_val = np.array([sol.get_value(v) for v in slack], dtype=float)
    obj_val = float(sol.objective_value)
    return wplus_val, wminus_val, gamma_val, slack_val, obj_val


# -----------------------------
# Restricted mixed-binary model
#   - force_allowed_one=True: y_k fixed to 1 for k in allowed_set, else 0.
#   - force_allowed_one=False: y_k decision only for allowed_set, forced 0 outside.
# -----------------------------
def solve_restricted_mip(
    X: np.ndarray,
    y: np.ndarray,
    C: float,
    D: float,
    u: np.ndarray,
    allowed_set: np.ndarray,
    u_override: Optional[float] = None,
    time_limit: Optional[float] = None,
    mip_gap: float = 0.0,
    log_output: bool = False,
    force_allowed_one: bool = False,
) -> RestrictedMIPResult:
    n, p = X.shape
    I = range(n)
    K = range(p)
    allowed = set(int(k) for k in allowed_set.tolist())

    mdl = Model("restricted_mip")
    mdl.parameters.mip.tolerances.mipgap = float(mip_gap)
    if time_limit is not None:
        mdl.parameters.timelimit = float(time_limit)

    w_plus = [mdl.continuous_var(lb=0.0, name=f"wplus_{k}") for k in K]
    w_minus = [mdl.continuous_var(lb=0.0, name=f"wminus_{k}") for k in K]
    gamma = mdl.continuous_var(lb=-mdl.infinity, ub=mdl.infinity, name="gamma")
    slack = [mdl.continuous_var(lb=0.0, name=f"slack_{i}") for i in I]
    ybin = [mdl.binary_var(name=f"y_{k}") for k in K]

    mdl.minimize(
        mdl.sum(w_plus[k] + w_minus[k] for k in K)
        + float(C) * mdl.sum(slack[i] for i in I)
        + float(D) * mdl.sum(ybin[k] for k in K)
    )

    for i in I:
        expr = mdl.sum(float(X[i, k]) * (w_plus[k] - w_minus[k]) for k in K) + gamma
        if int(y[i]) == 1:
            mdl.add_constraint(expr >= 1.0 - slack[i])
        else:
            mdl.add_constraint(expr <= -1.0 + slack[i])

    for k in K:
        if k in allowed:
            uk = float(u_override) if u_override is not None else float(u[k])
            mdl.add_constraint(w_plus[k] + w_minus[k] <= uk * ybin[k])
            if force_allowed_one:
                mdl.add_constraint(ybin[k] == 1)
        else:
            mdl.add_constraint(ybin[k] == 0)
            mdl.add_constraint(w_plus[k] == 0)
            mdl.add_constraint(w_minus[k] == 0)

    sol = mdl.solve(log_output=log_output)
    if sol is None:
        raise RuntimeError("Restricted MIP failed (no solution).")

    wplus_val = np.array([sol.get_value(v) for v in w_plus], dtype=float)
    wminus_val = np.array([sol.get_value(v) for v in w_minus], dtype=float)
    gamma_val = float(sol.get_value(gamma))
    slack_val = np.array([sol.get_value(v) for v in slack], dtype=float)
    y_val = np.array([int(round(sol.get_value(v))) for v in ybin], dtype=int)

    return RestrictedMIPResult(
        objective=float(sol.objective_value),
        w_plus=wplus_val,
        w_minus=wminus_val,
        gamma=gamma_val,
        slack=slack_val,
        ybin=y_val,
    )


# -----------------------------
# Step 3: max ||w||1 under budget <= z_hat
# -----------------------------
def solve_max_l1_under_budget(
    X: np.ndarray,
    y: np.ndarray,
    C: float,
    D: float,
    z_hat: float,
    u_big: float,
    log_output: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    n, p = X.shape
    K = range(p)

    mdl = Model("max_l1_under_budget")

    w_plus = [mdl.continuous_var(lb=0.0, name=f"wplus_{k}") for k in K]
    w_minus = [mdl.continuous_var(lb=0.0, name=f"wminus_{k}") for k in K]
    gamma = mdl.continuous_var(lb=-mdl.infinity, ub=mdl.infinity, name="gamma")

    pos = [i for i in range(n) if int(y[i]) == 1]
    neg = [i for i in range(n) if int(y[i]) == -1]
    xi = {i: mdl.continuous_var(lb=0.0, name=f"xi_{i}") for i in pos}
    zeta = {i: mdl.continuous_var(lb=0.0, name=f"zeta_{i}") for i in neg}

    y_cont = [mdl.continuous_var(lb=0.0, ub=1.0, name=f"y_{k}") for k in K]

    mdl.maximize(mdl.sum(w_plus[k] + w_minus[k] for k in K))

    for i in pos:
        expr = mdl.sum(float(X[i, k]) * (w_plus[k] - w_minus[k]) for k in K) + gamma
        mdl.add_constraint(expr <= xi[i] - 1.0)
    for i in neg:
        expr = mdl.sum(float(X[i, k]) * (w_plus[k] - w_minus[k]) for k in K) + gamma
        mdl.add_constraint(-expr <= zeta[i] - 1.0)

    mdl.add_constraint(
        mdl.sum(w_plus[k] + w_minus[k] for k in K)
        + float(C) * (mdl.sum(xi[i] for i in pos) + mdl.sum(zeta[i] for i in neg))
        + float(D) * mdl.sum(y_cont[k] for k in K)
        <= float(z_hat)
    )

    for k in K:
        mdl.add_constraint(w_plus[k] <= float(u_big) * y_cont[k])
        mdl.add_constraint(w_minus[k] <= float(u_big) * y_cont[k])
        mdl.add_constraint(w_plus[k] <= float(u_big))
        mdl.add_constraint(w_minus[k] <= float(u_big))

    sol = mdl.solve(log_output=log_output)
    if sol is None:
        raise RuntimeError("Max-L1-under-budget model failed.")

    wplus_val = np.array([sol.get_value(v) for v in w_plus], dtype=float)
    wminus_val = np.array([sol.get_value(v) for v in w_minus], dtype=float)
    return wplus_val, wminus_val


# -----------------------------
# Preprocessing: build instance with u
# -----------------------------
def build_instance_with_preprocessing(
    X: np.ndarray,
    y: np.ndarray,
    sample_ids: List[str],
    feature_ids: List[str],
    C: float,
    D: float,
    eps_zero: float = 1e-9,
    u_big: float = 1e10,
    step2_timelimit: Optional[float] = None,
    log_output: bool = False,
) -> FeatureSelectionSVMInstance:
    wplus0, wminus0, _, _, _ = solve_lp_relaxation(X=X, y=y, C=C, log_output=log_output)
    wbar = wplus0 - wminus0
    k1_idx = np.where(np.abs(wbar) > eps_zero)[0]  # nonzero weights

    u_tmp = np.full(X.shape[1], float(u_big), dtype=float)

    # Step 2: fix y_k=1 on K1, 0 otherwise
    res2 = solve_restricted_mip(
        X=X, y=y, C=C, D=D, u=u_tmp,
        allowed_set=k1_idx,
        u_override=float(u_big),
        force_allowed_one=True,
        time_limit=step2_timelimit,
        mip_gap=0.0,
        log_output=log_output,
    )
    z_hat = float(res2.objective)

    # Step 3: derive u from max-L1-under-budget
    wplus_star, wminus_star = solve_max_l1_under_budget(
        X=X, y=y, C=C, D=D, z_hat=z_hat, u_big=float(u_big), log_output=log_output
    )
    absw_star = wplus_star + wminus_star
    u_vec = np.minimum(float(u_big), absw_star)
    u_vec = np.maximum(u_vec, 1e-12)

    return FeatureSelectionSVMInstance(
        X=X, y=y, sample_ids=sample_ids, feature_ids=feature_ids,
        C=float(C), D=float(D), u=u_vec, k1_idx=k1_idx, z_hat=z_hat
    )


# -----------------------------
# Lagrangian LP z(lambda)
# -----------------------------
def solve_lagrangian_lp(inst: FeatureSelectionSVMInstance, lam: np.ndarray, log_output: bool = False) -> Tuple[float, np.ndarray, np.ndarray]:
    X, y, C, D, u = inst.X, inst.y, float(inst.C), float(inst.D), inst.u
    n, p = X.shape
    pos = [i for i in range(n) if int(y[i]) == 1]
    neg = [i for i in range(n) if int(y[i]) == -1]

    mdl = Model("lagrangian_lp")

    w_plus = [mdl.continuous_var(lb=0.0, ub=float(u[k]), name=f"wplus_{k}") for k in range(p)]
    w_minus = [mdl.continuous_var(lb=0.0, ub=float(u[k]), name=f"wminus_{k}") for k in range(p)]
    gamma = mdl.continuous_var(lb=-mdl.infinity, ub=mdl.infinity, name="gamma")

    xi = {i: mdl.continuous_var(lb=0.0, name=f"xi_{i}") for i in pos}
    zeta = {i: mdl.continuous_var(lb=0.0, name=f"zeta_{i}") for i in neg}

    obj = mdl.sum(w_plus[k] + w_minus[k] for k in range(p)) + C * (mdl.sum(xi[i] for i in pos) + mdl.sum(zeta[i] for i in neg))
    for k in range(p):
        uk = float(u[k])
        if uk > 0:
            obj += (2.0 * float(lam[k]) - D / uk) * (w_plus[k] - w_minus[k])

    mdl.minimize(obj)

    for i in pos:
        expr = mdl.sum(float(X[i, k]) * (w_plus[k] - w_minus[k]) for k in range(p)) + gamma
        mdl.add_constraint(expr <= xi[i] - 1.0)
    for i in neg:
        expr = mdl.sum(float(X[i, k]) * (w_plus[k] - w_minus[k]) for k in range(p)) + gamma
        mdl.add_constraint(-expr <= zeta[i] - 1.0)

    sol = mdl.solve(log_output=log_output)
    if sol is None:
        raise RuntimeError("Lagrangian LP solve failed.")

    wplus_val = np.array([sol.get_value(v) for v in w_plus], dtype=float)
    wminus_val = np.array([sol.get_value(v) for v in w_minus], dtype=float)
    z_val = float(sol.objective_value)
    return z_val, wplus_val, wminus_val


# -----------------------------
# Projected subgradient method
# -----------------------------
def run_projected_subgradient(
    inst: FeatureSelectionSVMInstance,
    max_iter: int = 150,
    eps_nonzero_w: float = 1e-9,
    step_rule: str = "adaptive_improvement",   # adaptive_improvement | diminishing | polyak
    step0: float = 1.0,
    polyak_eta: float = 1.5,
    scale0: float = 2.0,
    stagnation_window: int = 1,
    scale_shrink: float = 0.5,
    scale_eps: float = 1e-12,
    ub_time_limit: Optional[float] = None,
    ub_mip_gap: float = 0.0,
    log_output_ub: bool = False,
    verbose: bool = True,
) -> SubgradientRunResult:
    t0 = time.perf_counter()
    X, y, C, D, u = inst.X, inst.y, float(inst.C), float(inst.D), inst.u
    _, p = X.shape

    lam_ub = np.array([(D / float(u[k])) if float(u[k]) > 0 else 0.0 for k in range(p)], dtype=float)
    lam = np.zeros(p, dtype=float)

    scale = float(scale0)
    no_improve = 0
    best_seen = -float("inf")

    best_lb = -float("inf")
    best_ub = float("inf")
    best_mip: Optional[RestrictedMIPResult] = None

    lb_hist, ub_hist, zlam_hist, step_hist, scale_hist, lam_norm_hist = [], [], [], [], [], []

    for it in range(1, max_iter + 1):
        zlam, wplus, wminus = solve_lagrangian_lp(inst, lam, log_output=False)
        best_lb = max(best_lb, zlam)

        w_signed = wplus - wminus
        k_set = np.where(np.abs(w_signed) > eps_nonzero_w)[0]
        if k_set.size == 0 and p > 0:
            k_set = np.array([int(np.argmax(np.abs(w_signed)))], dtype=int)

        ub_res = solve_restricted_mip(
            X=X, y=y, C=C, D=D, u=u, allowed_set=k_set,
            time_limit=ub_time_limit, mip_gap=ub_mip_gap,
            log_output=log_output_ub, force_allowed_one=False
        )
        if ub_res.objective < best_ub:
            best_ub = ub_res.objective
            best_mip = ub_res

        subgrad = 2.0 * (w_signed)

        if step_rule == "adaptive_improvement":
            if zlam > best_seen + 1e-9:
                best_seen = zlam
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= max(1, int(stagnation_window)):
                    scale = max(float(scale_eps), scale * float(scale_shrink))
                    no_improve = 0

            gnorm2 = float(np.dot(subgrad, subgrad))
            if gnorm2 <= 1e-18 or not math.isfinite(gnorm2):
                step = 0.0
            else:
                step = float(scale) * max(0.0, float(best_ub) - float(zlam)) / gnorm2

        elif step_rule == "diminishing":
            step = float(step0) / math.sqrt(it)

        elif step_rule == "polyak":
            gnorm2 = float(np.dot(subgrad, subgrad))
            if gnorm2 <= 1e-18 or not math.isfinite(gnorm2):
                step = 0.0
            else:
                step = float(polyak_eta) * max(0.0, float(best_ub) - float(zlam)) / gnorm2
        else:
            raise ValueError("step_rule must be one of: adaptive_improvement, diminishing, polyak")

        lam = np.minimum(lam_ub, np.maximum(0.0, lam + step * subgrad))

        lb_hist.append(best_lb)
        ub_hist.append(best_ub)
        zlam_hist.append(zlam)
        step_hist.append(step)
        scale_hist.append(scale)
        lam_norm_hist.append(float(np.linalg.norm(lam)))

        if verbose:
            gap = float(best_ub) - float(best_lb)
            print(
                f"iter={it:03d} | z(λ)={zlam:.6f} | LB*={best_lb:.6f} | UB*={best_ub:.6f} | "
                f"gap={gap:.6f} | |K(λ)|={int(k_set.size)} | step={step:.3e} | scale={scale:.4g}"
            )

        if (best_ub - best_lb) <= 1e-6:
            break
        if scale <= float(scale_eps) and step_rule == "adaptive_improvement":
            break

    return SubgradientRunResult(
        best_lb=float(best_lb),
        best_ub=float(best_ub),
        best_mip=best_mip,
        lb_hist=lb_hist,
        ub_hist=ub_hist,
        zlam_hist=zlam_hist,
        step_hist=step_hist,
        scale_hist=scale_hist,
        lam_norm_hist=lam_norm_hist,
        wall_time_sec=time.perf_counter() - t0,
    )


# -----------------------------
# Save debug outputs
# -----------------------------
def save_debug_outputs(inst: FeatureSelectionSVMInstance, res: SubgradientRunResult, debug_dir: str, zero_as: int = 1) -> None:
    out = Path(debug_dir)
    out.mkdir(parents=True, exist_ok=True)

    pd.DataFrame({"feature_id": inst.feature_ids, "u": inst.u}).to_csv(out / "u.csv", index=False)
    pd.DataFrame({"k1_idx": inst.k1_idx}).to_csv(out / "k1_idx.csv", index=False)
    if inst.z_hat is not None:
        pd.DataFrame([{"z_hat": float(inst.z_hat)}]).to_csv(out / "z_hat.csv", index=False)

    pd.DataFrame({
        "iter": np.arange(1, len(res.lb_hist) + 1),
        "best_lb": res.lb_hist,
        "best_ub": res.ub_hist,
        "z_lambda": res.zlam_hist,
        "step": res.step_hist,
        "scale": res.scale_hist,
        "lambda_norm": res.lam_norm_hist,
    }).to_csv(out / "subgradient_history.csv", index=False)

    if res.best_mip is not None:
        mip = res.best_mip
        w = mip.w_plus - mip.w_minus
        pred = predict_labels(inst.X, w, mip.gamma, zero_as=zero_as)
        tab = confusion_table(inst.y, pred)

        pd.DataFrame({
            "feature_id": inst.feature_ids,
            "w": w,
            "w_plus": mip.w_plus,
            "w_minus": mip.w_minus,
            "y": mip.ybin,
            "u": inst.u,
        }).to_csv(out / "best_solution_weights.csv", index=False)

        pd.DataFrame({
            "sample_id": inst.sample_ids,
            "y_true": inst.y,
            "y_pred": pred,
            "decision": (inst.X @ w + float(mip.gamma)),
            "slack": mip.slack,
        }).to_csv(out / "best_solution_predictions.csv", index=False)

        pd.DataFrame(tab, index=["true_-1", "true_+1"], columns=["pred_-1", "pred_+1"]).to_csv(out / "confusion.csv")

    print(f"[debug] Saved debug CSVs to: {out.resolve()}")


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", type=str, required=True)
    ap.add_argument("--C", type=float, required=True)
    ap.add_argument("--D", type=float, required=True)

    ap.add_argument("--u_big", type=float, default=1e10)
    ap.add_argument("--eps_zero", type=float, default=1e-9)
    ap.add_argument("--step2_timelimit", type=float, default=None)

    ap.add_argument("--solve_lr", action="store_true")
    ap.add_argument("--max_iter", type=int, default=150)
    ap.add_argument("--step_rule", type=str, default="adaptive_improvement", choices=["adaptive_improvement", "diminishing", "polyak"])
    ap.add_argument("--step0", type=float, default=1.0)
    ap.add_argument("--polyak_eta", type=float, default=1.5)

    ap.add_argument("--scale0", type=float, default=2.0)
    ap.add_argument("--stagnation_window", type=int, default=1)
    ap.add_argument("--scale_shrink", type=float, default=0.5)
    ap.add_argument("--scale_eps", type=float, default=1e-12)

    ap.add_argument("--ub_time_limit", type=float, default=None)
    ap.add_argument("--ub_mip_gap", type=float, default=0.0)
    ap.add_argument("--ub_log", action="store_true")

    ap.add_argument("--debug_dir", type=str, default="debug_out")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--zero_as", type=int, default=1, choices=[-1, 1])
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    X, y_vals, sample_ids, feature_ids = read_xy_from_excel(args.xlsx)

    inst = build_instance_with_preprocessing(
        X=X,
        y=y_vals.astype(int),
        sample_ids=sample_ids,
        feature_ids=feature_ids,
        C=float(args.C),
        D=float(args.D),
        eps_zero=float(args.eps_zero),
        u_big=float(args.u_big),
        step2_timelimit=args.step2_timelimit,
        log_output=(not args.quiet),
    )

    print(f"[preprocess] |K|={X.shape[1]} |K1|={int(inst.k1_idx.size)} z_hat={float(inst.z_hat):.6f}")
    print(f"[preprocess] u: min={float(np.min(inst.u)):.3e} max={float(np.max(inst.u)):.3e}")

    if not args.solve_lr:
        print("Nothing to do: pass --solve_lr to run the projected subgradient algorithm.")
        return

    res = run_projected_subgradient(
        inst=inst,
        max_iter=int(args.max_iter),
        eps_nonzero_w=float(args.eps_zero),
        step_rule=str(args.step_rule),
        step0=float(args.step0),
        polyak_eta=float(args.polyak_eta),
        scale0=float(args.scale0),
        stagnation_window=int(args.stagnation_window),
        scale_shrink=float(args.scale_shrink),
        scale_eps=float(args.scale_eps),
        ub_time_limit=args.ub_time_limit,
        ub_mip_gap=float(args.ub_mip_gap),
        log_output_ub=bool(args.ub_log),
        verbose=(not args.quiet),
    )

    print(f"\n=== Finished ===\nLB*={res.best_lb:.6f} | UB*={res.best_ub:.6f} | wall_time={res.wall_time_sec:.3f}s")

    if res.best_mip is not None:
        bd = objective_breakdown(inst.C, inst.D, res.best_mip.w_plus, res.best_mip.w_minus, res.best_mip.slack, res.best_mip.ybin)
        w = res.best_mip.w_plus - res.best_mip.w_minus
        y_pred = predict_labels(inst.X, w, res.best_mip.gamma, zero_as=int(args.zero_as))
        acc = float(np.mean(y_pred == inst.y))

        print("\n=== Best feasible (restricted MIP) ===")
        print(f"Objective: {bd['Total']:.6f} | L1={bd['L1']:.6f} | Slack={bd['Slack']:.6f} | D*|y|={bd['FeaturePenalty']:.6f}")
        print(f"n_selected: {int(np.sum(res.best_mip.ybin))} / {len(res.best_mip.ybin)}")
        print(f"accuracy: {acc:.4f}")
        tab = confusion_table(inst.y, y_pred)
        print("confusion (rows=true [-1,+1], cols=pred [-1,+1]):")
        print(tab)

    save_debug_outputs(inst, res, debug_dir=args.debug_dir, zero_as=int(args.zero_as))

    if args.plot:
        plot_bounds(res.lb_hist, res.ub_hist, out_png=str(Path(args.debug_dir) / "bounds.png"))


if __name__ == "__main__":
    main()


