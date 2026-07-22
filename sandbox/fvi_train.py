"""Phase 3 step 3-5: fit C_theta(phi(s)) to the fitted-value-iteration
residual target from fvi_training_data.csv, using grouped (by seed) cross-
validation. sklearn is not installed in this environment, so all model
classes are implemented directly on numpy/scipy (closed-form ridge, a
monotone-ish additive binned model, and a low-rank quadratic model via SVD)
-- consistent with this project's established preference for interpretable,
verifiable models over black-box NN/RF (per prior-phase guidance).

Model classes tested (per spec Section "Fit C_theta using at least these
model classes"):
  A. regularized linear model with interactions (ridge, closed-form)
  B. monotone/binned additive model (per-feature quantile bins + ridge,
     with a post-hoc monotonicity check/clip -- see fit_binned_additive)
  C. low-rank quadratic model (SVD to top-k components, ridge on quadratic
     expansion of those components)
  [gradient-boosted trees are skipped: sklearn/xgboost/lightgbm are not
  available in this environment; noted honestly rather than silently
  substituting something else and calling it the same thing]

Grouped CV: leave-one-training-seed-out (10 folds, one per TRAINING_SEEDS
entry), never a random row split (adjacent rows are highly correlated
within an episode).
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

_SANDBOX_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SANDBOX_DIR))
import fvi_common as fc
from fvi_generate_data import TRAINING_SEEDS

DATA_PATH = _SANDBOX_DIR / "fvi_training_data.csv"
OUT_PATH = _SANDBOX_DIR / "fvi_model.json"


def load_data():
    seeds, scenarios, X, y = [], [], [], []
    with open(DATA_PATH) as f:
        r = csv.DictReader(f)
        for row in r:
            seeds.append(int(row["seed"]))
            scenarios.append(row["scenario"])
            X.append([float(row[name]) for name in fc.FEATURE_NAMES])
            y.append(float(row["y"]))
    return np.array(seeds), np.array(scenarios), np.array(X), np.array(y)


def standardize_fit(X):
    mu = X.mean(axis=0)
    sd = X.std(axis=0) + 1e-9
    return mu, sd


def ridge_fit(X, y, lam):
    n, p = X.shape
    Xb = np.concatenate([X, np.ones((n, 1))], axis=1)
    A = Xb.T @ Xb + lam * np.eye(p + 1)
    A[-1, -1] -= lam  # don't regularize intercept
    theta = np.linalg.solve(A, Xb.T @ y)
    return theta  # theta[:-1]=weights, theta[-1]=intercept


def ridge_predict(X, theta):
    n = X.shape[0]
    Xb = np.concatenate([X, np.ones((n, 1))], axis=1)
    return Xb @ theta


def fit_linear_interactions(Xtr, ytr, mu, sd, lam=10.0, top_k_inter=8):
    Xs = (Xtr - mu) / sd
    # add a modest set of pairwise interactions among the highest-variance-explaining features
    corrs = np.abs(np.array([np.corrcoef(Xs[:, j], ytr)[0, 1] for j in range(Xs.shape[1])]))
    top = np.argsort(-corrs)[:top_k_inter]
    inter_cols = []
    inter_idx_pairs = []
    for a in range(len(top)):
        for b in range(a + 1, len(top)):
            inter_cols.append(Xs[:, top[a]] * Xs[:, top[b]])
            inter_idx_pairs.append((int(top[a]), int(top[b])))
    Xfull = np.concatenate([Xs] + ([np.stack(inter_cols, axis=1)] if inter_cols else []), axis=1)
    theta = ridge_fit(Xfull, ytr, lam)
    return dict(theta=theta.tolist(), mu=mu.tolist(), sd=sd.tolist(),
                inter_idx_pairs=inter_idx_pairs, lam=lam, kind="linear_interactions")


def predict_linear_interactions(X, model):
    mu, sd = np.array(model["mu"]), np.array(model["sd"])
    Xs = (X - mu) / sd
    inter_cols = [Xs[:, a] * Xs[:, b] for a, b in model["inter_idx_pairs"]]
    Xfull = np.concatenate([Xs] + ([np.stack(inter_cols, axis=1)] if inter_cols else []), axis=1)
    return ridge_predict(Xfull, np.array(model["theta"]))


def fit_binned_additive(Xtr, ytr, mu, sd, n_bins=5, lam=5.0):
    Xs = (Xtr - mu) / sd
    edges = [np.quantile(Xs[:, j], np.linspace(0, 1, n_bins + 1)[1:-1]) for j in range(Xs.shape[1])]
    cols = []
    for j in range(Xs.shape[1]):
        bin_idx = np.digitize(Xs[:, j], edges[j])
        onehot = np.zeros((Xs.shape[0], n_bins))
        onehot[np.arange(Xs.shape[0]), bin_idx] = 1.0
        cols.append(onehot)
    Xfull = np.concatenate(cols, axis=1)
    theta = ridge_fit(Xfull, ytr, lam)
    # monotonicity check/clip per feature: enforce non-decreasing bin means via cumulative max
    # if the fitted trend is broadly increasing, else non-increasing -- this is a POST-HOC
    # projection (pooled-adjacent-violators-style clip), not a hard constraint during fitting.
    w = theta[:-1].reshape(Xs.shape[1], n_bins)
    for j in range(Xs.shape[1]):
        direction = 1.0 if (w[j, -1] - w[j, 0]) >= 0 else -1.0
        seq = w[j] * direction
        seq = np.maximum.accumulate(seq)
        w[j] = seq * direction
    theta = np.concatenate([w.reshape(-1), theta[-1:]])
    return dict(theta=theta.tolist(), mu=mu.tolist(), sd=sd.tolist(), edges=[e.tolist() for e in edges],
                n_bins=n_bins, lam=lam, kind="binned_additive")


def predict_binned_additive(X, model):
    mu, sd = np.array(model["mu"]), np.array(model["sd"])
    n_bins = model["n_bins"]
    Xs = (X - mu) / sd
    cols = []
    for j in range(Xs.shape[1]):
        edges = np.array(model["edges"][j])
        bin_idx = np.digitize(Xs[:, j], edges)
        onehot = np.zeros((Xs.shape[0], n_bins))
        onehot[np.arange(Xs.shape[0]), bin_idx] = 1.0
        cols.append(onehot)
    Xfull = np.concatenate(cols, axis=1)
    return ridge_predict(Xfull, np.array(model["theta"]))


def fit_lowrank_quadratic(Xtr, ytr, mu, sd, k=5, lam=5.0):
    Xs = (Xtr - mu) / sd
    U, S, Vt = np.linalg.svd(Xs, full_matrices=False)
    Vk = Vt[:k].T  # (p, k)
    Z = Xs @ Vk  # (n, k)
    quad_cols = [Z[:, a] * Z[:, b] for a in range(k) for b in range(a, k)]
    Zfull = np.concatenate([Z, np.stack(quad_cols, axis=1)], axis=1)
    theta = ridge_fit(Zfull, ytr, lam)
    return dict(theta=theta.tolist(), mu=mu.tolist(), sd=sd.tolist(), Vk=Vk.tolist(), k=k, lam=lam,
                kind="lowrank_quadratic")


def predict_lowrank_quadratic(X, model):
    mu, sd = np.array(model["mu"]), np.array(model["sd"])
    Vk = np.array(model["Vk"])
    k = model["k"]
    Xs = (X - mu) / sd
    Z = Xs @ Vk
    quad_cols = [Z[:, a] * Z[:, b] for a in range(k) for b in range(a, k)]
    Zfull = np.concatenate([Z, np.stack(quad_cols, axis=1)], axis=1)
    return ridge_predict(Zfull, np.array(model["theta"]))


MODEL_FITTERS = dict(
    linear_interactions=(fit_linear_interactions, predict_linear_interactions),
    binned_additive=(fit_binned_additive, predict_binned_additive),
    lowrank_quadratic=(fit_lowrank_quadratic, predict_lowrank_quadratic),
)


def r2_score(y, yhat):
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2) + 1e-9
    return 1.0 - ss_res / ss_tot


def grouped_cv(seeds, X, y, model_name, **kwargs):
    fitter, predictor = MODEL_FITTERS[model_name]
    r2s, maes = [], []
    for held_out in TRAINING_SEEDS:
        tr_mask = seeds != held_out
        te_mask = seeds == held_out
        if te_mask.sum() == 0:
            continue
        mu, sd = standardize_fit(X[tr_mask])
        model = fitter(X[tr_mask], y[tr_mask], mu, sd, **kwargs)
        yhat = predictor(X[te_mask], model)
        r2s.append(r2_score(y[te_mask], yhat))
        maes.append(float(np.mean(np.abs(y[te_mask] - yhat))))
    return dict(mean_r2=float(np.mean(r2s)), std_r2=float(np.std(r2s)),
                mean_mae=float(np.mean(maes)), fold_r2=r2s)


def main():
    seeds, scenarios, X, y = load_data()
    print(f"loaded {len(y)} rows, {len(set(seeds.tolist()))} seeds, y mean={y.mean():.2f} std={y.std():.2f}")

    results = {}
    for name in MODEL_FITTERS:
        cv = grouped_cv(seeds, X, y, name)
        results[name] = cv
        print(f"{name:20s} mean_R2={cv['mean_r2']:+.4f} std_R2={cv['std_r2']:.4f} mean_MAE={cv['mean_mae']:.2f}")

    best_name = max(results, key=lambda k: results[k]["mean_r2"])
    print(f"\nbest model by grouped-CV R2: {best_name} (R2={results[best_name]['mean_r2']:+.4f})")

    # refit winner on full training data
    fitter, _ = MODEL_FITTERS[best_name]
    mu, sd = standardize_fit(X)
    final_model = fitter(X, y, mu, sd)
    final_model["cv_results"] = {k: {kk: vv for kk, vv in v.items() if kk != "fold_r2"} for k, v in results.items()}
    final_model["chosen"] = best_name
    with open(OUT_PATH, "w") as f:
        json.dump(final_model, f)
    print(f"saved {OUT_PATH}")


if __name__ == "__main__":
    main()
