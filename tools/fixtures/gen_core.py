"""Reference values for the numerical core, computed with NumPy / SciPy / scikit-learn.

Run from the repository root:  python tools/fixtures/gen_core.py
The JSON it writes is committed, so nobody needs Python to run the tests.
"""
import json
import pathlib

import numpy as np
from scipy import special, stats
from sklearn import metrics

OUT = pathlib.Path(__file__).resolve().parents[2] / 'packages' / 'core' / 'fixtures' / 'core.json'
rng = np.random.default_rng(7)

out = {}

# --- distributions ---------------------------------------------------------------------------
xs = [-3.2, -1.0, -0.3, 0.0, 0.7, 1.9, 4.1]
out['normal'] = {
    'x': xs,
    'cdf': [float(stats.norm.cdf(x)) for x in xs],
    'pdf': [float(stats.norm.pdf(x)) for x in xs],
    'p': [0.001, 0.025, 0.2, 0.5, 0.8, 0.975, 0.999],
    'ppf': [float(stats.norm.ppf(p)) for p in [0.001, 0.025, 0.2, 0.5, 0.8, 0.975, 0.999]],
}
out['t'] = [
    {'t': t, 'df': df, 'cdf': float(stats.t.cdf(t, df)), 'ppf': float(stats.t.ppf(p, df)), 'p': p}
    for t, df, p in [(-2.5, 3, 0.025), (0.4, 10, 0.5), (1.96, 30, 0.975), (3.1, 120, 0.995), (-0.2, 1, 0.3)]
]
out['f'] = [
    {'f': f, 'd1': a, 'd2': b, 'cdf': float(stats.f.cdf(f, a, b))}
    for f, a, b in [(0.5, 3, 20), (2.5, 5, 40), (10.0, 2, 8), (1.0, 7, 7)]
]
out['chi2'] = [{'x': x, 'k': k, 'cdf': float(stats.chi2.cdf(x, k))} for x, k in [(0.5, 1), (3.0, 4), (12.0, 7), (30.0, 20)]]
out['lgamma'] = [{'x': x, 'value': float(special.gammaln(x))} for x in [0.1, 0.5, 1.0, 2.5, 7.0, 30.5, 100.0]]
out['betainc'] = [{'x': x, 'a': a, 'b': b, 'value': float(special.betainc(a, b, x))} for x, a, b in [(0.2, 2, 3), (0.5, 0.5, 0.5), (0.9, 5, 1.5), (0.35, 12, 8)]]

# --- linear algebra --------------------------------------------------------------------------
a = rng.normal(size=(5, 5))
spd = a @ a.T + 5 * np.eye(5)
b = rng.normal(size=(5, 2))
sym = (a + a.T) / 2
w, v = np.linalg.eigh(sym)
order = np.argsort(-w)
out['linalg'] = {
    'spd': spd.tolist(),
    'general': a.tolist(),
    'b': b.tolist(),
    'solve_spd': np.linalg.solve(spd, b).tolist(),
    'solve_general': np.linalg.solve(a, b).tolist(),
    'inverse': np.linalg.inv(a).tolist(),
    'det': float(np.linalg.det(a)),
    'logdet_spd': float(np.linalg.slogdet(spd)[1]),
    'sym': sym.tolist(),
    'eigenvalues': w[order].tolist(),
    'eigenvectors_abs': np.abs(v[:, order]).tolist(),
}

# --- descriptive -----------------------------------------------------------------------------
data = rng.normal(size=(40, 3)) * [1, 2, 0.5] + [0, 3, -1]
out['descriptive'] = {
    'x': data.tolist(),
    'mean': data.mean(axis=0).tolist(),
    'cov': np.cov(data.T).tolist(),
    'corr01': float(np.corrcoef(data[:, 0], data[:, 1])[0, 1]),
    'q': [0.1, 0.25, 0.5, 0.9],
    'quantiles0': [float(np.quantile(data[:, 0], q)) for q in [0.1, 0.25, 0.5, 0.9]],
}

# --- metrics ---------------------------------------------------------------------------------
n = 60
y_true = (rng.random(n) < 0.4).astype(int)
scores = np.clip(y_true * 0.35 + rng.normal(size=n) * 0.3 + 0.35, 0, 1)
fpr, tpr, thr = metrics.roc_curve(y_true, scores, drop_intermediate=False)
labels_a = rng.integers(0, 3, size=50)
labels_b = np.where(rng.random(50) < 0.7, labels_a, rng.integers(0, 3, size=50))
pts = np.vstack([rng.normal(size=(25, 2)), rng.normal(size=(25, 2)) + 3])
cl = np.array([0] * 25 + [1] * 25)
y_reg = rng.normal(size=30)
p_reg = y_reg + rng.normal(size=30) * 0.5
out['metrics'] = {
    'y_true': y_true.tolist(),
    'scores': scores.tolist(),
    'auc': float(metrics.roc_auc_score(y_true, scores)),
    'roc_points': len(fpr),
    'fpr': fpr.tolist(),
    'tpr': tpr.tolist(),
    'labels_a': labels_a.tolist(),
    'labels_b': labels_b.tolist(),
    'ari': float(metrics.adjusted_rand_score(labels_a, labels_b)),
    'confusion': metrics.confusion_matrix(labels_a, labels_b).tolist(),
    'f1': metrics.f1_score(labels_a, labels_b, average=None).tolist(),
    'precision': metrics.precision_score(labels_a, labels_b, average=None).tolist(),
    'recall': metrics.recall_score(labels_a, labels_b, average=None).tolist(),
    'points': pts.tolist(),
    'cluster_labels': cl.tolist(),
    'silhouette': float(metrics.silhouette_score(pts, cl)),
    'y_reg': y_reg.tolist(),
    'p_reg': p_reg.tolist(),
    'mse': float(metrics.mean_squared_error(y_reg, p_reg)),
    'mae': float(metrics.mean_absolute_error(y_reg, p_reg)),
    'r2': float(metrics.r2_score(y_reg, p_reg)),
}

OUT.write_text(json.dumps(out, separators=(',', ':')))
print('wrote', OUT, OUT.stat().st_size, 'bytes')
