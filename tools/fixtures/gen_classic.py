"""Reference results for the classical models, from statsmodels / scikit-learn / SciPy.

Usage (from the repository root):   python tools/fixtures/gen_classic.py [name ...]
With no names every fixture is regenerated. JSON goes to packages/classic/fixtures/<name>.json and is
committed, so running the tests needs no Python.
"""
import json
import pathlib
import sys

import numpy as np

OUT = pathlib.Path(__file__).resolve().parents[2] / 'packages' / 'classic' / 'fixtures'
OUT.mkdir(parents=True, exist_ok=True)
REGISTRY = {}


def fixture(fn):
    REGISTRY[fn.__name__] = fn
    return fn


def save(name, payload):
    path = OUT / f'{name}.json'
    path.write_text(json.dumps(payload, separators=(',', ':')))
    print('wrote', path.name, path.stat().st_size, 'bytes')


@fixture
def linear():
    import statsmodels.api as sm
    from sklearn.linear_model import Ridge

    rng = np.random.default_rng(11)
    n, d = 60, 3
    x = rng.normal(size=(n, d))
    x[:, 1] = 0.6 * x[:, 0] + 0.8 * x[:, 1]  # correlated features
    y = 1.5 + x @ np.array([2.0, -1.0, 0.0]) + rng.normal(size=n) * 0.8
    fit = sm.OLS(y, sm.add_constant(x)).fit()
    infl = fit.get_influence()
    out = {
        'x': x.tolist(),
        'y': y.tolist(),
        'params': fit.params.tolist(),  # intercept first
        'bse': fit.bse.tolist(),
        'tvalues': fit.tvalues.tolist(),
        'pvalues': fit.pvalues.tolist(),
        'conf_int': fit.conf_int().tolist(),
        'rsquared': float(fit.rsquared),
        'rsquared_adj': float(fit.rsquared_adj),
        'fvalue': float(fit.fvalue),
        'f_pvalue': float(fit.f_pvalue),
        'aic': float(fit.aic),
        'bic': float(fit.bic),
        'sigma2': float(fit.scale),
        'hat': infl.hat_matrix_diag.tolist(),
        'cooks': infl.cooks_distance[0].tolist(),
        'vif': [float(1 / (1 - sm.OLS(x[:, j], sm.add_constant(np.delete(x, j, axis=1))).fit().rsquared)) for j in range(d)],
    }
    # ridge: sklearn minimises ||y - Xw||^2 + alpha ||w||^2 with an unpenalised intercept
    out['ridge'] = []
    for alpha in [0.1, 1.0, 10.0, 100.0]:
        r = Ridge(alpha=alpha).fit(x, y)
        out['ridge'].append({'alpha': alpha, 'coef': r.coef_.tolist(), 'intercept': float(r.intercept_)})
    save('linear', out)


@fixture
def regularized():
    from sklearn.linear_model import ElasticNet, Lasso, Ridge, lasso_path
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler

    rng = np.random.default_rng(5)
    n, d = 80, 8
    x = rng.normal(size=(n, d))
    x[:, 1] = 0.8 * x[:, 0] + 0.6 * x[:, 1]
    y = 0.7 + x @ np.array([3.0, 0.0, -2.0, 0.0, 0.0, 1.0, 0.0, 0.0]) + rng.normal(size=n) * 1.0
    out = {'x': x.tolist(), 'y': y.tolist(), 'lasso': [], 'enet': [], 'ridge': []}
    for alpha in [0.01, 0.1, 0.5, 1.0, 3.0]:
        m = Lasso(alpha=alpha, tol=1e-14, max_iter=200000).fit(x, y)
        out['lasso'].append({'alpha': alpha, 'coef': m.coef_.tolist(), 'intercept': float(m.intercept_)})
    for alpha, ratio in [(0.1, 0.5), (0.5, 0.2), (0.3, 0.9)]:
        m = ElasticNet(alpha=alpha, l1_ratio=ratio, tol=1e-14, max_iter=200000).fit(x, y)
        out['enet'].append({'alpha': alpha, 'l1_ratio': ratio, 'coef': m.coef_.tolist(), 'intercept': float(m.intercept_)})
    # lasso_path does not fit an intercept, so hand it centred data (equivalent to fit_intercept=True)
    alphas, coefs, _ = lasso_path(x - x.mean(axis=0), y - y.mean(), alphas=[2.0, 1.0, 0.5, 0.25, 0.1, 0.05], tol=1e-14)
    out['path'] = {'alphas': alphas.tolist(), 'coefs': coefs.T.tolist()}
    out['no_intercept'] = {
        'alpha': 0.2,
        'coef': Lasso(alpha=0.2, fit_intercept=False, tol=1e-14, max_iter=200000).fit(x, y).coef_.tolist(),
    }
    # polynomial regression on one feature
    xs = np.sort(rng.uniform(-3, 3, size=40))
    ys = np.sin(1.3 * xs) + rng.normal(size=40) * 0.3
    poly = {'x': xs.tolist(), 'y': ys.tolist(), 'fits': []}
    grid = np.linspace(-3.5, 3.5, 15)
    for degree, alpha in [(3, 0.0), (6, 0.0), (6, 1.0), (9, 0.1)]:
        reg = Ridge(alpha=alpha) if alpha > 0 else Ridge(alpha=1e-12)
        pipe = make_pipeline(PolynomialFeatures(degree, include_bias=False), StandardScaler(), reg)
        pipe.fit(xs[:, None], ys)
        poly['fits'].append({'degree': degree, 'alpha': alpha, 'grid': grid.tolist(), 'pred': pipe.predict(grid[:, None]).tolist()})
    poly['lasso'] = []
    for degree, alpha in [(8, 0.02)]:
        pipe = make_pipeline(PolynomialFeatures(degree, include_bias=False), StandardScaler(), Lasso(alpha=alpha, tol=1e-14, max_iter=200000))
        pipe.fit(xs[:, None], ys)
        poly['lasso'].append({'degree': degree, 'alpha': alpha, 'grid': grid.tolist(), 'pred': pipe.predict(grid[:, None]).tolist()})
    out['poly'] = poly
    save('regularized', out)


@fixture
def glm():
    import statsmodels.api as sm

    rng = np.random.default_rng(21)
    n = 120
    x = rng.normal(size=(n, 2))
    eta = 0.3 + 0.8 * x[:, 0] - 0.5 * x[:, 1]
    xc = sm.add_constant(x)
    out = {'x': x.tolist(), 'families': {}}
    data = {
        'gaussian': (eta + rng.normal(size=n) * 0.5, sm.families.Gaussian()),
        'binomial': ((rng.uniform(size=n) < 1 / (1 + np.exp(-eta))).astype(float), sm.families.Binomial()),
        'poisson': (rng.poisson(np.exp(eta)).astype(float), sm.families.Poisson()),
        'gamma': (rng.gamma(shape=4.0, scale=np.exp(eta) / 4.0), sm.families.Gamma(sm.families.links.Log())),
    }
    for name, (y, family) in data.items():
        fit = sm.GLM(y, xc, family=family).fit(tol=1e-12, maxiter=200)
        out['families'][name] = {
            'y': y.tolist(),
            'params': fit.params.tolist(),
            'bse': fit.bse.tolist(),
            'deviance': float(fit.deviance),
            'null_deviance': float(fit.null_deviance),
            'pearson_chi2': float(fit.pearson_chi2),
            'scale': float(fit.scale),
            'llf': float(fit.llf) if name != 'gamma' else None,
            'aic': float(fit.aic) if name != 'gamma' else None,
            'pvalues': fit.pvalues.tolist(),
        }
    save('glm', out)


@fixture
def robust():
    import statsmodels.api as sm
    from scipy.stats import theilslopes

    rng = np.random.default_rng(8)
    n = 60
    x = rng.uniform(-3, 3, size=(n, 1))
    y = 0.5 + 0.9 * x[:, 0] + rng.normal(size=n) * 0.3
    y[::9] += rng.choice([-1, 1], size=len(y[::9])) * (4 + rng.uniform(size=len(y[::9])) * 3)
    xc = sm.add_constant(x)
    huber = sm.RLM(y, xc, M=sm.robust.norms.HuberT()).fit(tol=1e-12, maxiter=500)
    out = {
        'x': x.tolist(),
        'y': y.tolist(),
        'huber': {'params': huber.params.tolist(), 'scale': float(huber.scale), 'weights': huber.weights.tolist()},
        'quantile': {},
    }
    for q in [0.1, 0.5, 0.9]:
        fit = sm.QuantReg(y, xc).fit(q=q, max_iter=5000, p_tol=1e-10)
        out['quantile'][str(q)] = fit.params.tolist()
    res = theilslopes(y, x[:, 0])
    out['theil_sen'] = {'slope': float(res[0]), 'intercept': float(res[1])}
    save('robust', out)


@fixture
def logistic():
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(3)
    n = 150
    x = rng.normal(size=(n, 3))
    logit = 0.4 + 1.5 * x[:, 0] - 1.0 * x[:, 1]
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype(int)
    out = {'x': x.tolist(), 'y': y.tolist(), 'binary': []}
    for c in [0.01, 0.1, 1.0, 100.0]:
        m = LogisticRegression(C=c, tol=1e-12, max_iter=10000).fit(x, y)
        out['binary'].append(
            {'C': c, 'coef': m.coef_[0].tolist(), 'intercept': float(m.intercept_[0]), 'proba': m.predict_proba(x[:5]).tolist()}
        )
    centers = np.array([[0, 0], [2.5, 1.0], [0.5, 3.0]])
    ym = rng.integers(0, 3, size=n)
    xm = centers[ym] + rng.normal(size=(n, 2)) * 0.9
    out['multi'] = {'x': xm.tolist(), 'y': ym.tolist(), 'fits': []}
    for c in [0.1, 1.0, 10.0]:
        m = LogisticRegression(C=c, tol=1e-12, max_iter=10000).fit(xm, ym)
        out['multi']['fits'].append(
            {
                'C': c,
                'coef': m.coef_.tolist(),
                'intercept': m.intercept_.tolist(),
                'proba': m.predict_proba(xm[:6]).tolist(),
                'pred': m.predict(xm).tolist(),
            }
        )
    save('logistic', out)


@fixture
def knn():
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

    rng = np.random.default_rng(13)
    n = 120
    centers = np.array([[0, 0], [2.0, 1.0], [0.5, 2.5]])
    y = rng.integers(0, 3, size=n)
    x = centers[y] + rng.normal(size=(n, 2)) * 0.9
    q = rng.uniform(-2, 4, size=(40, 2))
    out = {'x': x.tolist(), 'y': y.tolist(), 'queries': q.tolist(), 'classifier': []}
    for k, weights, metric in [(1, 'uniform', 'euclidean'), (5, 'uniform', 'euclidean'), (7, 'distance', 'euclidean'), (5, 'uniform', 'manhattan'), (9, 'distance', 'chebyshev')]:
        m = KNeighborsClassifier(n_neighbors=k, weights=weights, metric=metric).fit(x, y)
        out['classifier'].append(
            {'k': k, 'weights': weights, 'metric': metric, 'pred': m.predict(q).tolist(), 'proba': m.predict_proba(q).tolist()}
        )
    xr = rng.uniform(-3, 3, size=(60, 1))
    yr = np.sin(xr[:, 0]) + rng.normal(size=60) * 0.2
    grid = np.linspace(-3.5, 3.5, 30)[:, None]
    out['regressor'] = {'x': xr.tolist(), 'y': yr.tolist(), 'grid': grid.tolist(), 'fits': []}
    for k, weights in [(1, 'uniform'), (5, 'uniform'), (9, 'distance')]:
        m = KNeighborsRegressor(n_neighbors=k, weights=weights).fit(xr, yr)
        out['regressor']['fits'].append({'k': k, 'weights': weights, 'pred': m.predict(grid).tolist()})
    save('knn', out)


@fixture
def gaussian():
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
    from sklearn.naive_bayes import GaussianNB

    rng = np.random.default_rng(17)
    n = 150
    y = rng.integers(0, 3, size=n)
    centers = np.array([[0, 0, 0], [2.0, 1.0, -1.0], [0.5, 2.5, 1.0]])
    scales = np.array([[1.0, 0.7, 1.3], [0.6, 1.2, 0.9], [1.1, 0.8, 0.5]])
    x = centers[y] + rng.normal(size=(n, 3)) * scales[y]
    x[:, 1] += 0.5 * x[:, 0]
    q = rng.uniform(-2, 4, size=(30, 3))
    out = {'x': x.tolist(), 'y': y.tolist(), 'queries': q.tolist(), 'models': {}}

    def pack(model):
        return {'pred': model.predict(q).tolist(), 'proba': model.predict_proba(q).tolist()}

    nb = GaussianNB().fit(x, y)
    out['models']['nb'] = {**pack(nb), 'theta': nb.theta_.tolist(), 'var': nb.var_.tolist(), 'prior': nb.class_prior_.tolist()}
    lda = LinearDiscriminantAnalysis(solver='lsqr').fit(x, y)
    out['models']['lda'] = {**pack(lda), 'covariance': lda.covariance_.tolist(), 'means': lda.means_.tolist()}
    lda_s = LinearDiscriminantAnalysis(solver='lsqr', shrinkage=0.4).fit(x, y)
    out['models']['lda_shrink'] = {**pack(lda_s), 'covariance': lda_s.covariance_.tolist()}
    qda = QuadraticDiscriminantAnalysis().fit(x, y)
    out['models']['qda'] = pack(qda)
    qda_r = QuadraticDiscriminantAnalysis(reg_param=0.3).fit(x, y)
    out['models']['qda_reg'] = pack(qda_r)
    save('gaussian', out)


def tree_dump(t):
    """Pre-order dump of an sklearn tree: feature, threshold, samples, impurity, value."""
    tr = t.tree_
    out = []

    def walk(i, depth=0):
        out.append(
            {
                'depth': depth,
                'feature': int(tr.feature[i]),
                'threshold': float(tr.threshold[i]),
                'samples': int(tr.n_node_samples[i]),
                'weight': float(tr.weighted_n_node_samples[i]),
                'impurity': float(tr.impurity[i]),
                'value': tr.value[i].ravel().tolist(),
                'leaf': bool(tr.children_left[i] == -1),
            }
        )
        if tr.children_left[i] != -1:
            walk(tr.children_left[i], depth + 1)
            walk(tr.children_right[i], depth + 1)

    walk(0)
    return out


@fixture
def tree():
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

    rng = np.random.default_rng(23)
    n = 150
    y = rng.integers(0, 3, size=n)
    centers = np.array([[0, 0], [2.0, 1.0], [0.5, 2.5]])
    # scikit-learn trees work on float32: hand both implementations numbers float32 can represent
    f32 = lambda a: a.astype(np.float32).astype(np.float64)
    x = f32(centers[y] + rng.normal(size=(n, 2)) * 1.0)
    grid = f32(rng.uniform(-2, 4, size=(60, 2)))
    out = {'x': x.tolist(), 'y': y.tolist(), 'grid': grid.tolist(), 'classifier': []}
    configs = [
        {'criterion': 'gini'},
        {'criterion': 'entropy'},
        {'criterion': 'gini', 'max_depth': 3},
        {'criterion': 'gini', 'min_samples_leaf': 8},
        {'criterion': 'gini', 'min_samples_split': 20},
        {'criterion': 'gini', 'min_impurity_decrease': 0.01},
        {'criterion': 'gini', 'ccp_alpha': 0.01},
    ]
    for cfg in configs:
        m = DecisionTreeClassifier(random_state=0, **cfg).fit(x, y)
        out['classifier'].append(
            {
                'config': cfg,
                'nodes': tree_dump(m),
                'pred': m.predict(grid).tolist(),
                'proba': m.predict_proba(grid).tolist(),
                'importances': m.feature_importances_.tolist(),
            }
        )
    full = DecisionTreeClassifier(random_state=0).fit(x, y)
    path = full.cost_complexity_pruning_path(x, y)
    out['path'] = {'alphas': path.ccp_alphas.tolist(), 'impurities': path.impurities.tolist()}
    # weighted samples
    sw = rng.uniform(0.2, 3.0, size=n)
    mw = DecisionTreeClassifier(random_state=0, max_depth=4).fit(x, y, sample_weight=sw)
    out['weighted'] = {'sw': sw.tolist(), 'nodes': tree_dump(mw), 'pred': mw.predict(grid).tolist()}
    # regression
    xr = f32(rng.uniform(-3, 3, size=(120, 2)))
    yr = np.sin(xr[:, 0]) + 0.5 * xr[:, 1] ** 2 * 0.3 + rng.normal(size=120) * 0.2
    gr = f32(rng.uniform(-3, 3, size=(50, 2)))
    out['regressor'] = {'x': xr.tolist(), 'y': yr.tolist(), 'grid': gr.tolist(), 'fits': []}
    for cfg in [{}, {'max_depth': 3}, {'min_samples_leaf': 5}, {'ccp_alpha': 0.02}]:
        m = DecisionTreeRegressor(random_state=0, **cfg).fit(xr, yr)
        out['regressor']['fits'].append(
            {'config': cfg, 'nodes': tree_dump(m), 'pred': m.predict(gr).tolist(), 'importances': m.feature_importances_.tolist()}
        )
    save('tree', out)


@fixture
def ensemble():
    from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, GradientBoostingRegressor

    f32 = lambda a: a.astype(np.float32).astype(np.float64)
    rng = np.random.default_rng(31)
    n = 160
    y3 = rng.integers(0, 3, size=n)
    centers = np.array([[0, 0], [2.0, 1.0], [0.5, 2.5]])
    x = f32(centers[y3] + rng.normal(size=(n, 2)) * 1.0)
    y2 = (y3 == 1).astype(int)
    grid = f32(rng.uniform(-2, 4, size=(60, 2)))
    out = {'x': x.tolist(), 'y3': y3.tolist(), 'y2': y2.tolist(), 'grid': grid.tolist()}

    # AdaBoost with stumps (SAMME) and with depth-2 trees
    out['adaboost'] = []
    for cfg, y in [({'n_estimators': 15, 'learning_rate': 1.0}, y3), ({'n_estimators': 12, 'learning_rate': 0.5}, y2)]:
        m = AdaBoostClassifier(random_state=0, **cfg).fit(x, y)
        out['adaboost'].append(
            {
                'config': cfg,
                'y': y.tolist(),
                'alphas': m.estimator_weights_.tolist(),
                'errors': m.estimator_errors_.tolist(),
                'pred': m.predict(grid).tolist(),
                'proba': m.predict_proba(grid).tolist(),
            }
        )

    # gradient boosting
    out['gbm_classifier'] = []
    for cfg, y in [({'n_estimators': 20, 'learning_rate': 0.2, 'max_depth': 2}, y2), ({'n_estimators': 15, 'learning_rate': 0.3, 'max_depth': 2}, y3)]:
        m = GradientBoostingClassifier(random_state=0, **cfg).fit(x, y)
        out['gbm_classifier'].append(
            {
                'config': cfg,
                'y': y.tolist(),
                'proba': m.predict_proba(grid).tolist(),
                'staged5': list(m.staged_predict_proba(grid))[4].tolist(),
                'train_loss': m.train_score_.tolist(),
            }
        )
    xr = f32(rng.uniform(-3, 3, size=(120, 2)))
    yr = np.sin(xr[:, 0]) + 0.3 * xr[:, 1] ** 2 + rng.normal(size=120) * 0.2
    gr = f32(rng.uniform(-3, 3, size=(40, 2)))
    out['gbm_regressor'] = {'x': xr.tolist(), 'y': yr.tolist(), 'grid': gr.tolist(), 'fits': []}
    for cfg in [{'n_estimators': 30, 'learning_rate': 0.1, 'max_depth': 3}, {'n_estimators': 10, 'learning_rate': 0.5, 'max_depth': 2}]:
        m = GradientBoostingRegressor(random_state=0, **cfg).fit(xr, yr)
        out['gbm_regressor']['fits'].append(
            {'config': cfg, 'pred': m.predict(gr).tolist(), 'staged5': list(m.staged_predict(gr))[4].tolist(), 'train_loss': m.train_score_.tolist()}
        )
    save('ensemble', out)


@fixture
def svm():
    from sklearn.svm import SVC, SVR

    rng = np.random.default_rng(41)
    n = 100
    y3 = rng.integers(0, 3, size=n)
    centers = np.array([[0, 0], [2.0, 1.0], [0.5, 2.5]])
    x = centers[y3] + rng.normal(size=(n, 2)) * 0.9
    y2 = (y3 == 1).astype(int)
    grid = rng.uniform(-2, 4, size=(50, 2))
    out = {'x': x.tolist(), 'y2': y2.tolist(), 'y3': y3.tolist(), 'grid': grid.tolist(), 'binary': [], 'multi': None}
    configs = [
        {'kernel': 'rbf', 'C': 1.0},
        {'kernel': 'rbf', 'C': 10.0, 'gamma': 0.5},
        {'kernel': 'linear', 'C': 0.5},
        {'kernel': 'poly', 'C': 1.0, 'degree': 3, 'gamma': 0.3, 'coef0': 1.0},
        {'kernel': 'sigmoid', 'C': 1.0, 'gamma': 0.1, 'coef0': 0.0},
    ]
    for cfg in configs:
        m = SVC(tol=1e-9, cache_size=500, **cfg).fit(x, y2)
        out['binary'].append(
            {
                'config': cfg,
                'support': m.support_.tolist(),
                'dual_coef': m.dual_coef_[0].tolist(),
                'intercept': float(m.intercept_[0]),
                'decision': m.decision_function(grid).tolist(),
                'pred': m.predict(grid).tolist(),
                'gamma': float(m._gamma),
            }
        )
    m3 = SVC(kernel='rbf', C=2.0, tol=1e-9, decision_function_shape='ovo').fit(x, y3)
    out['multi'] = {'support': m3.support_.tolist(), 'pred': m3.predict(grid).tolist(), 'ovo': m3.decision_function(grid).tolist()}
    xr = rng.uniform(-3, 3, size=(80, 1))
    yr = np.sin(xr[:, 0]) + rng.normal(size=80) * 0.15
    gr = np.linspace(-3.2, 3.2, 40)[:, None]
    out['svr'] = {'x': xr.tolist(), 'y': yr.tolist(), 'grid': gr.tolist(), 'fits': []}
    for cfg in [{'kernel': 'rbf', 'C': 1.0, 'epsilon': 0.1}, {'kernel': 'rbf', 'C': 10.0, 'epsilon': 0.05, 'gamma': 1.0}, {'kernel': 'linear', 'C': 1.0, 'epsilon': 0.2}]:
        m = SVR(tol=1e-9, **cfg).fit(xr, yr)
        out['svr']['fits'].append(
            {'config': cfg, 'support': m.support_.tolist(), 'dual_coef': m.dual_coef_[0].tolist(), 'intercept': float(m.intercept_[0]), 'pred': m.predict(gr).tolist()}
        )
    save('svm', out)


@fixture
def kernel_models():
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel as C
    from sklearn.gaussian_process.kernels import ExpSineSquared, Matern, RBF
    from sklearn.kernel_ridge import KernelRidge
    from sklearn.linear_model import BayesianRidge

    rng = np.random.default_rng(51)
    xs = np.sort(rng.uniform(-3, 3, size=25))[:, None]
    ys = np.sin(1.4 * xs[:, 0]) + rng.normal(size=25) * 0.2
    grid = np.linspace(-4, 4, 30)[:, None]
    out = {'x': xs.tolist(), 'y': ys.tolist(), 'grid': grid.tolist(), 'gp': [], 'krr': []}
    kernels = {
        'rbf': lambda v, l: C(v) * RBF(l),
        'matern32': lambda v, l: C(v) * Matern(l, nu=1.5),
        'matern52': lambda v, l: C(v) * Matern(l, nu=2.5),
        'periodic': lambda v, l: C(v) * ExpSineSquared(length_scale=l, periodicity=2.5),
    }
    for name, make in kernels.items():
        for v, l, noise in [(1.0, 1.0, 0.05), (2.0, 0.6, 0.01)]:
            gp = GaussianProcessRegressor(kernel=make(v, l), alpha=noise, optimizer=None, normalize_y=False).fit(xs, ys)
            mean, std = gp.predict(grid, return_std=True)
            out['gp'].append(
                {'kernel': name, 'variance': v, 'length': l, 'noise': noise, 'mean': mean.tolist(), 'std': std.tolist(), 'lml': float(gp.log_marginal_likelihood_value_)}
            )
    for alpha, gamma in [(0.1, 0.5), (1.0, 2.0)]:
        m = KernelRidge(alpha=alpha, kernel='rbf', gamma=gamma).fit(xs, ys)
        out['krr'].append({'alpha': alpha, 'gamma': gamma, 'pred': m.predict(grid).tolist(), 'dual': m.dual_coef_.tolist()})
    # Bayesian ridge on polynomial-ish features
    xb = rng.normal(size=(60, 4))
    yb = 0.5 + xb @ np.array([1.5, -2.0, 0.0, 0.5]) + rng.normal(size=60) * 0.7
    qb = rng.normal(size=(10, 4))
    br = BayesianRidge(tol=1e-12, max_iter=1000).fit(xb, yb)
    pred, std = br.predict(qb, return_std=True)
    out['bayes'] = {
        'x': xb.tolist(), 'y': yb.tolist(), 'queries': qb.tolist(),
        'coef': br.coef_.tolist(), 'intercept': float(br.intercept_), 'alpha': float(br.alpha_), 'lambda': float(br.lambda_),
        'sigma': br.sigma_.tolist(), 'pred': pred.tolist(), 'std': std.tolist(),
    }
    save('kernel_models', out)


@fixture
def xgboost_fixture():
    import xgboost as xgb

    f32 = lambda a: a.astype(np.float32).astype(np.float64)
    rng = np.random.default_rng(61)
    n = 140
    y3 = rng.integers(0, 3, size=n)
    centers = np.array([[0, 0], [2.0, 1.0], [0.5, 2.5]])
    x = f32(centers[y3] + rng.normal(size=(n, 2)) * 1.0)
    y2 = (y3 == 1).astype(int)
    grid = f32(rng.uniform(-2, 4, size=(50, 2)))
    out = {'x': x.tolist(), 'y2': y2.tolist(), 'y3': y3.tolist(), 'grid': grid.tolist(), 'classifier': [], 'regressor': {'fits': []}}
    common = dict(tree_method='exact', base_score=0.5, n_jobs=1, random_state=0, reg_alpha=0.0, subsample=1.0, colsample_bytree=1.0)
    for cfg, y in [
        ({'n_estimators': 12, 'learning_rate': 0.3, 'max_depth': 3, 'reg_lambda': 1.0, 'gamma': 0.0, 'min_child_weight': 1.0}, y2),
        ({'n_estimators': 10, 'learning_rate': 0.2, 'max_depth': 4, 'reg_lambda': 5.0, 'gamma': 0.3, 'min_child_weight': 2.0}, y2),
        ({'n_estimators': 8, 'learning_rate': 0.3, 'max_depth': 3, 'reg_lambda': 1.0, 'gamma': 0.0, 'min_child_weight': 1.0}, y3),
    ]:
        m = xgb.XGBClassifier(**cfg, **common).fit(x, y)
        out['classifier'].append({'config': cfg, 'y': y.tolist(), 'proba': m.predict_proba(grid).tolist(), 'staged3': m.predict_proba(grid, iteration_range=(0, 3)).tolist()})
    xr = f32(rng.uniform(-3, 3, size=(120, 2)))
    yr = np.sin(xr[:, 0]) + 0.3 * xr[:, 1] ** 2 + rng.normal(size=120) * 0.2
    gr = f32(rng.uniform(-3, 3, size=(40, 2)))
    out['regressor'].update({'x': xr.tolist(), 'y': yr.tolist(), 'grid': gr.tolist()})
    for cfg in [
        {'n_estimators': 15, 'learning_rate': 0.3, 'max_depth': 3, 'reg_lambda': 1.0, 'gamma': 0.0, 'min_child_weight': 1.0},
        {'n_estimators': 12, 'learning_rate': 0.2, 'max_depth': 4, 'reg_lambda': 10.0, 'gamma': 0.5, 'min_child_weight': 3.0},
    ]:
        m = xgb.XGBRegressor(**cfg, **common).fit(xr, yr)
        out['regressor']['fits'].append({'config': cfg, 'pred': m.predict(gr).tolist(), 'staged3': m.predict(gr, iteration_range=(0, 3)).tolist()})
    save('xgboost', out)


@fixture
def kmeans():
    from sklearn.cluster import KMeans

    rng = np.random.default_rng(71)
    out = {'runs': []}
    for k, n, spread in [(3, 150, 0.9), (4, 200, 1.1), (5, 180, 1.6)]:
        centers = rng.uniform(-4, 4, size=(k, 2))
        y = rng.integers(0, k, size=n)
        x = centers[y] + rng.normal(size=(n, 2)) * spread
        init = x[rng.choice(n, size=k, replace=False)]
        m = KMeans(n_clusters=k, init=init, n_init=1, algorithm='lloyd', tol=1e-4, max_iter=300).fit(x)
        out['runs'].append(
            {
                'x': x.tolist(),
                'init': init.tolist(),
                'centers': m.cluster_centers_.tolist(),
                'labels': m.labels_.tolist(),
                'inertia': float(m.inertia_),
                'n_iter': int(m.n_iter_),
            }
        )
    save('kmeans', out)


@fixture
def hierarchical():
    from scipy.cluster.hierarchy import cophenet, dendrogram, fcluster, linkage
    from scipy.spatial.distance import pdist

    rng = np.random.default_rng(81)
    n = 60
    centers = np.array([[0, 0], [3, 1], [1, 3.5], [4, 4]])
    y = rng.integers(0, 4, size=n)
    x = centers[y] + rng.normal(size=(n, 2)) * 0.6
    out = {'x': x.tolist(), 'methods': {}}
    for method in ['single', 'complete', 'average', 'ward']:
        z = linkage(x, method=method)
        _, coph = cophenet(z, pdist(x))
        out['methods'][method] = {
            'z': z.tolist(),
            'leaves': [int(v) for v in dendrogram(z, no_plot=True)['leaves']],
            'cophenet': coph.tolist(),
            'cut3': fcluster(z, 3, criterion='maxclust').tolist(),
            'cut_height': fcluster(z, float(np.median(z[:, 2])), criterion='distance').tolist(),
            'median_height': float(np.median(z[:, 2])),
        }
    save('hierarchical', out)


@fixture
def density():
    from sklearn.cluster import DBSCAN, OPTICS, MeanShift, estimate_bandwidth, cluster_optics_dbscan
    from sklearn.datasets import make_blobs, make_moons

    rng = np.random.default_rng(91)
    xm, _ = make_moons(n_samples=120, noise=0.07, random_state=1)
    xb, _ = make_blobs(n_samples=120, centers=[[0, 0], [3, 3], [0, 4]], cluster_std=[0.4, 0.8, 0.3], random_state=2)
    out = {'moons': xm.tolist(), 'blobs': xb.tolist(), 'dbscan': [], 'optics': [], 'meanshift': []}
    for name, x, eps, ms in [('moons', xm, 0.2, 5), ('moons', xm, 0.12, 4), ('blobs', xb, 0.5, 5), ('blobs', xb, 0.3, 8)]:
        m = DBSCAN(eps=eps, min_samples=ms).fit(x)
        out['dbscan'].append({'data': name, 'eps': eps, 'min_samples': ms, 'labels': m.labels_.tolist(), 'core': m.core_sample_indices_.tolist()})
    for name, x, ms in [('moons', xm, 5), ('blobs', xb, 8)]:
        o = OPTICS(min_samples=ms, cluster_method='dbscan', eps=0.5).fit(x)
        rec = {
            'data': name,
            'min_samples': ms,
            'ordering': o.ordering_.tolist(),
            'reachability': [None if not np.isfinite(v) else float(v) for v in o.reachability_],
            'core': [None if not np.isfinite(v) else float(v) for v in o.core_distances_],
            'predecessor': o.predecessor_.tolist(),
            'extract': {},
        }
        for eps in [0.15, 0.3, 0.6]:
            rec['extract'][str(eps)] = cluster_optics_dbscan(
                reachability=o.reachability_, core_distances=o.core_distances_, ordering=o.ordering_, eps=eps
            ).tolist()
        out['optics'].append(rec)
    for name, x, bw in [('blobs', xb, None), ('blobs', xb, 1.0), ('moons', xm, 0.5)]:
        bandwidth = float(estimate_bandwidth(x, quantile=0.3)) if bw is None else bw
        m = MeanShift(bandwidth=bandwidth, bin_seeding=False).fit(x)
        out['meanshift'].append({'data': name, 'bandwidth': bandwidth, 'estimated': bw is None, 'centers': m.cluster_centers_.tolist(), 'labels': m.labels_.tolist()})
    save('density', out)


@fixture
def mixture():
    from sklearn.mixture import GaussianMixture

    rng = np.random.default_rng(101)
    n = 240
    centers = np.array([[0, 0], [3.5, 1.0], [1.0, 3.5]])
    scales = np.array([[1.0, 0.5], [0.6, 1.1], [0.9, 0.9]])
    y = rng.integers(0, 3, size=n)
    x = centers[y] + rng.normal(size=(n, 2)) * scales[y]
    x[:, 1] += 0.35 * x[:, 0]
    k = 3
    means0 = x[rng.choice(n, size=k, replace=False)]
    weights0 = np.array([0.4, 0.35, 0.25])
    var0 = np.cov(x.T)
    out = {'x': x.tolist(), 'means0': means0.tolist(), 'weights0': weights0.tolist(), 'var0': var0.tolist(), 'fits': []}
    for cov_type in ['full', 'diag', 'spherical', 'tied']:
        if cov_type == 'full':
            covs0 = [var0 for _ in range(k)]
            prec0 = np.array([np.linalg.inv(c) for c in covs0])
        elif cov_type == 'diag':
            covs0 = [np.diag(np.diag(var0)) for _ in range(k)]
            prec0 = np.array([1.0 / np.diag(c) for c in covs0])
        elif cov_type == 'spherical':
            covs0 = [np.eye(2) * np.trace(var0) / 2 for _ in range(k)]
            prec0 = np.array([1.0 / (np.trace(var0) / 2) for _ in covs0])
        else:
            covs0 = [var0 for _ in range(k)]
            prec0 = np.linalg.inv(var0)
        m = GaussianMixture(
            n_components=k, covariance_type=cov_type, weights_init=weights0, means_init=means0, precisions_init=prec0,
            tol=1e-3, max_iter=200, reg_covar=1e-6,
        ).fit(x)
        if cov_type == 'full':
            covs = [c.tolist() for c in m.covariances_]
        elif cov_type == 'diag':
            covs = [np.diag(c).tolist() for c in m.covariances_]
        elif cov_type == 'spherical':
            covs = [(np.eye(2) * c).tolist() for c in m.covariances_]
        else:
            covs = [m.covariances_.tolist() for _ in range(k)]
        out['fits'].append(
            {
                'type': cov_type,
                'covs0': [np.asarray(c).tolist() for c in covs0],
                'weights': m.weights_.tolist(),
                'means': m.means_.tolist(),
                'covariances': covs,
                'n_iter': int(m.n_iter_),
                'converged': bool(m.converged_),
                'lower_bound': float(m.lower_bound_),
                'score': float(m.score(x)),
                'bic': float(m.bic(x)),
                'aic': float(m.aic(x)),
                'proba': m.predict_proba(x[:8]).tolist(),
                'labels': m.predict(x).tolist(),
            }
        )
    save('mixture', out)


@fixture
def spectral():
    from sklearn.datasets import make_moons

    x, _ = make_moons(n_samples=90, noise=0.06, random_state=3)
    n = len(x)
    d2 = ((x[:, None, :] - x[None, :, :]) ** 2).sum(-1)
    out = {'x': x.tolist(), 'rbf': {}, 'knn': {}}
    gamma = 15.0
    w = np.exp(-gamma * d2)
    np.fill_diagonal(w, 0.0)
    deg = w.sum(1)
    m = w / np.sqrt(deg[:, None] * deg[None, :])
    vals = np.linalg.eigvalsh(np.eye(n) - m)
    out['rbf'] = {'gamma': gamma, 'eigenvalues': np.sort(vals).tolist()}
    k = 8
    near = np.zeros((n, n))
    for i in range(n):
        order = [j for j in np.argsort(d2[i], kind='stable') if j != i][:k]
        near[i, order] = 1
    wk = 0.5 * (near + near.T)
    degk = wk.sum(1)
    mk = wk / np.sqrt(degk[:, None] * degk[None, :])
    out['knn'] = {'neighbors': k, 'eigenvalues': np.sort(np.linalg.eigvalsh(np.eye(n) - mk)).tolist(), 'affinity_sum': float(wk.sum())}
    save('spectral', out)


@fixture
def pca():
    from sklearn.datasets import load_iris, load_wine
    from sklearn.decomposition import PCA, KernelPCA
    from sklearn.preprocessing import StandardScaler

    out = {'datasets': []}
    rng = np.random.default_rng(111)
    for name, x in [('iris', load_iris().data), ('wine', StandardScaler().fit_transform(load_wine().data))]:
        p = PCA().fit(x)
        p2 = PCA(n_components=2, whiten=True).fit(x)
        out['datasets'].append(
            {
                'name': name,
                'x': x.tolist(),
                'components': p.components_.tolist(),
                'variance': p.explained_variance_.tolist(),
                'ratio': p.explained_variance_ratio_.tolist(),
                'mean': p.mean_.tolist(),
                'scores2': PCA(n_components=2).fit_transform(x)[:10].tolist(),
                'whitened2': p2.transform(x)[:10].tolist(),
            }
        )
    # kernel PCA on a noisy circle pair
    t = rng.uniform(0, 2 * np.pi, size=70)
    r = np.where(np.arange(70) % 2 == 0, 1.0, 2.5)
    xk = np.c_[r * np.cos(t), r * np.sin(t)] + rng.normal(size=(70, 2)) * 0.05
    out['kpca'] = {'x': xk.tolist(), 'fits': []}
    for kernel, kw in [('rbf', {'gamma': 1.5}), ('poly', {'gamma': 0.5, 'degree': 2, 'coef0': 1.0}), ('linear', {}), ('cosine', {})]:
        m = KernelPCA(n_components=3, kernel=kernel, eigen_solver='dense', **kw).fit(xk)
        out['kpca']['fits'].append(
            {
                'kernel': kernel,
                'kw': kw,
                'eigenvalues': m.eigenvalues_.tolist(),
                'embedding': m.transform(xk)[:8].tolist(),
                'new': m.transform(xk[:3] * 1.1).tolist(),
            }
        )
    save('pca', out)


@fixture
def manifold():
    from sklearn.datasets import make_swiss_roll
    from sklearn.manifold import Isomap, LocallyLinearEmbedding, smacof, trustworthiness
    from sklearn.metrics import pairwise_distances

    x, t = make_swiss_roll(100, noise=0.05, random_state=1)
    rng = np.random.default_rng(121)
    out = {'x': x.tolist(), 't': t.tolist()}
    iso = Isomap(n_neighbors=10, n_components=2).fit(x)
    out['isomap'] = {
        'neighbors': 10,
        'embedding': iso.embedding_.tolist(),
        'geodesic_sample': iso.dist_matrix_[:6, :6].tolist(),
    }
    lle = LocallyLinearEmbedding(n_neighbors=10, n_components=2, eigen_solver='dense', reg=1e-3).fit(x)
    out['lle'] = {'neighbors': 10, 'embedding': lle.embedding_.tolist(), 'reconstruction_error': float(lle.reconstruction_error_)}
    d = pairwise_distances(x)
    init = rng.normal(size=(100, 2))
    xs, st, n_iter = smacof(d, n_components=2, init=init, n_init=1, eps=1e-3, max_iter=300, metric=True, return_n_iter=True, random_state=0)
    out['smacof'] = {'init': init.tolist(), 'embedding': xs.tolist(), 'stress': float(st), 'n_iter': int(n_iter)}
    z = rng.normal(size=(100, 2))
    out['trust'] = {
        'embedded': z.tolist(),
        'values': {str(k): float(trustworthiness(x, z, n_neighbors=k)) for k in [3, 5, 10]},
        'iso_values': {str(k): float(trustworthiness(x, iso.embedding_, n_neighbors=k)) for k in [3, 5, 10]},
    }
    save('manifold', out)


@fixture
def tsne():
    from scipy.spatial.distance import squareform
    from sklearn.datasets import load_iris
    from sklearn.manifold import TSNE
    from sklearn.manifold._t_sne import _joint_probabilities, _kl_divergence
    from sklearn.metrics import pairwise_distances

    rng = np.random.default_rng(131)
    iris = load_iris()
    idx = rng.choice(150, size=60, replace=False)
    x = iris.data[idx]
    d2 = pairwise_distances(x, squared=True)
    out = {'x': x.tolist(), 'labels': iris.target[idx].tolist(), 'cases': []}
    for perplexity in [5.0, 15.0]:
        p = _joint_probabilities(d2.astype(np.float32), perplexity, 0)
        y = rng.normal(size=(60, 2))
        kl, grad = _kl_divergence(y.ravel(), p, 1, 60, 2)
        out['cases'].append(
            {
                'perplexity': perplexity,
                'p_full': squareform(p).tolist(),
                'y': y.tolist(),
                'kl': float(kl),
                'grad': grad.reshape(60, 2).tolist(),
            }
        )
    model = TSNE(n_components=2, perplexity=15.0, init='pca', method='exact', random_state=0, max_iter=1000).fit(x)
    out['sklearn_final_kl'] = float(model.kl_divergence_)
    save('tsne', out)


@fixture
def anomaly():
    from sklearn.ensemble import IsolationForest
    from sklearn.ensemble._iforest import _average_path_length
    from sklearn.neighbors import LocalOutlierFactor

    rng = np.random.default_rng(141)
    x = np.vstack([rng.normal(size=(80, 2)), rng.normal(size=(40, 2)) * 0.4 + [4, 4], rng.uniform(-8, 8, size=(8, 2))])
    out = {'x': x.tolist(), 'lof': []}
    for k in [5, 10, 20]:
        m = LocalOutlierFactor(n_neighbors=k).fit(x)
        out['lof'].append({'k': k, 'scores': (-m.negative_outlier_factor_).tolist()})
    out['path_length'] = {str(n): float(_average_path_length(np.array([n]))[0]) for n in [1, 2, 3, 10, 64, 256]}
    save('anomaly', out)


@fixture
def timeseries():
    from statsmodels.tsa.ar_model import AutoReg
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.holtwinters import ExponentialSmoothing, Holt, SimpleExpSmoothing
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.stattools import acf, pacf

    rng = np.random.default_rng(2024)
    n = 96
    t = np.arange(n)
    x = 50 + 0.4 * t + 6 * np.sin(2 * np.pi * t / 12) + rng.normal(size=n) * 1.5
    out = {'x': x.tolist()}
    out['acf'] = acf(x, nlags=20, fft=False).tolist()
    out['pacf'] = pacf(x, nlags=15, method='ldb').tolist()
    # statsmodels' decomposition needs a positive series for the multiplicative case; x is positive
    for model in ['additive', 'multiplicative']:
        d = seasonal_decompose(x, model=model, period=12)
        out['decompose_' + model] = {
            'trend': [None if np.isnan(v) else float(v) for v in d.trend],
            'seasonal': d.seasonal.tolist(),
            'resid': [None if np.isnan(v) else float(v) for v in d.resid],
        }
    ar = AutoReg(x - x.mean(), lags=3, trend='c').fit()
    out['autoreg'] = {'params': ar.params.tolist(), 'sigma2': float(ar.sigma2)}
    a = 0.4
    ses = SimpleExpSmoothing(x, initialization_method='known', initial_level=x[0]).fit(smoothing_level=a, optimized=False)
    out['ses'] = {'alpha': a, 'fitted': ses.fittedvalues.tolist(), 'forecast': ses.forecast(6).tolist(), 'sse': float(ses.sse)}
    holt = Holt(x, initialization_method='known', initial_level=x[0], initial_trend=x[1] - x[0]).fit(
        smoothing_level=0.5, smoothing_trend=0.2, optimized=False)
    out['holt'] = {'alpha': 0.5, 'beta': 0.2, 'fitted': holt.fittedvalues.tolist(), 'forecast': holt.forecast(6).tolist(), 'sse': float(holt.sse)}
    l0 = x[:12].mean()
    b0 = (x[12:24].mean() - x[:12].mean()) / 12
    s0 = (x[:12] - l0)
    hw = ExponentialSmoothing(x, trend='add', seasonal='add', seasonal_periods=12, initialization_method='known',
                              initial_level=l0, initial_trend=b0, initial_seasonal=s0).fit(
        smoothing_level=0.4, smoothing_trend=0.1, smoothing_seasonal=0.3, optimized=False)
    out['hw'] = {'alpha': 0.4, 'beta': 0.1, 'gamma': 0.3, 'fitted': hw.fittedvalues.tolist(), 'forecast': hw.forecast(12).tolist(), 'sse': float(hw.sse)}
    # an ARMA(1,1) with known coefficients, fitted by exact maximum likelihood
    m = 600
    e = rng.normal(size=m + 50)
    y = np.zeros(m + 50)
    for i in range(1, m + 50):
        y[i] = 0.7 * y[i - 1] + e[i] + 0.4 * e[i - 1]
    y = y[50:] + 10
    fit = ARIMA(y, order=(1, 0, 1), trend='c').fit()
    fc = fit.get_forecast(8)
    out['arma'] = {'y': y.tolist(), 'params': fit.params.tolist(), 'names': list(fit.param_names),
                   'forecast': fc.predicted_mean.tolist(), 'se': fc.se_mean.tolist()}
    walk = np.cumsum(rng.normal(size=200)) + 100
    fit2 = ARIMA(walk, order=(0, 1, 0)).fit()
    fc2 = fit2.get_forecast(10)
    out['walk'] = {'y': walk.tolist(), 'forecast': fc2.predicted_mean.tolist(), 'se': fc2.se_mean.tolist(), 'sigma2': float(fit2.params[-1])}
    save('timeseries', out)


@fixture
def airline():
    """The Box-Jenkins airline model (0,1,1)(0,1,1)12 on the log of the passenger series, by exact maximum likelihood."""
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    air = np.array([
        112, 118, 132, 129, 121, 135, 148, 148, 136, 119, 104, 118, 115, 126, 141, 135, 125, 149, 170, 170,
        158, 133, 114, 140, 145, 150, 178, 163, 172, 178, 199, 199, 184, 162, 146, 166, 171, 180, 193, 181,
        183, 218, 230, 242, 209, 191, 172, 194, 196, 196, 236, 235, 229, 243, 264, 272, 237, 211, 180, 201,
        204, 188, 235, 227, 234, 264, 302, 293, 259, 229, 203, 229, 242, 233, 267, 269, 270, 315, 364, 347,
        312, 274, 237, 278, 284, 277, 317, 313, 318, 374, 413, 405, 355, 306, 271, 306, 315, 301, 356, 348,
        355, 422, 465, 467, 404, 347, 305, 336, 340, 318, 362, 348, 363, 435, 491, 505, 404, 359, 310, 337,
        360, 342, 406, 396, 420, 472, 548, 559, 463, 407, 362, 405, 417, 391, 419, 461, 472, 535, 622, 606,
        508, 461, 390, 432], dtype=float)
    y = np.log(air)
    fit = SARIMAX(y, order=(0, 1, 1), seasonal_order=(0, 1, 1, 12)).fit(disp=False)
    fc = fit.get_forecast(12)
    save('airline', {'y': y.tolist(), 'params': fit.params.tolist(), 'names': list(fit.param_names),
                     'forecast': fc.predicted_mean.tolist(), 'se': fc.se_mean.tolist()})


@fixture
def evaluation():
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import (average_precision_score, brier_score_loss, f1_score, matthews_corrcoef,
                                 precision_recall_curve, precision_score, recall_score)

    rng = np.random.default_rng(77)
    n = 400
    y = (rng.random(n) < 0.15).astype(int)
    score = np.clip(0.15 + 0.35 * y + rng.normal(size=n) * 0.2, 0.001, 0.999)
    pred = (score >= 0.4).astype(int)
    prec, rec, thr = precision_recall_curve(y, score)
    frac, mean_pred = calibration_curve(y, score, n_bins=10, strategy='uniform')
    bins = np.minimum((score * 10).astype(int), 9)
    counts = [int((bins == b).sum()) for b in range(10)]
    save('evaluation', {
        'y': y.tolist(), 'score': score.tolist(),
        'ap': float(average_precision_score(y, score)),
        'pr': {'precision': prec.tolist(), 'recall': rec.tolist(), 'thresholds': thr.tolist()},
        'calibration': {'observed': frac.tolist(), 'predicted': mean_pred.tolist(), 'counts': counts},
        'brier': float(brier_score_loss(y, score)),
        'at04': {'precision': float(precision_score(y, pred)), 'recall': float(recall_score(y, pred)),
                 'f1': float(f1_score(y, pred)), 'mcc': float(matthews_corrcoef(y, pred))},
    })


@fixture
def isotonic():
    from sklearn.isotonic import IsotonicRegression

    rng = np.random.default_rng(31)
    x = np.sort(np.round(rng.uniform(0, 10, 60), 1))  # rounded, so some x repeat
    y = np.sqrt(x) * 2 + rng.normal(size=60) * 0.9
    grid = np.linspace(-1, 11, 61)
    out = {'x': x.tolist(), 'y': y.tolist(), 'grid': grid.tolist()}
    for increasing in (True, False):
        yy = y if increasing else -y + 3
        m = IsotonicRegression(increasing=increasing, out_of_bounds='clip').fit(x, yy)
        out['up' if increasing else 'down'] = {
            'y': yy.tolist(),
            'fitted': m.predict(x).tolist(),
            'grid': m.predict(grid).tolist(),
            'pieces': int(len(np.unique(np.round(m.predict(x), 12)))),
        }
    save('isotonic', out)


@fixture
def survival():
    from statsmodels.duration.hazard_regression import PHReg
    from statsmodels.duration.survfunc import SurvfuncRight, survdiff

    rng = np.random.default_rng(1)
    n = 60
    group = rng.integers(0, 2, n)
    age = rng.normal(size=n)
    # exponential times, hazard multiplied by exp(0.8 group + 0.5 age); times rounded so that some tie
    hazard = 0.15 * np.exp(0.8 * group + 0.5 * age)
    t_event = rng.exponential(1 / hazard)
    t_cens = rng.uniform(2, 14, n)
    time = np.round(np.minimum(t_event, t_cens), 1) + 0.1
    status = (t_event <= t_cens).astype(int)
    sf = SurvfuncRight(time, status)
    chisq, p = survdiff(time, status, group)
    x = np.column_stack([group, age])
    fit = PHReg(time, x, status=status, ties='breslow').fit()
    save('survival', {
        'time': time.tolist(), 'event': status.tolist(), 'group': group.tolist(), 'age': age.tolist(),
        'km': {
            'times': sf.surv_times.tolist(), 'atRisk': sf.n_risk.tolist(), 'events': sf.n_events.tolist(),
            'survival': sf.surv_prob.tolist(),
            # statsmodels reports nan once the curve has reached zero
            'se': [None if not np.isfinite(v) else float(v) for v in sf.surv_prob_se],
            'median': float(sf.quantile(0.5)),
        },
        'logrank': {'statistic': float(chisq), 'pValue': float(p)},
        'cox': {'coef': fit.params.tolist(), 'se': fit.bse.tolist(), 'llf': float(fit.llf)},
    })


if __name__ == '__main__':
    names = sys.argv[1:] or list(REGISTRY)
    for name in names:
        REGISTRY[name]()
