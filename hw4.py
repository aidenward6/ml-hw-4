import itertools
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog, brentq
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from numpy.polynomial.legendre import legvander

np.random.seed(0)
os.makedirs("figs", exist_ok=True)


def f_m(h, m, eta=0.05):
    return np.sqrt((h * (np.log(2 * m / h) + 1) - np.log(eta / 4)) / m)


# ---------------- P1 ----------------
print("===== P1 =====")

plt.figure()
for m in [10**3, 10**4, 10**5]:
    ratios = np.linspace(1 / m, 1, 200)
    hs = np.unique(np.maximum(1, np.round(ratios * m)).astype(int))
    plt.plot(hs / m, f_m(hs, m), label=f"m = {m}")
plt.axhline(1, color="gray", linestyle="--", label="f = 1")
plt.xlabel("h / m")
plt.ylabel("f_m(h)")
plt.title("VC confidence term vs h/m (eta = 0.05)")
plt.legend()
plt.savefig("figs/p1a_vc_confidence.png", dpi=150)
plt.close()

# (b)
for m in [10**3, 10**4, 10**5]:
    r = brentq(lambda r: f_m(r * m, m) - 1, 1e-6, 1)
    print(f"m = {m}: f_m(h) = 1 at h/m = {r:.4f}")

# (c)
m = 10**4
for remp in [0, 0.05]:
    h = 1
    while remp + f_m(h + 1, m) <= 0.1:
        h += 1
    print(f"m = {m}, Remp = {remp}: largest h = {h}  (bound = {remp + f_m(h, m):.4f})")


# ---------------- P2 ----------------
print("\n===== P2 =====")


def separable(X, y):
    m, n = X.shape
    A_ub = -y[:, None] * np.hstack([X, np.ones((m, 1))])
    b_ub = -np.ones(m)
    res = linprog(c=np.zeros(n + 1), A_ub=A_ub, b_ub=b_ub,
                  bounds=[(None, None)] * (n + 1), method="highs")
    return res.status == 0


def can_shatter(X):
    for y in itertools.product([-1, 1], repeat=len(X)):
        if not separable(X, np.array(y)):
            return False
    return True


def bad_labelings(X):
    bad = []
    for y in itertools.product([-1, 1], repeat=len(X)):
        if not separable(X, np.array(y)):
            bad.append(y)
    return bad


# (a)
X_tri = np.array([[0, 0], [1, 0], [0, 1]], dtype=float)
X_col = np.array([[0, 0], [1, 0], [2, 0]], dtype=float)
print("non-collinear shattered:", can_shatter(X_tri))
print("collinear shattered:", can_shatter(X_col))
print("collinear non-separable labelings:", bad_labelings(X_col))

# (b)
print("\n n | n+1 points | n+2 points")
for n in [2, 3, 4, 5]:
    c1 = 0
    c2 = 0
    for t in range(1000):
        if can_shatter(np.random.randn(n + 1, n)):
            c1 += 1
        if can_shatter(np.random.randn(n + 2, n)):
            c2 += 1
    print(f" {n} | {c1 / 1000:.3f}      | {c2 / 1000:.3f}")

# (c)
rng = np.random.RandomState(4)
X4 = rng.randn(4, 2)
print("\n4 random points:\n", X4)
bad = bad_labelings(X4)
print("non-separable labelings:", bad)


def in_triangle(p, a, b, c):
    def cross(o, u, v):
        return (u[0] - o[0]) * (v[1] - o[1]) - (u[1] - o[1]) * (v[0] - o[0])
    d1 = cross(a, b, p)
    d2 = cross(b, c, p)
    d3 = cross(c, a, p)
    return (d1 > 0 and d2 > 0 and d3 > 0) or (d1 < 0 and d2 < 0 and d3 < 0)


interior = None
for i in range(4):
    others = [X4[j] for j in range(4) if j != i]
    if in_triangle(X4[i], *others):
        interior = i

if interior is not None:
    print(f"case: point {interior} is inside the triangle of the other three")
    pred = []
    for s in [1, -1]:
        y = [-s] * 4
        y[interior] = s
        pred.append(tuple(y))
else:
    print("case: convex quadrilateral")
    # find the diagonal pairing: two segments that cross
    pred = []
    for (a, b), (c, d) in [((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2))]:
        # check if segment a-b crosses segment c-d
        def side(p, q, r):
            return np.sign((q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]))
        if side(X4[a], X4[b], X4[c]) != side(X4[a], X4[b], X4[d]) and \
                side(X4[c], X4[d], X4[a]) != side(X4[c], X4[d], X4[b]):
            for s in [1, -1]:
                y = [0] * 4
                y[a] = y[b] = s
                y[c] = y[d] = -s
                pred.append(tuple(y))
print("predicted non-separable:", sorted(pred))
print("matches:", sorted(pred) == sorted(bad))

y_bad = np.array(bad[0])
plt.figure()
plt.scatter(X4[y_bad == 1, 0], X4[y_bad == 1, 1], c="red", s=80, label="+1")
plt.scatter(X4[y_bad == -1, 0], X4[y_bad == -1, 1], c="blue", s=80, label="-1")
for i in range(4):
    plt.annotate(str(i), X4[i] + 0.05)
plt.xlabel("x1")
plt.ylabel("x2")
plt.title(f"4 random points with non-separable labeling {bad[0]}")
plt.legend()
plt.savefig("figs/p2c_nonseparable.png", dpi=150)
plt.close()


# (d)
def rect_realizable(X, y):
    pos = X[y == 1]
    neg = X[y == -1]
    if len(pos) == 0:
        return True
    lo = pos.min(axis=0)
    hi = pos.max(axis=0)
    inside = np.all((neg >= lo) & (neg <= hi), axis=1)
    return not inside.any()


def rect_can_shatter(X):
    for y in itertools.product([-1, 1], repeat=len(X)):
        if not rect_realizable(X, np.array(y)):
            return False
    return True


X_diamond = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]], dtype=float)
print("\nrectangles shatter (+-1,0),(0,+-1):", rect_can_shatter(X_diamond))
cnt = 0
for t in range(1000):
    if rect_can_shatter(np.random.randn(5, 2)):
        cnt += 1
print("random 5-point sets shattered by rectangles:", cnt, "/ 1000")


# ---------------- P3 ----------------
print("\n===== P3 =====")
c_lo = np.array([0.2, 0.3])
c_hi = np.array([0.7, 0.6])
area_c = 0.5 * 0.3


def true_error(m):
    X = np.random.rand(m, 2)
    pos = X[np.all((X >= c_lo) & (X <= c_hi), axis=1)]
    if len(pos) == 0:
        return area_c
    side = pos.max(axis=0) - pos.min(axis=0)
    return area_c - side[0] * side[1]


T = 2000
eps = 0.05
delta = 0.05
ms = [10, 20, 50, 100, 200, 500, 1000]
mean_err = []
p_bad = []
for m in ms:
    errs = np.array([true_error(m) for _ in range(T)])
    mean_err.append(errs.mean())
    p_bad.append(np.mean(errs > eps))
    print(f"m = {m:5d}: mean R = {errs.mean():.5f}, P(R > eps) = {p_bad[-1]:.4f}")

m_bound = 4 / eps * np.log(4 / delta)
print(f"PAC bound m >= {m_bound:.1f}")

# (c) smallest m (step 10) with empirical P(R > eps) <= delta
m = 10
while True:
    errs = np.array([true_error(m) for _ in range(T)])
    if np.mean(errs > eps) <= delta:
        break
    m += 10
m_emp = m
print(f"smallest empirical m = {m_emp}, P(R > eps) = {np.mean(errs > eps):.4f}")
print(f"bound is pessimistic by a factor of {m_bound / m_emp:.2f}")

plt.figure()
plt.loglog(ms, mean_err, "o-", label="mean true error")
plt.xlabel("m (training set size)")
plt.ylabel("mean R")
plt.title("Rectangle learner: mean true error vs m")
plt.legend()
plt.savefig("figs/p3b_mean_error.png", dpi=150)
plt.close()

plt.figure()
plt.semilogx(ms, p_bad, "o-", label="empirical P(R > 0.05)")
plt.axhline(delta, color="gray", linestyle="--", label="delta = 0.05")
plt.axvline(m_bound, color="red", linestyle="--", label=f"PAC bound m = {m_bound:.0f}")
plt.axvline(m_emp, color="green", linestyle=":", label=f"empirical m = {m_emp}")
plt.xlabel("m (training set size)")
plt.ylabel("fraction of trials with R > eps")
plt.title("Rectangle learner: P(R > 0.05) vs m")
plt.legend()
plt.savefig("figs/p3b_prob_bad.png", dpi=150)
plt.close()

# (d)
slope, intercept = np.polyfit(np.log(ms), np.log(mean_err), 1)
print(f"log-log fit slope = {slope:.3f}  ->  R ~ m^(-{-slope:.3f})")


# ---------------- P4 ----------------
print("\n===== P4 =====")


def make_data(n):
    x = np.random.uniform(-1, 1, n)
    y = np.sign(np.sin(3 * np.pi * x))
    y[y == 0] = 1
    flip = np.random.rand(n) < 0.1
    y[flip] = -y[flip]
    return x, y.astype(int)


def features(x, d):
    return legvander(x, d)[:, 1:]


x_tr, y_tr = make_data(200)
x_te, y_te = make_data(10**4)

ds = list(range(1, 16))
train_err = []
bound = []
test_err = []
cv_err = []
for d in ds:
    clf = LinearSVC(C=1e4, max_iter=200000).fit(features(x_tr, d), y_tr)
    tr = np.mean(clf.predict(features(x_tr, d)) != y_tr)
    te = np.mean(clf.predict(features(x_te, d)) != y_te)
    cv = 1 - cross_val_score(LinearSVC(C=1e4, max_iter=200000), features(x_tr, d), y_tr, cv=5).mean()
    train_err.append(tr)
    test_err.append(te)
    bound.append(tr + f_m(d + 1, 200))
    cv_err.append(cv)
    print(f"d = {d:2d}: train = {tr:.3f}, bound = {bound[-1]:.3f}, cv = {cv:.3f}, test = {te:.3f}")

print("SRM picks d =", ds[np.argmin(bound)])
print("5-fold CV picks d =", ds[np.argmin(cv_err)])
print("best test error at d =", ds[np.argmin(test_err)])
print("min bound value =", min(bound))

plt.figure()
plt.plot(ds, train_err, "o-", label="training error")
plt.plot(ds, bound, "s-", label="VC bound")
plt.plot(ds, test_err, "^-", label="test error")
plt.plot(ds, cv_err, "x--", label="5-fold CV error")
plt.axhline(0.5, color="gray", linestyle=":", label="random guessing")
plt.xlabel("polynomial degree d")
plt.ylabel("error")
plt.title("SRM vs cross-validation for sign(p(x))")
plt.legend()
plt.savefig("figs/p4c_srm_vs_cv.png", dpi=150)
plt.close()

# (e)
knn = KNeighborsClassifier(n_neighbors=1).fit(x_tr.reshape(-1, 1), y_tr)
print("1-NN train error:", np.mean(knn.predict(x_tr.reshape(-1, 1)) != y_tr))
print("1-NN test error:", np.mean(knn.predict(x_te.reshape(-1, 1)) != y_te))


# ---------------- Bonus ----------------
print("\n===== Bonus =====")
# B1
delta = 0.02
print("h threshold:", 1 + 32 * np.log(1 / delta))
h = 127
eps = 0.01
m_nec = (h - 1) / (32 * eps)
print("necessary m ~", m_nec)
m_suf = brentq(lambda m: f_m(h, m, eta=0.02) - 0.01, h, 1e12)
print(f"sufficient m from f_m(127) = 0.01: {m_suf:.0f}")
print(f"ratio sufficient / necessary: {m_suf / m_nec:.0f}")

# B2
print("\nnotebook classifier:")
for m in [10, 50, 100, 500, 1000]:
    h = m
    risks = []
    emps = []
    for t in range(200):
        x = np.random.rand(m)
        y = np.random.choice([-1, 1], m)
        memory = dict(zip(x, y))

        def predict(xs):
            return np.array([memory.get(v, 1) for v in xs])

        emps.append(np.mean(predict(x) != y))
        x_test = np.random.rand(5000)
        y_test = np.random.choice([-1, 1], 5000)
        risks.append(np.mean(predict(x_test) != y_test))
    risks = np.array(risks)
    emps = np.array(emps)
    b = emps + f_m(h, m)
    print(f"m = h = {m:4d}: Remp = {emps.mean():.3f}, R = {risks.mean():.3f}, "
          f"bound = {b.mean():.3f}, violations = {np.sum(risks > b)}")
