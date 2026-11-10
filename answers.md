# HW4 Part I answers

## P1

**(a)** see figs/p1a_vc_confidence.png. All three curves look almost the same when plotted against h/m. They go up fast at first and then flatten out, ending around 1.30 at h/m = 1.

**(b)** I got h/m = 0.3708 (m = 1000), 0.3731 (m = 10^4), and 0.3733 (m = 10^5). These match the ~0.37 from the notes. With r = h/m,

f_m(h)^2 = r(ln(2/r) + 1) - ln(η/4)/m

The first term only depends on r. The only part that depends on m is -ln(η/4)/m = ln(80)/m ≈ 4.38/m, which is tiny for big m. So the root is basically wherever r(ln(2/r)+1) = 1, no matter what m is. The small shift for m = 1000 comes from that 4.38/m term.

**(c)** With m = 10^4 and Remp = 0, the largest h is 11 (bound = 0.099). With Remp = 0.05 the largest h is only 2. So even with 10,000 examples the bound can only promise 10% error for a really simple model (VC dim around 11), and if there's any training error the budget is almost all used up. The bound is very restrictive/pessimistic. In practice people get good test error with way more complex models and fewer examples.

## P2

**(a)** (0,0),(1,0),(0,1) are shattered. (0,0),(1,0),(2,0) are not, since the labelings (-1,+1,-1) and (+1,-1,+1) can't be done (the middle point can't be on a different side from both ends). This doesn't contradict VC dim = 3 because VC dimension only needs *some* set of 3 points to be shattered, not every set. The non-collinear set works, so VC ≥ 3.

**(b)**

| n | n+1 points | n+2 points |
|---|---|---|
| 2 | 1.000 | 0.000 |
| 3 | 1.000 | 0.000 |
| 4 | 1.000 | 0.000 |
| 5 | 1.000 | 0.000 |

Random Gaussian points are in general position with probability 1, so every set of n+1 points gets shattered, and no set of n+2 points does. That matches VC dim = n+1 exactly: n+1 is the most points hyperplanes in R^n can shatter.

**(c)** My points (seed 4) were (0.05, 0.50), (-1.00, 0.69), (-0.42, -1.58), (-0.65, 0.60). The non-separable labelings were (-1,-1,-1,+1) and (+1,+1,+1,-1). Point 3 is inside the triangle made by the other three, so this is the "interior point" case. The predicted bad labelings are interior = +, others = - and its negation, and those are exactly what the checker found. Plot: figs/p2c_nonseparable.png.

**(d)** The four points (±1,0),(0,±1) are shattered by axis-aligned rectangles. 0 out of 1000 random 5-point sets were shattered, which fits with VC dim = 4. (For any 5 points, take the leftmost, rightmost, top, and bottom points as positive. Their bounding box has to contain the 5th point, so labeling that one negative can't be done.)

## P3

**(a)** ĥ is the tightest box around the positive training points. All the positives are inside it, so they're labeled +1. Every negative training point is outside c*. ĥ is inside c* because c* is a rectangle that contains all the positives, and ĥ is the *smallest* such rectangle. So negatives fall outside ĥ too, which means Remp = 0. Since ĥ ⊆ c*, the only mistakes are points in c* but not in ĥ (labeled + but predicted -). That region has probability area(c*) - area(ĥ) = 0.15 - area(ĥ) under the uniform distribution.

**(b)** see figs/p3b_mean_error.png and figs/p3b_prob_bad.png.

| m | mean R | P(R > 0.05) |
|---|---|---|
| 10 | 0.1376 | 0.994 |
| 20 | 0.1160 | 0.965 |
| 50 | 0.0680 | 0.688 |
| 100 | 0.0371 | 0.214 |
| 200 | 0.0193 | 0.010 |
| 500 | 0.0078 | 0.000 |
| 1000 | 0.0040 | 0.000 |

**(c)** The bound gives m ≥ (4/0.05) ln(80) ≈ 350.6, so 351. Empirically m = 150 is already enough (P(R > ε) = 0.0375 ≤ 0.05). So the bound is about 2.3x pessimistic. That's honestly not that bad compared to the VC bound.

**(d)** The log-log slope is -0.80, so R ~ m^-0.8. That's closer to 1/m than to 1/√m. For small m, the R = 0.15 cap (no positives found) flattens the curve, which pulls the slope down a bit. If you only fit the large m points it gets closer to -1. Bound (2) gives ε ~ (ln|H| + ln 1/δ)/m, which is the 1/m rate for a consistent learner. Bound (1) has a square root, so it gives 1/√m (well, √(ln m / m)).

## P4

**(a)** A nonzero polynomial of degree ≤ d has at most d real roots, so sign(p(x)) can switch sign at most d times. If you sort d+2 points and alternate the labels +,-,+,-,..., you need d+1 sign changes, which is impossible. So no set of d+2 points can be shattered. For d+1 sorted points x_1 < ... < x_{d+1} and any labeling, put a root between x_i and x_{i+1} wherever the label changes (at most d roots). Then multiply by ±1 to get the first point's sign right. That's a polynomial of degree ≤ d that realizes the labeling. (If all labels are the same, use a constant.) So VC(H_d) = d+1.

**(b)/(c)** see figs/p4c_srm_vs_cv.png and the table in output.txt.

- SRM (min of the VC bound) picks **d = 5**
- 5-fold CV picks **d = 15** (d = 10 to 15 are all within about 0.005 of each other, so it's basically a tie)
- best test error is at **d = 9** (0.135, d = 10 is tied)

SRM picks a smaller degree because the f_m(d+1) penalty grows quickly with d when m = 200, so it gets more conservative than it should be. The target sin(3πx) changes sign 6 times on [-1,1], so you need around degree 6+ to fit it, and SRM stops right at the first big drop in training error. CV estimates the test error directly, so it goes higher. The CV curve is pretty flat from d = 9 on, so its pick is kind of noisy. CV and test error agree much better than the bound does.

**(d)** The bound is never informative. The smallest value is 0.62 (at d = 5), and every d is above 0.5, which is worse than random guessing. So you can't use (1) to actually guarantee anything here with m = 200. It's only useful qualitatively: it shows the tradeoff between training error and capacity, and it gives a shape to minimize for picking a model. The actual numbers don't mean much.

**(e)** 1-NN gets 0 training error and 0.200 test error. 1-NN has infinite VC dimension (it can memorize any labeling), so the VC bound is useless or undefined for it. It still generalizes OK, though, because the true function is smooth in 1D and neighbors usually share labels. The error is about 2 × 0.1 noise rate, which matches the classic 1-NN result that it's at most about twice the Bayes error. VC theory is a worst-case bound over all distributions for the whole class. It doesn't say a high-capacity method *has* to do badly, just that it can't guarantee it does well.

## Bonus

**B1.** With δ = 0.02, the threshold is h > 1 + 32 ln(50) ≈ 126.2, so h = 127 just passes. Necessary m ≈ 126/(32 × 0.01) ≈ 394. Sufficient m from solving f_m(127) = 0.01 is about 1.72 × 10^7. That's a gap of around 44,000x. The upper bound has the 1/√m rate, so getting ε = 0.01 needs roughly 1/ε² times as much, plus loose constants. The lower bound goes like 1/ε. So the real answer is somewhere in a huge range.

**B2.** With the notebook classifier at m = h, Remp = 0 every time, and the true risk is 0.50 (it's just guessing on new points). The bound Remp + f_m(m) is about 1.30-1.46, which is always above 1. So it's never violated (0 violations in all runs), but it's also completely vacuous. This is the example where training error tells you nothing, and the bound correctly says so.
