"""Exact finite-state checks for the KL article. Python 3, no dependencies.

No Monte Carlo noise, no model weights, no files written.
Derivatives are with respect to probability p, not its logit.
"""
from math import isclose, log


def close(a, b, tol=1e-7):
    assert isclose(a, b, rel_tol=tol, abs_tol=tol), (a, b)


def kl(P, Q):
    return sum(p * log(p / q) for p, q in zip(P, Q))


def diff(fn, x, eps=1e-6):
    return (fn(x + eps) - fn(x - eps)) / (2 * eps)


def check_binary(p, q):
    P, Q = [p, 1-p], [q, 1-q]
    u = [log(a/b) for a, b in zip(P, Q)]
    w = [b/a for a, b in zip(P, Q)]
    g = [1/p, -1/(1-p)]
    avg = lambda values: sum(a*b for a, b in zip(P, values))
    reverse = u[0] - u[1]
    forward = -q/p + (1-q)/(1-p)
    close(avg(u), kl(P, Q))
    close(avg([ui+wi-1 for ui, wi in zip(u, w)]), kl(P, Q))
    close(avg(g), 0)
    close(avg([ui*gi for ui, gi in zip(u, g)]), reverse)
    close(avg([(1-wi)*gi for wi, gi in zip(w, g)]), forward)
    close(diff(lambda x: kl([x, 1-x], Q), p), reverse)
    close(diff(lambda x: kl(Q, [x, 1-x]), p), forward)
    # Exact head A plus a sampled tail B: the +1 is essential.
    head = u[0] + 1
    naive = head + P[1]*u[1]*g[1]
    corrected = head + P[1]*(u[1]+1)*g[1]
    close(naive, reverse+1)
    close(corrected, reverse)
    return reverse, forward


def check_sequence():
    # First-step probability and both conditional probabilities depend on p.
    # This tests both earlier-action effects and local conditional derivatives.
    Q0, Qnext = [.4, .6], [[.7, .3], [.2, .8]]

    def paths(p):
        P0 = [p, 1-p]
        Pnext = [[.2+.4*p, .8-.4*p], [.8-.3*p, .2+.3*p]]
        dP0, dPnext = [1, -1], [[.4, -.4], [-.3, .3]]
        result = []
        for i in range(2):
            for j in range(2):
                mass = P0[i]*Pnext[i][j]
                u1, u2 = log(P0[i]/Q0[i]), log(Pnext[i][j]/Qnext[i][j])
                g1, g2 = dP0[i]/P0[i], dPnext[i][j]/Pnext[i][j]
                result.append((mass, u1, u2, g1, g2))
        return result

    for p in [.15, .4, .8]:
        rows = paths(p)
        full = sum(m*(u1+u2)*(g1+g2) for m,u1,u2,g1,g2 in rows)
        rtg = sum(m*((u1+u2)*g1+u2*g2) for m,u1,u2,g1,g2 in rows)
        local = sum(m*(u1*g1+u2*g2) for m,u1,u2,g1,g2 in rows)
        future = sum(m*u2*g1 for m,u1,u2,g1,g2 in rows)
        exact = diff(lambda x: sum(m*(u1+u2) for m,u1,u2,_,_ in paths(x)), p)
        close(full, exact)
        close(rtg, exact)
        close(full-local, future)
        assert abs(future) > .01
    print("Sequence KL: full derivative and reward-to-go match finite differences.")


def check_variance():
    P = [[.5,.5],[.5,.5],[.01,.99]]
    Q = [[.25,.75],[.5,.5],[.99,.01]]
    rows = []
    for i in range(2):
        for j in range(2):
            mass = P[0][i]*P[i+1][j]
            mc = log(P[0][i]/Q[0][i])+log(P[i+1][j]/Q[i+1][j])
            conditional = kl(P[0],Q[0])+kl(P[i+1],Q[i+1])
            rows.append((mass,mc,conditional))
    means = [sum(row[0]*row[k] for row in rows) for k in [1,2]]
    variances = [sum(row[0]*(row[k]-means[k-1])**2 for row in rows) for k in [1,2]]
    close(means[0],means[1])
    close(variances[0],3.315913584927973)
    close(variances[1],5.069741857547936)
    assert variances[1] > variances[0]
    print(f"Sequence variance: sampled={variances[0]:.8f}, conditional={variances[1]:.8f}")


if __name__ == "__main__":
    for p in [.05,.2,.5,.8,.95]:
        for q in [.1,.3,.5,.9]:
            check_binary(p,q)
    reverse, forward = check_binary(.8,.5)
    print(f"Binary gradients: reverse={reverse:.8f}, forward={forward:.8f}")
    check_sequence()
    check_variance()
    print("All KL checks passed (20 distributions, sequence gradients, Top-k, variance).")
