"""Compatibility of the leaf strains with a mean dilation."""
def dilation_residual(design, alpha):
    """Return the compatibility and mean-dilation residuals for the stored amplitudes."""
    import mpmath as mp
    from .laminate import mp_frame_to_e3, mp_rot6
    D = design['D']
    s2 = mp.sqrt(2)

    def sym6mp(n, p):
        n = [mp.mpf(x) for x in n]
        p = [mp.mpf(x) for x in p]
        nn = mp.sqrt(sum((x * x for x in n)))
        pn = mp.sqrt(sum((x * x for x in p)))
        n = [x / nn for x in n]
        p = [x / pn for x in p]
        M = [[(n[i] * p[j] + p[i] * n[j]) / 2 for j in range(3)] for i in range(3)]
        return mp.matrix([M[0][0], M[1][1], M[2][2], s2 * M[1][2], s2 * M[0][2], s2 * M[0][1]])
    E = [alpha[i] * sym6mp(design['leaf_n'][i][0], design['leaf_p'][i][0]) for i in range(2 ** D)]
    res = []
    for l in reversed(range(D)):
        new = []
        for j in range(len(E) // 2):
            m = design['m'][l][j]
            f = mp.mpf(design['f'][l][j])
            Q = mp_rot6(mp_frame_to_e3(m))
            dlt = Q * (E[2 * j] - E[2 * j + 1])
            for c in (2, 3, 4):
                dlt[c] = 0
            dlt = Q.T * dlt
            res += list(dlt)
            new.append(f * E[2 * j] + (1 - f) * E[2 * j + 1])
        E = new
    I = mp.matrix([1, 1, 1, 0, 0, 0])
    res += list(E[0] - I)
    return res
