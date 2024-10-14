import numpy as np
import numpy.linalg as lin
import numpy.random as npr

Q = 2 ** 32 - 1
N = 2 ** 10
# x^n + 1
IDEAL = np.array([1] + [0] * (N - 1) + [1])

A_MIN = -Q + 1
A_MAX = Q - 1

S_MU = 0.0
S_SIGMA = 1.0

E_MU = 0.0
E_SIGMA = 1.0


def ring_mod(p):
    return (np.polydiv(p % Q, IDEAL)[1] % Q).astype(int)


def add(a, b):
    return np.polyadd(a, b) % Q


def mul(a, b):
    return ring_mod(np.polymul(a, b))


def discrete_gaussian(mu, sigma):
    return ring_mod(np.rint(npr.normal(mu, sigma, N)))


def gen_a():
    return npr.randint(A_MIN, A_MAX + 1, N)


def gen_s():
    return discrete_gaussian(S_MU, S_SIGMA)


def gen_e():
    return discrete_gaussian(E_MU, E_SIGMA)


def make_p(a, s, e):
    return add(mul(a, s), e)


def make_c(p, s):
    return mul(p, s)


def main():
    a = gen_a()

    s_0 = gen_s()
    e_0 = gen_e()
    p_0 = make_p(a, s_0, e_0)

    s_1 = gen_s()
    e_1 = gen_e()
    p_1 = make_p(a, s_1, e_1)

    c_0 = make_c(p_1, s_0)
    c_1 = make_c(p_0, s_1)

    print(f'c-distance = {lin.norm(c_0 - c_1)}')
    # NOTE: The dot product must be done in float because it can easily overflow int.
    print(f'c-similarity = {np.dot(c_0.astype(float), c_1.astype(float)) / (lin.norm(c_0) * lin.norm(c_1))}')


if __name__ == '__main__':
    main()
