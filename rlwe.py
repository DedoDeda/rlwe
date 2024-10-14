import numpy as np
import numpy.random as npr

# Universe
INT_TYPE = np.int64
INT_SIZE = np.dtype(INT_TYPE).itemsize
Q = 2 ** 32 - 1
N = 2 ** 10
# x^n + 1
IDEAL = np.array([1] + [0] * (N - 1) + [1])

# Base Parameters
MIN_BASE = -Q + 1
MAX_BASE = Q - 1

# Private Key Parameters
PRIVATE_KEY_GAUSSIAN_MU = 0.0
PRIVATE_KEY_GAUSSIAN_SIGMA = 1.0

# Public Key Error Parameters
PUBLIC_KEY_ERROR_GAUSSIAN_MU = 0.0
PUBLIC_KEY_ERROR_GAUSSIAN_SIGMA = 1.0

# Shared Key Parameters
SHARED_KEY_SELECT_INTERVAL_0 = np.array([0.0, 0.25]) * Q
SHARED_KEY_SELECT_INTERVAL_1 = np.array([0.5, 0.75]) * Q
SHARED_KEY_BIT_INTERVAL_0 = np.array([0.125, 0.625]) * Q
SHARED_KEY_BIT_INTERVAL_1 = np.array([0.375, 0.875]) * Q


def mod_poly(p):
    return (np.polydiv(p % Q, IDEAL)[1] % Q).astype(np.int64)


def add_poly(a, b):
    return np.polyadd(a, b) % Q


def mul_poly(a, b):
    return mod_poly(np.polymul(a, b))


def discrete_gaussian_poly(mu, sigma):
    return mod_poly(np.rint(npr.normal(mu, sigma, N)))


def gen_base():
    """
    Generates a base, over which the key pairs will be generated.
    """
    return npr.randint(MIN_BASE, MAX_BASE + 1, N)


def gen_private_key():
    """
    Generates a private key.
    """
    return discrete_gaussian_poly(PRIVATE_KEY_GAUSSIAN_MU, PRIVATE_KEY_GAUSSIAN_SIGMA)


def gen_public_key(base, private_key):
    """
    Generates a public key, given a base to generate over and a private key.
    """

    def gen_error():
        return discrete_gaussian_poly(PUBLIC_KEY_ERROR_GAUSSIAN_MU, PUBLIC_KEY_ERROR_GAUSSIAN_SIGMA)

    return add_poly(mul_poly(base, private_key), gen_error())


def gen_key_pair(base):
    """
    Generates a key pair, given a base to generate over.
    """
    private_key = gen_private_key()
    public_key = gen_public_key(base, private_key)
    return private_key, public_key


def compute_shared_key(other_public_key, private_key, ref_public_key):
    """
    Computes a shared key, given another's public key, a private key, and a reference public key.
    """

    def reconcile(shared_key):
        select_mask_0 = (
            (SHARED_KEY_SELECT_INTERVAL_0[0] <= ref_public_key) &
            (SHARED_KEY_SELECT_INTERVAL_0[1] >= ref_public_key))
        select_mask_1 = (
            (SHARED_KEY_SELECT_INTERVAL_1[0] <= ref_public_key) &
            (SHARED_KEY_SELECT_INTERVAL_1[1] >= ref_public_key))
        select_mask = select_mask_0 | select_mask_1
        bit_mask_0 = (
            (SHARED_KEY_BIT_INTERVAL_0[0] <= shared_key) &
            (SHARED_KEY_BIT_INTERVAL_0[1] >= shared_key))
        bit_mask_1 = (
            (SHARED_KEY_BIT_INTERVAL_1[0] <= shared_key) &
            (SHARED_KEY_BIT_INTERVAL_1[1] >= shared_key))
        return np.where(
            select_mask,
            bit_mask_0,
            bit_mask_1).astype(np.int64)

    return reconcile(mul_poly(other_public_key, private_key))


def example():
    # Generate a base on either side (up to the protocol).
    base = gen_base()

    # Generate a key pair on each side.
    alice_private_key, alice_public_key = gen_key_pair(base)
    bob_private_key, bob_public_key = gen_key_pair(base)

    # Decide on a reference public key (up to the protocol).
    ref_public_key = alice_public_key
    # Compute the shared key on each side.
    alice_shared_key = compute_shared_key(bob_public_key, alice_private_key, ref_public_key)
    bob_shared_key = compute_shared_key(alice_public_key, bob_private_key, ref_public_key)

    # Verify that the shared key matches on both sides (HIGHLY probable).
    if (alice_shared_key == bob_shared_key).all():
        print(f'Alice and Bob exchanged keys successfully. Shared key is {alice_shared_key}.')
    else:
        print('Alice and bob failed to exchange keys. '
              'There was a mismatch between their shared keys.')


if __name__ == '__main__':
    example()
