from functools import partial

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from haiku.initializers import RandomUniform
from scipy import linalg as la
from scipy import signal
from scipy import special as ss


def _batched(batch_size, shape):
    return [batch_size, *shape] if batch_size else shape


def transition(measure, N, **measure_args):
    """A, B transition matrices for different measures
    measure: the type of measure
      legt - Legendre (translated)
      legs - Legendre (scaled)
      glagt - generalized Laguerre (translated)
      lagt, tlagt - previous versions of (tilted) Laguerre with slightly different normalization
    """
    # Laguerre (translated)
    if measure == "lagt":
        b = measure_args.get("beta", 1.0)
        A = np.eye(N) / 2 - np.tril(np.ones((N, N)))
        B = b * np.ones((N, 1))
    if measure == "tlagt":
        # beta = 1 corresponds to no tilt
        b = measure_args.get("beta", 1.0)
        A = (1.0 - b) / 2 * np.eye(N) - np.tril(np.ones((N, N)))
        B = b * np.ones((N, 1))
    # Generalized Laguerre
    # alpha 0, beta small is most stable (limits to the 'lagt' measure)
    # alpha 0, beta 1 has transition matrix A = [lower triangular 1]
    if measure == "glagt":
        alpha = measure_args.get("alpha", 0.0)
        beta = measure_args.get("beta", 0.01)
        A = -np.eye(N) * (1 + beta) / 2 - np.tril(np.ones((N, N)), -1)
        B = ss.binom(alpha + np.arange(N), np.arange(N))[:, None]

        L = np.exp(
            0.5 * (ss.gammaln(np.arange(N) + alpha + 1) - ss.gammaln(np.arange(N) + 1))
        )
        A = (1.0 / L[:, None]) * A * L[None, :]
        B = (
            (1.0 / L[:, None])
            * B
            * np.exp(-0.5 * ss.gammaln(1 - alpha))
            * beta ** ((1 - alpha) / 2)
        )
    # Legendre (translated)
    elif measure == "legt":
        Q = np.arange(N, dtype=np.float64)
        R = (2 * Q + 1) ** 0.5
        j, i = np.meshgrid(Q, Q)
        A = R[:, None] * np.where(i < j, (-1.0) ** (i - j), 1) * R[None, :]
        B = R[:, None]
        A = -A
    # LMU: equivalent to LegT up to normalization
    elif measure == "lmu":
        Q = np.arange(N, dtype=np.float64)
        R = (2 * Q + 1)[:, None]  # / theta
        j, i = np.meshgrid(Q, Q)
        A = np.where(i < j, -1, (-1.0) ** (i - j + 1)) * R
        B = (-1.0) ** Q[:, None] * R
    # Legendre (scaled)
    elif measure == "legs":
        q = np.arange(N, dtype=np.float64)
        col, row = np.meshgrid(q, q)
        r = 2 * q + 1
        M = -(np.where(row >= col, r, 0) - np.diag(q))
        T = np.sqrt(np.diag(2 * q + 1))
        A = T @ M @ np.linalg.inv(T)
        B = np.diag(T)[:, None]

    return A, B


class LegT(hk.RNNCore):
    def __init__(self, units, theta, order, measure="legt", name=None):
        super().__init__(name=name)
        self.order = order
        self.units = units
        A, B = transition(measure, order)
        # Construct A and B matrices
        C = np.ones((1, order))
        D = np.zeros((1,))
        dA, dB, _, _, _ = signal.cont2discrete((A, B, C, D), dt=1.0 / theta)

        self.A = dA - np.eye(order)  # puts into form: x += Ax
        self.B = dB

    def initial_state(self, batch_size=None):
        shapes = [[self.units], [self.order]]
        h, m = map(jnp.zeros, map(partial(_batched, batch_size), shapes))
        return h, m

    def __call__(self, x, state):
        h0, m = state
        u = sum(
            hk.Linear(
                1, w_init=jnp.zeros if v is m else RandomUniform(), with_bias=False
            )(v)
            for v in (x, h0, m)
        )
        m = m @ self.A.T + u @ self.B.T
        h = jax.nn.tanh(
            sum(hk.Linear(self.units, with_bias=False)(v) for v in (x, h0, m))
        )
        return h, (h, m)


class LegS(hk.RNNCore):
    def __init__(
        self,
        units,
        max_length,
        order,
        gate=None,
        measure="legs",
        name=None,
    ):
        super().__init__(name=name)
        self.units = units
        self.max_length = max_length
        self.gate = gate
        self.order = order
        A, B = transition(measure, order)
        # Construct A and B matrices

        A_stacked = np.empty((max_length, order, order), dtype=A.dtype)
        B_stacked = np.empty((max_length, order), dtype=B.dtype)
        B = B[:, 0]
        for t in range(1, max_length + 1):
            A_stacked[t - 1] = la.expm(A * (np.log(t + 1) - np.log(t)))
            B_stacked[t - 1] = la.solve_triangular(
                A, A_stacked[t - 1] @ B - B, lower=True
            )
        B_stacked = B_stacked[:, :, None]

        A_stacked -= np.eye(order)  # puts into form: x += Ax
        self.A = jnp.array(A_stacked - np.eye(order))  # puts into form: x += Ax
        self.B = jnp.array(B_stacked)

    def initial_state(self, batch_size=None):
        shapes = [[self.units], [self.order]]
        h, m = map(jnp.zeros, map(partial(_batched, batch_size), shapes))
        return h, m, jnp.array(0, dtype=int)

    def __call__(self, x, state):
        h0, m, t = state
        t = jnp.clip(t, 0, self.max_length)

        u = sum(
            hk.Linear(
                1, w_init=jnp.zeros if v is m else RandomUniform(), with_bias=False
            )(v)
            for v in (x, h0, m)
        )
        m = m @ self.A[t].T + u @ self.B[t].T
        h = jax.nn.tanh(
            sum(hk.Linear(self.units, with_bias=False)(v) for v in (x, h0, m))
        )
        if self.gate:
            g = jax.nn.sigmoid(
                hk.Linear(self.units, with_bias=False)(
                    jnp.concatenate([h0, m], axis=-1)
                )
            )
            h = (1.0 - g) * h0 + g * h
        return h, (h, m, t + 1)
