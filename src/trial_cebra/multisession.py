"""Trial-aware multisession sampler.

Implements CEBRA's cross-session alignment philosophy on top of
:py:class:`trial_cebra.distribution.TrialAwareDistribution`:

* Per-session anchor distribution (class-balanced or uniform as configured
  per session).
* Cross-session query shuffle — queries computed in each session's y-space
  are redistributed across sessions with a **strict derangement** (no query
  stays in its own session) before being used to search for positives in
  target sessions.  This is the alignment mechanism: ref and pos end up in
  different sessions, forcing shared encoders to map semantically equivalent
  states to nearby embeddings.
* Per-session positive search respecting trial-aware constraints (same-class,
  class-conditional trial_emb, etc.) in the **target** session.

Conditionals:

* ``"delta"`` — full support (Mode A / Mode B / no-discrete).  Mode C
  (``_DISC_MODE_PER_TP_2D``) cannot be reached via normal init because 2-D
  y is auto-broadcast to 3-D in ``TrialAwareDistribution.__init__``, but the
  sampler still rejects it defensively.
* ``"time_delta"`` — supported; the ±time_offsets window is dropped in
  multisession because relative time positions do not transfer meaningfully
  across sessions with heterogeneous ``ntime``.  Behaviour becomes joint
  argmin in y-space.
* ``"time"`` — **not supported** in multisession, matching CEBRA's native
  behaviour (``cebra.integrations.sklearn.cebra._init_loader`` rejects
  multisession without a behavioural index).

User-facing invariants (validated at init):

* ``num_sessions >= 2``
* all per-session distributions use the same ``conditional``
* all per-session y_continuous have the same feature dim ``nd``
* if any session has ``y_discrete``: all must have the same sorted ``unique``
  class set (same values, same order)
* no session is in Mode C (``_DISC_MODE_PER_TP_2D``) — defensive check;
  unreachable via normal init since 2-D y is auto-broadcast to 3-D
* all per-session distributions on the same device
"""

from __future__ import annotations

import dataclasses
from typing import List, Optional, Tuple

import cebra.data as cebra_data
import cebra.distributions.base as abc_
import numpy as np
import torch
from cebra.data.multi_session import ContinuousMultiSessionDataLoader

from trial_cebra.distribution import (
    _DISC_MODE_PER_TP_2D,
    TrialAwareDistribution,
)


def _invert_index(idx: np.ndarray) -> np.ndarray:
    """Invert a permutation. ``idx[_invert_index(idx)] == arange(len(idx))``."""
    out = np.zeros_like(idx)
    out[idx] = np.arange(len(idx))
    return out


def _random_derangement(n: int, rng: np.random.Generator) -> np.ndarray:
    """Return a random derangement of ``range(n)`` — no fixed points.

    Uses rejection sampling: fast for moderate ``n`` (probability of a
    random permutation being a derangement → 1/e ≈ 0.37).
    """
    if n < 2:
        raise ValueError(f"derangement requires n >= 2, got {n}")
    while True:
        perm = rng.permutation(n)
        if not np.any(perm == np.arange(n)):
            return perm


def _strict_cross_session_permutation(
    num_sessions: int, batch_size: int, rng: np.random.Generator
) -> np.ndarray:
    """Build a flat permutation over ``num_sessions * batch_size`` such that
    every batch element ends up in a session different from its origin.

    The permutation has the block structure ``idx[s' * B + b] = s * B + b``
    where ``s != s'`` and ``b`` is preserved (within-batch-position shuffling
    across the session axis).  This mirrors CEBRA's flat permutation format
    so downstream ``mix`` / ``_invert_index`` work unchanged.
    """
    if num_sessions < 2:
        raise ValueError("num_sessions must be >= 2 for cross-session shuffle")
    # For each batch position, derange the session indices.
    idx_sb = np.empty((num_sessions, batch_size), dtype=np.int64)
    for b in range(batch_size):
        idx_sb[:, b] = _random_derangement(num_sessions, rng)
    idx_flat = (idx_sb * batch_size + np.arange(batch_size)[None, :]).ravel()
    return idx_flat


class TrialAwareMultisessionSampler(abc_.PriorDistribution, abc_.ConditionalDistribution):
    """Cross-session trial-aware sampler (CEBRA philosophy + trial structure).

    See module docstring for the full picture.

    Args:
        per_session_dists: list of :py:class:`TrialAwareDistribution`, one
            per session, constructed independently (each with its own
            ``ntrial_s / ntime_s / y_s``).  Trial-aware constraints apply
            inside each session.
        conditional: ``"delta"`` or ``"time_delta"``.
        seed: integer seed for the numpy RNG used for cross-session shuffle.
    """

    def __init__(
        self,
        per_session_dists: List[TrialAwareDistribution],
        conditional: str,
        seed: Optional[int] = None,
    ):
        if conditional == "time":
            raise NotImplementedError(
                "conditional='time' is not supported in multisession. "
                "Use 'delta' or 'time_delta'. This matches CEBRA's native behaviour."
            )
        if conditional not in ("delta", "time_delta"):
            raise ValueError(
                f"Unknown conditional {conditional!r}; expected 'delta' or 'time_delta'."
            )
        num_sessions = len(per_session_dists)
        if num_sessions < 2:
            raise ValueError(f"multisession sampler requires >= 2 sessions, got {num_sessions}.")

        # All sessions use the same conditional
        for s, d in enumerate(per_session_dists):
            if d.conditional != conditional:
                raise ValueError(
                    f"session {s} has conditional={d.conditional!r}, expected {conditional!r}"
                )

        # All sessions have the same y feature dim (nd)
        def _nd(d: TrialAwareDistribution) -> int:
            if d.conditional == "time_delta":
                return d._y_flat.shape[-1]
            return d.trial_emb.shape[-1]

        nds = [_nd(d) for d in per_session_dists]
        if len(set(nds)) != 1:
            raise ValueError(f"all sessions must share the same y feature dim; got {nds}")
        self._nd = nds[0]

        # Mode C defensive check: unreachable via normal init since 2-D y is
        # auto-broadcast to 3-D in TrialAwareDistribution.__init__, but kept
        # as a safety net in case distributions are constructed manually.
        for s, d in enumerate(per_session_dists):
            if getattr(d, "_disc_mode", None) == _DISC_MODE_PER_TP_2D:
                raise ValueError(
                    f"session {s} is in Mode C (per-timepoint y_discrete + 2-D "
                    f"y_continuous) which is not allowed in multisession. "
                    f"Pass 3-D y_continuous for every session."
                )

        # If any session has y_discrete, all must have identical sorted class sets
        has_disc = [hasattr(d, "_y_discrete") for d in per_session_dists]
        if any(has_disc) and not all(has_disc):
            raise ValueError(
                "y_discrete presence inconsistent across sessions: "
                f"{has_disc}. Provide y_discrete for all sessions or none."
            )
        if all(has_disc):
            ref_classes = per_session_dists[0]._classes_sorted
            for s in range(1, num_sessions):
                if not torch.equal(per_session_dists[s]._classes_sorted, ref_classes):
                    raise ValueError(
                        f"session {s} has classes "
                        f"{per_session_dists[s]._classes_sorted.tolist()} but "
                        f"session 0 has {ref_classes.tolist()}. "
                        "All sessions must share the same sorted class set."
                    )

        # All sessions on the same device
        devices = [str(d.device) for d in per_session_dists]
        if len(set(devices)) != 1:
            raise ValueError(f"sessions span multiple devices: {devices}")
        self.device = per_session_dists[0].device

        self._dists = per_session_dists
        self._conditional = conditional
        self._num_sessions = num_sessions
        # session_lengths[s] = ntrial_s * ntime_s (total flat timepoints per session)
        self._session_lengths = np.array(
            [d.ntrial * d.ntime for d in per_session_dists], dtype=np.int64
        )
        self._np_rng = np.random.default_rng(seed)

    @property
    def num_sessions(self) -> int:
        return self._num_sessions

    @property
    def session_lengths(self) -> np.ndarray:
        return self._session_lengths

    @property
    def conditional(self) -> str:
        return self._conditional

    # ------------------------------------------------------------------
    # Prior / conditional
    # ------------------------------------------------------------------

    def sample_prior(self, num_samples: int) -> np.ndarray:
        """Return per-session reference indices.

        Each session uses its own :py:meth:`TrialAwareDistribution.sample_prior`
        (respects ``sample_prior`` mode: ``balanced`` or ``uniform``).

        Returns:
            ``(num_sessions, num_samples)`` numpy int array — flat indices
            into each session's own ``[0, session_length_s)``.
        """
        out = np.empty((self._num_sessions, num_samples), dtype=np.int64)
        for s, d in enumerate(self._dists):
            idx_s = d.sample_prior(num_samples)
            out[s] = idx_s.cpu().numpy()
        return out

    def sample_conditional(self, idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Cross-session conditional sampling.

        Algorithm (for ``delta`` / ``time_delta``):

        1. Per-session: ``compute_query(ref_s)`` produces ``(query_s, anchor_class_s)``.
        2. Stack into ``(num_sessions, batch_size, nd)`` and ``(num_sessions, batch_size)``.
        3. Build a strict cross-session derangement ``idx_flat`` (each element
           assigned to a session different from its origin).
        4. Shuffle queries and anchor_class along the flat axis.
        5. Per target session: :py:meth:`search_given_query` returns positives.
        6. Return ``(pos_idx, idx_flat, idx_rev_flat)``.

        Args:
            idx: ``(num_sessions, batch_size)`` reference indices.

        Returns:
            ``(pos_idx, idx_flat, idx_rev_flat)``:

            * pos_idx: ``(num_sessions, batch_size)`` positive flat indices
              (each row indexes into that session's ``[0, session_length_s)``).
            * idx_flat: ``(num_sessions * batch_size,)`` permutation applied
              to queries before per-session search.  Compatible with
              :py:func:`cebra.distributions.multisession._invert_index`.
            * idx_rev_flat: inverse permutation, to align positives back to
              references after encoder forward pass.
        """
        if isinstance(idx, torch.Tensor):
            idx = idx.cpu().numpy()
        if idx.ndim != 2 or idx.shape[0] != self._num_sessions:
            raise ValueError(
                f"idx must have shape (num_sessions={self._num_sessions}, batch_size); "
                f"got {idx.shape}"
            )
        batch_size = idx.shape[1]

        # Step 1+2: compute queries per session
        queries = torch.empty(self._num_sessions, batch_size, self._nd, device=self.device)
        has_classes = hasattr(self._dists[0], "_y_discrete")
        anchor_classes = (
            torch.empty(self._num_sessions, batch_size, dtype=torch.long, device=self.device)
            if has_classes
            else None
        )
        for s in range(self._num_sessions):
            ref_s = torch.from_numpy(idx[s]).to(self.device)
            q_s, ac_s = self._dists[s].compute_query(ref_s)
            queries[s] = q_s
            if has_classes:
                anchor_classes[s] = ac_s

        # Step 3: strict cross-session derangement
        idx_flat = _strict_cross_session_permutation(self._num_sessions, batch_size, self._np_rng)

        # Step 4: shuffle queries and anchor_class along flat axis
        queries_flat = queries.reshape(self._num_sessions * batch_size, self._nd)
        shuffled = queries_flat[idx_flat].reshape(self._num_sessions, batch_size, self._nd)
        if has_classes:
            ac_flat = anchor_classes.reshape(self._num_sessions * batch_size)
            ac_shuffled = ac_flat[idx_flat].reshape(self._num_sessions, batch_size)
        else:
            ac_shuffled = [None] * self._num_sessions

        # Step 5: per target session search
        pos_idx = np.empty((self._num_sessions, batch_size), dtype=np.int64)
        for sp in range(self._num_sessions):
            ac_sp = ac_shuffled[sp] if has_classes else None
            pos_flat = self._dists[sp].search_given_query(shuffled[sp], anchor_class=ac_sp)
            pos_idx[sp] = pos_flat.cpu().numpy()

        idx_rev_flat = _invert_index(idx_flat)
        return pos_idx, idx_flat, idx_rev_flat

    def mix(self, array: np.ndarray, idx: np.ndarray) -> np.ndarray:
        """Re-order array elements according to index mapping.

        Mirrors :py:meth:`cebra.distributions.multisession.MultisessionSampler.mix`.
        ``array`` is assumed to have shape ``(num_sessions, batch_size, ...)``.
        """
        n, m = array.shape[:2]
        return array.reshape(n * m, -1)[idx].reshape(array.shape)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class TrialAwareMultisessionLoader(ContinuousMultiSessionDataLoader):
    """Multi-session loader that defers sampler construction.

    Subclasses :py:class:`cebra.data.multi_session.ContinuousMultiSessionDataLoader`
    and overrides ``__post_init__`` to skip the vanilla
    :py:class:`cebra.distributions.multisession.MultisessionSampler` instantiation.
    The trial-aware sampler is set externally after construction:

    .. code-block:: python

        loader = TrialAwareMultisessionLoader(dataset=ds, batch_size=B,
                                              num_steps=N, time_offset=T)
        loader.sampler = TrialAwareMultisessionSampler(per_session_dists, ...)

    ``get_indices`` is inherited from
    :py:class:`cebra.data.multi_session.MultiSessionLoader` and works unchanged
    (calls ``self.sampler.sample_prior / sample_conditional`` and packages into
    a :py:class:`cebra.data.datatypes.BatchIndex`).
    """

    def __post_init__(self):
        # Skip MultiSessionLoader.__post_init__ which constructs the wrong
        # (vanilla y-only) sampler; call grandparent's __post_init__ for
        # basic Loader setup, then leave self.sampler unset (assigned externally).
        cebra_data.Loader.__post_init__(self)
        self.sampler = None  # caller must set this
