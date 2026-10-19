"""
Test doubles shared by the engine suite.

`StubPredictor` has the interface of `engine.model.predict.Predictor` (Amendment 1):

    predict(snapshots, histories) -> {"win": (N,), "hazard": (N, 10),
                                      "plant15": (N,), "defuse15": (N,)}

Each head is driven by a rule: a callable `rule(snapshot, history)` returning a float
(a sequence of 10 floats for `hazard`). Attribution and value tests use this to give
W(s) a known, hand-checkable shape, e.g. "a decoy is worth exactly nothing":

    stub = StubPredictor(win=lambda s, h: 0.5 + 0.1 * (s.alive_ct - s.alive_t))
    out  = stub.predict([s0, s1], [[], [s0]])

A head without a rule returns NaN for every element (not 0: 0 is a legitimate
probability), so a test that forgets to set a rule it relies on fails loudly.
`calls` and `n_snapshots` count invocations, for "at most one Predictor call per
window" style checks. Extend this file rather than writing a second stub.

`encode(snapshots) -> (N, embed_dim)` mirrors `Predictor.encode` (the h_t snapshot
embedding used by P9 neighbours and the P7.2.7 OOD detector); it is driven by an
optional `embed` rule `embed(snapshot) -> sequence of embed_dim floats` and returns
NaN without one. `encode_calls` counts its invocations.
"""
from typing import Callable, Sequence

import numpy as np

Rule = Callable[[object, object], float]
HazardRule = Callable[[object, object], Sequence[float]]
EmbedRule = Callable[[object], Sequence[float]]

N_PLAYERS = 10


def const(value: float) -> Rule:
    """A rule that ignores the state and always returns `value`."""
    return lambda snapshot, history: value


class StubPredictor:
    """Predictor double whose four heads are pluggable rules."""

    def __init__(self, win: Rule, hazard: HazardRule | None = None,
                 plant15: Rule | None = None, defuse15: Rule | None = None,
                 embed: EmbedRule | None = None, embed_dim: int = 8):
        self.rules = {"win": win, "hazard": hazard, "plant15": plant15, "defuse15": defuse15}
        self.embed        = embed
        self.embed_dim    = int(embed_dim)
        self.calls        = 0
        self.n_snapshots  = 0
        self.encode_calls = 0

    def predict(self, snapshots: Sequence, histories: Sequence) -> dict[str, np.ndarray]:
        if len(snapshots) != len(histories):
            raise ValueError(f"{len(snapshots)} snapshots but {len(histories)} histories")
        self.calls       += 1
        self.n_snapshots += len(snapshots)
        pairs = list(zip(snapshots, histories))
        return {
            "win":      self._scalar("win", pairs),
            "hazard":   self._hazard(pairs),
            "plant15":  self._scalar("plant15", pairs),
            "defuse15": self._scalar("defuse15", pairs),
        }

    def _scalar(self, head: str, pairs: list) -> np.ndarray:
        rule = self.rules[head]
        if rule is None:
            return np.full(len(pairs), np.nan)
        return np.array([float(rule(s, h)) for s, h in pairs], dtype=np.float64)

    def _hazard(self, pairs: list) -> np.ndarray:
        rule = self.rules["hazard"]
        if rule is None:
            return np.full((len(pairs), N_PLAYERS), np.nan)
        if not pairs:
            return np.empty((0, N_PLAYERS))
        out = np.array([list(rule(s, h)) for s, h in pairs], dtype=np.float64)
        if out.shape != (len(pairs), N_PLAYERS):
            raise ValueError(f"hazard rule must return {N_PLAYERS} values per snapshot, got shape {out.shape}")
        return out

    def encode(self, snapshots: Sequence) -> np.ndarray:
        """(N, embed_dim) embeddings from the `embed` rule; NaN without one."""
        self.encode_calls += 1
        if self.embed is None:
            return np.full((len(snapshots), self.embed_dim), np.nan)
        if not len(snapshots):
            return np.empty((0, self.embed_dim))
        out = np.array([list(self.embed(s)) for s in snapshots], dtype=np.float64)
        if out.shape != (len(snapshots), self.embed_dim):
            raise ValueError(f"embed rule must return {self.embed_dim} values per snapshot, got shape {out.shape}")
        return out
