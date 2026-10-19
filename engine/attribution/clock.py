"""
Team clock buckets (P8.4.2): the drift remainder never assigned to a player;
clock_ct and clock_t both non-negative.

    from engine.attribution.clock import ClockBuckets
    c = ClockBuckets()
    c.add(+0.03, "drift")          # remainder ΔW − Σ φ_p of one interval, CT-positive
    c.add(-0.01, "window_gap")
    c.clock_ct, c.clock_t          # 0.03, 0.01  ("value the clock gave this team")
    c.net                          # Σ remainders = clock_ct − clock_t

Everything the engine cannot assign to a player lands here, each with a source tag
kept in `by_source`: drift remainders (time passing while nobody's movement explains
the change), the 1-2 tick clock gap of an event window (`v(∅)` keeps `s_post`'s clock,
so `W(apply(∅)) − W(s_pre)` is time, not any member), `info` enabler shares (P8.5.1),
and the whole ΔW of a window whose interventions failed (flagged).

Filled by: P8.4.2.
"""
import math
from dataclasses import dataclass, field

# ── buckets ───────────────────────────────────────────────────────────────────

@dataclass
class ClockBuckets:
    clock_ct:   float = 0.0
    clock_t:    float = 0.0
    remainders: list[float] = field(default_factory=list)
    by_source:  dict[str, float] = field(default_factory=dict)

    def add(self, remainder: float, source: str = "drift") -> None:
        """Book one signed CT-positive remainder: positive to `clock_ct`, the magnitude
        of a negative one to `clock_t`."""
        r = float(remainder)
        if math.isnan(r):
            raise ValueError(f"clock remainder from {source} is NaN")
        if r > 0:
            self.clock_ct += r
        elif r < 0:
            self.clock_t += -r
        self.remainders.append(r)
        self.by_source[source] = self.by_source.get(source, 0.0) + r

    @property
    def net(self) -> float:
        return self.clock_ct - self.clock_t


# ── identities ────────────────────────────────────────────────────────────────

def interval_remainder(phis, dW: float) -> float:
    """The clock share of one interval: `ΔW − Σ φ_p` (CT-positive, signed)."""
    return float(dW) - math.fsum(phis)


def interval_identity(phis, clock: float, dW: float, tol: float = 1e-9) -> bool:
    """Σ φ_p + clock == ΔW within `tol` (P8.4.2 per-interval identity)."""
    return abs(math.fsum(list(phis) + [clock]) - dW) <= tol


def round_identity(buckets: ClockBuckets, tol: float = 1e-9) -> bool:
    """clock_ct − clock_t == Σ remainders within `tol` (P8.4.2 per-round identity)."""
    return (buckets.clock_ct >= 0 and buckets.clock_t >= 0
            and abs(buckets.net - math.fsum(buckets.remainders)) <= tol)
