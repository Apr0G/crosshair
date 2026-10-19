"""
Callouts over the nav mesh, for eyeballing (P2.3.3). No radar image is cached for the
pinned awpy-data release, so the backdrop is the nav mesh itself: one dot per nav-area
centroid, coloured by the callout `lookup` gives it (fallback `area_<id>` dots are
grey), with the callout polygons outlined and labelled.

    .venv\\Scripts\\python.exe -m engine.maps.plot de_mirage [out.png]

Default output: `data/engine/eval/callouts_<map>.png`.
"""
import sys
from pathlib import Path

from engine import paths
from engine.maps import assets as map_assets_mod
from engine.maps import callouts as callouts_mod

FALLBACK_COLOR = "#b0b0b0"


def plot(map_name: str, out: Path | None = None, *, map_assets=None, callouts=None) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    a  = map_assets if map_assets is not None else map_assets_mod.ensure(map_name)
    cs = callouts if callouts is not None else callouts_mod.load(map_name)
    areas = a.nav.areas
    xs, ys, zs = (areas[c].to_list() for c in ("centroid_x", "centroid_y", "centroid_z"))
    labels = [callouts_mod.lookup(a, cs, x, y, z) for x, y, z in zip(xs, ys, zs)]

    names  = sorted({lab for lab in labels if lab and not callouts_mod.is_fallback(lab)})
    cmap   = plt.get_cmap("tab20", max(len(names), 1))
    colour = {n: cmap(i % 20) for i, n in enumerate(names)}

    fig, ax = plt.subplots(figsize=(10, 10), dpi=120)
    fb = [i for i, lab in enumerate(labels) if not lab or callouts_mod.is_fallback(lab)]
    ax.scatter([xs[i] for i in fb], [ys[i] for i in fb], s=4, c=FALLBACK_COLOR,
               label=f"fallback area ids ({len(fb)})", linewidths=0)
    for n in names:
        idx = [i for i, lab in enumerate(labels) if lab == n]
        ax.scatter([xs[i] for i in idx], [ys[i] for i in idx], s=6, color=colour[n], linewidths=0)
    for c in cs:
        for poly in c.polygons:
            px, py = [p[0] for p in poly], [p[1] for p in poly]
            ax.plot(px + px[:1], py + py[:1], color=colour.get(c.name, "black"), linewidth=1.2)
            ax.annotate(c.name, (sum(px) / len(px), sum(py) / len(py)), fontsize=7, ha="center")
    ax.set_aspect("equal")
    ax.set_title(f"{map_name} callouts ({len(cs)} callouts) over nav centroids, release {a.release}")
    ax.set_xlabel("x (u)")
    ax.set_ylabel("y (u)")
    ax.legend(loc="lower right", fontsize=8)
    out = Path(out) if out is not None else paths.EVAL_DIR / f"callouts_{map_name}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python -m engine.maps.plot <map> [out.png]")
        sys.exit(2)
    try:
        print(plot(sys.argv[1], Path(sys.argv[2]) if len(sys.argv) > 2 else None))
    except (map_assets_mod.MapAssetError, FileNotFoundError) as exc:   # missing prerequisite (LOW-19)
        print(f"cannot plot {sys.argv[1]}: {exc}", file=sys.stderr)
        sys.exit(2)
