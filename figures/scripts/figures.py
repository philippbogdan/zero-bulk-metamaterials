"""Draw the stiffness map, corner-gap comparison and wall-convergence chart.

Inputs are the tables in figures/data; PNG and SVG outputs go in figures.
Selection and display transformations are saved in figures/tmp/figures_info.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

HERE = Path(__file__).resolve().parents[1]
DATA = HERE / "data"
OUT = HERE

DPI = 240          # Nine inches at this resolution gives a 2160-pixel-wide image.
WIDTH_IN = 9.0

INK = "#1A1A1A"
GREY = "#6E6E6E"
DEEP_BLUE = "#1D3F6E"
MID_BLUE = "#3F7FC4"
COPPER = "#B05A26"
BG = "#FFFFFF"

ROUNDOFF = 1e-12   # Display tiny bulk coordinates as zero without changing the stored table.

TRANSFORMS = []


def style():
    mpl.rcParams.update({
        "font.family": "Arial",
        "mathtext.fontset": "stixsans",   # Use a distinct Greek nu in mathematical labels.
        "font.size": 11,
        "axes.labelsize": 12,
        "xtick.labelsize": 10.5,
        "ytick.labelsize": 10.5,
        "legend.fontsize": 11,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "text.color": INK,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.minor.width": 0.6,
        "ytick.minor.width": 0.6,
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.minor.size": 2.2,
        "ytick.minor.size": 2.2,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": BG,
        "axes.facecolor": BG,
        "savefig.facecolor": BG,
        "svg.fonttype": "path",
        "svg.hashsalt": "zero-bulk-metamaterials",
        "path.simplify": False,
        "legend.frameon": False,
        "legend.handletextpad": 0.5,
        "legend.columnspacing": 1.6,
    })


# Construct a colour sequence with decreasing perceptual lightness.


def _lab_to_srgb(L, a, b):
    fy = (L + 16) / 116
    fx = fy + a / 500
    fz = fy - b / 200
    d = 6 / 29

    def finv(t):
        return np.where(t > d, t ** 3, 3 * d * d * (t - 4 / 29))
    X, Y, Z = 0.95047 * finv(fx), 1.0 * finv(fy), 1.08883 * finv(fz)
    M = np.array([[3.2404542, -1.5371385, -0.4985314],
                  [-0.9692660, 1.8760108, 0.0415560],
                  [0.0556434, -0.2040259, 1.0572252]])
    rgb = M @ np.array([X, Y, Z])
    inside = bool(np.all((rgb >= -1e-6) & (rgb <= 1 + 1e-6)))
    rgb = np.clip(rgb, 0, 1)
    srgb = np.where(rgb <= 0.0031308, 12.92 * rgb, 1.055 * rgb ** (1 / 2.4) - 0.055)
    return srgb, inside


def srgb_to_lab(hexcol):
    c = np.array([int(hexcol[i:i + 2], 16) for i in (1, 3, 5)]) / 255
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    M = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]])
    X, Y, Z = M @ lin
    X, Z = X / 0.95047, Z / 1.08883

    def f(t):
        return np.where(t > (6 / 29) ** 3, np.cbrt(t), t / (3 * (6 / 29) ** 2) + 4 / 29)
    L = 116 * f(Y) - 16
    return float(L), float(500 * (f(X) - f(Y))), float(200 * (f(Y) - f(Z)))


def theta_ramp(n=8):
    """Return colours whose CIE lightness decreases from copper through plum to blue."""
    L = np.linspace(76, 22, n)
    hue = np.linspace(62, -95, n)          # Hue angles pass from copper through pink and purple to blue.
    chroma = 34 + 14 * np.sin(np.linspace(0, np.pi, n))
    cols, ok = [], True
    for l_, h_, c_ in zip(L, hue, chroma):
        rgb, inside = _lab_to_srgb(l_, c_ * np.cos(np.radians(h_)), c_ * np.sin(np.radians(h_)))
        ok &= inside
        cols.append("#%02X%02X%02X" % tuple(int(round(v * 255)) for v in rgb))
    return cols, ok


def darker(hexcol, dl=22):
    L, a, b = srgb_to_lab(hexcol)
    rgb, _ = _lab_to_srgb(max(L - dl, 5), a, b)
    return "#%02X%02X%02X" % tuple(int(round(v * 255)) for v in rgb)


PLOTTED = {}


def scatter(ax, x, y, key, **kw):
    """Draw points and check that their coordinates exactly match the supplied arrays."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    coll = ax.scatter(x, y, **kw)
    off = np.asarray(coll.get_offsets(), float)
    assert off.shape == (len(x), 2) and np.array_equal(off[:, 0], x) and np.array_equal(off[:, 1], y), key
    PLOTTED[key] = PLOTTED.get(key, 0) + len(x)
    return coll


def save(fig, stem, svg=True):
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / f"{stem}.png", dpi=DPI, metadata={"Software": None})
    # Discard the fully opaque alpha channel while preserving every RGB value.
    from PIL import Image
    im = Image.open(OUT / f"{stem}.png")
    rgba = np.asarray(im)
    assert rgba.shape[2] == 4 and rgba[..., 3].min() == 255
    Image.fromarray(rgba[..., :3]).save(OUT / f"{stem}.png", optimize=True)
    if svg:
        fig.savefig(OUT / f"{stem}.svg", metadata={"Date": None, "Creator": None})
    plt.close(fig)


# Compare certified coordinates with the baseline and the HS square.


def stiffness_map():
    cert = pd.read_csv(DATA / "certified.csv")
    base = pd.read_csv(DATA / "baseline.csv")
    acc = cert[cert.accepted == 1].copy()
    rejected = cert[cert.accepted != 1]
    TRANSFORMS.append(dict(figure="stiffness-map", step="filter accepted == 1",
                           kept=int(len(acc)), dropped=rejected[["file", "defect"]].to_dict("records")))
    tiny = acc.x.abs() < ROUNDOFF
    neg = acc.x < 0
    TRANSFORMS.append(dict(figure="stiffness-map",
                           step=f"x drawn as 0 where |x| < {ROUNDOFF:g} (float round-off of exact zero bulk modulus)",
                           rows=int(tiny.sum()), negative_rows=int((neg & tiny).sum()),
                           min_x=float(acc.x.min()),
                           max_abs_x_set_to_zero=float(acc.x[tiny].abs().max())))
    assert not (neg & ~tiny).any(), "a negative x beyond round-off would need a different treatment"
    acc["x_plot"] = np.where(tiny, 0.0, acc.x)

    thetas = sorted(acc.theta.unique())
    cols, in_gamut = theta_ramp(len(thetas))
    colour = dict(zip(thetas, cols))
    cmap = ListedColormap(cols)
    norm = BoundaryNorm(np.arange(len(thetas) + 1) - 0.5, len(thetas))

    is_zb = acc.family.str.startswith("zero_bulk")
    fig = plt.figure(figsize=(WIDTH_IN, 4.8), layout="constrained")
    fig.get_layout_engine().set(w_pad=0.06, h_pad=0.04, wspace=0.03)
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.045])
    axes = [fig.add_subplot(gs[0])]
    axes.append(fig.add_subplot(gs[1], sharey=axes[0]))
    cax = fig.add_subplot(gs[2])
    lim = (-0.035, 1.035)
    for ax, nu in zip(axes, (0.3, 0.45)):
        ax.set_aspect("equal")
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        # The normalised HS limits enclose a unit square.
        ax.add_patch(Rectangle((0, 0), 1, 1, fill=False, ec=INK, lw=0.9, zorder=2))
        # Include each dust curve and the three limiting baseline points.
        b = base[base.nu == nu]
        for th in thetas:
            d = b[(b.theta == th) & (b.family == "dust")]
            ax.plot(d.x, d.y, color=GREY, lw=0.85, zorder=3, solid_capstyle="round")
        corners = b[b.family.isin(["void", "slip_limit", "hs"])].drop_duplicates(["x", "y"])
        scatter(ax, corners.x, corners.y, f"stiffness-map nu={nu} known corner points", s=26, marker="s", facecolor="white", edgecolor=GREY,
                   linewidth=1.1, zorder=6, clip_on=False)
        a = acc[acc.nu == nu]
        trees = a[~a.family.str.startswith("zero_bulk")].sort_values("theta", ascending=False)
        zb = a[a.family.str.startswith("zero_bulk")].sort_values("theta", ascending=False)
        scatter(ax, trees.x_plot, trees.y, f"stiffness-map nu={nu} certified trees", s=20, marker="o",
                   c=[colour[t] for t in trees.theta],
                   edgecolors=[darker(colour[t]) for t in trees.theta],
                   linewidths=0.45, zorder=4, clip_on=False)
        scatter(ax, zb.x_plot, zb.y, f"stiffness-map nu={nu} zero-bulk designs", s=34, marker="D",
                   c=[colour[t] for t in zb.theta], edgecolors=INK,
                   linewidths=0.75, zorder=5, clip_on=False)
        ax.text(0.965, 0.075, rf"$\nu_\mathrm{{s}}$ = {nu:g}", ha="right", va="bottom",
                fontsize=12, color=INK)
        ax.spines["left"].set_position(("outward", 6))
        ax.spines["bottom"].set_position(("outward", 6))
        ax.spines["left"].set_bounds(0, 1)
        ax.spines["bottom"].set_bounds(0, 1)
        ax.set_xticks(np.linspace(0, 1, 6))
        ax.set_yticks(np.linspace(0, 1, 6))
        ax.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%g"))
        ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%g"))
        ax.set_xlabel(r"Bulk modulus $\kappa^{*}/\kappa_\mathrm{HS}$ (1)")
    axes[0].set_ylabel(r"Shear modulus $\mu^{*}/\mu_\mathrm{HS}$ (1)")
    axes[1].tick_params(labelleft=False)

    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, cax=cax, ticks=np.arange(len(thetas)))
    cb.ax.set_yticklabels([f"{t:g}" for t in thetas])
    cb.set_label(r"Solid fraction $\theta$ (1)")
    cb.outline.set_linewidth(0.6)
    cb.ax.tick_params(length=0, pad=4)

    mid = cols[len(cols) // 2]
    handles = [
        Line2D([], [], ls="none", marker="o", ms=6.2, mfc=mid, mec=darker(mid), mew=0.6),
        Line2D([], [], ls="none", marker="D", ms=6.0, mfc=mid, mec=INK, mew=0.75),
        Line2D([], [], color=GREY, lw=0.95, marker="s", ms=5.4, mfc="white", mec=GREY, mew=1.1),
        Line2D([], [], color=INK, lw=0.9),
    ]
    labels = ["Certified trees", "Zero-bulk designs", "Known constructions", "HS bound"]
    fig.legend(handles, labels, loc="outside upper center", ncols=4, handlelength=1.8)
    n_drawn = sum(v for k, v in PLOTTED.items() if k.startswith("stiffness-map") and "corner" not in k)
    assert n_drawn == len(acc), (n_drawn, len(acc))
    # Align the colour key with the plotted panels after their equal-aspect layout.
    fig.canvas.draw()
    fig.set_layout_engine("none")
    p = axes[1].get_position()
    c = cax.get_position()
    cax.set_position([c.x0, p.y0, c.width, p.height])
    save(fig, "stiffness-map")
    return dict(colours=colour, in_gamut=in_gamut, n_points=int(len(acc)),
                n_zero_bulk=int(is_zb.sum()), n_trees=int((~is_zb).sum()))


# Show both support gaps at each tested fraction.


def corner_gap():
    g = pd.read_csv(DATA / "gaps.csv")
    used = g[g.theta < 1].copy()
    TRANSFORMS.append(dict(figure="corner-gap", step="omit theta == 1 (headline set is the seven fractions below one)",
                           kept=int(len(used)), dropped=int((g.theta >= 1).sum())))
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH_IN, 3.9), sharey=True, layout="constrained")
    fig.get_layout_engine().set(w_pad=0.06, h_pad=0.04, wspace=0.06)
    for ax, nu in zip(axes, (0.3, 0.45)):
        d = used[used.nu == nu].sort_values("theta")
        for _, r in d.iterrows():
            ax.plot([r.theta, r.theta], [r.after, r.before], color="#B9B9B9", lw=1.4, zorder=1,
                    solid_capstyle="butt")
        scatter(ax, d.theta, d.before, f"corner-gap nu={nu} before", s=46, marker="o", facecolor="white", edgecolor=GREY,
                   linewidth=1.5, zorder=3)
        scatter(ax, d.theta, d.after, f"corner-gap nu={nu} after", s=46, marker="o", facecolor=DEEP_BLUE, edgecolor=DEEP_BLUE,
                   linewidth=1.0, zorder=3)
        ax.set_xlim(-0.03, 0.73)
        ax.set_ylim(0, 0.8)
        ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        ax.set_yticks(np.arange(0, 0.81, 0.1))
        ax.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%g"))
        ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%g"))
        ax.spines["left"].set_position(("outward", 6))
        ax.spines["bottom"].set_position(("outward", 6))
        ax.spines["bottom"].set_bounds(0, 0.7)
        ax.set_xlabel(r"Solid fraction $\theta$ (1)")
        ax.text(0.97, 0.40, rf"$\nu_\mathrm{{s}}$ = {nu:g}", transform=ax.transAxes,
                ha="right", va="center", fontsize=12)
        ax.grid(axis="y", color="#EBEBEB", lw=0.7, zorder=0)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Maximum corner gap (1)")
    axes[1].tick_params(labelleft=False)
    handles = [
        Line2D([], [], ls="none", marker="o", ms=7.2, mfc="white", mec=GREY, mew=1.5),
        Line2D([], [], ls="none", marker="o", ms=7.2, mfc=DEEP_BLUE, mec=DEEP_BLUE),
    ]
    fig.legend(handles, ["Known constructions", "With certified designs"],
               loc="outside upper center", ncols=2)
    assert sum(v for k, v in PLOTTED.items() if k.startswith("corner-gap")) == 2 * len(used)
    save(fig, "corner-gap")
    return dict(rows=int(len(used)))


# Show the measured tensor errors as the separation ratio increases.


def wall_convergence():
    f = pd.read_csv(DATA / "finite_scale.csv")
    f["percent"] = 100 * f.relative_error
    TRANSFORMS.append(dict(figure="wall-convergence", step="percent = 100 * relative_error", rows=int(len(f))))
    rank2 = f[f.level == "A"].sort_values("ratio")
    h16 = f[(f.level == "H") & (f.p2 == 16)]
    h32 = f[(f.level == "H") & (f.p2 == 32)].sort_values("ratio")
    dup = h16[h16.ratio == 4]
    assert len(dup) == 2 and set(dup.N) == {128, 256}
    h16_main = h16[~((h16.ratio == 4) & (h16.N == 128))].sort_values("ratio")
    h16_dup = dup[dup.N == 128]

    fig, ax = plt.subplots(figsize=(WIDTH_IN, 4.6), layout="constrained")
    fig.get_layout_engine().set(w_pad=0.08, h_pad=0.06)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=10)
    # Place an inverse-ratio reference above the observations; its height is not fitted.
    xr = np.array([2.0, 16.0])
    guide = 25.0 * 2.0 / xr
    ax.plot(xr, guide, ls=(0, (5, 3.5)), color=GREY, lw=1.1, zorder=1)
    ax.text(16.0 * 1.06, guide[-1], "1/ratio", ha="left", va="center", color=GREY, fontsize=11)

    ax.plot(rank2.ratio, rank2.percent, color=COPPER, lw=1.3, zorder=2)
    ax.plot(h16_main.ratio, h16_main.percent, color=DEEP_BLUE, lw=1.3, zorder=2)
    ax.plot(h32.ratio, h32.percent, color=MID_BLUE, lw=1.3, ls=(0, (1.2, 2.2)), zorder=2,
            dash_capstyle="round")
    scatter(ax, rank2.ratio, rank2.percent, "convergence rank-two", s=48, marker="s", facecolor=COPPER, edgecolor=COPPER, zorder=4)
    scatter(ax, h16_main.ratio, h16_main.percent, "convergence herringbone p2=16 filled", s=44, marker="o", facecolor=DEEP_BLUE,
               edgecolor=DEEP_BLUE, zorder=5)
    scatter(ax, h16_dup.ratio, h16_dup.percent, "convergence herringbone p2=16 N=128 hollow ring", s=190, marker="o", facecolor="none",
               edgecolor=DEEP_BLUE, linewidth=1.3, zorder=5)
    scatter(ax, h32.ratio, h32.percent, "convergence herringbone p2=32", s=95, marker="^", facecolor="none", edgecolor=MID_BLUE,
               linewidth=1.5, zorder=6)

    ax.set_xlim(1.75, 21)
    ax.set_ylim(1.0, 30)
    ax.set_xticks([2, 4, 8, 16])
    ax.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%g"))
    ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())
    ax.set_yticks([1, 2, 5, 10, 20])
    ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%g"))
    ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    ax.spines["left"].set_position(("outward", 6))
    ax.spines["bottom"].set_position(("outward", 6))
    ax.set_xlabel("Scale ratio (1)")
    ax.set_ylabel("Tensor error (%)")
    handles = [
        Line2D([], [], color=COPPER, lw=1.3, marker="s", ms=7, mfc=COPPER, mec=COPPER),
        Line2D([], [], color=DEEP_BLUE, lw=1.3, marker="o", ms=6.6, mfc=DEEP_BLUE, mec=DEEP_BLUE),
        Line2D([], [], color=MID_BLUE, lw=1.3, ls=(0, (1.2, 2.2)), marker="^", ms=9, mfc="none",
               mec=MID_BLUE, mew=1.5),
    ]
    labels = ["Rank-two wall", r"Herringbone, $p_2$ = 16", r"Herringbone, $p_2$ = 32"]
    ax.legend(handles, labels, loc="lower left", bbox_to_anchor=(0.02, 0.03), handlelength=2.6)
    assert sum(v for k, v in PLOTTED.items() if k.startswith("convergence")) == len(f)
    save(fig, "wall-convergence")
    return dict(rows=int(len(f)), dup=dup[["file", "N", "ratio", "relative_error"]].to_dict("records"))


def main():
    style()
    info = dict(stiffness_map=stiffness_map(), corner_gap=corner_gap(),
                wall_convergence=wall_convergence(), transforms=TRANSFORMS, plotted_points=PLOTTED)
    (HERE / "tmp").mkdir(exist_ok=True)
    (HERE / "tmp" / "figures_info.json").write_text(json.dumps(info, indent=1, default=str))
    print(json.dumps(info["stiffness_map"], indent=1, default=str))


if __name__ == "__main__":
    main()
