import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Ellipse, Circle, Rectangle, FancyArrowPatch

plt.rcParams.update({"font.family": "DejaVu Sans", "mathtext.fontset": "cm", "pdf.fonttype": 42, "svg.fonttype": "none"})
INK, MUTE = "#1b2a41", "#5c6b7a"
COL = {"blue": "#2f6fb7", "green": "#4c9a2a", "orange": "#e08a1e", "purple": "#7b4fb3", "red": "#d1362f"}

fig = plt.figure(figsize=(13.0, 4.5))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 130); ax.set_ylim(0, 45); ax.axis("off")

W, H, Y0, GAP = 22.4, 39.0, 3.0, 3.6
X = [2.0 + i * (W + GAP) for i in range(5)]
EDGE = [COL["blue"], COL["green"], COL["orange"], COL["purple"], COL["blue"]]
TITLES = [
    "Extract frozen PTM\nfeatures $Z_t$ from\nthe task data",
    "For each class $c$:\nsample count $N_c$ $\\rightarrow$\nrectification budget $M_c$",
    "Generate $M_c$ intra-class\nspherical mixup\nsamples $\\tilde{z}$",
    "Dual-stream update\nof $G_t$ and $Q_t$:\nreal + mixed features",
    "Closed-form\nanalytic solution",
]
for i, x in enumerate(X):
    ax.add_patch(FancyBboxPatch((x, Y0), W, H, boxstyle="round,pad=0,rounding_size=1.6", fc="white", ec=EDGE[i], lw=2.0))
    ax.add_patch(Circle((x + W / 2, Y0 + H - 3.4), 1.9, fc=EDGE[i], ec="none"))
    ax.text(x + W / 2, Y0 + H - 3.45, str(i + 1), ha="center", va="center", color="white", fontsize=12, fontweight="bold")
    ax.text(x + W / 2, Y0 + H - 6.6, TITLES[i], ha="center", va="top", color=INK, fontsize=10.5, linespacing=1.35)
    if i < 4:
        ax.add_patch(FancyArrowPatch((x + W + 0.5, Y0 + H / 2), (x + W + GAP - 0.5, Y0 + H / 2), arrowstyle="-|>", mutation_scale=16, lw=1.8, color=INK))

# ---- 1: frozen backbone + features -------------------------------------------------------------
x = X[0]; cx = x + 6.4; cy = Y0 + 17.0
for k in range(3):                       # database cylinder
    ax.add_patch(Rectangle((cx - 2.6, cy - 1.0 + 2.1 * k - 2.1), 5.2, 2.1, fc="white", ec=INK, lw=1.4))
for k in range(4):
    ax.add_patch(Ellipse((cx, cy - 3.1 + 2.1 * k), 5.2, 1.5, fc="white", ec=INK, lw=1.4))
sx, sy = x + 15.6, cy                     # snowflake = frozen
for a in np.arange(0, np.pi, np.pi / 3):
    dx, dy = 3.0 * np.cos(a), 3.0 * np.sin(a)
    ax.plot([sx - dx, sx + dx], [sy - dy, sy + dy], color=COL["blue"], lw=2.0, solid_capstyle="round")
    for s in (-1, 1):
        px, py = sx + s * 0.62 * dx, sy + s * 0.62 * dy
        for b in (a + 0.75, a - 0.75):
            ax.plot([px, px + s * 0.95 * np.cos(b)], [py, py + s * 0.95 * np.sin(b)], color=COL["blue"], lw=1.6, solid_capstyle="round")
ax.text(cx, cy - 6.0, "data", ha="center", va="top", fontsize=8.5, color=MUTE)
ax.text(sx, cy - 6.0, "frozen PTM", ha="center", va="top", fontsize=8.5, color=MUTE)
rng = np.random.RandomState(3)
pts = np.c_[rng.uniform(x + 4.5, x + W - 4.5, 16), rng.uniform(Y0 + 3.8, Y0 + 7.6, 16)]
ax.scatter(pts[:, 0], pts[:, 1], s=20, fc="white", ec=COL["blue"], lw=1.3, zorder=3)
ax.text(x + W / 2, Y0 + 2.0, "$Z_t$", ha="center", va="center", fontsize=12, color=INK)

# ---- 2: class-adaptive budget ------------------------------------------------------------------
x = X[1]; px0, py0, pw, ph = x + 4.2, Y0 + 7.6, W - 7.0, 12.2
ax.annotate("", xy=(px0 + pw + 0.8, py0), xytext=(px0, py0), arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.3))
ax.annotate("", xy=(px0, py0 + ph + 0.8), xytext=(px0, py0), arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.3))
xt = px0 + 0.62 * pw                                   # N_c = T_t
ax.plot([px0, xt, px0 + pw], [py0 + 0.86 * ph, py0, py0], color=COL["green"], lw=2.4, solid_capstyle="round")
ax.plot([xt, xt], [py0, py0 - 0.7], color=INK, lw=1.2)
ax.text(xt, py0 - 1.0, "$T_t$", ha="center", va="top", fontsize=10, color=INK)
ax.text(px0 + pw + 0.9, py0 - 1.0, "$N_c$", ha="center", va="top", fontsize=10.5, color=INK)
ax.text(px0 - 0.6, py0 + ph + 0.6, "$M_c$", ha="right", va="center", fontsize=10.5, color=INK)
tx, ty = px0 + 0.05 * pw, py0 + 0.86 * ph - 0.05 * 0.86 * ph / 0.62
ax.scatter([tx], [ty], s=46, color=COL["red"], zorder=4); ax.text(tx + 1.2, ty + 0.9, "tail", fontsize=9.5, color=COL["red"], va="bottom")
hx = px0 + 0.86 * pw
ax.scatter([hx], [py0], s=46, color=COL["blue"], zorder=4); ax.text(hx, py0 + 1.3, "head", fontsize=9.5, color=COL["blue"], ha="center", va="bottom")
ax.text(x + W / 2, Y0 + 2.6, "$M_c=\\max(T_t-N_c,\\,0)$", ha="center", va="center", fontsize=10.5, color=INK,
        bbox=dict(boxstyle="round,pad=0.35", fc="#eef6ea", ec=COL["green"], lw=1.0))

# ---- 3: spherical mixup ------------------------------------------------------------------------
x = X[2]; cx, cy, R = x + W / 2, Y0 + 15.2, 7.6
ax.add_patch(Circle((cx, cy), R, fc="#fafafa", ec=INK, lw=1.4))
t = np.linspace(0, np.pi, 80)
ax.plot(cx + R * np.cos(t), cy - 0.30 * R * np.sin(t), color=INK, lw=1.0)
ax.plot(cx + R * np.cos(t), cy + 0.30 * R * np.sin(t), color=INK, lw=0.8, ls=(0, (3, 3)))
def on_sphere(u):                          # project a 3-d unit vector to the picture plane
    u = np.asarray(u, float); u = u / np.linalg.norm(u)
    return cx + R * u[0], cy + R * (0.95 * u[1] + 0.30 * u[2])
zi, zj = np.array([-0.62, 0.55, 0.56]), np.array([0.70, 0.30, 0.65])
arc = np.array([on_sphere(g * zi + (1 - g) * zj) for g in np.linspace(0, 1, 60)])
ax.plot(arc[:, 0], arc[:, 1], color=COL["orange"], lw=2.4, zorder=3)
ax.plot([arc[0, 0], arc[-1, 0]], [arc[0, 1], arc[-1, 1]], color=MUTE, lw=1.0, ls=(0, (2, 2)), zorder=2)
pi, pj, pm = on_sphere(zi), on_sphere(zj), on_sphere(0.45 * zi + 0.55 * zj)
ax.scatter(*pi, s=60, color=COL["blue"], zorder=5); ax.scatter(*pj, s=60, color=COL["red"], zorder=5)
ax.scatter(*pm, s=70, color=COL["green"], zorder=6, ec="white", lw=1.0)
ax.text(pi[0] - 1.0, pi[1] + 1.0, "$z_i$", ha="right", va="bottom", fontsize=11.5, color=COL["blue"])
ax.text(pj[0] + 1.0, pj[1] + 1.0, "$z_j$", ha="left", va="bottom", fontsize=11.5, color=COL["red"])
ax.text(pm[0] + 0.2, pm[1] - 1.5, "$\\tilde{z}$", ha="center", va="top", fontsize=12, color=COL["green"])
ax.text(x + W / 2, Y0 + 3.3, "$\\tilde{z}=\\frac{\\gamma z_i+(1-\\gamma)z_j}{\\Vert\\gamma z_i+(1-\\gamma)z_j\\Vert_2}$", ha="center", va="center", fontsize=12.5, color=INK)

# ---- 4: dual-stream update ---------------------------------------------------------------------
x = X[3]
def grid(gx, gy, s, n, fill, label, color):
    for r in range(n):
        for c in range(n):
            ax.add_patch(Rectangle((gx + c * s, gy + r * s), s, s, fc=fill(r, c), ec=INK, lw=0.9))
    ax.text(gx + n * s / 2, gy + n * s + 0.8, label, ha="center", va="bottom", fontsize=9, color=color, linespacing=1.2)
s, n = 1.75, 4; gy = Y0 + 11.6
g1x, g2x = x + 2.3, x + W - 2.3 - n * s
real = lambda r, c: "#cfe0f3" if abs(r - (n - 1 - c)) <= 0 or (r >= 2 and c <= 1) else "white"
augm = lambda r, c: "#f7d9b0" if (r <= 1 and c >= 2) or abs(r - (n - 1 - c)) <= 0 else "white"
grid(g1x, gy, s, n, real, "fidelity\n(real)", COL["blue"])
grid(g2x, gy, s, n, augm, "rectification\n(mixed)", COL["orange"])
ax.text(x + W / 2, gy + n * s / 2, "$+\\,\\beta$", ha="center", va="center", fontsize=13, color=INK)
ax.text(g1x + n * s / 2, gy - 1.0, "$\\sum_{i=1}^{N_c} z z^{\\top}$", ha="center", va="top", fontsize=10.5, color=INK)
ax.text(g2x + n * s / 2, gy - 1.0, "$\\sum_{m=1}^{M_c} \\tilde{z}\\tilde{z}^{\\top}$", ha="center", va="top", fontsize=10.5, color=INK)
ax.text(x + W / 2, Y0 + 3.3, "$G_t = G_{t-1}+\\sum_c[\\,\\cdot\\,]$", ha="center", va="center", fontsize=11, color=INK)

# ---- 5: closed-form solution -------------------------------------------------------------------
x = X[4]; cx, cy = x + W / 2, Y0 + 16.5
ax.add_patch(Circle((cx, cy), 4.6, fc="white", ec=COL["green"], lw=2.6))
ax.plot([cx - 2.1, cx - 0.5, cx + 2.4], [cy + 0.1, cy - 1.7, cy + 1.9], color=COL["green"], lw=3.4, solid_capstyle="round", solid_joinstyle="round")
ax.text(cx, Y0 + 6.2, "$W_t=(G_t+\\tau I)^{-1}Q_t$", ha="center", va="center", fontsize=12, color=INK)
ax.text(cx, Y0 + 2.6, "no gradients, no exemplars", ha="center", va="center", fontsize=8.8, color=MUTE)

for ext in ("pdf", "png", "svg"):
    fig.savefig(f"assets/method.{ext}", dpi=200, bbox_inches="tight", pad_inches=0.08, facecolor="white")
