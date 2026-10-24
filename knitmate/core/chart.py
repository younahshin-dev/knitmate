"""글 도안 → 그림 도안(기호 차트) 렌더러 (결정적 Tool).

LLM이 그림을 그리지 않는다. 검증된 단별 시퀀스(RowResult.sequence)를 받아
matplotlib 도형으로 기호를 그린다. 그래서 코 수가 도안과 정확히 일치한다.
"""
from __future__ import annotations

import io
from functools import lru_cache

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.patches import Circle, Ellipse, Rectangle  # noqa: E402

from . import glossary  # noqa: E402
from .stitch_counter import ValidationReport  # noqa: E402

INK = "#3b2f2f"
GRID = "#d9cfc4"
BG = "#fffaf3"
ACCENT = "#c0573e"
MUTED = "#efe7dc"

WS_SWAP = {"k": "p", "p": "k", "ktbl": "ptbl", "ptbl": "ktbl", "p2tog": "k2tog"}


@lru_cache(maxsize=1)
def korean_font() -> str | None:
    """한글 폰트 찾기: koreanize-matplotlib(나눔고딕 내장) → 시스템 폰트."""
    try:
        import koreanize_matplotlib  # noqa: F401

        return plt.rcParams["font.family"][0] if isinstance(plt.rcParams["font.family"], list) else plt.rcParams["font.family"]
    except Exception:
        pass
    names = {f.name for f in font_manager.fontManager.ttflist}
    for cand in ("NanumGothic", "Malgun Gothic", "Noto Sans CJK KR", "Noto Sans KR", "AppleGothic", "UnDotum"):
        if cand in names:
            plt.rcParams["font.family"] = cand
            plt.rcParams["axes.unicode_minus"] = False
            return cand
    return None


def _line(ax, xs, ys, lw=1.6, color=INK):
    ax.plot(xs, ys, color=color, lw=lw, solid_capstyle="round")


def draw_symbol(ax, code: str, x: float, y: float, s: float = 1.0):
    """(x, y) = 셀 왼쪽 아래. s = 셀 크기"""
    cx, cy = x + s / 2, y + s / 2
    lo, hi = y + s * 0.2, y + s * 0.8
    le, ri = x + s * 0.2, x + s * 0.8
    c = code.lower()
    if c == "k":
        _line(ax, [cx, cx], [lo, hi])
    elif c == "p":
        _line(ax, [le, ri], [cy, cy])
    elif c == "yo":
        ax.add_patch(Circle((cx, cy), s * 0.28, fill=False, ec=INK, lw=1.6))
    elif c == "k2tog":
        _line(ax, [le, ri], [lo, hi])
    elif c == "ssk":
        _line(ax, [le, ri], [hi, lo])
    elif c == "p2tog":
        _line(ax, [le, ri], [lo, hi])
        _line(ax, [le, cx], [cy - 0.05, cy - 0.05], lw=1.2)
    elif c == "k3tog":
        _line(ax, [le, ri], [lo, hi])
        _line(ax, [cx, ri], [lo, cy + 0.1], lw=1.2)
    elif c == "sk2p":
        _line(ax, [le, ri], [hi, lo])
        _line(ax, [le, cx], [cy + 0.1, lo], lw=1.2)
    elif c in ("s2kp", "dec"):
        _line(ax, [le, cx, ri], [lo, hi, lo])
        if c == "s2kp":
            _line(ax, [cx, cx], [lo, hi])
    elif c in ("inc", "kfb"):
        _line(ax, [le, cx, ri], [hi, lo, hi])
        if c == "kfb":
            _line(ax, [cx, cx], [lo, hi], lw=1.0)
    elif c == "m1":
        ax.text(cx, cy, "M", ha="center", va="center", fontsize=9 * s, color=INK, fontweight="bold")
    elif c == "sl":
        _line(ax, [le, cx, ri], [hi, lo, hi], lw=1.2)
    elif c in ("ktbl", "ptbl"):
        ax.text(cx, cy, "Q" if c == "ktbl" else "q", ha="center", va="center", fontsize=9 * s, color=INK)
    elif c == "bo":
        ax.add_patch(Circle((cx, cy), s * 0.16, color=INK))
    elif c == "co":
        ax.add_patch(Circle((cx, cy), s * 0.12, fill=False, ec=INK, lw=1.0))
    elif c == "ch":
        ax.add_patch(Ellipse((cx, cy), s * 0.6, s * 0.34, fill=False, ec=INK, lw=1.5))
    elif c == "sc":
        _line(ax, [le + 0.05, ri - 0.05], [lo + 0.05, hi - 0.05])
        _line(ax, [le + 0.05, ri - 0.05], [hi - 0.05, lo + 0.05])
    elif c in ("hdc", "dc", "tr"):
        _line(ax, [cx, cx], [lo, hi])
        _line(ax, [le, ri], [hi, hi])
        slashes = {"hdc": 0, "dc": 1, "tr": 2}[c]
        for i in range(slashes):
            yy = cy + 0.08 - i * 0.18
            _line(ax, [cx - 0.14, cx + 0.14], [yy - 0.07, yy + 0.07], lw=1.2)
    elif c == "slst":
        ax.add_patch(Circle((cx, cy), s * 0.1, color=INK))
    elif c == "mr":
        ax.add_patch(Circle((cx, cy), s * 0.32, fill=False, ec=ACCENT, lw=1.8))
    elif c == "sk":
        ax.add_patch(Rectangle((x, y), s, s, color=MUTED))
    else:
        ax.text(cx, cy, c[:4], ha="center", va="center", fontsize=6, color=ACCENT)


def _display_cells(r, craft: str, row_index: int) -> tuple[list[str], str]:
    """차트에 그릴 순서(왼→오)와 읽는 방향 반환"""
    seq = list(r.sequence)
    side = r.side
    if r.kind == "round":
        return list(reversed(seq)), "rtl"
    if side == "any":
        side = "RS" if row_index % 2 == 0 else "WS"
    if side == "RS":
        return list(reversed(seq)), "rtl"
    # WS 단: 왼쪽→오른쪽으로 읽고, 대바늘은 겉면에서 보이는 기호로 바꿔 그린다
    if craft == "knit":
        seq = [WS_SWAP.get(c, c) for c in seq]
    return seq, "ltr"


def render_chart(report: ValidationReport, craft: str, title: str = "", max_rows: int = 60, max_width: int = 60, highlight_row: int | None = None) -> bytes:
    has_ko = korean_font() is not None
    rows = report.rows[:max_rows]
    width = max((len(r.sequence) for r in rows), default=1)
    width = min(max(width, 1), max_width)
    n = len(rows)

    used: list[str] = []
    for r in rows:
        for c in r.sequence:
            if c not in used:
                used.append(c)

    col_w = 8  # 범례 한 칸 폭(셀 단위)
    per_line = max(1, int((width + 4) // col_w))
    legend_lines = (len(used) + per_line - 1) // per_line
    total_h = n + 1.5 + legend_lines * 1.4
    fig_w = min(max((width + 5) * 0.3, 6.5), 22)
    fig_h = min(total_h * 0.3 + 0.9, 40)
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=130, facecolor=BG)
    ax = fig.add_axes([0.03, 0.02, 0.94, 0.9])
    ax.set_facecolor(BG)

    for i, r in enumerate(rows):
        cells, direction = _display_cells(r, craft, i)
        cells = cells[:max_width]
        offset = (width - len(cells)) / 2
        y = i
        if highlight_row is not None and r.row == highlight_row:
            ax.add_patch(Rectangle((offset - 0.15, y - 0.05), len(cells) + 0.3, 1.1, color="#f6d7c3", zorder=0))
        for j, code in enumerate(cells):
            x = offset + j
            ax.add_patch(Rectangle((x, y), 1, 1, fill=False, ec=GRID, lw=0.7))
            draw_symbol(ax, code, x, y)
        color = ACCENT if r.status == "error" else INK
        label = f"{r.row}"
        if direction == "rtl":
            ax.text(offset + len(cells) + 0.4, y + 0.5, label + " ←", va="center", ha="left", fontsize=7, color=color)
        else:
            ax.text(offset - 0.4, y + 0.5, "→ " + label, va="center", ha="right", fontsize=7, color=color)

    # 범례 (차트 아래, 같은 좌표계라 기호 모양이 차트와 동일)
    x0 = -2.0
    for k, code in enumerate(used):
        line, col = k // per_line, k % per_line
        lx = x0 + col * col_w
        ly = -2.0 - line * 1.4
        ax.add_patch(Rectangle((lx, ly), 1, 1, fill=False, ec=GRID, lw=0.7))
        draw_symbol(ax, code, lx, ly)
        name = glossary.ko_name(code) if has_ko else code
        if len(name) > 9:
            name = name[:9] + "…"
        ax.text(lx + 1.3, ly + 0.5, f"{code} {name}", va="center", ha="left", fontsize=6.5, color=INK)

    ax.set_xlim(-2.5, max(width + 2.5, x0 + per_line * col_w))
    ax.set_ylim(-2.0 - (legend_lines - 1) * 1.4 - 0.4, n + 0.5)
    ax.set_aspect("equal")
    ax.axis("off")
    craft_ko = "대바늘" if craft == "knit" else "코바늘"
    t = f"{title}  ({craft_ko} 기호 차트)" if has_ko else f"{title} ({craft} chart)"
    ax.set_title(t, fontsize=11, color=INK, loc="left", pad=8)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=BG)
    plt.close(fig)
    return buf.getvalue()
