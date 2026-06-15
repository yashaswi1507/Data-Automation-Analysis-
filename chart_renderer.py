"""
chart_renderer.py
Converts plotly figure JSON → matplotlib PNG
No Chrome, no kaleido, no browser needed.
Works on any server.
"""
import io
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import to_rgba


# ── Color palette matching plotly Set2 ──────────────────────
COLORS = [
    "#66c2a5","#fc8d62","#8da0cb","#e78ac3",
    "#a6d854","#ffd92f","#e5c494","#b3b3b3",
]

def _clean_title(fig_json):
    try:
        d = json.loads(fig_json) if isinstance(fig_json, str) else fig_json
        return d.get("layout",{}).get("title",{}).get("text","Chart")
    except:
        return "Chart"


# ─────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────

def fig_to_png(fig_json, width_px=600, height_px=380):
    """
    Convert a plotly figure (JSON string or dict) to PNG bytes.
    Detects chart type and renders with matplotlib.
    Returns PNG bytes or None.
    """
    try:
        data = json.loads(fig_json) if isinstance(fig_json, str) else fig_json
    except Exception:
        return _fallback_png("Could not parse chart", width_px, height_px)

    traces = data.get("data", [])
    layout = data.get("layout", {})
    title  = layout.get("title", {}).get("text", "")

    if not traces:
        return _fallback_png(title or "Empty chart", width_px, height_px)

    chart_type = traces[0].get("type", "scatter")

    try:
        if chart_type == "bar":
            return _render_bar(traces, layout, title, width_px, height_px)
        elif chart_type in ("scatter", "scattergl"):
            return _render_scatter(traces, layout, title, width_px, height_px)
        elif chart_type == "histogram":
            return _render_histogram(traces, layout, title, width_px, height_px)
        elif chart_type == "box":
            return _render_box(traces, layout, title, width_px, height_px)
        elif chart_type == "pie":
            return _render_pie(traces, layout, title, width_px, height_px)
        elif chart_type == "heatmap":
            return _render_heatmap(traces, layout, title, width_px, height_px)
        else:
            return _fallback_png(title, width_px, height_px)
    except Exception as e:
        return _fallback_png(f"{title}\n(render error: {e})", width_px, height_px)


# ─────────────────────────────────────────────────────────────
# RENDERERS
# ─────────────────────────────────────────────────────────────

def _setup_fig(width_px, height_px, title):
    dpi = 100
    fig, ax = plt.subplots(figsize=(width_px/dpi, height_px/dpi), dpi=dpi)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#f9f9f9")
    if title:
        clean = title.replace("<b>","").replace("</b>","")
        ax.set_title(clean, fontsize=11, fontweight="bold", pad=8, color="#2c3e50")
    ax.spines[["top","right"]].set_visible(False)
    ax.spines[["left","bottom"]].set_color("#cccccc")
    ax.tick_params(colors="#555555", labelsize=8)
    return fig, ax


def _to_png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return buf.getvalue()


def _render_bar(traces, layout, title, w, h):
    fig, ax = _setup_fig(w, h, title)

    for i, trace in enumerate(traces):
        x = trace.get("x", [])
        y = trace.get("y", [])
        if not x or not y:
            continue
        color = COLORS[i % len(COLORS)]
        bars  = ax.bar(range(len(x)), y, color=color, alpha=0.85,
                       width=0.6, zorder=2)
        # Value labels on bars
        for bar, val in zip(bars, y):
            if val is not None:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                        f"{val:,.1f}" if isinstance(val, float) else str(val),
                        ha="center", va="bottom", fontsize=7, color="#333")

    ax.set_xticks(range(len(traces[0].get("x",[]))))
    ax.set_xticklabels(
        [str(v) for v in traces[0].get("x",[])],
        rotation=30, ha="right", fontsize=7
    )
    ax.yaxis.grid(True, color="#e0e0e0", zorder=0)
    ax.set_axisbelow(True)

    xaxis_title = layout.get("xaxis",{}).get("title",{})
    yaxis_title = layout.get("yaxis",{}).get("title",{})
    if isinstance(xaxis_title, dict): xaxis_title = xaxis_title.get("text","")
    if isinstance(yaxis_title, dict): yaxis_title = yaxis_title.get("text","")
    if xaxis_title: ax.set_xlabel(xaxis_title, fontsize=8)
    if yaxis_title: ax.set_ylabel(yaxis_title, fontsize=8)

    fig.tight_layout()
    return _to_png(fig)


def _render_scatter(traces, layout, title, w, h):
    fig, ax = _setup_fig(w, h, title)

    for i, trace in enumerate(traces):
        x    = trace.get("x", [])
        y    = trace.get("y", [])
        name = trace.get("name", "")
        if not x or not y:
            continue
        color = COLORS[i % len(COLORS)]
        ax.scatter(x, y, color=color, alpha=0.6, s=18, label=name, zorder=2)

    ax.yaxis.grid(True, color="#e0e0e0", zorder=0)
    ax.set_axisbelow(True)

    xaxis_title = layout.get("xaxis",{}).get("title",{})
    yaxis_title = layout.get("yaxis",{}).get("title",{})
    if isinstance(xaxis_title, dict): xaxis_title = xaxis_title.get("text","")
    if isinstance(yaxis_title, dict): yaxis_title = yaxis_title.get("text","")
    if xaxis_title: ax.set_xlabel(xaxis_title, fontsize=8)
    if yaxis_title: ax.set_ylabel(yaxis_title, fontsize=8)

    if len(traces) > 1:
        ax.legend(fontsize=7, framealpha=0.7)

    fig.tight_layout()
    return _to_png(fig)


def _render_histogram(traces, layout, title, w, h):
    fig, ax = _setup_fig(w, h, title)

    for i, trace in enumerate(traces):
        x = trace.get("x", [])
        if not x:
            continue
        nbins = trace.get("nbinsx", 25)
        color = COLORS[i % len(COLORS)]
        ax.hist(x, bins=nbins, color=color, alpha=0.8, edgecolor="white", zorder=2)

    ax.yaxis.grid(True, color="#e0e0e0", zorder=0)
    ax.set_axisbelow(True)

    xaxis_title = layout.get("xaxis",{}).get("title",{})
    if isinstance(xaxis_title, dict): xaxis_title = xaxis_title.get("text","")
    if xaxis_title: ax.set_xlabel(xaxis_title, fontsize=8)
    ax.set_ylabel("Count", fontsize=8)

    fig.tight_layout()
    return _to_png(fig)


def _render_box(traces, layout, title, w, h):
    fig, ax = _setup_fig(w, h, title)

    plot_data = []
    labels    = []

    for i, trace in enumerate(traces):
        y = trace.get("y", [])
        if y:
            plot_data.append([v for v in y if v is not None])
            labels.append(trace.get("name", f"Group {i+1}"))

    if plot_data:
        bp = ax.boxplot(
            plot_data, labels=labels,
            patch_artist=True,
            medianprops=dict(color="#e74c3c", linewidth=2),
            whiskerprops=dict(color="#555"),
            capprops=dict(color="#555"),
            flierprops=dict(marker="o", markerfacecolor="#aaa", markersize=3),
        )
        for patch, color in zip(bp["boxes"], COLORS):
            patch.set_facecolor(color); patch.set_alpha(0.7)

    ax.yaxis.grid(True, color="#e0e0e0", zorder=0)
    ax.set_axisbelow(True)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right", fontsize=7)

    fig.tight_layout()
    return _to_png(fig)


def _render_pie(traces, layout, title, w, h):
    fig, ax = _setup_fig(w, h, title)
    ax.set_facecolor("white")

    trace  = traces[0]
    labels = trace.get("labels", [])
    values = trace.get("values", [])
    hole   = trace.get("hole", 0)

    if not labels or not values:
        return _fallback_png(title, w, h)

    colors_use = COLORS[:len(labels)]
    wedges, texts, autotexts = ax.pie(
        values,
        labels=None,
        colors=colors_use,
        autopct="%1.1f%%",
        pctdistance=0.75,
        startangle=90,
        wedgeprops=dict(width=1-hole if hole else 1, edgecolor="white", linewidth=1.5),
    )
    for at in autotexts:
        at.set_fontsize(7)

    ax.legend(
        wedges, [str(l) for l in labels],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=min(3, len(labels)),
        fontsize=7,
        framealpha=0.5,
    )
    fig.tight_layout()
    return _to_png(fig)


def _render_heatmap(traces, layout, title, w, h):
    trace = traces[0]
    z     = trace.get("z", [])
    x     = trace.get("x", [])
    y_    = trace.get("y", [])

    if not z:
        return _fallback_png(title, w, h)

    import numpy as np
    z_arr = np.array(z, dtype=float)
    h_adj = max(h, len(z_arr) * 35 + 60)

    fig, ax = _setup_fig(w, h_adj, title)
    im = ax.imshow(z_arr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, shrink=0.8)

    if x:
        ax.set_xticks(range(len(x)))
        ax.set_xticklabels(x, rotation=30, ha="right", fontsize=7)
    if y_:
        ax.set_yticks(range(len(y_)))
        ax.set_yticklabels(y_, fontsize=7)

    # Annotate cells
    for i in range(z_arr.shape[0]):
        for j in range(z_arr.shape[1]):
            val = z_arr[i,j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                       fontsize=6, color="white" if abs(val) > 0.6 else "black")

    fig.tight_layout()
    return _to_png(fig)


def _fallback_png(msg, w, h):
    dpi = 100
    fig, ax = plt.subplots(figsize=(w/dpi, h/dpi), dpi=dpi)
    fig.patch.set_facecolor("white")
    ax.text(0.5, 0.5, str(msg), ha="center", va="center",
            fontsize=10, color="#555", transform=ax.transAxes, wrap=True)
    ax.axis("off")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return buf.getvalue()
