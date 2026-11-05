"""Generate per-market chained equity curves for slides 19–21.

Agent values are loaded from results/_canonical/canonical_daily_data.json (each
period starts from $1M in the canonical evaluation). The four periods (val,
test_bear, test_2023, test_bull) are stitched into one continuous wealth path
by carrying forward the ending wealth of each period as the starting wealth of
the next. BH and EW are chained the same way.

Output: one PNG per market with solid-navy background matching the template.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# ----------------------------- Palette ----------------------------
BG = "#0E1B3A"
TEXT = "white"
TITLE = "#7DC8FF"
SUBTEXT = "#C9D5EA"
AGENT_C = "#7DC8FF"           # cyan agent line
BH_C    = "#C9D5EA"           # light grey-blue BH
EW_C    = "#E66060"           # soft red EW
PERIOD_LINES = "#6E83AB"
PERIOD_SHADE = {
    "Val 2021":      "#3B5D9C",
    "Bear 2022":     "#E66060",
    "Recovery 2023": "#EE9554",
    "Bull 2024":     "#3FBF7F",
}


# ----------------------------- Config -----------------------------
MARKETS = {
    "DJI 30": {
        "key_in_daily": "DJI",
        "index_csv": "dji_index_ext.csv",
        "period_csvs": {
            "Val 2021":      "val_data_ext.csv",
            "Bear 2022":     "test_bear_data_ext.csv",
            "Recovery 2023": "test_2023_data_ext.csv",
            "Bull 2024":     "test_bull_data_ext.csv",
        },
        "bh_label": "DJI buy-and-hold",
    },
    "S&P 100": {
        "key_in_daily": "SP100",
        "index_csv": "sp500_index_ext.csv",
        "period_csvs": {
            "Val 2021":      "val_data_sp100.csv",
            "Bear 2022":     "test_bear_data_sp100.csv",
            "Recovery 2023": "test_2023_data_sp100.csv",
            "Bull 2024":     "test_bull_data_sp100.csv",
        },
        "bh_label": "S&P 500 buy-and-hold",
    },
    "NASDAQ 100": {
        "key_in_daily": "NDX",
        "index_csv": "ndx_index_ext.csv",
        "period_csvs": {
            "Val 2021":      "val_data_ndx_2017.csv",
            "Bear 2022":     "test_bear_data_ndx_2017.csv",
            "Recovery 2023": "test_2023_data_ndx_2017.csv",
            "Bull 2024":     "test_bull_data_ndx_2017.csv",
        },
        "bh_label": "NDX buy-and-hold",
    },
}


# ----------------------- Helpers -----------------------------------
def bh_series(period_csv, index_csv, data_dir):
    """Index daily close over the period's trading days, normalised to 1.0 at start."""
    p = pd.read_csv(data_dir / period_csv)
    p["date"] = pd.to_datetime(p["date"])
    dates = pd.to_datetime(sorted(p["date"].unique()))
    start, end = dates[0], dates[-1]
    idx = pd.read_csv(data_dir / index_csv)
    idx["date"] = pd.to_datetime(idx["date"])
    idx = idx[(idx["date"] >= start) & (idx["date"] <= end)].sort_values("date")
    closes = idx["close"].values.astype(float)
    closes_norm = closes / closes[0]
    return idx["date"].values, closes_norm


def ew_series(period_csv, data_dir, tx_cost=0.001):
    """Equal-weight portfolio of constituents, rebalanced daily, net of cost.

    Returns dates and a wealth path starting at 1.0 at the period's first day.
    """
    df = pd.read_csv(data_dir / period_csv)
    df["date"] = pd.to_datetime(df["date"])
    rets = df.pivot(index="date", columns="ticker",
                    values="daily_return").dropna(how="all").fillna(0.0)
    n_stocks = rets.shape[1]
    target = np.full(n_stocks, 1.0 / n_stocks)
    weights = target.copy()
    pv = 1.0
    values = [pv]
    for t in range(rets.shape[0]):
        r_t = rets.iloc[t].values
        port_ret = float(np.dot(weights, r_t))
        pv *= (1.0 + port_ret)
        if abs(1.0 + port_ret) > 1e-12:
            drifted = weights * (1.0 + r_t) / (1.0 + port_ret)
        else:
            drifted = target.copy()
        turnover = float(np.abs(target - drifted).sum())
        pv *= (1.0 - turnover * tx_cost)
        weights = target.copy()
        values.append(pv)
    dates_out = np.concatenate([
        rets.index.values[:1],
        rets.index.values,
    ])
    return dates_out[:len(values)], np.array(values)


def chain_series(legs):
    """Chain per-period wealth multipliers into one continuous path.

    legs: list of (dates_array, normalised_wealth_array) — wealth starts at 1.0 in each leg.
    Returns single (dates, wealth) where each leg starts where the previous ended.
    """
    chained_dates = []
    chained_wealth = []
    running = 1.0
    for i, (dates, wealth) in enumerate(legs):
        dates = pd.to_datetime(dates)
        wealth = np.asarray(wealth, dtype=float)
        # The wealth array has shape n+1 (start + n days). Skip the leading 1.0 on legs after the first
        # so we don't repeat the boundary date.
        if i == 0:
            chained_dates.append(dates)
            chained_wealth.append(running * wealth)
        else:
            chained_dates.append(dates[1:])
            chained_wealth.append(running * wealth[1:])
        running *= wealth[-1] / wealth[0]
    return np.concatenate(chained_dates), np.concatenate(chained_wealth)


# ----------------------- Main ---------------------------------------
def build_market_figure(market_name, market_cfg, agent_daily, data_dir, out_dir):
    fig = plt.figure(figsize=(15, 7.5), dpi=240)
    ax = fig.add_subplot(111)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    # ---- Build agent leg list (dates + normalised wealth per period) ----
    agent_legs = []
    bh_legs = []
    ew_legs = []
    period_ranges = []

    for p_name, csv in market_cfg["period_csvs"].items():
        # Agent
        period_data = agent_daily[market_cfg["key_in_daily"]]
        # Map period name to canonical_daily_data key
        key_map = {
            "Val 2021":      "val",
            "Bear 2022":     "test_bear",
            "Recovery 2023": "test_2023",
            "Bull 2024":     "test_bull",
        }
        d = period_data[key_map[p_name]]
        dates = pd.to_datetime(d["dates"])
        values = np.asarray(d["values"], dtype=float)
        wealth_norm = values / values[0]
        agent_legs.append((dates, wealth_norm))

        # BH
        bh_dates, bh_norm = bh_series(csv, market_cfg["index_csv"], data_dir)
        bh_legs.append((bh_dates, bh_norm))

        # EW
        ew_dates, ew_norm = ew_series(csv, data_dir)
        ew_legs.append((ew_dates, ew_norm))

        period_ranges.append((p_name, dates[0], dates[-1]))

    a_dates, a_wealth = chain_series(agent_legs)
    b_dates, b_wealth = chain_series(bh_legs)
    e_dates, e_wealth = chain_series(ew_legs)

    # ---- Shade each period background ----
    for p_name, t_start, t_end in period_ranges:
        ax.axvspan(t_start, t_end,
                   color=PERIOD_SHADE[p_name], alpha=0.08, linewidth=0)

    # ---- Plot wealth lines ----
    ax.plot(a_dates, a_wealth, color=AGENT_C, linewidth=2.2,
            label="Agent",         alpha=1.0, zorder=5)
    ax.plot(b_dates, b_wealth, color=BH_C, linewidth=1.6,
            label=market_cfg["bh_label"], alpha=0.95, linestyle="--", zorder=4)
    ax.plot(e_dates, e_wealth, color=EW_C, linewidth=1.6,
            label=f"Equal-weight {market_name}", alpha=0.95, linestyle=":", zorder=3)

    # ---- Period labels at top ----
    ymax = max(a_wealth.max(), b_wealth.max(), e_wealth.max()) * 1.04
    ymin = min(a_wealth.min(), b_wealth.min(), e_wealth.min()) * 0.96
    for p_name, t_start, t_end in period_ranges:
        mid = t_start + (t_end - t_start) / 2
        ax.text(mid, ymax * 0.99, p_name,
                ha="center", va="top",
                color=PERIOD_SHADE[p_name],
                fontsize=10, fontweight="bold", family="sans-serif")

    # ---- Period boundary dashed verticals ----
    for _, t_start, _ in period_ranges[1:]:
        ax.axvline(t_start, color=PERIOD_LINES, linewidth=0.6, linestyle="--", alpha=0.6)

    # ---- Axes formatting ----
    ax.set_xlim(a_dates[0], a_dates[-1])
    ax.set_ylim(ymin, ymax)

    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.tick_params(axis="x", colors=SUBTEXT, labelsize=11, length=4)
    ax.tick_params(axis="y", colors=SUBTEXT, labelsize=10, length=3)

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("bottom", "left"):
        ax.spines[spine].set_color(SUBTEXT)
        ax.spines[spine].set_linewidth(0.7)

    ax.set_ylabel("Wealth multiplier  (start = 1.00)",
                  color=SUBTEXT, fontsize=11)
    ax.grid(True, axis="y", linestyle=":", color=SUBTEXT, alpha=0.18)
    ax.axhline(1.0, color=SUBTEXT, linewidth=0.5, alpha=0.4)

    # ---- Title ----
    fig.text(0.5, 0.96, f"{market_name}  ·  Equity curves (2021 – 2024)",
             ha="center", va="center",
             color=TITLE, fontsize=17, fontweight="bold")
    fig.text(0.5, 0.925,
             "Agent chained through all 4 periods · benchmarks chained the same way",
             ha="center", va="center",
             color=SUBTEXT, fontsize=11, style="italic")

    # ---- Legend at bottom ----
    leg = ax.legend(
        loc="upper left", frameon=False,
        fontsize=11, handlelength=2.5, borderaxespad=0.5,
    )
    for text in leg.get_texts():
        text.set_color(TEXT)

    plt.tight_layout(rect=[0, 0, 1, 0.90])

    fn = f"slide_equity_{market_cfg['key_in_daily'].lower()}.png"
    out_path = out_dir / fn
    plt.savefig(out_path, dpi=240,
                facecolor=BG, edgecolor=BG, transparent=False,
                pad_inches=0.20)
    plt.close(fig)

    # Strip alpha channel by flattening onto the navy background so Canva
    # never interprets any pixel as transparent.
    from PIL import Image
    img = Image.open(out_path).convert("RGBA")
    bg_rgb = tuple(int(BG[i:i + 2], 16) for i in (1, 3, 5))
    canvas = Image.new("RGB", img.size, bg_rgb)
    canvas.paste(img, mask=img.split()[-1])
    canvas.save(out_path, format="PNG")

    print(f"saved -> {out_path}")
    return out_path


def main():
    proj = Path(__file__).parent
    data_dir = proj / "data"
    out_dir = proj

    with open(proj / "results" / "_canonical" / "canonical_daily_data.json") as f:
        agent_daily = json.load(f)

    for name, cfg in MARKETS.items():
        print(f"\n=== {name} ===")
        build_market_figure(name, cfg, agent_daily, data_dir, out_dir)


if __name__ == "__main__":
    main()
