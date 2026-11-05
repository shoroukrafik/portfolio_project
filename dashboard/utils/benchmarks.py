"""
Central registry of benchmark returns, Sharpe ratios, and max drawdowns.

This is the SINGLE SOURCE OF TRUTH. All page comparisons read from here.

All cum-return numbers are cumulative percentage over the period.
Sharpe = annualized; MDD = percentage (negative).
"""

BENCHMARKS = {
    "DJI": {
        "val":       {"DJI BH": 20.23, "EW DJI": 25.09},
        "test_bear": {"DJI BH": -9.40, "EW DJI": -10.25},
        "test_2023": {"DJI BH": 13.74, "EW DJI": 18.05},
        "test_bull": {"DJI BH": 12.88, "EW DJI": 22.04},
    },
    "SP100": {
        "val":       {"SP500 BH": 28.79, "OEX BH": 29.33, "EW SP100": 33.11},
        "test_bear": {"SP500 BH": -19.95, "OEX BH": -22.99, "EW SP100": -9.48},
        "test_2023": {"SP500 BH": 24.73, "OEX BH": 31.39, "EW SP100": 21.17},
        "test_bull": {"SP500 BH": 24.54, "OEX BH": 30.90, "EW SP100": 18.16},
    },
    "NDX": {
        "val":       {"NDX BH": 26.63, "EW NDX": 31.00},
        "test_bear": {"NDX BH": -33.71, "EW NDX": -15.86},
        "test_2023": {"NDX BH": 54.90, "EW NDX": 37.00},
        "test_bull": {"NDX BH": 28.13, "EW NDX": 10.38},
    },
}


# Sharpe ratio per benchmark per period (annualized)
BENCHMARK_SHARPE = {
    "DJI": {
        "val":       {"DJI BH":  1.559, "EW DJI":  1.951},
        "test_bear": {"DJI BH": -0.403, "EW DJI": -0.459},
        "test_2023": {"DJI BH":  1.200, "EW DJI":  2.007},
        "test_bull": {"DJI BH":  1.129, "EW DJI":  1.664},
    },
    "SP100": {
        "val":       {"SP500 BH":  2.020, "OEX BH":  2.014, "EW SP100":  2.419},
        "test_bear": {"SP500 BH": -0.807, "OEX BH": -0.926, "EW SP100": -0.373},
        "test_2023": {"SP500 BH":  1.775, "OEX BH":  2.097, "EW SP100":  1.497},
        "test_bull": {"SP500 BH":  1.814, "OEX BH":  2.034, "EW SP100":  1.612},
    },
    "NDX": {
        "val":       {"NDX BH":  1.455, "EW NDX":  1.877},
        "test_bear": {"NDX BH": -1.111, "EW NDX": -0.472},
        "test_2023": {"NDX BH":  2.535, "EW NDX":  2.125},
        "test_bull": {"NDX BH":  1.465, "EW NDX":  0.744},
    },
}


# Max drawdown per benchmark per period (percentage, negative)
BENCHMARK_MDD = {
    "DJI": {
        "val":       {"DJI BH":  -6.62, "EW DJI":  -5.42},
        "test_bear": {"DJI BH": -21.94, "EW DJI": -23.40},
        "test_2023": {"DJI BH":  -9.02, "EW DJI":  -8.29},
        "test_bull": {"DJI BH":  -6.06, "EW DJI":  -6.17},
    },
    "SP100": {
        "val":       {"SP500 BH":  -5.21, "OEX BH":  -5.29, "EW SP100":  -5.21},
        "test_bear": {"SP500 BH": -25.43, "OEX BH": -27.23, "EW SP100": -22.26},
        "test_2023": {"SP500 BH": -10.28, "OEX BH":  -9.47, "EW SP100": -10.20},
        "test_bull": {"SP500 BH":  -8.49, "OEX BH":  -9.86, "EW SP100":  -6.07},
    },
    "NDX": {
        "val":       {"NDX BH": -10.93, "EW NDX":  -6.73},
        "test_bear": {"NDX BH": -35.28, "EW NDX": -24.85},
        "test_2023": {"NDX BH": -10.93, "EW NDX": -10.39},
        "test_bull": {"NDX BH": -13.58, "EW NDX":  -9.79},
    },
}


PERIOD_LABELS = {
    "val":       "Validation 2021",
    "test_bear": "Bear 2022",
    "test_2023": "Recovery 2023",
    "test_bull": "Bull 2024",
}


PERIOD_DESCRIPTIONS = {
    "val":       "Steady bull year used for model selection.",
    "test_bear": "Fed pivot bear market — first true out-of-sample stress test.",
    "test_2023": "Sharp V-shape recovery year.",
    "test_bull": "Mega-cap AI-led rally year.",
}


def get_benchmark(universe: str, period: str) -> dict:
    return BENCHMARKS.get(universe, {}).get(period, {})


def get_benchmark_sharpe(universe: str, period: str) -> dict:
    return BENCHMARK_SHARPE.get(universe, {}).get(period, {})


def get_benchmark_mdd(universe: str, period: str) -> dict:
    return BENCHMARK_MDD.get(universe, {}).get(period, {})
