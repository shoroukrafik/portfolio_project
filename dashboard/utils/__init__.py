"""Dashboard utility modules."""
from .results_loader import (
    load_all_experiments, get_universe_experiments,
    count_total_experiments, CATEGORIES, VARIANT_REGISTRY,
)
from .theme import apply_theme, badge
from .benchmarks import (
    BENCHMARKS, get_benchmark,
    BENCHMARK_SHARPE, get_benchmark_sharpe,
    BENCHMARK_MDD,    get_benchmark_mdd,
    PERIOD_LABELS, PERIOD_DESCRIPTIONS,
)
from .selection import (
    picks_summary, headline_pick, defensive_pick,
    load_sweep, MIN_STEP_FRACTION,
)
from .legacy_loader import (
    load_legacy_ablations, get_legacy_for_universe,
)
from .plots import (
    period_bar_chart, four_period_grid,
    risk_metric_chart, risk_metrics_grid,
    period_focus_panel,
)
from .training_curves import (
    training_curves_chart, load_val_history,
)
from .concentration import (
    load_weights_from_legacy,
    concentration_timeseries,
    concentration_chart,
    effective_n_chart,
    average_allocation_chart,
    VARIANT_TO_LEGACY_SLUG,
)
from .tickers import get_tickers
