# ============================================
# PHASE 3: Portfolio Trading Environment
# ============================================
# This file creates a virtual stock market where
# the RL agent can practice trading.
#
# The agent:
#   1. SEES the current market state (prices, indicators, regime)
#   2. DECIDES portfolio weights (how much in each stock)
#   3. GETS a reward (profit minus risk penalty minus costs)
#   4. Moves to the next trading day
#   5. Repeats until the last day

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd


class PortfolioEnv(gym.Env):
    """
    Custom trading environment for portfolio management.
    
    This is like a DRIVING SIMULATOR but for stock trading.
    The agent practices here before we test it on real data.
    """
    
    # This tells Gymnasium what kind of rendering we support
    metadata = {'render_modes': ['human']}
    
    def __init__(
        self,
        df,                          # Stock data (normalized, with regime)
        initial_balance=1_000_000,   # Starting money: \$1,000,000
        transaction_cost=0.001,      # 0.1% cost per trade
        risk_penalties=None,         # λ values for each regime
        cash_floors=None,            # Minimum cash % per regime (dict {0,1,2}).
                                     # Default {0: 0.0, 1: 0.5, 2: 0.1}.
                                     # Used by SP30-v2 universe calibration —
                                     # different universes may need different
                                     # defensive floors.
        window_size=20,              # How many past days the agent sees
        ablation_mode=None,          # None | 'no_floor' | 'no_dsr' | 'vanilla'
        risk_lambda_fixed=None,      # If set, overrides regime-conditioned λ
                                     # (used by risk-personality agents)
        include_regime=True,         # If False, drop the regime one-hot from
                                     # the observation (used by the vanilla
                                     # baseline so the agent is regime-blind).
        include_vix=False,           # If True, expects a 'vix_norm' column in
                                     # df and adds the day's VIX z-score as
                                     # one extra observation dim. Default off
                                     # to preserve backward-compatibility with
                                     # every existing trainer.
        include_regime_probs=False,  # If True, appends 3 extra dims to the
                                     # observation: per-day HMM regime
                                     # probabilities (bull/bear/sideways).
                                     # Used by the MTL multi-agent architecture
                                     # for soft mixing of regime-specific heads.
                                     # Default off for backward-compatibility.
        include_vol_features=False,  # If True, expects columns 'vol_20d_norm',
                                     # 'vol_60d_norm', 'vol_ratio_norm' (rolling
                                     # vols of the broad-market index, z-score
                                     # normalised on the train period) and
                                     # appends them as 3 extra observation dims.
                                     # Used by the Universal v2 design — gives
                                     # the agent vol-regime context independent
                                     # of the HMM (Paper 6 JPM precedent).
                                     # Default off for backward-compatibility.
        use_markowitz_reward=False,  # If True, REPLACES the DSR + downside
                                     # penalty reward with Paper 7's Markowitz
                                     # mean-variance form:
                                     #   reward_t = port_return_t
                                     #            − markowitz_beta × σ_30d
                                     # σ_30d is the rolling 30-day std of the
                                     # agent's daily portfolio returns. Used by
                                     # Universal v3 to swap the smoothness-
                                     # rewarding DSR for a volatility-penalising
                                     # reward that's friendlier to volatile
                                     # winners (mega-cap tech in SP100).
        markowitz_beta=0.005,        # Risk-aversion coefficient β in Markowitz
                                     # reward. Paper 7 found 0.005 optimal on
                                     # S&P 100 (annual return 26.91% on DJIA,
                                     # 36.92% on S&P 100). Lower = more
                                     # aggressive, higher = more defensive.
        use_hyperheuristic=False,    # If True, switches the action space from
                                     # continuous portfolio weights (n_stocks+1
                                     # softmax-ed logits) to a DISCRETE strategy
                                     # index (Paper 8 Cui et al. 2024 — hyper-
                                     # heuristic DRL). Each action selects one
                                     # of K hand-crafted strategies from
                                     # models.strategies.STRATEGY_REGISTRY,
                                     # which then computes the actual stock
                                     # weights. Solves the action-space
                                     # dilution problem for large universes.
        use_cvar_penalty=False,      # MARS v3 BEAR DEFENDER: add a Conditional
                                     # VaR penalty to the reward. Encourages
                                     # tail-risk control. The reward becomes:
                                     #   reward = base_reward − cvar_lambda ×
                                     #            |mean(worst-5% daily returns
                                     #             over last 60 days)|
                                     # Specifically helps defender specialist
                                     # in bear regimes (Paper 5 MARS, Paper 9).
        cvar_lambda=0.5,             # Strength of CVaR penalty when
                                     # use_cvar_penalty=True. 0.5 = moderate.
        cvar_window=60,              # Rolling window size (days) for CVaR.
        cvar_alpha=0.05,             # Tail size — 0.05 means worst-5% returns.
        use_concentration_bonus=False, # MARS v3 STEADY / HUNTER: add a positive
                                     # bonus when the top-K portfolio weights
                                     # sum above a threshold. Encourages
                                     # concentrated bull-capture portfolios.
                                     # Universal (same params on all universes);
                                     # naturally adaptive because uniform
                                     # softmax over a larger action space is
                                     # farther from the threshold.
                                     #   bonus = λ × max(0, top_K_sum − thresh)
                                     # See Paper 9 concentration-reward
                                     # precedent.
        concentration_top_k=10,      # K = how many top stocks to sum.
        concentration_threshold=0.5, # Threshold sum above which bonus fires.
        concentration_lambda=0.1,    # Bonus weight. Steady uses 0.1, hunter
                                     # uses 0.3 (per-role, not per-universe).
        concentration_bull_only=False, # If True, the concentration bonus ONLY
                                     # fires when HMM detects bull regime
                                     # (regime_id == 0). Used by NDX bull-only
                                     # experiment — regime-blind concentration
                                     # (exp2) showed no effect because the
                                     # bear-regime concentration penalty cancelled
                                     # bull-regime gains. Bull-only gating tries
                                     # to isolate the bull-side mechanism.
        use_max_weight_penalty=False, # MARS v3 SAFEGUARD: penalize single-stock
                                     # dominance to prevent the BULL HUNTER's
                                     # degenerate single-stock concentration
                                     # collapse (e.g. policy puts 99% in one
                                     # ticker and rides it). Without this,
                                     # high concentration_lambda + zero risk
                                     # penalty lets the policy find a local
                                     # optimum that's not real diversification.
                                     #   penalty = λ × max(0, top_1_weight − cap)
        max_single_weight=0.20,      # Per-stock cap above which the penalty
                                     # fires. 0.20 = no single stock > 20%.
        max_weight_lambda=1.0,       # Penalty strength. 1.0 = strong; the
                                     # daily penalty in a 100%-MMM portfolio
                                     # would be ≈ 1.0 × (0.99 − 0.20) = 0.79,
                                     # vastly outweighing any concentration
                                     # bonus and any plausible single-day
                                     # return — pushes policy to diversify.
        top_k_action_mask=None,      # MARS v3 HUNTER v6: structural action-space
                                     # constraint. If set to integer K, the env
                                     # masks all but the top-K stock logits
                                     # BEFORE softmax. Result: portfolio holds
                                     # exactly K stocks (+ cash) with real
                                     # weights instead of 90-stock dilution.
                                     # Paper 8 (Cui et al. 2024) discrete-action
                                     # hyper-heuristic insight — continuous
                                     # version. The policy is still continuous
                                     # in logit space; we just clip non-top-K
                                     # to a very negative value so softmax
                                     # zeros them. None = no masking (default).
    ):
        """
        Initialize the environment.
        
        Think of this as SETTING UP the driving simulator
        before the student starts practicing.
        
        Parameters:
        -----------
        df : pandas DataFrame
            The stock data with all features and regime labels.
            Each row = one stock on one day.
            
        initial_balance : float
            How much money the agent starts with.
            
        transaction_cost : float
            Fee for buying/selling (0.001 = 0.1%)
            Real brokers charge this.
            
        risk_penalties : dict
            λ values for regime-conditioned reward.
            Higher λ = more punishment for risky behavior.
            
        window_size : int
            How many past days the agent can see.
            Like looking at the last 20 days of a stock chart.
        """
        super().__init__()
        
        # ---- Store parameters ----
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.window_size = window_size
        self.ablation_mode = ablation_mode  # None for full model
        # If set, overrides regime-conditioned λ. Used by risk-personality
        # multi-agent variants where each agent has a fixed risk attitude
        # regardless of detected regime. None means "regime-conditioned"
        # (the original Phase 5 baseline behavior).
        self.risk_lambda_fixed = risk_lambda_fixed
        # Whether to expose the regime one-hot in the observation.
        # The vanilla baseline sets this to False so the agent has no access
        # to HMM regime information through any channel.
        self.include_regime = include_regime
        # Whether to expose the day's normalised VIX as an extra observation
        # dim. When True the dataframe must include a 'vix_norm' column.
        self.include_vix = include_vix
        # Whether to expose the HMM regime probabilities (3 dims) as extra
        # observation dims. When True the dataframe must include
        # 'regime_bull_prob', 'regime_bear_prob', 'regime_sideways_prob'.
        self.include_regime_probs = include_regime_probs
        # Whether to expose 3 rolling-volatility features of the broad-market
        # index (vol_20d, vol_60d, vol_ratio = vol_20d/vol_60d). Z-score
        # normalised columns 'vol_20d_norm', 'vol_60d_norm', 'vol_ratio_norm'
        # must be present in df. Used by Universal v2 (Paper 6 JPM precedent).
        self.include_vol_features = include_vol_features
        # Markowitz mean-variance reward switch (Universal v3, Paper 7).
        # When True, the per-step reward becomes:
        #   reward_t = port_return_t  −  markowitz_beta × σ_30d
        # Replaces DSR + regime-λ downside penalty.
        self.use_markowitz_reward = use_markowitz_reward
        self.markowitz_beta = float(markowitz_beta)
        # Rolling 30-day window of agent's daily portfolio returns, used to
        # compute σ_30d for the Markowitz reward. Initialised on reset().
        self._port_return_window = None
        # Hyper-heuristic discrete-action mode (Universal v4, Paper 8).
        self.use_hyperheuristic = use_hyperheuristic
        # CVaR penalty (BEAR DEFENDER specialist, MARS v3).
        self.use_cvar_penalty = bool(use_cvar_penalty)
        self.cvar_lambda      = float(cvar_lambda)
        self.cvar_window      = int(cvar_window)
        self.cvar_alpha       = float(cvar_alpha)
        self._cvar_returns    = None   # Initialised on reset()
        # Concentration bonus (STEADY / HUNTER specialists, MARS v3).
        self.use_concentration_bonus  = bool(use_concentration_bonus)
        self.concentration_top_k      = int(concentration_top_k)
        self.concentration_threshold  = float(concentration_threshold)
        self.concentration_lambda     = float(concentration_lambda)
        self.concentration_bull_only  = bool(concentration_bull_only)
        # Max-weight penalty (safeguard against single-stock dominance).
        self.use_max_weight_penalty   = bool(use_max_weight_penalty)
        self.max_single_weight        = float(max_single_weight)
        self.max_weight_lambda        = float(max_weight_lambda)
        # Top-K action mask (HUNTER v6 structural concentration).
        self.top_k_action_mask        = (int(top_k_action_mask)
                                          if top_k_action_mask is not None
                                          else None)

        # Default risk penalties per regime
        # 0 = bull, 1 = bear, 2 = sideways
        if risk_penalties is None:
            self.risk_penalties = {
                0: 0.1,   # Bull:     low  penalty (be aggressive)
                1: 0.5,   # Bear:     high penalty (be defensive)
                2: 0.3,   # Sideways: medium penalty
            }
        else:
            self.risk_penalties = risk_penalties

        # ============================================================
        # REGIME-CONDITIONED CASH FLOOR (rule-based risk overlay)
        # ============================================================
        # After the policy proposes a portfolio, we enforce a MINIMUM
        # cash percentage that depends on the detected regime.
        # The agent's stock-picking decisions are PRESERVED — we only
        # rescale their total weight to leave room for the cash floor.
        #
        # Bull:     no floor — agent can be 100% in stocks
        # Sideways: 10% cash minimum
        # Bear:     50% cash minimum — direct structural defence
        #           against losses during slow grinding declines.
        #           Higher floors (0.75) tested but did not improve
        #           bear return (sideways days dominate 2022) — see
        #           thesis Section X discussion.
        if cash_floors is None:
            # Default — tuned for the DJI 30 universe.
            self.cash_floors = {
                0: 0.0,    # Bull
                1: 0.5,    # Bear
                2: 0.1,    # Sideways
            }
        else:
            # Universe-specific calibration — used by SP30 v2 etc.
            self.cash_floors = {int(k): float(v) for k, v in cash_floors.items()}

        # Ablation: disable the regime cash floor entirely
        # ('vanilla' implies both no_floor and no_dsr — see also the DSR
        # check inside step() below.)
        if ablation_mode in ('no_floor', 'vanilla'):
            self.cash_floors = {0: 0.0, 1: 0.0, 2: 0.0}
        
        # ---- Process the data ----
        self.df = df.copy()
        
        # Get list of unique stocks and dates
        self.tickers = sorted(df['ticker'].unique())
        self.n_stocks = len(self.tickers)
        self.dates = sorted(df['date'].unique())
        self.n_days = len(self.dates)
        
        # Define which columns are FEATURES the agent sees
        # (not date, ticker, or regime_name — those are metadata)
        self.feature_columns = [
            'close', 'open', 'high', 'low', 'volume',
            'rsi', 'macd', 'macd_signal', 'macd_diff',
            'bb_upper', 'bb_lower', 'bb_width',
            'atr', 'sma_20', 'sma_50', 'adx', 'cci',
            'daily_return', 'volatility_20d',
            # Universal v2 addition — z-score normalised 60-day per-stock
            # momentum. Only present in data/*.csv files that have been
            # passed through data/add_universal_features.py. Auto-skipped
            # by the existence filter below for legacy trainers.
            'mom_60d_norm',
            # Long-horizon momentum (NDX 2017 experimental extension).
            # Only present in data/*.csv files that have been passed
            # through data/add_long_momentum_features.py.
            'mom_120d_norm', 'mom_252d_norm',
            # Universal v3 additions (Paper 6 + Paper 2 inspired).
            # Multi-horizon return lags per stock (z-score normed):
            'r_lag_1_norm', 'r_lag_5_norm', 'r_lag_10_norm',
            'r_lag_20_norm', 'r_lag_40_norm', 'r_lag_60_norm',
            # Per-stock trend categories (1 = stock falls in that bucket
            # of today's cross-sectional 60-day momentum distribution):
            'is_uptrend', 'is_sideways', 'is_downtrend',
        ]
        # Only keep features that exist in the data
        self.feature_columns = [
            f for f in self.feature_columns if f in df.columns
        ]
        self.n_features = len(self.feature_columns)
        
        # Regime columns (one-hot)
        self.regime_columns = ['regime_bull', 'regime_bear', 'regime_sideways']
        
        # ---- Organize data into a 3D array for fast access ----
        # Shape: (n_days, n_stocks, n_features)
        # This way we can quickly get all stocks' data for any day
        self._organize_data()
        
        # ---- Define action space ----
        # Agent outputs UNBOUNDED raw scores (one per stock + cash).
        # The environment applies a SOFTMAX to convert them into a valid
        # portfolio (positive, summing to 1). This is cleaner than
        # clip+normalize, which collapsed many distinct policies into the
        # same portfolio and broke PPO's gradient.
        # Stable-Baselines3 requires finite bounds. ±10 is plenty of headroom
        # for softmax (exp(10)/exp(-10) ≈ 5e8, more than enough dynamic range).
        self.n_actions = self.n_stocks + 1  # +1 for cash
        if self.use_hyperheuristic:
            # Hyper-heuristic mode: agent picks 1 of K pre-defined strategies.
            from strategies import N_STRATEGIES
            self.action_space = spaces.Discrete(N_STRATEGIES)
        else:
            # Continuous portfolio weights (legacy + v1/v2/v3).
            self.action_space = spaces.Box(
                low=-10.0,
                high=10.0,
                shape=(self.n_actions,),
                dtype=np.float32
            )
        
        # ---- Define observation space ----
        # What the agent sees each step:
        # For each stock: n_features values
        # Plus: 3 regime values (one-hot)
        # Plus: current portfolio weights (n_stocks + 1)
        # Regime one-hot (3 dims) is included by default; the vanilla baseline
        # drops it via include_regime=False.
        regime_dims = 3 if self.include_regime else 0
        vix_dims    = 1 if self.include_vix    else 0
        prob_dims   = 3 if self.include_regime_probs else 0
        vol_dims    = 3 if self.include_vol_features else 0
        obs_size = (self.n_stocks * self.n_features) + regime_dims + vix_dims + prob_dims + vol_dims + self.n_actions
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_size,),
            dtype=np.float32
        )
        
        # ---- Initialize state variables ----
        self.current_step = 0
        self.portfolio_weights = np.zeros(self.n_actions, dtype=np.float32)
        self.portfolio_weights[-1] = 1.0  # Start with 100% cash
        self.portfolio_value = initial_balance
        self.portfolio_history = []

        # Differential Sharpe Ratio running estimates (Moody & Saffell, 1998)
        # A_t = EMA of returns; B_t = EMA of squared returns.
        # alpha controls the time horizon — 0.01 ≈ ~100-day half-life.
        self._dsr_A = 0.0
        self._dsr_B = 0.0
        self._dsr_alpha = 0.01
        
        print(f"Environment created! ✅")
        print(f"  Stocks: {self.n_stocks}")
        print(f"  Features per stock: {self.n_features}")
        print(f"  Trading days: {self.n_days}")
        print(f"  Observation size: {obs_size}")
        print(f"  Action size: {self.n_actions} (30 stocks + cash)")
        print(f"  Initial balance: ${initial_balance:,.0f}")
        print(f"  Transaction cost: {transaction_cost*100}%")
        print(f"  Risk penalties: bull={self.risk_penalties[0]}, "
              f"bear={self.risk_penalties[1]}, "
              f"sideways={self.risk_penalties[2]}")
        print(f"  Cash floors:    bull={self.cash_floors[0]}, "
              f"bear={self.cash_floors[1]}, "
              f"sideways={self.cash_floors[2]}")
        if self.ablation_mode is not None:
            print(f"  ⚠  Ablation mode active: {self.ablation_mode}")
        if not self.include_regime:
            print(f"  ⚠  Regime one-hot DROPPED from observation "
                  f"(agent is regime-blind)")
        if self.include_vix:
            print(f"  ⚙  VIX z-score ADDED to observation (+1 dim)")
        if self.include_regime_probs:
            print(f"  ⚙  HMM regime probabilities ADDED to observation (+3 dims)")
        if self.include_vol_features:
            print(f"  ⚙  Broad-market vol features (vol_20d/vol_60d/ratio) "
                  f"ADDED to observation (+3 dims)")
        if self.use_markowitz_reward:
            print(f"  ⚙  Markowitz mean-variance REWARD: "
                  f"r_t − β × σ_30d  (β={self.markowitz_beta})  "
                  f"— REPLACES DSR + downside penalty")
        if self.use_hyperheuristic:
            from strategies import N_STRATEGIES, strategy_names
            print(f"  ⚙  HYPER-HEURISTIC ACTION SPACE: "
                  f"agent picks 1 of {N_STRATEGIES} strategies "
                  f"({', '.join(strategy_names())})")
        if self.risk_lambda_fixed is not None:
            print(f"  ⚙  Fixed risk λ: {self.risk_lambda_fixed} "
                  f"(overrides regime-conditioned penalties)")
    
    def _organize_data(self):
        """
        Reorganize data from a flat table into structured arrays.
        
        BEFORE: One big table where each row is one stock on one day
        AFTER:  Organized arrays where we can quickly look up:
                "Give me all stocks' features for day 42"
        """
        # Create data array: (n_days, n_stocks, n_features)
        self.data_array = np.zeros(
            (self.n_days, self.n_stocks, self.n_features),
            dtype=np.float32
        )
        
        # Create regime array: (n_days, 3)
        self.regime_array = np.zeros((self.n_days, 3), dtype=np.float32)
        
        # Create regime ID array: (n_days,)
        self.regime_id_array = np.zeros(self.n_days, dtype=np.int32)
        
        # Create daily return array: (n_days, n_stocks)
        # Used to calculate portfolio return
        self.returns_array = np.zeros(
            (self.n_days, self.n_stocks), dtype=np.float32
        )

        # Optional: per-day VIX z-score array (n_days,).
        # Only populated if include_vix=True. If the column is missing the
        # env raises immediately so the experiment fails loudly, not silently.
        if self.include_vix:
            if 'vix_norm' not in self.df.columns:
                raise ValueError(
                    "include_vix=True but the dataframe has no 'vix_norm' "
                    "column. Run data/add_vix_feature.py first."
                )
            self.vix_array = np.zeros(self.n_days, dtype=np.float32)
        else:
            self.vix_array = None

        # Optional: per-day HMM regime probability array (n_days, 3).
        # Order: [bull_prob, bear_prob, sideways_prob].
        if self.include_regime_probs:
            required = ['regime_bull_prob', 'regime_bear_prob',
                        'regime_sideways_prob']
            missing = [c for c in required if c not in self.df.columns]
            if missing:
                raise ValueError(
                    f"include_regime_probs=True but dataframe missing "
                    f"columns {missing}. Run data/add_regime_probs_to_data.py."
                )
            self.regime_prob_array = np.zeros((self.n_days, 3), dtype=np.float32)
        else:
            self.regime_prob_array = None

        # Optional: per-day rolling-vol features of the broad-market index.
        # 3 dims: vol_20d_norm, vol_60d_norm, vol_ratio_norm.
        # Universe-specific (DJI uses DJI index vol, SP30/SP100 use S&P 500 vol),
        # already z-score normalised on the train period by the prep script.
        if self.include_vol_features:
            required = ['vol_20d_norm', 'vol_60d_norm', 'vol_ratio_norm']
            missing = [c for c in required if c not in self.df.columns]
            if missing:
                raise ValueError(
                    f"include_vol_features=True but dataframe missing "
                    f"columns {missing}. Run data/add_universal_features.py."
                )
            self.vol_features_array = np.zeros((self.n_days, 3), dtype=np.float32)
        else:
            self.vol_features_array = None

        # Vectorised population — replaces the previous triple-nested
        # per-day × per-stock × per-feature loop which made env init the
        # dominant bottleneck on SP100 (~10 min per env). The new path
        # does ~5 fancy-index assignments and finishes in seconds.
        # SAME arrays populated with SAME values, just way faster.

        # Index maps: date → 0..n_days-1, ticker → 0..n_stocks-1
        date_to_idx   = {d: i for i, d in enumerate(self.dates)}
        ticker_to_idx = {t: i for i, t in enumerate(self.tickers)}

        df = self.df.copy()
        df['_day_idx']   = df['date'].map(date_to_idx)
        df['_stock_idx'] = df['ticker'].map(ticker_to_idx)
        df = df.dropna(subset=['_day_idx', '_stock_idx']).copy()
        df['_day_idx']   = df['_day_idx'].astype(int)
        df['_stock_idx'] = df['_stock_idx'].astype(int)
        days   = df['_day_idx'].values
        stocks = df['_stock_idx'].values

        # Per-stock features → data_array[day, stock, feat]
        for feat_idx, feat in enumerate(self.feature_columns):
            if feat in df.columns:
                self.data_array[days, stocks, feat_idx] = \
                    df[feat].astype(np.float32).values

        # Daily returns → returns_array[day, stock]
        if 'daily_return' in df.columns:
            self.returns_array[days, stocks] = \
                df['daily_return'].astype(np.float32).values

        # Per-day globals (same value across tickers on a given day).
        # Take one row per date, then fancy-index by day_idx.
        day_first = (df.drop_duplicates(subset='_day_idx')
                       .sort_values('_day_idx'))
        di = day_first['_day_idx'].values

        for col_idx, col in enumerate(self.regime_columns):
            if col in day_first.columns:
                self.regime_array[di, col_idx] = \
                    day_first[col].astype(np.float32).values

        if 'regime' in day_first.columns:
            self.regime_id_array[di] = day_first['regime'].astype(int).values

        if self.include_vix and 'vix_norm' in day_first.columns:
            self.vix_array[di] = day_first['vix_norm'].astype(np.float32).values

        if self.include_regime_probs:
            self.regime_prob_array[di, 0] = \
                day_first['regime_bull_prob'].astype(np.float32).values
            self.regime_prob_array[di, 1] = \
                day_first['regime_bear_prob'].astype(np.float32).values
            self.regime_prob_array[di, 2] = \
                day_first['regime_sideways_prob'].astype(np.float32).values

        if self.include_vol_features:
            self.vol_features_array[di, 0] = \
                day_first['vol_20d_norm'].astype(np.float32).values
            self.vol_features_array[di, 1] = \
                day_first['vol_60d_norm'].astype(np.float32).values
            self.vol_features_array[di, 2] = \
                day_first['vol_ratio_norm'].astype(np.float32).values
    
    def reset(self, seed=None, options=None):
        """
        Reset the environment to the BEGINNING.
        
        Like pressing "restart" on a video game.
        The agent starts over with fresh money and no positions.
        
        Returns:
            observation: The first state the agent sees
            info: Extra information (for debugging)
        """
        super().reset(seed=seed)
        
        # Start from the beginning (after window_size days so we have history)
        self.current_step = self.window_size
        
        # Reset portfolio: 100% cash
        self.portfolio_weights = np.zeros(self.n_actions, dtype=np.float32)
        self.portfolio_weights[-1] = 1.0  # 100% cash

        # Reset portfolio value
        self.portfolio_value = self.initial_balance

        # Reset Markowitz rolling-vol window
        self._port_return_window = []
        # Reset CVaR rolling-return window (MARS v3 defender)
        self._cvar_returns = []

        # Reset history tracking
        self.portfolio_history = [{
            'step': self.current_step,
            'date': self.dates[self.current_step],
            'value': self.portfolio_value,
            'weights': self.portfolio_weights.copy(),
            'regime': int(self.regime_id_array[self.current_step]),
        }]

        # Reset Differential Sharpe Ratio running estimates
        self._dsr_A = 0.0
        self._dsr_B = 0.0
        
        # Get the first observation
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action):
        """
        Execute ONE trading day.
        
        This is the CORE function. Every time the agent makes a decision,
        this function:
          1. Takes the agent's action (new portfolio weights)
          2. Calculates transaction costs
          3. Moves to the next day
          4. Calculates the portfolio return
          5. Calculates the reward (return - risk penalty - costs)
          6. Returns the new state and reward
        
        Parameters:
            action: numpy array of shape (n_stocks + 1,)
                    Raw weights from the agent's neural network
        
        Returns:
            observation: New state after this day
            reward: How good/bad this day was
            terminated: Is the episode over?
            truncated: Was the episode cut short?
            info: Extra information
        """
        # ---- Step 1: Action → portfolio weights ----
        if self.use_hyperheuristic:
            # Universal v4 hyper-heuristic: action is a discrete index that
            # selects 1 of K pre-defined strategies. The strategy function
            # consumes today's state and produces weights directly.
            from strategies import apply_strategy
            step = min(self.current_step, self.n_days - 1)
            # Build the context the strategies need
            mom_scores       = None
            vol_scores       = None
            is_uptrend_mask  = None
            # Look up per-stock features for today by feature_columns index
            if 'mom_60d_norm' in self.feature_columns:
                idx = self.feature_columns.index('mom_60d_norm')
                mom_scores = self.data_array[step, :, idx]
            if 'volatility_20d' in self.feature_columns:
                idx = self.feature_columns.index('volatility_20d')
                vol_scores = self.data_array[step, :, idx]
            if 'is_uptrend' in self.feature_columns:
                idx = self.feature_columns.index('is_uptrend')
                is_uptrend_mask = self.data_array[step, :, idx]
            context = dict(
                regime_id=int(self.regime_id_array[step]),
                mom_scores=mom_scores,
                vol_scores=vol_scores,
                is_uptrend_mask=is_uptrend_mask,
            )
            strategy_idx = int(action) if np.isscalar(action) else int(np.asarray(action).flatten()[0])
            new_weights = apply_strategy(strategy_idx, self.n_stocks,
                                         self.n_actions, context)
            new_weights = np.array(new_weights, dtype=np.float32)
        else:
            # Continuous mode: softmax over raw scores → weights.
            # The agent outputs unbounded scores. Softmax converts them to a
            # proper probability distribution (positive, sums to 1).
            action = np.array(action, dtype=np.float32)
            action = np.nan_to_num(action, nan=0.0, posinf=10.0, neginf=-10.0)

            # ---- Optional Top-K Action Mask (HUNTER v6) ----
            # Force the policy to hold exactly K stocks + cash by zeroing
            # all but the top-K STOCK logits before softmax. This is the
            # continuous-space version of Paper 8 Cui's discrete-action
            # hyper-heuristic — solves the action-space dilution problem
            # in 90-stock universes by structural constraint.
            # The policy still freely picks WHICH K stocks each step.
            if self.top_k_action_mask is not None:
                k = min(self.top_k_action_mask, self.n_stocks)
                stock_logits = action[:-1]            # all but cash
                cash_logit   = action[-1]
                # Indices of top-K stock logits (highest values)
                top_k_idx = np.argpartition(stock_logits, -k)[-k:]
                # Build masked logits: keep top-K, suppress rest
                masked_stocks = np.full_like(stock_logits, -1e9)
                masked_stocks[top_k_idx] = stock_logits[top_k_idx]
                action = np.concatenate([masked_stocks,
                                          np.array([cash_logit])])

            action = action - action.max()           # numerical stability
            exp_action = np.exp(action)
            new_weights = exp_action / exp_action.sum()

        # ---- Step 1b: Regime-conditioned cash-floor overlay ----
        # If the floor for today's regime is higher than the agent's
        # proposed cash weight, scale down stocks proportionally and
        # push the rest into cash. The relative ordering of stocks is
        # preserved (the agent's stock PICKS still matter).
        regime_id_at_decision = int(self.regime_id_array[self.current_step])
        cash_floor = self.cash_floors.get(regime_id_at_decision, 0.0)
        if new_weights[-1] < cash_floor:
            current_stocks_total = new_weights[:-1].sum()
            if current_stocks_total > 1e-8:
                target_stocks_total = 1.0 - cash_floor
                new_weights[:-1] *= (target_stocks_total / current_stocks_total)
            new_weights[-1] = cash_floor
        
        # ---- Step 2: Calculate transaction costs ----
        # Cost = how much the weights CHANGED × cost rate
        # If you change from 10% AAPL to 30% AAPL, 
        # you pay 0.1% × |30%-10%| = 0.1% × 20% = 0.02%
        weight_change = np.abs(new_weights - self.portfolio_weights)
        turnover = weight_change.sum()  # Total change in weights
        cost = turnover * self.transaction_cost
        
        # ---- Step 3: Update portfolio weights ----
        old_weights = self.portfolio_weights.copy()
        self.portfolio_weights = new_weights
        
        # ---- Step 4: Move to next day ----
        self.current_step += 1
        
        # Check if we've reached the end
        terminated = self.current_step >= self.n_days - 1
        
        if terminated:
            # At the end, return final state
            observation = self._get_observation()
            reward = 0.0
            info = self._get_info()
            return observation, reward, terminated, False, info
        
        # ---- Step 5: Calculate portfolio return ----
        # Get today's returns for all stocks
        stock_returns = self.returns_array[self.current_step]
        
        # Portfolio return = weighted sum of stock returns
        # Cash return = 0 (cash doesn't grow or shrink)
        # stock weights are new_weights[0:n_stocks]
        # cash weight is new_weights[-1]
        stock_weights = new_weights[:-1]  # All except cash
        portfolio_return = np.dot(stock_weights, stock_returns)
        
        # ---- Step 6: Update portfolio value ----
        # Subtract costs, add returns
        self.portfolio_value *= (1 + portfolio_return - cost)
        
        # ---- Step 7: Calculate reward ----
        # Reward = Differential Sharpe Ratio (DSR)
        #          − λ(regime) × downside loss
        #          − transaction cost
        #
        # DSR (Moody & Saffell 1998) measures the marginal contribution of
        # today's return to the running Sharpe ratio — a much stronger
        # learning signal than raw daily return.
        # The regime-conditioned downside penalty preserves the thesis
        # novelty: λ depends on detected regime (bull=0.1, sideways=0.3,
        # bear=0.5). Only downside is penalised — gains are never punished.

        regime_id = int(self.regime_id_array[self.current_step])
        # Risk-personality override: if a fixed λ is set, use it regardless
        # of detected regime. Otherwise use regime-conditioned λ.
        if self.risk_lambda_fixed is not None:
            risk_lambda = float(self.risk_lambda_fixed)
        else:
            risk_lambda = self.risk_penalties.get(regime_id, 0.3)

        # Universal v3 Markowitz reward path. Replaces DSR + downside penalty
        # with Paper 7's mean-variance form:
        #   reward = port_return  −  β × σ_30d  −  cost
        # σ_30d = rolling 30-day std of agent's daily portfolio returns.
        # On startup (fewer than 30 observations) σ_30d is computed on whatever
        # history exists; this matches Paper 7's online estimator.
        if self.use_markowitz_reward:
            # Maintain rolling window of past 30 daily portfolio returns
            if self._port_return_window is None:
                self._port_return_window = []
            self._port_return_window.append(float(portfolio_return))
            if len(self._port_return_window) > 30:
                self._port_return_window.pop(0)
            if len(self._port_return_window) >= 2:
                sigma_30d = float(np.std(self._port_return_window))
            else:
                sigma_30d = 0.0
            reward = portfolio_return - self.markowitz_beta * sigma_30d - cost
        else:
            # Legacy DSR + downside penalty reward (FULL / v2 / ablations).
            # Ablation: 'no_dsr' swaps the DSR term for raw daily return
            # (keeps regime-conditioned downside penalty as the control variable
            # so the test isolates the reward shape itself).
            if self.ablation_mode in ('no_dsr', 'vanilla'):
                reward_signal = portfolio_return
                # Still update the DSR running estimates so behaviour stays
                # comparable if logging reads them. Discard the value.
                self._compute_dsr(portfolio_return)
            else:
                reward_signal = self._compute_dsr(portfolio_return)
            downside = max(0.0, -portfolio_return)
            reward = reward_signal - risk_lambda * downside - cost

        # ---- Optional CVaR penalty (MARS v3 BEAR DEFENDER) ----
        # Track the most recent `cvar_window` daily returns and subtract a
        # penalty proportional to the magnitude of the worst-α% tail.
        # Specifically rewards tail-risk avoidance — pulls the defensive
        # specialist toward portfolios that don't suffer large drawdowns.
        if self.use_cvar_penalty:
            self._cvar_returns.append(float(portfolio_return))
            if len(self._cvar_returns) > self.cvar_window:
                self._cvar_returns.pop(0)
            if len(self._cvar_returns) >= 10:
                arr = np.array(self._cvar_returns)
                k = max(1, int(np.ceil(self.cvar_alpha * len(arr))))
                tail = np.sort(arr)[:k]   # worst-α% returns
                cvar = float(np.mean(tail))
                if cvar < 0:
                    reward = reward - self.cvar_lambda * abs(cvar)

        # ---- Optional concentration bonus (MARS v3 STEADY / HUNTER) ----
        # Adds a positive reward when the policy's top-K stock weights sum
        # above the UNIFORM BASELINE for this universe.  The baseline is
        # auto-computed as K / n_stocks so it adapts naturally per universe:
        #   DJI   (30 stocks): baseline = 10/30 = 0.333
        #   SP100 (90 stocks): baseline = 10/90 = 0.111
        # The bonus fires from training step 1 as soon as the policy
        # concentrates ABOVE its universe's natural uniform-softmax level.
        # Replaces the earlier fixed-threshold (>0.5) version, which created
        # a dead zone on SP100 where the bonus never fired during training.
        # Universal recipe: same top_k / lambda on every universe; the
        # baseline derives from a structural property (universe size).
        # The optional `concentration_threshold` is kept for backward compat
        # but now ONLY raises the effective floor: bonus fires above
        # max(uniform_baseline, threshold) — so threshold=0 means pure
        # baseline-relative bonus.
        if self.use_concentration_bonus:
            # Bull-only gating: skip the bonus outside bull regime if requested.
            # regime_id was computed earlier in step() (post-action).
            apply_bonus = True
            if self.concentration_bull_only and regime_id != 0:
                apply_bonus = False
            if apply_bonus:
                stock_weights = self.portfolio_weights[:-1]  # exclude cash
                k = min(self.concentration_top_k, len(stock_weights))
                top_k_sum = float(np.sort(stock_weights)[-k:].sum())
                uniform_baseline = k / self.n_stocks
                effective_floor = max(uniform_baseline, self.concentration_threshold)
                if top_k_sum > effective_floor:
                    reward = reward + self.concentration_lambda * (
                        top_k_sum - effective_floor
                    )

        # ---- Optional max-weight penalty (MARS v3 anti-collapse safeguard) ----
        # Without this, high concentration_lambda + low risk penalty can let
        # the policy collapse into a degenerate single-stock bet (e.g. MMM
        # 99% — observed in hunter v2 training). This penalty makes such
        # collapse strictly negative-expected-value.
        if self.use_max_weight_penalty:
            stock_weights = self.portfolio_weights[:-1]
            top1 = float(stock_weights.max())
            if top1 > self.max_single_weight:
                reward = reward - self.max_weight_lambda * (
                    top1 - self.max_single_weight
                )

        # ---- Step 8: Record history ----
        self.portfolio_history.append({
            'step': self.current_step,
            'date': self.dates[self.current_step],
            'value': self.portfolio_value,
            'weights': self.portfolio_weights.copy(),
            'regime': regime_id,
            'return': portfolio_return,
            'cost': cost,
            'reward': reward,
        })
        
        # ---- Step 9: Get new observation ----
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, reward, terminated, False, info
    
    def _compute_dsr(self, portfolio_return):
        """
        Differential Sharpe Ratio (Moody & Saffell, 1998).

        Tracks running EMAs of returns (A) and squared returns (B), then
        returns the marginal change in Sharpe caused by today's return.

        This gives a naturally-scaled signal (~ ±0.1 per step in steady
        state) that aligns directly with our evaluation metric.
        """
        R_t = float(portfolio_return)
        A_prev = self._dsr_A
        B_prev = self._dsr_B
        alpha = self._dsr_alpha

        delta_A = R_t - A_prev
        delta_B = R_t * R_t - B_prev

        denom = B_prev - A_prev * A_prev
        if denom > 1e-6:
            dsr = (B_prev * delta_A - 0.5 * A_prev * delta_B) / (denom ** 1.5)
            dsr = float(np.clip(dsr, -1.0, 1.0))   # safety clip
        else:
            # Not enough variance accumulated yet — fall back to raw return
            dsr = R_t

        # Update EMAs for next step
        self._dsr_A = A_prev + alpha * delta_A
        self._dsr_B = B_prev + alpha * delta_B

        return dsr

    def _get_observation(self):
        """
        Build the state vector that the agent sees.
        
        The observation includes:
          1. Current day's features for ALL stocks (flattened)
          2. Current regime (one-hot: [bull, bear, sideways])
          3. Current portfolio weights
          
        All combined into one flat vector.
        """
        step = min(self.current_step, self.n_days - 1)
        
        # Get today's features for all stocks
        # Shape: (n_stocks, n_features) → flatten to 1D
        stock_features = self.data_array[step].flatten()
        
        # Current portfolio weights
        weights = self.portfolio_weights

        # Build the observation. The vanilla baseline omits the regime
        # one-hot so the agent is truly regime-blind (no information about
        # the HMM's output reaches it through any channel).
        # If include_vix=True, append the day's VIX z-score as one extra dim
        # so the agent can see fast stress information alongside (or instead
        # of) the slow regime label.
        parts = [stock_features]
        if self.include_regime:
            parts.append(self.regime_array[step])    # regime one-hot (3,)
        if self.include_vix:
            parts.append(np.array([self.vix_array[step]], dtype=np.float32))
        if self.include_regime_probs:
            parts.append(self.regime_prob_array[step])  # HMM probs (3,)
        if self.include_vol_features:
            parts.append(self.vol_features_array[step])  # vol features (3,)
        parts.append(weights)
        observation = np.concatenate(parts).astype(np.float32)
        
        # Replace any NaN or inf with 0
        observation = np.nan_to_num(observation, nan=0.0, posinf=0.0, neginf=0.0)
        
        return observation
    
    def _get_info(self):
        """
        Return extra information (useful for debugging and analysis).
        """
        step = min(self.current_step, self.n_days - 1)
        regime_id = int(self.regime_id_array[step])
        regime_names = {0: 'bull', 1: 'bear', 2: 'sideways'}
        
        return {
            'step': self.current_step,
            'date': str(self.dates[step]),
            'portfolio_value': self.portfolio_value,
            'regime': regime_names.get(regime_id, 'unknown'),
            'regime_id': regime_id,
        }
    
    def get_portfolio_history(self):
        """
        Return the complete trading history.
        Useful for plotting and analysis after an episode.
        """
        return pd.DataFrame(self.portfolio_history)


# ============================================
# TEST THE ENVIRONMENT
# ============================================
# This section runs when you execute this file directly.
# It creates the environment and runs a few test steps
# to make sure everything works.

if __name__ == "__main__":
    
    print("=" * 60)
    print("   PHASE 3: Testing Portfolio Environment")
    print("=" * 60)
    
    # Load training data
    print("\n--- Loading data ---\n")
    train_df = pd.read_csv('data/train_data.csv')
    train_df['date'] = pd.to_datetime(train_df['date'])
    print(f"Loaded: {len(train_df)} rows, {train_df['ticker'].nunique()} stocks")
    
    # Create environment
    print("\n--- Creating environment ---\n")
    env = PortfolioEnv(
        df=train_df,
        initial_balance=1_000_000,
        transaction_cost=0.001,
    )
    
    # Test reset
    print("\n--- Testing reset ---\n")
    obs, info = env.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Starting info: {info}")
    
    # Test a few random steps
    print("\n--- Testing random actions (5 days) ---\n")
    
    for day in range(5):
        # Random action: random portfolio weights
        random_action = np.random.random(env.n_actions)
        
        # Take a step
        obs, reward, terminated, truncated, info = env.step(random_action)
        
        print(f"  Day {day+1}:")
        print(f"    Date: {info['date']}")
        print(f"    Regime: {info['regime']}")
        print(f"    Portfolio value: ${info['portfolio_value']:,.2f}")
        print(f"    Reward: {reward:.6f}")
        print(f"    Terminated: {terminated}")
        
        if terminated:
            break
    
    # Test running a FULL episode with equal weights
    print("\n--- Testing full episode (equal weights) ---\n")
    
    obs, info = env.reset()
    total_reward = 0
    steps = 0
    
    # Equal weight: put equal money in all stocks
    equal_weights = np.ones(env.n_actions, dtype=np.float32)
    equal_weights = equal_weights / equal_weights.sum()
    
    while True:
        obs, reward, terminated, truncated, info = env.step(equal_weights)
        total_reward += reward
        steps += 1
        
        if terminated:
            break
    
    # Get history
    history = env.get_portfolio_history()
    
    final_value = info['portfolio_value']
    total_return = (final_value - 1_000_000) / 1_000_000 * 100
    
    print(f"  Total steps: {steps}")
    print(f"  Final value: ${final_value:,.2f}")
    print(f"  Total return: {total_return:.2f}%")
    print(f"  Total reward: {total_reward:.4f}")
    print(f"  Start date: {history['date'].iloc[0]}")
    print(f"  End date: {history['date'].iloc[-1]}")
    
    # Quick regime breakdown
    if 'regime' in history.columns:
        regime_counts = history['regime'].value_counts()
        regime_map = {0: 'bull', 1: 'bear', 2: 'sideways'}
        print(f"\n  Regime breakdown during episode:")
        for regime_id, count in regime_counts.items():
            name = regime_map.get(regime_id, 'unknown')
            print(f"    {name}: {count} days")
    
    print("\n" + "=" * 60)
    print("   PHASE 3 COMPLETE! Environment works! ✅")
    print("=" * 60)
    print(f"""
WHAT WE BUILT:
──────────────
  ✅ Custom Gymnasium environment (PortfolioEnv)
  ✅ State: stock features + regime + portfolio weights
  ✅ Action: continuous portfolio weights (30 stocks + cash)
  ✅ Reward: regime-conditioned (return - λ×risk - cost)
  ✅ Transaction costs modeled
  ✅ Portfolio history tracking
  ✅ Tested with random and equal-weight strategies

NEXT: Phase 4 - Attention Network + PPO Training
""")