"""
Train MAC (Meta-Adaptive Controller) — MARS-Pareto v2.

Universal trainer: same architecture, same recipe, same hyperparams for DJI
and SP100. Two specialist lineups supported:

  --lineup v6 (default): all 3 specialists are v6 checkpoints (defensive=
                          step 25k, balanced=mid, aggressive=late). Single
                          master env. Same architecture as v1.

  --lineup v1c:           3 DISTINCT training recipes:
                            defensive  = v1 FULL  (legacy + cash-floor)
                            balanced   = v2       (regime-λ + bear-floor)
                            aggressive = v6       (no-floor)
                          Per-role envs + cash-floor preconditioning.

Optionally pre-trains the MAC supervised on HMM-direct routing targets so
fine-tuning starts from a sensible point (mimics v1's regime-conditional
routing).

Usage:
  python3 models/train_mac.py --universe dji --lineup v1c --warm_start
  python3 models/train_mac.py --universe sp100 --lineup v1c --warm_start

Saves (under --lineup v1c, with --warm_start):
  models/saved/ppo_mac_{universe}_v1c.zip                       # best-by-val-Sharpe
  models/saved/checkpoints/mac_{universe}_v1c/step_*.zip        # every 25k
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from stable_baselines3 import PPO
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from mac_meta_env import MACMetaEnv
from mac_meta_env_v1c import MACMetaEnvV1C
from mac_meta_env_v3 import (
    MACMetaEnvV3,
    SPECIALISTS_DJI_V3, SPECIALISTS_SP100_V3,
    V3_TRAINING_FLOORS,
)
from mac_meta_env_v5 import (
    MACMetaEnvV5,
    SPECIALISTS_DJI_V5, SPECIALISTS_SP100_V5,
    V5_TRAINING_FLOORS,
)
from mac_meta_env_v6 import (
    MACMetaEnvV6,
    SPECIALISTS_DJI_V6_ABL, SPECIALISTS_SP100_V6_ABL,
    V6_ABL_TRAINING_FLOORS,
)
from mac_meta_env_v7 import (
    MACMetaEnvV7,
    SPECIALISTS_DJI_V7_ABL, SPECIALISTS_SP100_V7_ABL,
    V7_ABL_TRAINING_FLOORS,
)
from mac_obs_wrappers import ShuffleRegimeMACObs, NoVolMACObs
from train_final_extended import (
    DetailedCallback,
    ValidationCheckpointCallback,
)
from mars_pareto_controller import (
    SPECIALISTS_DJI, SPECIALISTS_SP100,
    SPECIALISTS_DJI_V1C, SPECIALISTS_SP100_V1C,
    V1C_ENV_CONFIG_DJI, V1C_ENV_CONFIG_SP100,
    V1C_TRAINING_FLOORS_DJI, V1C_TRAINING_FLOORS_SP100,
)


V3_DROP = ['r_lag_1_norm','r_lag_5_norm','r_lag_10_norm',
           'r_lag_20_norm','r_lag_40_norm','r_lag_60_norm',
           'is_uptrend','is_sideways','is_downtrend']


def load(data_dir, name, drop_v3=True):
    df = pd.read_csv(os.path.join(data_dir, name))
    df['date'] = pd.to_datetime(df['date'])
    if drop_v3:
        df = df.drop(columns=[c for c in V3_DROP if c in df.columns])
    return df


def supervised_warm_start(model, train_env, n_samples=4096, lr=3e-3,
                           epochs=20, batch_size=128, verbose=True,
                           target_mode='soft'):
    """Pre-train MAC's policy to output HMM-derived routing targets.

    Auto-detects 3-output (v3/v4: def/bal/agg) vs 2-output (v5: def/agg) MACs
    from the model's action_space.

    target_mode='soft' (v3 default for 3-out, v5 default for 2-out):
        3-out: target = (bear_p, sideways_p, bull_p)
        2-out: target = (bear_p, bull_p) normalized (drops sideways)

    target_mode='one_hot' (v4 / v5 sharper variant):
        target = one-hot on HMM argmax over included regimes.
    """
    n_out = int(model.action_space.shape[0])  # 3 for v3/v4, 2 for v5
    if verbose:
        print(f"\n--- Supervised warm-start ({target_mode}, n_out={n_out}): "
              f"{n_samples} samples × {epochs} epochs ---")
    obs_buf = []
    tgt_buf = []
    obs, _ = train_env.reset()
    for _ in range(n_samples):
        # MAC's obs[0:3] is [bull_p, bear_p, sideways_p]
        bull_p, bear_p, side_p = float(obs[0]), float(obs[1]), float(obs[2])
        if n_out == 3:
            # 3-output: (def, bal, agg) ↔ (bear, sideways, bull)
            if target_mode == 'one_hot':
                regime_idx = int(np.argmax([bull_p, bear_p, side_p]))
                target_idx = {0: 2, 1: 0, 2: 1}[regime_idx]
                target = np.full(3, 1e-3, dtype=np.float32)
                target[target_idx] = 1.0 - 2e-3
            else:
                target = np.array([bear_p, side_p, bull_p], dtype=np.float32)
                target = np.maximum(target, 1e-3)
                target = target / target.sum()
        else:
            # 2-output (v5): (def, agg) ↔ (bear, bull); sideways prob is dropped
            # On sideways days, the bull/bear residual lean drives routing.
            if target_mode == 'one_hot':
                # argmax over bull and bear only
                target_idx = 0 if bear_p >= bull_p else 1
                target = np.full(2, 1e-3, dtype=np.float32)
                target[target_idx] = 1.0 - 1e-3
            else:
                target = np.array([bear_p, bull_p], dtype=np.float32)
                target = np.maximum(target, 1e-3)
                target = target / target.sum()
        obs_buf.append(obs.copy())
        tgt_buf.append(target)
        # Step with uniform action just to advance time
        obs, _, term, trunc, _ = train_env.step(
            np.array([0.0, 0.0, 0.0], dtype=np.float32))
        if term or trunc:
            obs, _ = train_env.reset()

    X = torch.tensor(np.array(obs_buf), dtype=torch.float32)
    Y = torch.tensor(np.array(tgt_buf), dtype=torch.float32)

    policy = model.policy
    optim = torch.optim.Adam(policy.parameters(), lr=lr)

    for ep in range(epochs):
        idx = torch.randperm(len(X))
        ep_loss = 0.0
        n_batches = 0
        for i in range(0, len(X), batch_size):
            batch_idx = idx[i:i+batch_size]
            obs_b = X[batch_idx]
            tgt_b = Y[batch_idx]
            features = policy.extract_features(obs_b)
            if isinstance(features, tuple):
                features = features[0]  # SB3 >=2 returns (pi_feats, vf_feats)
            latent_pi, _ = policy.mlp_extractor(features)
            mean_actions = policy.action_net(latent_pi)
            pred = torch.softmax(mean_actions, dim=-1)
            loss = -(tgt_b * torch.log(pred + 1e-8)).sum(dim=-1).mean()
            optim.zero_grad()
            loss.backward()
            optim.step()
            ep_loss += loss.item()
            n_batches += 1
        if verbose and (ep % 5 == 0 or ep == epochs-1):
            print(f"  epoch {ep+1:>2}/{epochs}  loss={ep_loss/n_batches:.4f}")
    if verbose:
        print("✅ Warm-start done — MAC now mimics HMM-direct routing")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--universe', choices=['dji','sp100'], required=True)
    parser.add_argument('--variant',  choices=['v2','v3','v4','v5','v6','v7'], default=None,
                        help='Production / ablation presets: '
                             'v2 = lineup v1c + warm-start (legacy production). '
                             'v3 = lineup v3 + soft  warm-start (MARS v3 final). '
                             'v4 = lineup v3 + one-hot warm-start (B-1 sharper routing). '
                             'v5 = lineup v5 (drop balanced) + soft warm-start (ablation). '
                             'v6 = lineup v6 (drop defender) + soft warm-start (ablation). '
                             'v7 = lineup v7 (drop hunter)   + soft warm-start (ablation). '
                             'Overrides --lineup / --warm_start / --warm_target.')
    parser.add_argument('--lineup',   choices=['v6','v1c','v3','v5','v6abl','v7abl'], default='v6')
    parser.add_argument('--warm_start', action='store_true',
                        help='Pre-train MAC supervised on HMM routing targets')
    parser.add_argument('--no_warm_start', action='store_true',
                        help='Tier-1 ablation A: skip supervised warm-start entirely')
    parser.add_argument('--warm_target', choices=['soft','one_hot'], default='soft',
                        help='Warm-start target: soft HMM probs (v3) or one-hot argmax (v4)')
    parser.add_argument('--shuffle_regime', action='store_true',
                        help='Tier-1 ablation B: scramble MAC obs[:3] (HMM regime probs) each step')
    parser.add_argument('--no_vol', action='store_true',
                        help='Tier-1 ablation C: drop vol features from MAC obs')
    parser.add_argument('--ent_coef', type=float, default=0.01,
                        help='Tier-1 ablation F: PPO entropy coefficient (default 0.01)')
    parser.add_argument('--total_steps', type=int, default=250_000)
    parser.add_argument('--eval_freq',   type=int, default=25_000)
    parser.add_argument('--seed',        type=int, default=42)
    args = parser.parse_args()

    # Production presets:
    if args.variant == 'v2':
        args.lineup = 'v1c'
        args.warm_start = True
    elif args.variant == 'v3':
        args.lineup = 'v3'
        args.warm_start = True
        args.warm_target = 'soft'
    elif args.variant == 'v4':
        args.lineup = 'v3'
        args.warm_start = True
        args.warm_target = 'one_hot'
    elif args.variant == 'v5':
        args.lineup = 'v5'
        args.warm_start = True
        args.warm_target = 'soft'
    elif args.variant == 'v6':
        args.lineup = 'v6abl'
        args.warm_start = True
        args.warm_target = 'soft'
    elif args.variant == 'v7':
        args.lineup = 'v7abl'
        args.warm_start = True
        args.warm_target = 'soft'

    # --no_warm_start overrides everything (Tier-1 ablation A)
    if args.no_warm_start:
        args.warm_start = False

    base_dir    = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(base_dir)
    data_dir    = os.path.join(project_dir, 'data')
    saved_dir   = os.path.join(base_dir, 'saved')

    # Resolve filename suffix. The production combo (lineup=v1c + warm_start)
    # is canonicalised to "_v2" so on-disk artefacts use the thesis name.
    if args.lineup == 'v3' and args.warm_start and args.warm_target == 'one_hot':
        suffix = '_v4'
    elif args.lineup == 'v3' and args.warm_start:
        suffix = '_v3'
    elif args.lineup == 'v5' and args.warm_start:
        suffix = '_v5'
    elif args.lineup == 'v6abl' and args.warm_start:
        suffix = '_v6'
    elif args.lineup == 'v7abl' and args.warm_start:
        suffix = '_v7'
    elif args.lineup == 'v1c' and args.warm_start:
        suffix = '_v2'
    elif args.lineup == 'v3' and not args.warm_start:
        # Tier-1 ablation A: v3 lineup but no warm-start
        suffix = '_v3'
    else:
        suffix = f"_{args.lineup}" if args.lineup != 'v6' else ''
        if args.warm_start:
            suffix += '_ws'

    # Append ablation modifiers (Tier-1 A/B/C/F)
    if args.no_warm_start:
        suffix += '_nowarm'
    if args.shuffle_regime:
        suffix += '_shuf'
    if args.no_vol:
        suffix += '_novol'
    if abs(args.ent_coef - 0.01) > 1e-6:
        suffix += f'_ent{int(round(args.ent_coef*100)):02d}'
    # Seed tag: only append when non-default (keeps existing seed=42 production paths)
    if args.seed != 42:
        suffix += f'_s{args.seed}'
    checkpoint_dir = os.path.join(saved_dir, 'checkpoints',
                                   f'mac_{args.universe}{suffix}')
    results_dir   = os.path.join(project_dir, 'results',
                                  f'mac_{args.universe}{suffix}')
    os.makedirs(saved_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    if args.universe == 'dji':
        train_csv, val_csv = 'train_data_ext.csv', 'val_data_ext.csv'
        specs_v6  = SPECIALISTS_DJI
        specs_v1c = SPECIALISTS_DJI_V1C
        specs_v3  = SPECIALISTS_DJI_V3
        specs_v5  = SPECIALISTS_DJI_V5
        specs_v6abl = SPECIALISTS_DJI_V6_ABL
        specs_v7abl = SPECIALISTS_DJI_V7_ABL
        env_cfg   = V1C_ENV_CONFIG_DJI
        floors    = V1C_TRAINING_FLOORS_DJI
    else:
        train_csv, val_csv = 'train_data_sp100.csv', 'val_data_sp100.csv'
        specs_v6  = SPECIALISTS_SP100
        specs_v1c = SPECIALISTS_SP100_V1C
        specs_v3  = SPECIALISTS_SP100_V3
        specs_v5  = SPECIALISTS_SP100_V5
        specs_v6abl = SPECIALISTS_SP100_V6_ABL
        specs_v7abl = SPECIALISTS_SP100_V7_ABL
        env_cfg   = V1C_ENV_CONFIG_SP100
        floors    = V1C_TRAINING_FLOORS_SP100

    env_kwargs = dict(
        initial_balance=1_000_000,
        cash_floors={0:0.0, 1:0.0, 2:0.0},
        risk_penalties={0:0.1, 1:0.5, 2:0.3},
        include_regime=True,
    )

    print("=" * 72)
    print(f"   MAC ({args.universe.upper()}) — lineup={args.lineup}, "
          f"warm_start={args.warm_start}")
    print(f"   {args.total_steps:,} steps, checkpoint every {args.eval_freq:,}")
    print("=" * 72)

    if args.lineup == 'v3':
        # MARS v3 final: single master env (all 3 specialists share obs shape)
        # + per-specialist cash-floor preconditioning
        specs = {k: v for k, v in specs_v3.items()}
        train_df = load(data_dir, train_csv, drop_v3=True)
        val_df   = load(data_dir, val_csv,   drop_v3=True)
        env_kwargs.update(include_regime_probs=True, include_vol_features=True)
        train_env = MACMetaEnvV3(train_df, specs, env_kwargs, V3_TRAINING_FLOORS)
        val_env   = MACMetaEnvV3(val_df,   specs, env_kwargs, V3_TRAINING_FLOORS)
    elif args.lineup == 'v5':
        # Ablation v5: drop balanced specialist → 2-way (def/agg) router.
        specs = {k: v for k, v in specs_v5.items()}
        train_df = load(data_dir, train_csv, drop_v3=True)
        val_df   = load(data_dir, val_csv,   drop_v3=True)
        env_kwargs.update(include_regime_probs=True, include_vol_features=True)
        train_env = MACMetaEnvV5(train_df, specs, env_kwargs, V5_TRAINING_FLOORS)
        val_env   = MACMetaEnvV5(val_df,   specs, env_kwargs, V5_TRAINING_FLOORS)
    elif args.lineup == 'v6abl':
        # Ablation v6: drop defender specialist → 2-way (bal/agg) router.
        specs = {k: v for k, v in specs_v6abl.items()}
        train_df = load(data_dir, train_csv, drop_v3=True)
        val_df   = load(data_dir, val_csv,   drop_v3=True)
        env_kwargs.update(include_regime_probs=True, include_vol_features=True)
        train_env = MACMetaEnvV6(train_df, specs, env_kwargs, V6_ABL_TRAINING_FLOORS)
        val_env   = MACMetaEnvV6(val_df,   specs, env_kwargs, V6_ABL_TRAINING_FLOORS)
    elif args.lineup == 'v7abl':
        # Ablation v7: drop hunter specialist → 2-way (def/bal) router.
        specs = {k: v for k, v in specs_v7abl.items()}
        train_df = load(data_dir, train_csv, drop_v3=True)
        val_df   = load(data_dir, val_csv,   drop_v3=True)
        env_kwargs.update(include_regime_probs=True, include_vol_features=True)
        train_env = MACMetaEnvV7(train_df, specs, env_kwargs, V7_ABL_TRAINING_FLOORS)
        val_env   = MACMetaEnvV7(val_df,   specs, env_kwargs, V7_ABL_TRAINING_FLOORS)
    elif args.lineup == 'v1c':
        specs = {k:v for k,v in specs_v1c.items()
                 if k in ('defensive','balanced','aggressive')}
        train_df = load(data_dir, train_csv, drop_v3=False)
        val_df   = load(data_dir, val_csv,   drop_v3=False)
        train_env = MACMetaEnvV1C(train_df, specs, env_kwargs, env_cfg, floors)
        val_env   = MACMetaEnvV1C(val_df,   specs, env_kwargs, env_cfg, floors)
    else:  # v6 (legacy)
        specs = {k:v for k,v in specs_v6.items()
                 if k in ('defensive','balanced','aggressive')}
        train_df = load(data_dir, train_csv, drop_v3=True)
        val_df   = load(data_dir, val_csv,   drop_v3=True)
        env_kwargs.update(include_regime_probs=True, include_vol_features=True)
        train_env = MACMetaEnv(train_df, specs, env_kwargs)
        val_env   = MACMetaEnv(val_df,   specs, env_kwargs)

    for role, p in specs.items():
        print(f"   specialist {role:<11} = {os.path.basename(p)}")

    # Tier-1 obs-side ablations B / C — only modify the MAC controller's input.
    # Applied AFTER env construction so master env + specialists see the real signal.
    if args.shuffle_regime:
        print("   ⚠ ablation B: HMM regime probs in MAC obs SCRAMBLED (Dirichlet)")
        train_env = ShuffleRegimeMACObs(train_env, seed=args.seed)
        val_env   = ShuffleRegimeMACObs(val_env,   seed=args.seed + 1)
    if args.no_vol:
        print("   ⚠ ablation C: vol features DROPPED from MAC obs (3D instead of 6D)")
        train_env = NoVolMACObs(train_env)
        val_env   = NoVolMACObs(val_env)

    policy_kwargs = dict(net_arch=[32, 16])
    model = PPO(
        "MlpPolicy", train_env,
        policy_kwargs=policy_kwargs,
        learning_rate=3e-4,
        n_steps=1024, batch_size=128,
        ent_coef=args.ent_coef,
        n_epochs=10, gamma=0.99, gae_lambda=0.95, clip_range=0.2,
        verbose=0, seed=args.seed, device='cpu',
    )
    n_params = sum(p.numel() for p in model.policy.parameters())
    print(f"\n   MAC policy params: {n_params:,}")

    if args.warm_start:
        supervised_warm_start(model, train_env, n_samples=8192,
                              lr=1e-3, epochs=5,
                              target_mode=args.warm_target)

    print(f"\n--- Training MAC for {args.total_steps:,} steps ---\n")
    detailed = DetailedCallback(eval_freq=50_000,
                                 total_timesteps=args.total_steps)
    best_path = os.path.join(saved_dir, f'ppo_mac_{args.universe}{suffix}')
    val_cb = ValidationCheckpointCallback(
        val_env=val_env, save_path=best_path,
        eval_freq=args.eval_freq, checkpoint_dir=checkpoint_dir,
    )
    model.learn(total_timesteps=args.total_steps,
                callback=[detailed, val_cb], progress_bar=True)

    final_path = os.path.join(saved_dir, f'ppo_mac_{args.universe}{suffix}_final')
    model.save(final_path)
    print(f"\n✅ Best by val Sharpe: {val_cb.best_val_sharpe:.4f} "
          f"at step {val_cb.best_step:,}")

    with open(os.path.join(results_dir, 'val_sharpe_history.json'), 'w') as f:
        json.dump({'history': val_cb.history,
                   'best_step': val_cb.best_step,
                   'best_val_sharpe': val_cb.best_val_sharpe}, f, indent=2)

    print("\n" + "=" * 72)
    print(f"   MAC {args.universe.upper()} ({args.lineup}"
          f"{', warm-start' if args.warm_start else ''}) COMPLETE ✅")
    print(f"   → {best_path}.zip")
    print(f"   → checkpoints in {checkpoint_dir}/")
    print("=" * 72)


if __name__ == "__main__":
    main()
