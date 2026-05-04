"""Shared defaults for every sweep config.

Each sweep_d*.py imports BASE and overrides only `pfgmpp` and `aug_dim` so that
architecture, batch size, training budget, and seed stay identical across D values.

Two profiles are exposed:
  - `local`   : 3080Ti smoke run, batch 16, ~5k steps, 32x32
  - `cluster` : H100 production run, batch 512, 200k steps, 32x32

run_sweep.py picks the profile by --profile=<name> and merges it onto BASE.

All keys map directly to `pfgmpp/train.py` click options unless prefixed with `_`
(internal). `_total_kimg` and `_snapshot_kimg` are converted to --duration / --tick
/ --snap / --dump by the orchestrator.
"""

# ---------------------------------------------------------------------------
# Architecture / data — held fixed across the entire sweep.
# ---------------------------------------------------------------------------

BASE = dict(
    data="datasets/cifar10-32x32.zip",   # relative to pfgmpp/ working dir
    cond=False,
    arch="ncsnpp",                       # more memory-efficient than ddpmpp
    precond="edm",
    seed=42,                             # fixed across all D for reproducibility
    workers=2,
    bench=True,
    cache=True,
    fp16=False,                          # bfloat16 ok on H100; default fp32 for safety
    augment=0.12,
    dropout=0.13,
    xflip=False,
    ema=0.5,
    lr=10e-4,
)

# ---------------------------------------------------------------------------
# Profiles — pick with `run_sweep.py --profile=<name>`.
# ---------------------------------------------------------------------------

LOCAL = dict(
    batch=16,
    _total_kimg=80,         # 80 kimg = 5k steps at batch 16
    _snapshot_kimg=16,      # FID + sample grid every 1k steps (16 kimg)
    workers=1,
)

CLUSTER = dict(
    batch=512,
    _total_kimg=102400,     # 102.4 Mimg = 200k steps at batch 512
    _snapshot_kimg=5120,    # FID every 10k steps (5120 kimg) per the spec
    workers=4,
)

PROFILES = {"local": LOCAL, "cluster": CLUSTER}


def materialize(profile_name: str, d_overrides: dict) -> dict:
    """Merge BASE + profile + per-D overrides. Per-D overrides are: pfgmpp, aug_dim, name."""
    out = dict(BASE)
    if profile_name not in PROFILES:
        raise ValueError(f"Unknown profile {profile_name!r}; expected one of {list(PROFILES)}")
    out.update(PROFILES[profile_name])
    out.update(d_overrides)
    return out
