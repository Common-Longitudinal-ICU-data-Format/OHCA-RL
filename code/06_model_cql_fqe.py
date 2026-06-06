# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: ohca-rl (3.11.11)
#     language: python
#     name: python3
# ---

# %%
# Force non-interactive matplotlib backend BEFORE any pyplot import.
# Required when running as a script under tee/pipes — otherwise the
# default macOSX backend tries to spin up a GUI event loop and hangs.
import matplotlib
matplotlib.use("Agg")

import sys, json, logging
from pathlib import Path
import yaml
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

# Notebook-compatible display() shim: real display in Jupyter, plain print as a script.
try:
    from IPython.display import display  # type: ignore
    _ = get_ipython()  # noqa: F821 — only defined inside an IPython kernel
except (ImportError, NameError):
    def display(obj):
        if hasattr(obj, "to_string"):
            print(obj.to_string())
        else:
            print(obj)

# ── Single source of truth: config/config.json ──
# Resolves relative to current working dir; sites override paths in their own config.json.
_config_path = Path("config/config.json")
if not _config_path.exists():
    # Fallback: search upward from current notebook location
    for p in Path.cwd().parents:
        if (p / "config" / "config.json").exists():
            _config_path = p / "config" / "config.json"
            break

with open(_config_path) as f:
    site_config = json.load(f)

SITE_NAME    = site_config["site_name"]
TABLES_PATH  = site_config["tables_path"]
FILE_TYPE    = site_config["file_type"]
TIMEZONE     = site_config["timezone"]
PROJECT_ROOT = Path(site_config["project_root"])

CODE_DIR     = PROJECT_ROOT / "code"
CONFIG_DIR   = PROJECT_ROOT / "config"
OUT_DIR      = PROJECT_ROOT / "output" / "intermediate"
MODEL_DIR    = PROJECT_ROOT / "output" / "model"
FINAL_DIR    = PROJECT_ROOT / "output" / "final"
SHARED_DIR   = PROJECT_ROOT / "shared"

for _d in (OUT_DIR, MODEL_DIR, FINAL_DIR, SHARED_DIR):
    _d.mkdir(parents=True, exist_ok=True)

import os as _os
MODE = _os.environ.get("OHCA_RL_MODE", "train").lower()
assert MODE in ("train", "validate"), f"OHCA_RL_MODE must be 'train' or 'validate', got {MODE}"

print(f"Site: {SITE_NAME}  |  tables: {TABLES_PATH}  |  mode: {MODE}")

from utils import init_log_capture
init_log_capture(__file__, PROJECT_ROOT, mode_label=MODE)


SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

def deep_merge(base, override):
    out = dict(base) if base else {}
    for k, v in (override or {}).items():
        out[k] = deep_merge(out[k], v) if (k in out and isinstance(out[k], dict)
                                            and isinstance(v, dict)) else v
    return out

with open(CONFIG_DIR / "ohca_rl_config.yaml") as f:
    _base = yaml.safe_load(f)
_local_path = CONFIG_DIR / "ohca_rl_config_local.yaml"
_local = (yaml.safe_load(open(_local_path)) or {}) if _local_path.exists() else {}
ohca_config = deep_merge(_base, _local)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("05_training")

device = torch.device("cpu")
print(f"Torch device: {device}")
print(f"PyTorch version: {torch.__version__}")

# %%
df = pd.read_parquet(OUT_DIR / "bucketed_with_reward_reviewed.parquet")
df["hospitalization_id"] = df["hospitalization_id"].astype(str)

print(f"\nLoaded bucketed_with_reward.parquet:")
print(f"  Rows     : {len(df):,}")
print(f"  Cols     : {df.shape[1]}")
print(f"  Patients : {df['hospitalization_id'].nunique():,}")

_required = ["hospitalization_id", "hour", "action_tier", "reward", "return",
             "mask_off", "mask_low", "mask_med", "mask_high", "mask_vhigh"]
_missing = [c for c in _required if c not in df.columns]
print(f"\n{'⚠️  MISSING: ' + str(_missing) if _missing else '✓ All required columns present'}")

print(f"\nAction tier distribution (training data):")
for tier, n in df["action_tier"].value_counts().sort_index().items():
    print(f"  Tier {tier}: {n:>6,}  ({n/len(df)*100:5.1f}%)")

print(f"\nReward distribution:")
print(df["reward"].describe(percentiles=[.05, .5, .95]).round(2).to_string())

# %%
# ============================================================
# DIAGNOSTIC: how much does per-step shaping matter vs the terminal reward?
# Decomposes each patient's discounted return G0 into intermediate vs terminal
# contributions. Proves the balance is scale-invariant (rescaling can't fix it).
# Requires: df (with r_intermediate, r_terminal, return), loaded in cell 1.
# ============================================================
import numpy as np, pandas as pd, matplotlib.pyplot as plt

GAMMA = globals().get("GAMMA", 0.99)

d = df.sort_values(["hospitalization_id", "hour"]).copy()
d["k"]    = d.groupby("hospitalization_id").cumcount()        # decision-step index 0..T
d["disc"] = GAMMA ** d["k"]
d["disc_int"]  = d["disc"] * d["r_intermediate"]
d["disc_term"] = d["disc"] * d["r_terminal"]
d["abs_int"]   = d["r_intermediate"].abs()
d["abs_term"]  = d["r_terminal"].abs()

per = d.groupby("hospitalization_id").agg(
    disc_int=("disc_int", "sum"), disc_term=("disc_term", "sum"),
    gross_abs_int=("abs_int", "sum"), abs_term=("abs_term", "max"), T=("k", "size"))

aterm, aint = per["disc_term"].abs(), per["disc_int"].abs()
ratio = aterm / aint.replace(0, np.nan)
share = aterm / (aterm + aint)                                # terminal's share of |G0|

print("=== Discounted return decomposition at t0 (per patient) ===")
print(f"  median |disc terminal|     : {aterm.median():.2f}")
print(f"  median |disc intermediate| : {aint.median():.2f}  (net, signed)")
print(f"  RATIO median|term| / median|int|     : {aterm.median()/aint.median():.1f}x")
print(f"  per-patient ratio |term|/|int|       : median {ratio.median():.1f}x  (IQR {ratio.quantile(.25):.1f}-{ratio.quantile(.75):.1f})")
print(f"  share of |G0| from terminal          : median {share.median():.1%}, mean {share.mean():.1%}")
print(f"  patients with |terminal| > |interm|  : {(aterm > aint).mean():.1%}")
print("\n=== Scale invariance (this is why rescaling doesn't help the balance) ===")
r2 = (0.01*per['disc_term']).abs() / (0.01*per['disc_int']).abs().replace(0, np.nan)
print(f"  median ratio  raw : {ratio.median():.3f}x")
print(f"  median ratio x0.01: {r2.median():.3f}x   (identical — global scale cancels)")
print("\n=== Per-transition ===")
nt = (df['r_terminal'] != 0).sum()
print(f"  transitions carrying ANY terminal reward: {nt}/{len(df)} = {nt/len(df):.2%}")
print(f"  net intermediate ({aint.median():.1f}) << gross |intermediate| ({per['gross_abs_int'].median():.1f}): shaping partially self-cancels (signed)")

fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
ax[0].hist(share, bins=30, color="#7b1113", alpha=.85, edgecolor="white")
ax[0].axvline(share.median(), color="k", ls="--", lw=1.5, label=f"median {share.median():.1%}")
ax[0].set(title="Terminal reward's share of |discounted return G0|",
        xlabel="|disc terminal| / (|disc terminal| + |disc intermediate|)", ylabel="patients")
ax[0].legend()
ax[1].bar(["intermediate\n(net, discounted)", "terminal\n(discounted)"],
        [aint.median(), aterm.median()], color=["#1f77b4", "#7b1113"])
ax[1].set(title=f"Median per-patient contribution to G0  ({aterm.median()/aint.median():.0f}x gap)",
        ylabel="|discounted reward|")
for i, v in enumerate([aint.median(), aterm.median()]):
    ax[1].text(i, v, f"{v:.1f}", ha="center", va="bottom")
plt.tight_layout()

# %%
_drop_id_meta = [
    "hospitalization_id", "hour", "anchor_dttm", "anchor_source",
    "exit_hour", "is_scaffold", "window_close_reason",
    "cpc_tier", "cpc_tier_x", "cpc_tier_y", "survival_status",
]
_LEAKY_MED_COLS = ['med_cont_norepinephrine', 'med_cont_epinephrine', 'med_cont_phenylephrine',
                   'med_cont_vasopressin', 'med_cont_dopamine', 'med_cont_angiotensin']
_LEAKY_DIR_COLS = ['nee_changes_in_hour', 'nee_dir_none', 'nee_dir_esc', 'nee_dir_desc', 'nee_dir_mixed']
_LEAKY_OTHER    = ['hours_since_last_on_pressor']
_drop_action = (["action_tier", "action_label", "med_cont_nee"]
                + _LEAKY_MED_COLS + _LEAKY_DIR_COLS + _LEAKY_OTHER)
_drop_mask   = ["mask_off", "mask_low", "mask_med", "mask_high", "mask_vhigh"]
_drop_reward = ["r_intermediate", "r_terminal", "reward", "return",
                "raw_intermediate", "raw_intermediate_unit"]
_drop_categorical = [c for c in df.columns
                     if c.startswith(("resp_device_", "resp_mode_", "resp_vent_brand_",
                                      "resp_tracheostomy", "adt_"))]
_drop_extra = ["in_decision_window", "first_vaso_hour"]

drop_cols = set(_drop_id_meta + _drop_action + _drop_mask + _drop_reward +
                _drop_categorical + _drop_extra)
state_cols = [c for c in df.columns if c not in drop_cols]

_non_numeric = [c for c in state_cols if not pd.api.types.is_numeric_dtype(df[c])]
if _non_numeric:
    print(f"⚠️  Non-numeric state cols (will be dropped): {_non_numeric}")
    state_cols = [c for c in state_cols if c not in _non_numeric]

_still_leaky = [c for c in state_cols if c in (_LEAKY_MED_COLS + _LEAKY_DIR_COLS + _LEAKY_OTHER)]
assert not _still_leaky, f"LEAKY cols still in state: {_still_leaky}"
print(f"\nState features ({len(state_cols)} total):")
for i, c in enumerate(state_cols):
    _tag = " (lagged)" if c.endswith("_prev") or c == "prev_action_tier" else ""
    print(f"  {i+1:>2}. {c}{_tag}")

_leaks = [c for c in state_cols if "reward" in c or "return" in c or "raw_inter" in c]
print(f"\n{'⚠️  LEAKAGE: ' + str(_leaks) if _leaks else '✓ No reward leakage in state features'}")

# %%
unique_pts = df["hospitalization_id"].unique()
np.random.shuffle(unique_pts)
n_train = int(0.8 * len(unique_pts))
train_pts = set(unique_pts[:n_train])
val_pts   = set(unique_pts[n_train:])

df_train = df[df["hospitalization_id"].isin(train_pts)].copy()
df_val   = df[df["hospitalization_id"].isin(val_pts)].copy()

print(f"\nPatient-level split (seed={SEED}, 80/20):")
print(f"  Train: {len(train_pts):,} patients, {len(df_train):,} decision points")
print(f"  Val  : {len(val_pts):,} patients, {len(df_val):,} decision points")

print(f"\nAction tier balance — train vs val:")
for tier in range(5):
    _t = (df_train["action_tier"] == tier).mean() * 100
    _v = (df_val["action_tier"] == tier).mean() * 100
    print(f"  Tier {tier}: train {_t:5.1f}%, val {_v:5.1f}%")

patient_disp = pd.read_parquet(OUT_DIR / "patient_disposition_reviewed.parquet")
patient_disp["hospitalization_id"] = patient_disp["hospitalization_id"].astype(str)
_disp_train = patient_disp[patient_disp["hospitalization_id"].isin(train_pts)]
_disp_val   = patient_disp[patient_disp["hospitalization_id"].isin(val_pts)]

print(f"\nCPC tier balance — train vs val:")
for tier in ["CPC1_2", "CPC3", "CPC4", "CPC5"]:
    _tn = (_disp_train["cpc_tier"] == tier).sum(); _vn = (_disp_val["cpc_tier"] == tier).sum()
    print(f"  {tier:6s}: train {_tn:>3} ({_tn/len(_disp_train)*100:4.1f}%),  "
          f"val {_vn:>3} ({_vn/len(_disp_val)*100:4.1f}%)")

# %%
mask_cols = ["mask_off", "mask_low", "mask_med", "mask_high", "mask_vhigh"]

def build_transitions(df_subset, state_cols, mask_cols):
    df_subset = df_subset.sort_values(["hospitalization_id", "hour"]).reset_index(drop=True)
    grouped = df_subset.groupby("hospitalization_id", sort=False)
    done = (grouped.cumcount(ascending=False) == 0).astype(int)
    next_state = grouped[state_cols].shift(-1).fillna(0)
    next_mask  = grouped[mask_cols].shift(-1).fillna(1)
    return {
        "state":      df_subset[state_cols].values.astype(np.float32),
        "action":     df_subset["action_tier"].values.astype(np.int64),
        "reward":     df_subset["reward"].values.astype(np.float32),
        "next_state": next_state.values.astype(np.float32),
        "next_mask":  next_mask.values.astype(np.float32),
        "done":       done.values.astype(np.float32),
        "patient_id": df_subset["hospitalization_id"].values,
        "hour":       df_subset["hour"].values,
    }

print("Building transition tuples...")
train_data = build_transitions(df_train, state_cols, mask_cols)
val_data   = build_transitions(df_val,   state_cols, mask_cols)

print(f"\nTrain transitions: {len(train_data['state']):,}  "
      f"(terminal {int(train_data['done'].sum())}, non-terminal {int((1-train_data['done']).sum())})")
print(f"Val   transitions: {len(val_data['state']):,}  "
      f"(terminal {int(val_data['done'].sum())}, non-terminal {int((1-val_data['done']).sum())})")

n_train_pts = len(set(train_data['patient_id'])); n_val_pts = len(set(val_data['patient_id']))
print(f"Sanity (terminals = patients): "
      f"train {'✓' if train_data['done'].sum()==n_train_pts else '✗'}, "
      f"val {'✓' if val_data['done'].sum()==n_val_pts else '✗'}")

# %%
_train_state_df = pd.DataFrame(train_data["state"], columns=state_cols)
_val_state_df   = pd.DataFrame(val_data["state"],   columns=state_cols)
_train_next_df  = pd.DataFrame(train_data["next_state"], columns=state_cols)
_val_next_df    = pd.DataFrame(val_data["next_state"],   columns=state_cols)

feature_medians = _train_state_df.median(axis=0).fillna(0)
for _d in (_train_state_df, _val_state_df, _train_next_df, _val_next_df):
    _d.fillna(feature_medians, inplace=True)

feature_mean = _train_state_df.mean(axis=0)
feature_std  = _train_state_df.std(axis=0).replace(0, 1.0)

def normalize(df_state):
    return ((df_state - feature_mean) / feature_std).values.astype(np.float32)

train_data["state"]      = normalize(_train_state_df)
train_data["next_state"] = normalize(_train_next_df)
val_data["state"]        = normalize(_val_state_df)
val_data["next_state"]   = normalize(_val_next_df)

norm_stats = pd.DataFrame({"feature": state_cols, "median": feature_medians.values,
                           "mean": feature_mean.values, "std": feature_std.values})
norm_stats.to_parquet(OUT_DIR / "normalization_stats_reviewed.parquet", index=False)

CLIP_STD = 5.0
for k in ("state", "next_state"):
    train_data[k] = np.clip(train_data[k], -CLIP_STD, CLIP_STD)
    val_data[k]   = np.clip(val_data[k],   -CLIP_STD, CLIP_STD)
print(f"Normalized + clipped to ±{CLIP_STD} std. "
      f"Train state range [{train_data['state'].min():.2f}, {train_data['state'].max():.2f}]")


# %%
class DuelingDQN(nn.Module):
    def __init__(self, state_dim, n_actions=5, hidden_dim=128):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.value_head     = nn.Linear(hidden_dim, 1)
        self.advantage_head = nn.Linear(hidden_dim, n_actions)
        nn.init.uniform_(self.value_head.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.advantage_head.weight, -3e-3, 3e-3)

    def forward(self, state):
        h = self.trunk(state)
        v = self.value_head(h)
        a = self.advantage_head(h)
        return v + (a - a.mean(dim=1, keepdim=True))

    def predict_action(self, state, mask):
        with torch.no_grad():
            q = self.forward(state).masked_fill(mask == 0, float("-inf"))
            return q.argmax(dim=1)

STATE_DIM, N_ACTIONS, HIDDEN_DIM = len(state_cols), 5, 128
print(f"DuelingDQN: input {STATE_DIM}, hidden {HIDDEN_DIM}×2, output V+A({N_ACTIONS})")


# %%
def _to_tensors(d):
    return {
        "state":      torch.tensor(d["state"],      dtype=torch.float32, device=device),
        "action":     torch.tensor(d["action"],     dtype=torch.long,    device=device),
        "reward":     torch.tensor(d["reward"],     dtype=torch.float32, device=device),
        "next_state": torch.tensor(d["next_state"], dtype=torch.float32, device=device),
        "next_mask":  torch.tensor(d["next_mask"],  dtype=torch.float32, device=device),
        "done":       torch.tensor(d["done"],       dtype=torch.float32, device=device),
    }

train_tensors = _to_tensors(train_data)
val_tensors   = _to_tensors(val_data)
print(f"Train tensors: state={tuple(train_tensors['state'].shape)}")
print(f"Val tensors  : state={tuple(val_tensors['state'].shape)}")

# %%
# ⚠️ REVISED CELL (built during review, not from your original paste).
# Verify hyperparameters match your last working run before relying on output.
LR, BATCH_SIZE, TARGET_TAU = 1e-4, 256, 0.001
GRAD_CLIP, GAMMA, HUBER_DELTA = 10.0, 0.99, 1.0
CQL_ALPHA, REWARD_SCALE = 0.1, 0.01
N_EPOCHS, MIN_EPOCHS, STABILITY_THRESHOLD = 300, 25, 0.02
LOG_EVERY = 10

online_net = DuelingDQN(STATE_DIM, N_ACTIONS, HIDDEN_DIM).to(device)
target_net = DuelingDQN(STATE_DIM, N_ACTIONS, HIDDEN_DIM).to(device)
target_net.load_state_dict(online_net.state_dict()); target_net.eval()
optimizer = torch.optim.Adam(online_net.parameters(), lr=LR)

if not globals().get("_reward_scaled", False):
    train_tensors["reward"] = train_tensors["reward"] * REWARD_SCALE
    val_tensors["reward"]   = val_tensors["reward"]   * REWARD_SCALE
    _reward_scaled = True
    print(f"Rewards scaled by {REWARD_SCALE} "
          f"(train range [{train_tensors['reward'].min():.2f}, {train_tensors['reward'].max():.2f}])")

def compute_td_target(r, sp, mp, d, gamma):
    with torch.no_grad():
        qn = online_net(sp).masked_fill(mp == 0, float("-inf"))
        a_star = qn.argmax(dim=1)
        qt = target_net(sp).gather(1, a_star.unsqueeze(1)).squeeze(1)
        return r + gamma * qt * (1 - d)

def soft_update(tau):
    with torch.no_grad():
        for tp, op in zip(target_net.parameters(), online_net.parameters()):
            tp.data.mul_(1 - tau).add_(tau * op.data)

def train_epoch(gs):
    online_net.train()
    idx = torch.randperm(len(train_tensors["state"]))
    tt = tc = 0.0; nb = 0
    for st in range(0, len(idx), BATCH_SIZE):
        b = idx[st:st+BATCH_SIZE]
        s, a, r = train_tensors["state"][b], train_tensors["action"][b], train_tensors["reward"][b]
        sp, mp, d = train_tensors["next_state"][b], train_tensors["next_mask"][b], train_tensors["done"][b]
        q_all = online_net(s)
        q_sa  = q_all.gather(1, a.unsqueeze(1)).squeeze(1)
        td = F.smooth_l1_loss(q_sa, compute_td_target(r, sp, mp, d, GAMMA), beta=HUBER_DELTA)
        cql = (torch.logsumexp(q_all, dim=1) - q_sa).mean()
        loss = td + CQL_ALPHA * cql
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(online_net.parameters(), GRAD_CLIP)
        optimizer.step(); soft_update(TARGET_TAU); gs += 1
        tt += td.item(); tc += cql.item(); nb += 1
    return tt/nb, tc/nb, gs

def validate(t):
    online_net.eval()
    tt = tc = 0.0; nb = 0; ac = np.zeros(5, dtype=int)
    with torch.no_grad():
        for st in range(0, len(t["state"]), BATCH_SIZE):
            e = min(st+BATCH_SIZE, len(t["state"]))
            s, a, r = t["state"][st:e], t["action"][st:e], t["reward"][st:e]
            sp, mp, d = t["next_state"][st:e], t["next_mask"][st:e], t["done"][st:e]
            q_all = online_net(s); q_sa = q_all.gather(1, a.unsqueeze(1)).squeeze(1)
            td = F.smooth_l1_loss(q_sa, compute_td_target(r, sp, mp, d, GAMMA), beta=HUBER_DELTA)
            cql = (torch.logsumexp(q_all, dim=1) - q_sa).mean()
            tt += td.item(); tc += cql.item(); nb += 1
            pa = q_all.argmax(dim=1).cpu().numpy()
            for ai in range(5): ac[ai] += (pa == ai).sum()
    return tt/nb, tc/nb, ac

import os
import hashlib
import shutil

if MODE == "train":
    print(f"TRAINING — CQL α={CQL_ALPHA}, reward×{REWARD_SCALE}, fixed {N_EPOCHS} epochs")
    print("="*70)
    CKPT_DIR = MODEL_DIR / "cql_ckpts_tmp"; CKPT_DIR.mkdir(parents=True, exist_ok=True)
    for f in CKPT_DIR.glob("ep*.pt"): os.remove(f)

    history = {"val_td": [], "policy_l1": [], "stable": [], "dist": []}
    prev_dist = None; gs = 0
    for epoch in range(1, N_EPOCHS + 1):
        tr_td, tr_cql, gs = train_epoch(gs)
        va_td, va_cql, ac = validate(val_tensors)
        dist = ac / ac.sum()
        l1 = float(np.abs(dist - prev_dist).sum()) if prev_dist is not None else 1.0
        prev_dist = dist
        is_stable = (epoch >= MIN_EPOCHS) and (l1 < STABILITY_THRESHOLD)
        history["val_td"].append(va_td); history["policy_l1"].append(l1)
        history["stable"].append(is_stable); history["dist"].append(dist)
        torch.save(online_net.state_dict(), CKPT_DIR / f"ep{epoch:04d}.pt")
        if epoch % LOG_EVERY == 0 or epoch == 1:
            a_str = " ".join(f"{i}={100*dist[i]:.0f}%" for i in range(5))
            print(f"Epoch {epoch:>3}/{N_EPOCHS}  train[td={tr_td:6.3f} cql={tr_cql:6.3f}]  "
                  f"val[td={va_td:6.3f} cql={va_cql:6.3f}]  L1={l1:.4f} "
                  f"[{'stable' if is_stable else '      '}]  dist: {a_str}")

    td_arr, stable_arr = np.array(history["val_td"]), np.array(history["stable"])
    if stable_arr.any():
        best_idx = int(np.argmin(np.where(stable_arr, td_arr, np.inf))); reason = "min val-TD among stable"
    else:
        best_idx = int(np.argmin(td_arr)); reason = "min val-TD overall (NO stable epoch — inspect L1)"
        print(f"\n⚠️  No stable epoch (L1<{STABILITY_THRESHOLD} after ep {MIN_EPOCHS}).")
    best_epoch = best_idx + 1
    online_net.load_state_dict(torch.load(CKPT_DIR / f"ep{best_epoch:04d}.pt")); online_net.eval()
    torch.save(online_net.state_dict(), MODEL_DIR / "ddqn_cql_reviewed.pt")
    for f in CKPT_DIR.glob("ep*.pt"): os.remove(f)

    _fp = hashlib.md5(b"".join(p.detach().cpu().numpy().tobytes() for p in online_net.parameters())).hexdigest()[:8]
    print(f"\nSelected epoch {best_epoch} ({reason}); n_stable={int(stable_arr.sum())}; fingerprint={_fp}")

    # ── Copy shareable artifacts to shared/ for external sites ──
    shutil.copy(MODEL_DIR / "ddqn_cql_reviewed.pt", SHARED_DIR / "ddqn_cql_reviewed.pt")
    norm_stats.to_parquet(SHARED_DIR / "normalization_stats.parquet", index=False)
    feature_meta = {
        "state_features": list(state_cols),
        "n_features": STATE_DIM,
        "n_actions": N_ACTIONS,
        "hidden_dim": HIDDEN_DIM,
        "training_site": SITE_NAME,
        "selected_epoch": best_epoch,
        "selection_rule": reason,
        "fingerprint": _fp,
    }
    with open(SHARED_DIR / "feature_metadata.json", "w") as f:
        json.dump(feature_meta, f, indent=2)
    print(f"Copied shareable artifacts → {SHARED_DIR}")
else:
    _ckpt = SHARED_DIR / "ddqn_cql_reviewed.pt"
    assert _ckpt.exists(), f"Shared model checkpoint missing: {_ckpt}"
    online_net.load_state_dict(torch.load(_ckpt, map_location=device)); online_net.eval()
    history = {"val_td": [], "policy_l1": [], "stable": [], "dist": []}
    best_epoch, reason = None, "validate mode (loaded from shared/)"
    stable_arr = np.array([], dtype=bool)
    _fp = hashlib.md5(b"".join(p.detach().cpu().numpy().tobytes() for p in online_net.parameters())).hexdigest()[:8]
    print(f"[validate] Loaded shared model from {_ckpt} (fingerprint={_fp})")

# %%
# ============================================================
# External validation package for frozen RL policy
# Directional disagreement + descriptive + sensitivity analyses
# ============================================================

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.miscmodels.ordinal_model import OrderedModel
from pathlib import Path

# ------------------------------------------------------------
# 0. Site settings
# ------------------------------------------------------------

SITE_ID = site_config.get("site_id", SITE_NAME.lower().replace(" ", "_"))

SAVE_DIR = FINAL_DIR / SITE_ID
SAVE_DIR.mkdir(parents=True, exist_ok=True)

BASELINE_COVARS = [
    "age_at_admission",
    "sex_male",
    "sofa_total_0_24",
]
# ── Build val_df_for_eval: per-decision-point validation table ──
# Needed by external-validation cell (clinician_action, policy_action per decision)

# Compute policy's argmax action on val states (mask-aware)
with torch.no_grad():
    v_state_t = torch.tensor(val_data["state"], dtype=torch.float32, device=device)
    # current-state mask: from df_val in original row order
    v_mask_t = torch.tensor(
        df_val.sort_values(["hospitalization_id", "hour"])[mask_cols].values.astype(np.float32),
        device=device,
    )
    q_val = online_net(v_state_t)
    q_masked = q_val.masked_fill(v_mask_t == 0, float("-inf"))
    policy_action_val = q_masked.argmax(dim=1).cpu().numpy()

val_df_for_eval = pd.DataFrame({
    "hospitalization_id": val_data["patient_id"].astype(str),
    "hour":               val_data["hour"],
    "clinician_action":   val_data["action"].astype(int),
    "policy_action":      policy_action_val.astype(int),
})

print(f"Built val_df_for_eval: {len(val_df_for_eval):,} decision points, "
      f"{val_df_for_eval['hospitalization_id'].nunique():,} patients")
print(f"  Exact agreement: {(val_df_for_eval['clinician_action'] == val_df_for_eval['policy_action']).mean()*100:.1f}%")


# ------------------------------------------------------------
# 1. Start from decision-level validation dataframe
# ------------------------------------------------------------

df_eval = val_df_for_eval.copy()
df_eval["hospitalization_id"] = df_eval["hospitalization_id"].astype(str)

required_decision_cols = [
    "hospitalization_id",
    "clinician_action",
    "policy_action",
]

missing_decision_cols = [
    c for c in required_decision_cols if c not in df_eval.columns
]

if missing_decision_cols:
    raise ValueError(f"Missing decision-level columns: {missing_decision_cols}")

df_eval["clinician_action"] = df_eval["clinician_action"].astype(int)
df_eval["policy_action"] = df_eval["policy_action"].astype(int)

if "hour" not in df_eval.columns:
    df_eval["hour"] = np.nan

# ------------------------------------------------------------
# 2. Decision-level disagreement features
# ------------------------------------------------------------

df_eval["signed_diff"] = df_eval["policy_action"] - df_eval["clinician_action"]
df_eval["abs_diff"] = df_eval["signed_diff"].abs()

df_eval["exact_agree"] = (df_eval["signed_diff"] == 0).astype(int)
df_eval["within_1"] = (df_eval["abs_diff"] <= 1).astype(int)

df_eval["policy_higher"] = (df_eval["signed_diff"] > 0).astype(int)
df_eval["policy_lower"] = (df_eval["signed_diff"] < 0).astype(int)
df_eval["large_disagree"] = (df_eval["abs_diff"] >= 2).astype(int)

for a in range(5):
    df_eval[f"clinician_action_{a}"] = (df_eval["clinician_action"] == a).astype(int)
    df_eval[f"policy_action_{a}"] = (df_eval["policy_action"] == a).astype(int)

# ------------------------------------------------------------
# 3. Patient-level aggregation
# ------------------------------------------------------------

agg_dict = {
    "n_dp": ("signed_diff", "size"),

    "pct_exact_agree": ("exact_agree", "mean"),
    "pct_within_1": ("within_1", "mean"),

    "mean_signed_diff": ("signed_diff", "mean"),
    "median_signed_diff": ("signed_diff", "median"),

    "mean_abs_diff": ("abs_diff", "mean"),
    "pct_large_disagree": ("large_disagree", "mean"),

    "pct_policy_higher": ("policy_higher", "mean"),
    "pct_policy_lower": ("policy_lower", "mean"),

    "mean_clinician_action": ("clinician_action", "mean"),
    "mean_policy_action": ("policy_action", "mean"),
}

for a in range(5):
    agg_dict[f"pct_clinician_action_{a}"] = (f"clinician_action_{a}", "mean")
    agg_dict[f"pct_policy_action_{a}"] = (f"policy_action_{a}", "mean")

per_patient_behavior = (
    df_eval
    .groupby("hospitalization_id")
    .agg(**agg_dict)
    .reset_index()
)

# ------------------------------------------------------------
# 4. Load patient-level outcome/baseline files
# ------------------------------------------------------------

patient_static = pd.read_parquet(OUT_DIR / "patient_static.parquet")
patient_static["hospitalization_id"] = patient_static["hospitalization_id"].astype(str)

patient_disp = pd.read_parquet(OUT_DIR / "patient_disposition_reviewed.parquet")
patient_disp["hospitalization_id"] = patient_disp["hospitalization_id"].astype(str)

sofa_0_24 = pd.read_parquet(OUT_DIR / "sofa_0_24_reviewed.parquet")
sofa_0_24["hospitalization_id"] = sofa_0_24["hospitalization_id"].astype(str)

per_patient = (
    per_patient_behavior
    .merge(
        patient_disp[["hospitalization_id", "cpc_tier"]],
        on="hospitalization_id",
        how="left"
    )
    .merge(
        patient_static[["hospitalization_id", "age_at_admission", "sex_category"]],
        on="hospitalization_id",
        how="left"
    )
    .merge(
        sofa_0_24[["hospitalization_id", "sofa_total_0_24"]],
        on="hospitalization_id",
        how="left"
    )
)

# ------------------------------------------------------------
# 5. Create outcome variables
# ------------------------------------------------------------

per_patient["sex_male"] = (
    per_patient["sex_category"]
    .astype(str)
    .str.lower()
    .eq("male")
    .astype(int)
)

cpc_map_good = {
    "CPC1_2": 4,
    "CPC3": 3,
    "CPC4": 2,
    "CPC5": 1,
}

per_patient["cpc_ord_good"] = per_patient["cpc_tier"].map(cpc_map_good)

# 1 = favorable neurological outcome
per_patient["good_outcome"] = (per_patient["cpc_tier"] == "CPC1_2").astype(int)

# 1 = survived, 0 = CPC5/death
per_patient["survived"] = (per_patient["cpc_tier"] != "CPC5").astype(int)

for c in [
    "age_at_admission",
    "sex_male",
    "sofa_total_0_24",
    "cpc_ord_good",
    "good_outcome",
    "survived",
]:
    per_patient[c] = pd.to_numeric(per_patient[c], errors="coerce")

per_patient["site_id"] = SITE_ID

# ------------------------------------------------------------
# 6. Descriptive tables
# ------------------------------------------------------------

cohort_summary = pd.DataFrame([{
    "site_id": SITE_ID,
    "n_patients": per_patient["hospitalization_id"].nunique(),
    "n_decision_points": len(df_eval),
    "median_decision_points_per_patient": per_patient["n_dp"].median(),
    "mean_decision_points_per_patient": per_patient["n_dp"].mean(),
    "age_mean": per_patient["age_at_admission"].mean(),
    "age_sd": per_patient["age_at_admission"].std(),
    "sex_male_pct": per_patient["sex_male"].mean(),
    "sofa_mean": per_patient["sofa_total_0_24"].mean(),
    "sofa_sd": per_patient["sofa_total_0_24"].std(),
    "n_CPC1_2": (per_patient["cpc_tier"] == "CPC1_2").sum(),
    "n_CPC3": (per_patient["cpc_tier"] == "CPC3").sum(),
    "n_CPC4": (per_patient["cpc_tier"] == "CPC4").sum(),
    "n_CPC5": (per_patient["cpc_tier"] == "CPC5").sum(),
    "survival_rate": per_patient["survived"].mean(),
    "good_outcome_rate": per_patient["good_outcome"].mean(),
}])

action_distribution = pd.DataFrame([{
    "site_id": SITE_ID,
    "mean_clinician_action": per_patient["mean_clinician_action"].mean(),
    "mean_policy_action": per_patient["mean_policy_action"].mean(),
    **{
        f"clinician_action_{a}_pct": (df_eval["clinician_action"] == a).mean()
        for a in range(5)
    },
    **{
        f"policy_action_{a}_pct": (df_eval["policy_action"] == a).mean()
        for a in range(5)
    },
}])

disagreement_summary = pd.DataFrame([{
    "site_id": SITE_ID,
    "pct_exact_agree": df_eval["exact_agree"].mean(),
    "pct_within_1": df_eval["within_1"].mean(),
    "mean_signed_diff_decision_level": df_eval["signed_diff"].mean(),
    "mean_abs_diff_decision_level": df_eval["abs_diff"].mean(),
    "pct_policy_higher": df_eval["policy_higher"].mean(),
    "pct_policy_lower": df_eval["policy_lower"].mean(),
    "pct_large_disagree": df_eval["large_disagree"].mean(),
    "patient_mean_pct_exact_agree": per_patient["pct_exact_agree"].mean(),
    "patient_mean_pct_within_1": per_patient["pct_within_1"].mean(),
    "patient_mean_signed_diff": per_patient["mean_signed_diff"].mean(),
    "patient_mean_abs_diff": per_patient["mean_abs_diff"].mean(),
    "patient_mean_pct_policy_higher": per_patient["pct_policy_higher"].mean(),
    "patient_mean_pct_policy_lower": per_patient["pct_policy_lower"].mean(),
    "patient_mean_pct_large_disagree": per_patient["pct_large_disagree"].mean(),
}])

# ------------------------------------------------------------
# 7. Model helpers
# ------------------------------------------------------------

def run_logit(data, outcome, exposures, covars=BASELINE_COVARS, model_name="model"):
    model_df = data[[outcome] + exposures + covars].dropna().copy()

    if model_df[outcome].nunique() < 2:
        return pd.DataFrame([{
            "site_id": SITE_ID,
            "model": model_name,
            "outcome": outcome,
            "term": "MODEL_FAILED_ONE_OUTCOME_CLASS",
            "coef": np.nan,
            "OR": np.nan,
            "CI_lower": np.nan,
            "CI_upper": np.nan,
            "p_value": np.nan,
            "n_patients": len(model_df),
        }])

    y = model_df[outcome].astype(float)
    X = model_df[exposures + covars].astype(float)
    X = sm.add_constant(X, has_constant="add")

    try:
        model = sm.Logit(y, X).fit(disp=False)

        params = model.params
        conf = model.conf_int()
        pvals = model.pvalues

        out = pd.DataFrame({
            "site_id": SITE_ID,
            "model": model_name,
            "outcome": outcome,
            "term": params.index,
            "coef": params.values,
            "OR": np.exp(params.values),
            "CI_lower": np.exp(conf[0].values),
            "CI_upper": np.exp(conf[1].values),
            "p_value": pvals.values,
            "n_patients": len(model_df),
        })

    except Exception as e:
        out = pd.DataFrame([{
            "site_id": SITE_ID,
            "model": model_name,
            "outcome": outcome,
            "term": f"MODEL_FAILED: {e}",
            "coef": np.nan,
            "OR": np.nan,
            "CI_lower": np.nan,
            "CI_upper": np.nan,
            "p_value": np.nan,
            "n_patients": len(model_df),
        }])

    return out


def run_ordinal_cpc(data, exposures, covars=BASELINE_COVARS, model_name="ordinal_model"):
    model_df = data[["cpc_ord_good"] + exposures + covars].dropna().copy()

    if model_df["cpc_ord_good"].nunique() < 2:
        return pd.DataFrame([{
            "site_id": SITE_ID,
            "model": model_name,
            "outcome": "ordinal_CPC_higher_is_better",
            "term": "MODEL_FAILED_ONE_OUTCOME_CLASS",
            "coef": np.nan,
            "OR": np.nan,
            "CI_lower": np.nan,
            "CI_upper": np.nan,
            "p_value": np.nan,
            "n_patients": len(model_df),
        }])

    y = model_df["cpc_ord_good"].astype(int)
    X = model_df[exposures + covars].astype(float)

    try:
        model = OrderedModel(
            endog=y,
            exog=X,
            distr="logit"
        ).fit(method="bfgs", disp=False)

        params = model.params
        conf = model.conf_int()
        pvals = model.pvalues

        rows = []
        for term in exposures + covars:
            rows.append({
                "site_id": SITE_ID,
                "model": model_name,
                "outcome": "ordinal_CPC_higher_is_better",
                "term": term,
                "coef": params[term],
                "OR": np.exp(params[term]),
                "CI_lower": np.exp(conf.loc[term, 0]),
                "CI_upper": np.exp(conf.loc[term, 1]),
                "p_value": pvals[term],
                "n_patients": len(model_df),
            })

        out = pd.DataFrame(rows)

    except Exception as e:
        out = pd.DataFrame([{
            "site_id": SITE_ID,
            "model": model_name,
            "outcome": "ordinal_CPC_higher_is_better",
            "term": f"MODEL_FAILED: {e}",
            "coef": np.nan,
            "OR": np.nan,
            "CI_lower": np.nan,
            "CI_upper": np.nan,
            "p_value": np.nan,
            "n_patients": len(model_df),
        }])

    return out


def add_or_per_10pp(results):
    percent_terms = [
        "pct_exact_agree",
        "pct_within_1",
        "pct_policy_higher",
        "pct_policy_lower",
        "pct_large_disagree",
    ]

    out = results.copy()

    out["OR_per_10pp"] = np.where(
        out["term"].isin(percent_terms),
        np.exp(out["coef"] * 0.10),
        np.nan
    )

    out["CI_lower_per_10pp"] = np.where(
        out["term"].isin(percent_terms),
        np.exp(np.log(out["CI_lower"]) * 0.10),
        np.nan
    )

    out["CI_upper_per_10pp"] = np.where(
        out["term"].isin(percent_terms),
        np.exp(np.log(out["CI_upper"]) * 0.10),
        np.nan
    )

    return out

# ------------------------------------------------------------
# 8. Main models
# ------------------------------------------------------------

model_outputs = []

model_outputs.append(
    run_logit(
        per_patient,
        outcome="survived",
        exposures=["pct_policy_higher", "pct_policy_lower"],
        model_name="PRIMARY_survival_policy_higher_lower"
    )
)

model_outputs.append(
    run_ordinal_cpc(
        per_patient,
        exposures=["pct_policy_higher", "pct_policy_lower"],
        model_name="SECONDARY_ordinalCPC_policy_higher_lower"
    )
)

model_outputs.append(
    run_logit(
        per_patient,
        outcome="good_outcome",
        exposures=["pct_policy_higher", "pct_policy_lower"],
        model_name="SECONDARY_goodCPC_policy_higher_lower"
    )
)

model_outputs.append(
    run_logit(
        per_patient,
        outcome="survived",
        exposures=["pct_exact_agree"],
        model_name="CONTRAST_survival_exact_agreement"
    )
)

model_outputs.append(
    run_ordinal_cpc(
        per_patient,
        exposures=["pct_exact_agree"],
        model_name="CONTRAST_ordinalCPC_exact_agreement"
    )
)

model_outputs.append(
    run_logit(
        per_patient,
        outcome="survived",
        exposures=["mean_abs_diff"],
        model_name="SENS_survival_mean_abs_diff"
    )
)

model_outputs.append(
    run_ordinal_cpc(
        per_patient,
        exposures=["mean_abs_diff"],
        model_name="SENS_ordinalCPC_mean_abs_diff"
    )
)

model_outputs.append(
    run_logit(
        per_patient,
        outcome="survived",
        exposures=[
            "pct_policy_higher",
            "pct_policy_lower",
            "mean_policy_action",
            "mean_clinician_action",
        ],
        model_name="SENS_survival_higher_lower_adjust_absolute_intensity"
    )
)

model_outputs.append(
    run_ordinal_cpc(
        per_patient,
        exposures=[
            "pct_policy_higher",
            "pct_policy_lower",
            "mean_policy_action",
            "mean_clinician_action",
        ],
        model_name="SENS_ordinalCPC_higher_lower_adjust_absolute_intensity"
    )
)

if "pct_policy_action_4" in per_patient.columns:
    model_outputs.append(
        run_logit(
            per_patient,
            outcome="survived",
            exposures=[
                "pct_policy_higher",
                "pct_policy_lower",
                "pct_policy_action_4",
            ],
            model_name="SENS_survival_higher_lower_adjust_policy_action4"
        )
    )

    model_outputs.append(
        run_ordinal_cpc(
            per_patient,
            exposures=[
                "pct_policy_higher",
                "pct_policy_lower",
                "pct_policy_action_4",
            ],
            model_name="SENS_ordinalCPC_higher_lower_adjust_policy_action4"
        )
    )

all_model_results = pd.concat(model_outputs, ignore_index=True)
all_model_results_scaled = add_or_per_10pp(all_model_results)

# ------------------------------------------------------------
# 9. First-24-hour analysis
# ------------------------------------------------------------

early_outputs = []

if df_eval["hour"].notna().any():
    df_early = df_eval[df_eval["hour"] <= 24].copy()

    if len(df_early) > 0:
        early_patient_behavior = (
            df_early
            .groupby("hospitalization_id")
            .agg(
                n_dp_early=("signed_diff", "size"),
                pct_exact_agree=("exact_agree", "mean"),
                pct_within_1=("within_1", "mean"),
                mean_signed_diff=("signed_diff", "mean"),
                mean_abs_diff=("abs_diff", "mean"),
                pct_policy_higher=("policy_higher", "mean"),
                pct_policy_lower=("policy_lower", "mean"),
                pct_large_disagree=("large_disagree", "mean"),
                mean_clinician_action=("clinician_action", "mean"),
                mean_policy_action=("policy_action", "mean"),
            )
            .reset_index()
        )

        early_patient = (
            early_patient_behavior
            .merge(
                per_patient[
                    [
                        "hospitalization_id",
                        "survived",
                        "good_outcome",
                        "cpc_ord_good",
                        "age_at_admission",
                        "sex_male",
                        "sofa_total_0_24",
                    ]
                ],
                on="hospitalization_id",
                how="left"
            )
        )

        early_outputs.append(
            run_logit(
                early_patient,
                outcome="survived",
                exposures=["pct_policy_higher", "pct_policy_lower"],
                model_name="EARLY24_survival_policy_higher_lower"
            )
        )

        early_outputs.append(
            run_ordinal_cpc(
                early_patient,
                exposures=["pct_policy_higher", "pct_policy_lower"],
                model_name="EARLY24_ordinalCPC_policy_higher_lower"
            )
        )

if early_outputs:
    early_results = pd.concat(early_outputs, ignore_index=True)
    early_results_scaled = add_or_per_10pp(early_results)
else:
    early_results = pd.DataFrame()
    early_results_scaled = pd.DataFrame()

# ------------------------------------------------------------
# 10. SOFA-stratified analysis
# ------------------------------------------------------------

stratified_outputs = []

# Convert SOFA safely to float; avoids pd.NA ambiguity
per_patient["sofa_total_0_24"] = (
    pd.to_numeric(per_patient["sofa_total_0_24"], errors="coerce")
    .astype(float)
)

sofa_nonmissing = per_patient["sofa_total_0_24"].dropna()

if len(sofa_nonmissing) > 0:
    sofa_median = sofa_nonmissing.median()

    per_patient["sofa_group"] = "missing_SOFA"

    nonmissing_mask = per_patient["sofa_total_0_24"].notna()

    per_patient.loc[
        nonmissing_mask & (per_patient["sofa_total_0_24"] <= sofa_median),
        "sofa_group"
    ] = "low_SOFA"

    per_patient.loc[
        nonmissing_mask & (per_patient["sofa_total_0_24"] > sofa_median),
        "sofa_group"
    ] = "high_SOFA"

    for sofa_group, g in per_patient.groupby("sofa_group"):

        if sofa_group == "missing_SOFA":
            continue

        if g["hospitalization_id"].nunique() >= 30:

            stratified_outputs.append(
                run_logit(
                    g,
                    outcome="survived",
                    exposures=["pct_policy_higher", "pct_policy_lower"],
                    covars=["age_at_admission", "sex_male"],
                    model_name=f"STRAT_{sofa_group}_survival_policy_higher_lower"
                )
            )

            stratified_outputs.append(
                run_ordinal_cpc(
                    g,
                    exposures=["pct_policy_higher", "pct_policy_lower"],
                    covars=["age_at_admission", "sex_male"],
                    model_name=f"STRAT_{sofa_group}_ordinalCPC_policy_higher_lower"
                )
            )

if stratified_outputs:
    stratified_results = pd.concat(stratified_outputs, ignore_index=True)
    stratified_results_scaled = add_or_per_10pp(stratified_results)
else:
    stratified_results = pd.DataFrame()
    stratified_results_scaled = pd.DataFrame()

# ------------------------------------------------------------
# 11. Optional behavior-support diagnostics
# ------------------------------------------------------------

support_summary = pd.DataFrame()

if {"pi_b_clinician_action", "pi_b_policy_action"}.issubset(df_eval.columns):
    support_summary = pd.DataFrame([{
        "site_id": SITE_ID,
        "mean_pi_b_clinician_action": df_eval["pi_b_clinician_action"].mean(),
        "median_pi_b_clinician_action": df_eval["pi_b_clinician_action"].median(),
        "p05_pi_b_clinician_action": df_eval["pi_b_clinician_action"].quantile(0.05),
        "mean_pi_b_policy_action": df_eval["pi_b_policy_action"].mean(),
        "median_pi_b_policy_action": df_eval["pi_b_policy_action"].median(),
        "p05_pi_b_policy_action": df_eval["pi_b_policy_action"].quantile(0.05),
        "pct_policy_action_support_lt_0_05": (df_eval["pi_b_policy_action"] < 0.05).mean(),
        "pct_policy_action_support_lt_0_01": (df_eval["pi_b_policy_action"] < 0.01).mean(),
    }])

# ------------------------------------------------------------
# 12. Outcome-stratified descriptive summaries
# ------------------------------------------------------------

behavior_cols = [
    "pct_exact_agree",
    "pct_within_1",
    "mean_signed_diff",
    "mean_abs_diff",
    "pct_policy_higher",
    "pct_policy_lower",
    "pct_large_disagree",
    "mean_clinician_action",
    "mean_policy_action",
]

by_cpc = (
    per_patient
    .groupby("cpc_tier")[behavior_cols]
    .agg(["mean", "median", "std"])
    .reset_index()
)

by_survival = (
    per_patient
    .groupby("survived")[behavior_cols]
    .agg(["mean", "median", "std"])
    .reset_index()
)

by_good_outcome = (
    per_patient
    .groupby("good_outcome")[behavior_cols]
    .agg(["mean", "median", "std"])
    .reset_index()
)

# ------------------------------------------------------------
# 13. Save all outputs
# ------------------------------------------------------------

per_patient.to_csv(SAVE_DIR / f"{SITE_ID}_patient_level_policy_features.csv", index=False)
cohort_summary.to_csv(SAVE_DIR / f"{SITE_ID}_cohort_summary.csv", index=False)
action_distribution.to_csv(SAVE_DIR / f"{SITE_ID}_action_distribution.csv", index=False)
disagreement_summary.to_csv(SAVE_DIR / f"{SITE_ID}_disagreement_summary.csv", index=False)

all_model_results.to_csv(SAVE_DIR / f"{SITE_ID}_all_model_results_raw.csv", index=False)
all_model_results_scaled.to_csv(SAVE_DIR / f"{SITE_ID}_all_model_results_scaled_per10pp.csv", index=False)

early_results_scaled.to_csv(SAVE_DIR / f"{SITE_ID}_early24_results_scaled_per10pp.csv", index=False)
stratified_results_scaled.to_csv(SAVE_DIR / f"{SITE_ID}_sofa_stratified_results_scaled_per10pp.csv", index=False)

by_cpc.to_csv(SAVE_DIR / f"{SITE_ID}_behavior_by_cpc.csv", index=False)
by_survival.to_csv(SAVE_DIR / f"{SITE_ID}_behavior_by_survival.csv", index=False)
by_good_outcome.to_csv(SAVE_DIR / f"{SITE_ID}_behavior_by_good_outcome.csv", index=False)

if not support_summary.empty:
    support_summary.to_csv(SAVE_DIR / f"{SITE_ID}_behavior_support_summary.csv", index=False)

print("\nSaved external validation package to:")
print(SAVE_DIR)

print("\nMain files:")
print(SAVE_DIR / f"{SITE_ID}_cohort_summary.csv")
print(SAVE_DIR / f"{SITE_ID}_action_distribution.csv")
print(SAVE_DIR / f"{SITE_ID}_disagreement_summary.csv")
print(SAVE_DIR / f"{SITE_ID}_all_model_results_scaled_per10pp.csv")
print(SAVE_DIR / f"{SITE_ID}_early24_results_scaled_per10pp.csv")
print(SAVE_DIR / f"{SITE_ID}_sofa_stratified_results_scaled_per10pp.csv")

# ------------------------------------------------------------
# 14. Display key results
# ------------------------------------------------------------

print("\nCohort summary:")
display(cohort_summary)

print("\nAction distribution:")
display(action_distribution)

print("\nDisagreement summary:")
display(disagreement_summary)

print("\nMain model results scaled per +10 percentage points:")
display(
    all_model_results_scaled[
        all_model_results_scaled["model"].isin([
            "PRIMARY_survival_policy_higher_lower",
            "SECONDARY_ordinalCPC_policy_higher_lower",
            "SECONDARY_goodCPC_policy_higher_lower",
            "CONTRAST_survival_exact_agreement",
            "CONTRAST_ordinalCPC_exact_agreement",
            "SENS_survival_higher_lower_adjust_absolute_intensity",
            "SENS_ordinalCPC_higher_lower_adjust_absolute_intensity",
            "SENS_survival_higher_lower_adjust_policy_action4",
            "SENS_ordinalCPC_higher_lower_adjust_policy_action4",
        ])
    ].round(4)
)

print("\nEarly 24h results scaled per +10 percentage points:")
display(early_results_scaled.round(4))

print("\nSOFA-stratified results scaled per +10 percentage points:")
display(stratified_results_scaled.round(4))

if not support_summary.empty:
    print("\nBehavior-support summary:")
    display(support_summary)

# %%
# ============================================================
# FQE-only OPE block for external validation
# Stable version:
#   1. Behavior policy model pi_b(a|s), masked to valid actions
#   2. Softened frozen RL target policy pi_e(a|s)
#   3. FQE for RL target policy
#   4. FQE for clinician behavior policy
#   5. Target-stickiness sensitivity: 0.90, 0.80, 0.70
#   6. Patient-level bootstrap CI for FQE difference
#
# Run this after RL training, where online_net exists.
# Required existing objects:
#   df_train, df_val
#   train_data, val_data
#   train_tensors, val_tensors
#   state_cols, mask_cols
#   online_net, device, OUT_DIR
# ============================================================

import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score

# ------------------------------------------------------------
# 0. Settings
# ------------------------------------------------------------

GAMMA_OPE = 0.99
N_ACTIONS = 5

FQE_HIDDEN_DIM = 128
FQE_LR = 3e-5
FQE_BATCH_SIZE = 256
FQE_EPOCHS = 300
FQE_TARGET_UPDATE_EVERY = 10
FQE_GRAD_CLIP = 1.0

FQE_PATIENCE = 40
FQE_MIN_DELTA = 1e-4

MIN_BEHAVIOR_PROB = 1e-3

TARGET_STICKINESS_LIST = [1, 0.9]

N_BOOT = 1000
BOOT_SEED = 42

np.random.seed(BOOT_SEED)
torch.manual_seed(BOOT_SEED)

online_net.eval()

# ------------------------------------------------------------
# 1. Helper: current-state masks for train/val rows
# ------------------------------------------------------------

def get_current_mask_from_df(df_subset, mask_cols):
    df_sorted = (
        df_subset
        .sort_values(["hospitalization_id", "hour"])
        .reset_index(drop=True)
    )
    return df_sorted[mask_cols].values.astype(np.float32)


train_mask_now = get_current_mask_from_df(df_train, mask_cols)
val_mask_now = get_current_mask_from_df(df_val, mask_cols)

train_data["mask"] = train_mask_now
val_data["mask"] = val_mask_now

train_tensors["mask"] = torch.tensor(
    train_mask_now,
    dtype=torch.float32,
    device=device
)

val_tensors["mask"] = torch.tensor(
    val_mask_now,
    dtype=torch.float32,
    device=device
)

print("Added current-state masks:")
print("  train mask:", train_tensors["mask"].shape)
print("  val mask  :", val_tensors["mask"].shape)

# ------------------------------------------------------------
# 2. Behavior policy model pi_b(a | s)
# ------------------------------------------------------------

X_train = train_data["state"]
a_train = train_data["action"]

X_val = val_data["state"]
a_val = val_data["action"]

print("\nTraining behavior policy model pi_b(a|s)...")

behavior_model = LogisticRegression(
    max_iter=2000,
    C=1.0,
    class_weight=None,
    solver="lbfgs",
    random_state=BOOT_SEED,
)

behavior_model.fit(X_train, a_train)


def predict_behavior_proba_all_actions(
    model,
    X,
    masks_np=None,
    n_actions=5,
    min_prob=1e-3,
):
    """
    Predicts behavior probabilities for all actions.

    Important:
      - If masks_np is provided, invalid actions receive probability 0.
      - Probability floor is applied only to valid actions.
      - Rows are renormalized after masking/flooring.
    """

    raw = model.predict_proba(X)
    classes = model.classes_

    proba = np.zeros((X.shape[0], n_actions), dtype=np.float64)

    for j, cls in enumerate(classes):
        proba[:, int(cls)] = raw[:, j]

    if masks_np is not None:
        masks_np = masks_np.astype(bool)

        # Zero out invalid actions
        proba = proba * masks_np

        # Handle rows where all probability mass disappeared
        row_sums = proba.sum(axis=1, keepdims=True)
        bad_rows = row_sums.squeeze() <= 0

        if bad_rows.any():
            valid_counts = masks_np[bad_rows].sum(axis=1, keepdims=True)
            valid_counts = np.maximum(valid_counts, 1)
            proba[bad_rows] = masks_np[bad_rows] / valid_counts

        # Normalize
        proba = proba / proba.sum(axis=1, keepdims=True)

        # Floor only valid actions
        proba = np.where(
            masks_np,
            np.clip(proba, min_prob, 1.0),
            0.0
        )

        proba = proba / proba.sum(axis=1, keepdims=True)

    else:
        proba = np.clip(proba, min_prob, 1.0)
        proba = proba / proba.sum(axis=1, keepdims=True)

    return proba


pi_b_train = predict_behavior_proba_all_actions(
    behavior_model,
    X_train,
    masks_np=train_data["mask"],
    n_actions=N_ACTIONS,
    min_prob=MIN_BEHAVIOR_PROB,
)

pi_b_val = predict_behavior_proba_all_actions(
    behavior_model,
    X_val,
    masks_np=val_data["mask"],
    n_actions=N_ACTIONS,
    min_prob=MIN_BEHAVIOR_PROB,
)

pi_b_train_next = predict_behavior_proba_all_actions(
    behavior_model,
    train_data["next_state"],
    masks_np=train_data["next_mask"],
    n_actions=N_ACTIONS,
    min_prob=MIN_BEHAVIOR_PROB,
)

pi_b_val_next = predict_behavior_proba_all_actions(
    behavior_model,
    val_data["next_state"],
    masks_np=val_data["next_mask"],
    n_actions=N_ACTIONS,
    min_prob=MIN_BEHAVIOR_PROB,
)

behavior_pred_val = pi_b_val.argmax(axis=1)

print("\nBehavior policy diagnostics on validation:")
print(f"  Accuracy predicting clinician action: {accuracy_score(a_val, behavior_pred_val):.3f}")
print(f"  Log loss: {log_loss(a_val, pi_b_val, labels=list(range(N_ACTIONS))):.3f}")

obs_b_prob_val = pi_b_val[np.arange(len(a_val)), a_val]

print("\nObserved clinician action support under pi_b:")
print(
    pd.Series(obs_b_prob_val)
    .describe(percentiles=[0.01, 0.05, 0.10, 0.25, 0.50])
    .round(4)
)

# ------------------------------------------------------------
# 3. Frozen RL target policy
# ------------------------------------------------------------

def rl_greedy_action_numpy(states_np, masks_np):
    with torch.no_grad():
        s = torch.tensor(states_np, dtype=torch.float32, device=device)
        m = torch.tensor(masks_np, dtype=torch.float32, device=device)

        q = online_net(s)
        q = q.masked_fill(m == 0, float("-inf"))

        a = q.argmax(dim=1).cpu().numpy()

    return a


def softened_target_policy_probs(
    states_np,
    masks_np,
    stickiness=0.90,
    n_actions=5,
):
    """
    Softened deterministic RL policy.

    If multiple valid actions:
      greedy RL action gets stickiness
      other valid actions share 1 - stickiness

    If only one valid action:
      that action gets probability 1.
    """

    greedy = rl_greedy_action_numpy(states_np, masks_np)

    probs = np.zeros((states_np.shape[0], n_actions), dtype=np.float64)

    for i in range(states_np.shape[0]):

        valid_actions = np.where(masks_np[i] > 0)[0]

        if len(valid_actions) == 0:
            probs[i, :] = 1.0 / n_actions
            continue

        if len(valid_actions) == 1:
            probs[i, valid_actions[0]] = 1.0
            continue

        g = greedy[i]

        if g not in valid_actions:
            g = valid_actions[0]

        other_actions = [a for a in valid_actions if a != g]

        probs[i, g] = stickiness
        probs[i, other_actions] = (1.0 - stickiness) / len(other_actions)

    probs = probs / probs.sum(axis=1, keepdims=True)

    return probs


# ------------------------------------------------------------
# 4. FQE model
# ------------------------------------------------------------

class FQENet(nn.Module):
    def __init__(self, state_dim, n_actions=5, hidden_dim=128):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),

            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, x):
        return self.net(x)


def train_fqe(
    policy_name,
    train_tensors,
    train_policy_next_probs_np,
    gamma=0.99,
    n_epochs=300,
    batch_size=256,
    lr=3e-5,
    hidden_dim=128,
    target_update_every=10,
    grad_clip=1.0,
    patience=40,
    min_delta=1e-4,
):
    """
    Fitted Q Evaluation for a fixed policy pi.

    Bellman target:
      y = r + gamma * E_{a' ~ pi(.|s')} Q_target(s', a')

    Stabilization:
      - smooth L1 loss
      - target network
      - gradient clipping
      - early stopping by training Bellman loss
    """

    state_dim = train_tensors["state"].shape[1]

    q_net = FQENet(
        state_dim=state_dim,
        n_actions=N_ACTIONS,
        hidden_dim=hidden_dim
    ).to(device)

    target_q_net = FQENet(
        state_dim=state_dim,
        n_actions=N_ACTIONS,
        hidden_dim=hidden_dim
    ).to(device)

    target_q_net.load_state_dict(q_net.state_dict())
    target_q_net.eval()

    optimizer = torch.optim.AdamW(
        q_net.parameters(),
        lr=lr,
        weight_decay=1e-4
    )

    pi_next = torch.tensor(
        train_policy_next_probs_np,
        dtype=torch.float32,
        device=device
    )

    n = train_tensors["state"].shape[0]

    loss_history = []

    best_loss = np.inf
    best_state = None
    bad_epochs = 0

    for epoch in range(1, n_epochs + 1):

        q_net.train()

        idx = torch.randperm(n, device=device)

        epoch_loss = 0.0
        n_batches = 0

        for st in range(0, n, batch_size):
            b = idx[st:st + batch_size]

            s = train_tensors["state"][b]
            a = train_tensors["action"][b]
            r = train_tensors["reward"][b]
            sp = train_tensors["next_state"][b]
            d = train_tensors["done"][b]
            pi_next_b = pi_next[b]

            q_all = q_net(s)
            q_sa = q_all.gather(1, a.unsqueeze(1)).squeeze(1)

            with torch.no_grad():
                q_next_all = target_q_net(sp)
                v_next = (pi_next_b * q_next_all).sum(dim=1)
                target = r + gamma * (1.0 - d) * v_next

                # Defensive target clipping.
                # Adjust these bounds if your reward scale is very different.
                target = torch.clamp(target, -100.0, 100.0)

            loss = F.smooth_l1_loss(q_sa, target)

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                q_net.parameters(),
                grad_clip
            )

            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        if epoch % target_update_every == 0:
            target_q_net.load_state_dict(q_net.state_dict())

        avg_loss = epoch_loss / max(n_batches, 1)
        loss_history.append(avg_loss)

        improved = avg_loss < best_loss - min_delta

        if improved:
            best_loss = avg_loss
            best_state = copy.deepcopy(q_net.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1

        if epoch == 1 or epoch % 25 == 0 or epoch == n_epochs:
            print(
                f"[FQE {policy_name}] epoch {epoch:>3}/{n_epochs} "
                f"| loss={avg_loss:.5f} | best={best_loss:.5f}"
            )

        if bad_epochs >= patience:
            print(
                f"[FQE {policy_name}] early stopping at epoch {epoch}; "
                f"best loss={best_loss:.5f}"
            )
            break

    if best_state is not None:
        q_net.load_state_dict(best_state)

    q_net.eval()

    return q_net, loss_history, best_loss


def fqe_value_for_policy(q_net, states_np, policy_probs_np):
    with torch.no_grad():
        s = torch.tensor(states_np, dtype=torch.float32, device=device)
        pi = torch.tensor(policy_probs_np, dtype=torch.float32, device=device)

        q = q_net(s)
        v = (pi * q).sum(dim=1)

    return v.cpu().numpy()


# ------------------------------------------------------------
# 5. Initial states per validation patient
# ------------------------------------------------------------

def get_initial_indices(patient_ids, hours):
    tmp = pd.DataFrame({
        "idx": np.arange(len(patient_ids)),
        "hospitalization_id": patient_ids.astype(str),
        "hour": hours,
    })

    tmp = tmp.sort_values(["hospitalization_id", "hour"])

    first = tmp.groupby("hospitalization_id", sort=False).head(1)

    return first["idx"].values, first["hospitalization_id"].values


val_init_idx, val_init_patient_ids = get_initial_indices(
    val_data["patient_id"],
    val_data["hour"]
)

print(f"\nValidation initial states: {len(val_init_idx)} patients")

# ------------------------------------------------------------
# 6. Bootstrap helper
# ------------------------------------------------------------

def bootstrap_fqe_values(
    v_rl_init,
    v_behavior_init,
    n_boot=1000,
    seed=42,
):
    rng = np.random.default_rng(seed)

    n = len(v_rl_init)
    rows = []

    for b in range(n_boot):
        idx = rng.choice(np.arange(n), size=n, replace=True)

        rows.append({
            "fqe_rl": float(np.mean(v_rl_init[idx])),
            "fqe_behavior": float(np.mean(v_behavior_init[idx])),
            "fqe_diff_rl_minus_behavior": float(
                np.mean(v_rl_init[idx] - v_behavior_init[idx])
            ),
        })

    return pd.DataFrame(rows)


def summarize_bootstrap(boot_df, cols):
    out = []

    for c in cols:
        x = boot_df[c].dropna().values

        out.append({
            "estimate": c,
            "mean": np.mean(x),
            "median": np.median(x),
            "ci_lower_2.5": np.percentile(x, 2.5),
            "ci_upper_97.5": np.percentile(x, 97.5),
        })

    return pd.DataFrame(out)


# ------------------------------------------------------------
# 7. Run FQE sensitivity across target stickiness levels
# ------------------------------------------------------------

all_point_rows = []
all_boot_rows = []
all_boot_summary_rows = []
all_loss_rows = []
all_support_rows = []

for TARGET_STICKINESS in TARGET_STICKINESS_LIST:

    print("\n" + "=" * 70)
    print(f"Running FQE-only OPE with TARGET_STICKINESS = {TARGET_STICKINESS}")
    print("=" * 70)

    # Target policy probabilities
    pi_e_train = softened_target_policy_probs(
        train_data["state"],
        train_data["mask"],
        stickiness=TARGET_STICKINESS,
        n_actions=N_ACTIONS,
    )

    pi_e_val = softened_target_policy_probs(
        val_data["state"],
        val_data["mask"],
        stickiness=TARGET_STICKINESS,
        n_actions=N_ACTIONS,
    )

    pi_e_train_next = softened_target_policy_probs(
        train_data["next_state"],
        train_data["next_mask"],
        stickiness=TARGET_STICKINESS,
        n_actions=N_ACTIONS,
    )

    pi_e_val_next = softened_target_policy_probs(
        val_data["next_state"],
        val_data["next_mask"],
        stickiness=TARGET_STICKINESS,
        n_actions=N_ACTIONS,
    )

    rl_action_val = pi_e_val.argmax(axis=1)

    rl_support_prob_val = pi_b_val[
        np.arange(len(rl_action_val)),
        rl_action_val
    ]

    support_row = {
        "target_stickiness": TARGET_STICKINESS,
        "rl_support_mean": np.mean(rl_support_prob_val),
        "rl_support_median": np.median(rl_support_prob_val),
        "rl_support_p01": np.quantile(rl_support_prob_val, 0.01),
        "rl_support_p05": np.quantile(rl_support_prob_val, 0.05),
        "rl_support_p10": np.quantile(rl_support_prob_val, 0.10),
        "rl_support_p25": np.quantile(rl_support_prob_val, 0.25),
        "pct_rl_support_lt_0_05": np.mean(rl_support_prob_val < 0.05),
        "pct_rl_support_lt_0_01": np.mean(rl_support_prob_val < 0.01),
    }

    for a in range(N_ACTIONS):
        support_row[f"target_policy_action_{a}_pct"] = np.mean(
            rl_action_val == a
        )

    all_support_rows.append(support_row)

    print("\nRL target policy support under clinician behavior model:")
    print(
        pd.Series(rl_support_prob_val)
        .describe(percentiles=[0.01, 0.05, 0.10, 0.25, 0.50])
        .round(4)
    )

    print("\nTarget policy action distribution on validation:")
    for a in range(N_ACTIONS):
        print(f"  Action {a}: {(rl_action_val == a).mean() * 100:5.1f}%")

    # --------------------------------------------------------
    # FQE for RL target policy
    # --------------------------------------------------------

    print("\nTraining FQE for RL target policy...")

    fqe_rl_net, fqe_rl_loss, fqe_rl_best_loss = train_fqe(
        policy_name=f"RL_stickiness_{TARGET_STICKINESS}",
        train_tensors=train_tensors,
        train_policy_next_probs_np=pi_e_train_next,
        gamma=GAMMA_OPE,
        n_epochs=FQE_EPOCHS,
        batch_size=FQE_BATCH_SIZE,
        lr=FQE_LR,
        hidden_dim=FQE_HIDDEN_DIM,
        target_update_every=FQE_TARGET_UPDATE_EVERY,
        grad_clip=FQE_GRAD_CLIP,
        patience=FQE_PATIENCE,
        min_delta=FQE_MIN_DELTA,
    )

    # --------------------------------------------------------
    # FQE for clinician behavior policy
    # --------------------------------------------------------

    print("\nTraining FQE for clinician behavior policy...")

    fqe_behavior_net, fqe_behavior_loss, fqe_behavior_best_loss = train_fqe(
        policy_name=f"Behavior_stickiness_{TARGET_STICKINESS}",
        train_tensors=train_tensors,
        train_policy_next_probs_np=pi_b_train_next,
        gamma=GAMMA_OPE,
        n_epochs=FQE_EPOCHS,
        batch_size=FQE_BATCH_SIZE,
        lr=FQE_LR,
        hidden_dim=FQE_HIDDEN_DIM,
        target_update_every=FQE_TARGET_UPDATE_EVERY,
        grad_clip=FQE_GRAD_CLIP,
        patience=FQE_PATIENCE,
        min_delta=FQE_MIN_DELTA,
    )

    # --------------------------------------------------------
    # FQE value on validation initial states
    # --------------------------------------------------------

    v_rl_init = fqe_value_for_policy(
        fqe_rl_net,
        val_data["state"][val_init_idx],
        pi_e_val[val_init_idx],
    )

    v_behavior_init = fqe_value_for_policy(
        fqe_behavior_net,
        val_data["state"][val_init_idx],
        pi_b_val[val_init_idx],
    )

    fqe_rl_value = float(np.mean(v_rl_init))
    fqe_behavior_value = float(np.mean(v_behavior_init))
    fqe_diff = fqe_rl_value - fqe_behavior_value

    print("\nFQE value estimates on validation initial states:")
    print(f"  Target stickiness : {TARGET_STICKINESS}")
    print(f"  FQE V_RL          : {fqe_rl_value:.5f}")
    print(f"  FQE V_behavior    : {fqe_behavior_value:.5f}")
    print(f"  Difference        : {fqe_diff:.5f}")

    # --------------------------------------------------------
    # Bootstrap CI
    # --------------------------------------------------------

    boot = bootstrap_fqe_values(
        v_rl_init=v_rl_init,
        v_behavior_init=v_behavior_init,
        n_boot=N_BOOT,
        seed=BOOT_SEED,
    )

    boot["target_stickiness"] = TARGET_STICKINESS

    boot_summary = summarize_bootstrap(
        boot,
        [
            "fqe_rl",
            "fqe_behavior",
            "fqe_diff_rl_minus_behavior",
        ]
    )

    boot_summary["target_stickiness"] = TARGET_STICKINESS

    display(boot_summary.round(5))

    # --------------------------------------------------------
    # Store point estimates
    # --------------------------------------------------------

    point_row = {
        "target_stickiness": TARGET_STICKINESS,
        "gamma": GAMMA_OPE,
        "min_behavior_prob": MIN_BEHAVIOR_PROB,
        "fqe_lr": FQE_LR,
        "fqe_grad_clip": FQE_GRAD_CLIP,
        "fqe_hidden_dim": FQE_HIDDEN_DIM,
        "fqe_rl_best_loss": fqe_rl_best_loss,
        "fqe_behavior_best_loss": fqe_behavior_best_loss,
        "n_val_patients": len(val_init_idx),
        "fqe_rl": fqe_rl_value,
        "fqe_behavior": fqe_behavior_value,
        "fqe_diff_rl_minus_behavior": fqe_diff,
        "fqe_rl_init_sd": float(np.std(v_rl_init)),
        "fqe_behavior_init_sd": float(np.std(v_behavior_init)),
        "fqe_diff_init_sd": float(np.std(v_rl_init - v_behavior_init)),
    }

    all_point_rows.append(point_row)

    boot = boot[
        [
            "target_stickiness",
            "fqe_rl",
            "fqe_behavior",
            "fqe_diff_rl_minus_behavior",
        ]
    ]

    all_boot_rows.append(boot)
    all_boot_summary_rows.append(boot_summary)

    # --------------------------------------------------------
    # Store loss histories
    # --------------------------------------------------------

    rl_loss_df = pd.DataFrame({
        "target_stickiness": TARGET_STICKINESS,
        "policy": "RL",
        "epoch": np.arange(1, len(fqe_rl_loss) + 1),
        "loss": fqe_rl_loss,
    })

    behavior_loss_df = pd.DataFrame({
        "target_stickiness": TARGET_STICKINESS,
        "policy": "behavior",
        "epoch": np.arange(1, len(fqe_behavior_loss) + 1),
        "loss": fqe_behavior_loss,
    })

    all_loss_rows.append(rl_loss_df)
    all_loss_rows.append(behavior_loss_df)

# ------------------------------------------------------------
# 8. Combine and save outputs
# ------------------------------------------------------------

fqe_point_estimates = pd.DataFrame(all_point_rows)
fqe_bootstrap_draws = pd.concat(all_boot_rows, ignore_index=True)
fqe_bootstrap_summary = pd.concat(all_boot_summary_rows, ignore_index=True)
fqe_loss_history = pd.concat(all_loss_rows, ignore_index=True)
fqe_support_summary = pd.DataFrame(all_support_rows)

for df_, name in [
    (fqe_point_estimates,  f"{SITE_ID}_fqe_only_point_estimates.parquet"),
    (fqe_bootstrap_draws,  f"{SITE_ID}_fqe_only_bootstrap_draws.parquet"),
    (fqe_bootstrap_summary, f"{SITE_ID}_fqe_only_bootstrap_summary.parquet"),
    (fqe_loss_history,     f"{SITE_ID}_fqe_only_loss_history.parquet"),
    (fqe_support_summary,  f"{SITE_ID}_fqe_only_support_summary.parquet"),
]:
    df_.to_parquet(SAVE_DIR / name, index=False)

print("\nSaved FQE-only outputs to:", SAVE_DIR)
for name in (
    f"{SITE_ID}_fqe_only_point_estimates.parquet",
    f"{SITE_ID}_fqe_only_bootstrap_draws.parquet",
    f"{SITE_ID}_fqe_only_bootstrap_summary.parquet",
    f"{SITE_ID}_fqe_only_loss_history.parquet",
    f"{SITE_ID}_fqe_only_support_summary.parquet",
):
    print(" ", SAVE_DIR / name)

print("\nFQE point estimates:")
display(fqe_point_estimates.round(5))

print("\nFQE bootstrap summary:")
display(fqe_bootstrap_summary.round(5))

print("\nFQE support summary:")
display(fqe_support_summary.round(5))


# %%
# ------------------------------------------------------------
# 9. Unified JSON summary — federated handoff package
# ------------------------------------------------------------
# One compact file the coordinating center can scan to verify a site's
# run without opening every parquet/CSV. Includes pointers to the
# detailed files plus inline key metrics.

import datetime as _dt

def _to_py(v):
    """Cast numpy/pandas scalars to plain Python for JSON serialization."""
    import numpy as _np
    if v is None or (isinstance(v, float) and _np.isnan(v)):
        return None
    if isinstance(v, (_np.floating, _np.integer)):
        return v.item()
    return v

def _primary_row(model, term):
    sub = all_model_results_scaled[
        (all_model_results_scaled["model"] == model)
        & (all_model_results_scaled["outcome"] == "survived")
        & (all_model_results_scaled["term"] == term)
    ]
    if len(sub) == 0:
        return None
    r = sub.iloc[0]
    return {
        "OR_per_10pp":       _to_py(r["OR_per_10pp"]),
        "CI_lower_per_10pp": _to_py(r["CI_lower_per_10pp"]),
        "CI_upper_per_10pp": _to_py(r["CI_upper_per_10pp"]),
        "p_value":           _to_py(r["p_value"]),
    }

def _fqe_block(stickiness):
    sub = fqe_bootstrap_summary[fqe_bootstrap_summary["target_stickiness"] == stickiness]
    out = {}
    for est in ("fqe_rl", "fqe_behavior", "fqe_diff_rl_minus_behavior"):
        row = sub[sub["estimate"] == est]
        if len(row) == 0:
            continue
        r = row.iloc[0]
        out[est] = {
            "mean":      _to_py(r["mean"]),
            "median":    _to_py(r["median"]),
            "ci_lower":  _to_py(r["ci_lower_2.5"]),
            "ci_upper":  _to_py(r["ci_upper_97.5"]),
        }
    return out

_cohort = cohort_summary.iloc[0].to_dict()
_action = action_distribution.iloc[0].to_dict()
_disag  = disagreement_summary.iloc[0].to_dict()

summary = {
    "site_id":   SITE_ID,
    "site_name": SITE_NAME,
    "mode":      MODE,
    "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
    "model": {
        "fingerprint":    _fp,
        "selected_epoch": _to_py(best_epoch),
        "selection_rule": reason,
        "n_stable_epochs": int(stable_arr.sum()) if stable_arr.size else None,
    },
    "cohort": {k: _to_py(v) for k, v in _cohort.items() if k != "site_id"},
    "action_distribution": {k: _to_py(v) for k, v in _action.items() if k != "site_id"},
    "concordance": {k: _to_py(v) for k, v in _disag.items() if k != "site_id"},
    "concordance_regression_primary": {
        "survival_pct_policy_higher": _primary_row(
            "PRIMARY_survival_policy_higher_lower", "pct_policy_higher"
        ),
        "survival_pct_policy_lower": _primary_row(
            "PRIMARY_survival_policy_higher_lower", "pct_policy_lower"
        ),
    },
    "fqe": {
        "stickiness_1.0": _fqe_block(1.0),
        "stickiness_0.9": _fqe_block(0.9),
    },
    "files_in_save_dir": sorted([p.name for p in SAVE_DIR.iterdir() if p.is_file()]),
}

_summary_path = SAVE_DIR / f"{SITE_ID}_summary.json"
with open(_summary_path, "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nSaved unified JSON summary → {_summary_path}")

# %%
