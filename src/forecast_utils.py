"""Shared forecasting utilities for the energy-demand project.

Provides a single, model-agnostic backtesting harness so that every model
(naive, ARIMA/SARIMA/ARIMAX, XGBoost, RNN/GRU/LSTM, TiDE) is evaluated on the
*same* set of (forecast origin, horizon) pairs with the *same* metrics.

Evaluation convention
---------------------
- One representative zone, half-hourly series (48 periods/day).
- Weather + calendar are treated as **known-future covariates** (the project
  forecasts demand for given weather *scenarios*), so a model may read covariate
  rows at any time but only demand values up to and including the origin.
- For an integer ``origin`` position, the forecast targets are the demand values
  at positions ``origin+1 .. origin+H``.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

# ---------------------------------------------------------------- paths / cols
_THIS = Path(__file__).resolve()
ROOT = _THIS.parent.parent
PANEL_PATH = ROOT / "data" / "processed" / "demand_weather_panel.parquet"

TARGET = "demand_gwh"
TZ = "Pacific/Auckland"

WEATHER_FEATS = [
    "temperature_2m", "relative_humidity_2m", "dew_point_2m",
    "apparent_temperature", "wind_speed_10m", "cloud_cover",
    "shortwave_radiation", "surface_pressure",
]
CALENDAR_FEATS = [
    "tod_sin", "tod_cos", "dow_sin", "dow_cos", "doy_sin", "doy_cos",
    "is_weekend", "is_holiday",
]
COVARIATES = WEATHER_FEATS + CALENDAR_FEATS

PERIODS_PER_DAY = 48
H_DAY = 48           # day-ahead horizon (steps)
H_WEEK = 336         # week-ahead horizon (steps)


# ----------------------------------------------------------------- data access
def _add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    """(Re)compute calendar covariates from the UTC index, in NZ local time."""
    local = df.index.tz_convert(TZ)
    df["hour"] = local.hour
    df["minute"] = local.minute
    df["dayofweek"] = local.dayofweek
    df["dayofyear"] = local.dayofyear
    df["is_weekend"] = (local.dayofweek >= 5).astype(float)
    tod = local.hour + local.minute / 60.0
    df["tod_sin"] = np.sin(2 * np.pi * tod / 24)
    df["tod_cos"] = np.cos(2 * np.pi * tod / 24)
    df["dow_sin"] = np.sin(2 * np.pi * local.dayofweek / 7)
    df["dow_cos"] = np.cos(2 * np.pi * local.dayofweek / 7)
    df["doy_sin"] = np.sin(2 * np.pi * local.dayofyear / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * local.dayofyear / 365.25)
    return df


def load_zone(zone: str = "UNI", panel_path: Path | str = PANEL_PATH) -> pd.DataFrame:
    """Load one zone as a clean, gap-free half-hourly frame indexed by UTC time.

    Demand is interpolated over the small EMI gaps; weather is interpolated then
    edge-filled; leading rows that have no weather coverage are dropped.
    """
    panel = pd.read_parquet(panel_path)
    df = panel[panel["zone"] == zone].copy()
    df = df.set_index("timestamp_utc").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    df = df.asfreq("30min")                       # enforce a regular grid

    # asfreq can insert rows -> recompute calendar; is_holiday is carried from the
    # panel, so coerce to float and fill the few inserted intra-day gaps.
    df = _add_calendar(df)
    df["is_holiday"] = df["is_holiday"].astype("float").ffill().bfill()
    df[WEATHER_FEATS] = df[WEATHER_FEATS].interpolate("time", limit_direction="both")
    df[TARGET] = df[TARGET].interpolate("time", limit=6)   # only short demand gaps
    df = df.dropna(subset=WEATHER_FEATS + [TARGET])
    keep = [TARGET] + WEATHER_FEATS + CALENDAR_FEATS
    return df[keep]


def split_time(df: pd.DataFrame, train_end: str):
    """Split into (train, test) at ``train_end`` (inclusive of train)."""
    cut = pd.Timestamp(train_end, tz="UTC")
    return df.loc[df.index <= cut], df.loc[df.index > cut]


# ---------------------------------------------------------------------- metrics
def mae(y, p):
    return float(np.mean(np.abs(y - p)))


def rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


def mape(y, p):
    y, p = np.asarray(y), np.asarray(p)
    m = np.abs(y) > 1e-9
    return float(np.mean(np.abs((y[m] - p[m]) / y[m])) * 100)


def smape(y, p):
    y, p = np.asarray(y), np.asarray(p)
    denom = (np.abs(y) + np.abs(p))
    m = denom > 1e-9
    return float(np.mean(2 * np.abs(y[m] - p[m]) / denom[m]) * 100)


# ------------------------------------------------------------------- backtester
def make_origins(df: pd.DataFrame, test_start: str, max_h: int = H_WEEK,
                 step_days: int = 7, max_origins: int | None = None) -> list[int]:
    """Integer origin positions in the test period, spaced ``step_days`` apart.

    Each origin leaves room for ``max_h`` future steps inside ``df``.
    """
    start = pd.Timestamp(test_start, tz="UTC")
    pos = np.where(df.index >= start)[0]
    if len(pos) == 0:
        return []
    step = step_days * PERIODS_PER_DAY
    last_valid = len(df) - max_h - 1
    origins = [o for o in range(pos[0], last_valid + 1, step)]
    if max_origins:
        origins = origins[:max_origins]
    return origins


def backtest(predict_fn, df: pd.DataFrame, origins: list[int], max_h: int = H_WEEK):
    """Run ``predict_fn(df, origin, max_h) -> np.ndarray[max_h]`` over all origins.

    Returns a dict with stacked predictions/truth of shape (n_origins, max_h).
    """
    y_true = np.full((len(origins), max_h), np.nan)
    y_pred = np.full((len(origins), max_h), np.nan)
    target = df[TARGET].values
    for i, o in enumerate(origins):
        y_true[i] = target[o + 1: o + 1 + max_h]
        y_pred[i] = np.asarray(predict_fn(df, o, max_h), dtype=float)[:max_h]
    return {"origins": origins, "y_true": y_true, "y_pred": y_pred}


def score(bt: dict, horizons=(H_DAY, H_WEEK), name: str = "model") -> pd.DataFrame:
    """Per-horizon MAE/RMSE/MAPE/sMAPE from a backtest result."""
    rows = []
    for h in horizons:
        yt = bt["y_true"][:, :h].ravel()
        yp = bt["y_pred"][:, :h].ravel()
        ok = ~np.isnan(yt) & ~np.isnan(yp)
        yt, yp = yt[ok], yp[ok]
        rows.append({"model": name, "horizon": h,
                     "MAE": mae(yt, yp), "RMSE": rmse(yt, yp),
                     "MAPE%": mape(yt, yp), "sMAPE%": smape(yt, yp)})
    return pd.DataFrame(rows)


def error_by_step(bt: dict) -> np.ndarray:
    """Mean absolute error at each lead step (1..max_h), for horizon curves."""
    err = np.abs(bt["y_true"] - bt["y_pred"])
    return np.nanmean(err, axis=0)


# --------------------------------------------------- probabilistic forecasting
def pinball_loss(y, q_pred, alpha: float) -> float:
    """Quantile (pinball) loss for a single quantile level ``alpha``."""
    y, q_pred = np.asarray(y), np.asarray(q_pred)
    d = y - q_pred
    return float(np.mean(np.maximum(alpha * d, (alpha - 1) * d)))


def coverage(y, lo, hi) -> float:
    """Empirical fraction of actuals falling inside [lo, hi]."""
    y, lo, hi = np.asarray(y), np.asarray(lo), np.asarray(hi)
    return float(np.mean((y >= lo) & (y <= hi)))


def interval_width(lo, hi) -> float:
    return float(np.mean(np.asarray(hi) - np.asarray(lo)))


def backtest_quantile(predict_fn, df: pd.DataFrame, origins: list[int],
                      quantiles, max_h: int = H_WEEK):
    """Like :func:`backtest` but ``predict_fn(df, o, h) -> array (n_quantiles, h)``.

    Returns y_true (n_origins, max_h) and y_q (n_origins, n_quantiles, max_h).
    Quantile predictions are sorted along the quantile axis to avoid crossing.
    """
    nq = len(quantiles)
    y_true = np.full((len(origins), max_h), np.nan)
    y_q = np.full((len(origins), nq, max_h), np.nan)
    target = df[TARGET].values
    for i, o in enumerate(origins):
        y_true[i] = target[o + 1: o + 1 + max_h]
        pred = np.asarray(predict_fn(df, o, max_h), dtype=float)[:, :max_h]
        y_q[i] = np.sort(pred, axis=0)
    return {"origins": origins, "y_true": y_true, "y_q": y_q, "quantiles": list(quantiles)}


def score_crps(btq: dict, horizons=(H_DAY, H_WEEK), name: str = "model") -> pd.DataFrame:
    """Approximate CRPS from a multi-quantile backtest.

    Uses ``CRPS = 2 * integral_0^1 pinball_tau d_tau`` approximated by the mean
    pinball loss over the (ideally dense, uniform) quantile grid in ``btq``.
    """
    qs = btq["quantiles"]
    rows = []
    for h in horizons:
        yt = btq["y_true"][:, :h]
        ok = ~np.isnan(yt)
        ytf = yt[ok]
        pbs = [pinball_loss(ytf, btq["y_q"][:, k, :h][ok], a) for k, a in enumerate(qs)]
        rows.append({"model": name, "horizon": h, "CRPS": 2.0 * float(np.mean(pbs))})
    return pd.DataFrame(rows)


def score_quantile(btq: dict, lo_idx: int, hi_idx: int,
                   horizons=(H_DAY, H_WEEK), name: str = "model") -> pd.DataFrame:
    """Per-horizon probabilistic scores: mean pinball, PI coverage and width.

    ``lo_idx``/``hi_idx`` index into ``btq['quantiles']`` for the interval bounds.
    """
    qs = btq["quantiles"]
    nominal = qs[hi_idx] - qs[lo_idx]
    rows = []
    for h in horizons:
        yt = btq["y_true"][:, :h]
        ok = ~np.isnan(yt)
        ytf = yt[ok]
        pin = np.mean([pinball_loss(ytf, btq["y_q"][:, k, :h][ok], a)
                       for k, a in enumerate(qs)])
        lo, hi = btq["y_q"][:, lo_idx, :h][ok], btq["y_q"][:, hi_idx, :h][ok]
        rows.append({"model": name, "horizon": h,
                     "pinball": float(pin),
                     f"cover@{nominal:.0%}": coverage(ytf, lo, hi),
                     "PI_width": interval_width(lo, hi)})
    return pd.DataFrame(rows)
