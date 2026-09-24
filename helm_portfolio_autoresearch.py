#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HELM FINSERV — ML-Driven Portfolio Construction System
=======================================================

A single-file "mini Aladdin / Bloomberg PORT" style terminal that answers:

    "Which stocks should I buy, how much should I allocate,
     and what risk can happen after I buy?"

Pipeline
--------
    NSE Data -> Feature Engineering -> Expected-Return Models
             -> Risk (Covariance) Models -> Optimizer Engine
             -> Portfolio -> Monte-Carlo Risk Simulator -> Report / Backtest

House conventions (HELM FINSERV)
--------------------------------
  * Single self-contained Python/Tkinter file.
  * Angel One SmartAPI integration with TOTP auth (getCandleData + ltpData),
    with an automatic volatility-clustered GBM synthetic fallback for demo/CI.
  * Headless `--selftest` / `--test` CI flag (never touches Tkinter).
  * Native Tkinter canvas charts (no matplotlib dependency).
  * Background threading for long jobs.
  * Light airy theme: #f4f6f9 bg, white cards, teal #0d9488 accent, Segoe UI, clam.

Optional dependencies (all degrade gracefully if missing)
--------------------------------------------------------
  * xgboost   -> falls back to sklearn HistGradientBoostingRegressor
  * torch     -> required for LSTM/Transformer; app shows an install message instead of silent fallback
  * arch      -> DCC-GARCH falls back to a DCC-lite (EWMA vol + EWMA corr) engine
  * SmartApi + pyotp -> live NSE data; otherwise synthetic demo data is used

Usage
-----
    python helm_ml_portfolio.py             # launch GUI
    python helm_ml_portfolio.py --selftest  # headless numeric self-tests (CI)
    python helm_ml_portfolio.py --demo      # headless end-to-end demo run
"""

from __future__ import annotations

import os
import sys
import math
import json
import time
import queue
import warnings
import argparse
import threading
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------- #
#  Optional imports (graceful degradation)                                     #
# --------------------------------------------------------------------------- #
try:
    import xgboost as xgb  # noqa
    HAS_XGB = True
except Exception:
    HAS_XGB = False

try:
    import torch  # noqa
    import torch.nn as nn  # noqa
    HAS_TORCH = True
    TORCH_IMPORT_ERROR = ""
except Exception as _torch_err:
    torch = None
    nn = None
    HAS_TORCH = False
    TORCH_IMPORT_ERROR = str(_torch_err)


def require_torch_for_deep(model_name="deep sequence model"):
    """Prevent fake LSTM/Transformer fallback results.

    The GUI should never report a surrogate as LSTM/Transformer. If PyTorch is
    missing, stop and tell the user exactly how to install it.
    """
    if HAS_TORCH:
        return
    raise RuntimeError(
        f"{model_name} needs PyTorch. The program will not use a fake fallback.\n\n"
        "Install it in the same Python used to run this app:\n"
        "python -m pip install --upgrade pip\n"
        "python -m pip install torch torchvision torchaudio\n\n"
        "Then verify:\n"
        "python -c \"import torch; print(torch.__version__)\"\n\n"
        f"Import error: {TORCH_IMPORT_ERROR or 'torch not installed'}"
    )

try:
    from arch import arch_model  # noqa
    HAS_ARCH = True
except Exception:
    HAS_ARCH = False

# SmartAPI/logzero writes to a relative ``logs`` directory.  Windows Store
# Python or a protected launch directory can make that path unwritable and cause
# ``[WinError 5] Access is denied: 'logs'`` before any broker request is sent.
# Move the process to a private writable runtime folder before importing SmartAPI.
def _prepare_smartapi_runtime_dir():
    candidates = []
    local_appdata = os.getenv("LOCALAPPDATA")
    if local_appdata:
        candidates.append(os.path.join(local_appdata, "HELM_FINSERV", "runtime"))
    candidates.append(os.path.join(os.path.expanduser("~"), ".helm_finserv", "runtime"))
    candidates.append(os.path.join(os.getenv("TEMP", os.path.expanduser("~")), "HELM_FINSERV", "runtime"))

    last_error = None
    for folder in candidates:
        try:
            os.makedirs(os.path.join(folder, "logs"), exist_ok=True)
            probe = os.path.join(folder, ".write_test")
            with open(probe, "w", encoding="utf-8") as fh:
                fh.write("ok")
            os.remove(probe)
            os.chdir(folder)
            return folder
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Could not create a writable SmartAPI runtime folder: {last_error}")


SMARTAPI_RUNTIME_DIR = ""
SMARTAPI_IMPORT_ERROR = ""
try:
    SMARTAPI_RUNTIME_DIR = _prepare_smartapi_runtime_dir()
    from SmartApi import SmartConnect  # type: ignore
    import pyotp  # type: ignore
    HAS_SMARTAPI = True
except Exception as _smartapi_err:
    HAS_SMARTAPI = False
    SMARTAPI_IMPORT_ERROR = str(_smartapi_err)

from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.covariance import LedoitWolf
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize
from scipy.stats import norm

# --------------------------------------------------------------------------- #
#  Theme / constants                                                           #
# --------------------------------------------------------------------------- #
APP_NAME = "HELM FINSERV — ML Portfolio Construction System"
VERSION = "2.1.0"

BG = "#f4f6f9"          # app background
CARD = "#ffffff"        # card surfaces
INK = "#1e293b"         # primary text
MUTED = "#64748b"       # secondary text
ACCENT = "#0d9488"      # teal accent
ACCENT_DK = "#0f766e"
GREEN = "#16a34a"
RED = "#dc2626"
AMBER = "#d97706"
GRID = "#e2e8f0"
FONT = "Segoe UI"

TRADING_DAYS = 252
MONTH_DAYS = 21         # ~1 trading month horizon

DEFAULT_UNIVERSE = ["SBIN", "RELIANCE", "TCS", "INFY", "HDFCBANK"]
BENCHMARK = "NIFTY"

# Angel One token defaults for common NSE cash symbols and NIFTY index.
# If a symbol is not listed here, the live loader will try SmartAPI searchScrip()
# automatically, so the GUI no longer needs a manual Token Map JSON box.
DEFAULT_TOKEN_MAP = {
    "RELIANCE": "2885",
    "TCS": "11536",
    "INFY": "1594",
    "HDFCBANK": "1333",
    "ICICIBANK": "4963",
    "SBIN": "3045",
    "ITC": "1660",
    "LT": "11483",
    "AXISBANK": "5900",
    "KOTAKBANK": "1922",
    "HINDUNILVR": "1394",
    "BHARTIARTL": "10604",
    "WIPRO": "3787",
    "MARUTI": "10999",
    "NIFTY": "99926000",
    "NIFTY50": "99926000",
}

# Rough sector map (used for sector-cap constraints in the optimizer).
SECTOR_MAP = {
    "SBIN": "Financials",
    "HDFCBANK": "Financials",
    "RELIANCE": "Energy",
    "TCS": "IT",
    "INFY": "IT",
    "NIFTY": "Index",
}

RISK_FREE_ANNUAL = 0.065   # ~6.5% India risk-free (RBI repo-ish)
RISK_FREE_DAILY = RISK_FREE_ANNUAL / TRADING_DAYS


# --------------------------------------------------------------------------- #
#  Small numeric utilities                                                     #
# --------------------------------------------------------------------------- #
def safe_div(a, b, default=0.0):
    b = np.asarray(b, dtype=float)
    out = np.divide(np.asarray(a, dtype=float), b,
                    out=np.full_like(np.asarray(a, dtype=float), default),
                    where=b != 0)
    return out


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def annualize_return(mu_daily):
    return (1.0 + mu_daily) ** TRADING_DAYS - 1.0


def annualize_vol(sigma_daily):
    return sigma_daily * math.sqrt(TRADING_DAYS)


def business_days(n, end=None):
    """Return a DatetimeIndex of n business days ending at `end`.

    Over-request then slice to dodge the classic bdate_range weekend
    off-by-one (a HELM recurring fix).
    """
    end = end or datetime.today()
    idx = pd.bdate_range(end=end, periods=int(n * 1.6) + 10)
    return idx[-n:]


def to_pct(x, dp=2):
    try:
        return f"{100.0 * float(x):.{dp}f}%"
    except Exception:
        return "—"


# =========================================================================== #
#  STEP 1 — DATA LAYER                                                         #
# =========================================================================== #
class SyntheticDataGenerator:
    """Volatility-clustered GBM price paths + synthetic fundamentals/sentiment.

    Used for demo / CI when Angel One SmartAPI is unavailable. Paths are tuned
    to look plausible (mild trends, GARCH-like vol clustering, sensible
    cross-correlation to a common market factor) rather than pathological.
    """

    def __init__(self, seed=42):
        self.rng = np.random.default_rng(seed)

    def _garch_vol(self, n, base=0.012, omega=None, alpha=0.08, beta=0.90):
        omega = omega if omega is not None else base ** 2 * (1 - alpha - beta)
        var = np.empty(n)
        var[0] = base ** 2
        shock = self.rng.standard_normal(n)
        for t in range(1, n):
            var[t] = omega + alpha * (shock[t - 1] * math.sqrt(var[t - 1])) ** 2 + beta * var[t - 1]
        return np.sqrt(np.maximum(var, 1e-8))

    def generate(self, tickers, n_days=750, end=None):
        """Return {ticker: DataFrame[open,high,low,close,volume]} + NIFTY."""
        end = end or datetime.today()
        dates = business_days(n_days, end)
        n = len(dates)

        # Common market factor (drives cross-correlation / beta).
        mkt_vol = self._garch_vol(n, base=0.009)
        mkt_ret = 0.00035 + mkt_vol * self.rng.standard_normal(n)

        specs = {
            "SBIN":     dict(mu=0.0006, beta=1.15, vol=0.014, p0=780),
            "RELIANCE": dict(mu=0.0005, beta=1.05, vol=0.013, p0=2900),
            "TCS":      dict(mu=0.0004, beta=0.85, vol=0.011, p0=3850),
            "INFY":     dict(mu=0.00035, beta=0.95, vol=0.012, p0=1550),
            "HDFCBANK": dict(mu=0.00045, beta=1.00, vol=0.012, p0=1680),
        }
        out = {}
        all_names = list(dict.fromkeys(list(tickers) + [BENCHMARK]))
        for tk in all_names:
            if tk == BENCHMARK:
                ret = mkt_ret.copy()
                p0 = 24000.0
            else:
                s = specs.get(tk, dict(mu=0.0004, beta=1.0, vol=0.013, p0=1000))
                idio_vol = self._garch_vol(n, base=s["vol"])
                idio = idio_vol * self.rng.standard_normal(n)
                ret = s["mu"] + s["beta"] * (mkt_ret - 0.00035) + idio
                p0 = s["p0"]
            close = p0 * np.exp(np.cumsum(ret))
            # Build OHLC around close with intraday noise.
            intraday = np.abs(self.rng.standard_normal(n)) * close * 0.006
            openp = np.concatenate([[close[0]], close[:-1]]) * (
                1 + self.rng.standard_normal(n) * 0.002)
            high = np.maximum(openp, close) + intraday
            low = np.minimum(openp, close) - intraday
            vol = (self.rng.lognormal(mean=13.5, sigma=0.4, size=n)).astype(float)
            df = pd.DataFrame(
                {"open": openp, "high": high, "low": low, "close": close, "volume": vol},
                index=dates,
            )
            out[tk] = df
        return out

    def fundamentals(self, tickers):
        base = {
            "SBIN":     dict(mcap=7.1e12, pe=9.8, pb=1.6, roe=17.0, de=0.15),
            "RELIANCE": dict(mcap=19.5e12, pe=24.5, pb=2.1, roe=9.0, de=0.35),
            "TCS":      dict(mcap=14.0e12, pe=28.0, pb=13.5, roe=48.0, de=0.05),
            "INFY":     dict(mcap=6.4e12, pe=24.0, pb=8.2, roe=32.0, de=0.08),
            "HDFCBANK": dict(mcap=12.8e12, pe=18.5, pb=2.7, roe=16.5, de=0.20),
        }
        rows = {}
        for tk in tickers:
            b = base.get(tk, dict(mcap=1e12, pe=20.0, pb=3.0, roe=15.0, de=0.3))
            # small jitter so demo isn't perfectly static
            j = 1 + self.rng.standard_normal() * 0.02
            rows[tk] = dict(market_cap=b["mcap"] * j, pe=b["pe"] * j,
                            pb=b["pb"] * j, roe=b["roe"], de=b["de"])
        return rows

    def sentiment(self, tickers):
        """Synthetic news sentiment in [-1, 1] with a few headlines each."""
        pos = ["beats profit estimates", "brokerage upgrade", "record quarterly revenue",
               "new order win", "management guides higher"]
        neg = ["margin pressure flagged", "downgrade on valuation", "regulatory probe",
               "weak guidance", "asset quality concerns"]
        out = {}
        for tk in tickers:
            score = float(np.clip(self.rng.normal(0.1, 0.35), -1, 1))
            heads = []
            for _ in range(3):
                pick = pos if self.rng.random() < (0.5 + score / 2) else neg
                heads.append(f"{tk}: {self.rng.choice(pick)}")
            out[tk] = dict(score=score, headlines=heads)
        return out


class AngelOneClient:
    """Thin Angel One SmartAPI wrapper (getCandleData + ltpData).

    Auth uses TOTP. Kept optional; if the SDK/credentials are unavailable the
    caller should fall back to the synthetic generator.
    """

    def __init__(self, api_key=None, client_id=None, password=None, totp_secret=None):
        self.api_key = api_key or os.getenv("ANGEL_API_KEY")
        self.client_id = client_id or os.getenv("ANGEL_CLIENT_ID")
        self.password = password or os.getenv("ANGEL_PASSWORD")
        self.totp_secret = totp_secret or os.getenv("ANGEL_TOTP_SECRET")
        self.smart = None
        self.connected = False

    def connect(self):
        if not HAS_SMARTAPI:
            raise RuntimeError(f"SmartApi/pyotp unavailable: {SMARTAPI_IMPORT_ERROR or 'not installed'}")
        if not all([self.api_key, self.client_id, self.password, self.totp_secret]):
            raise RuntimeError("Missing Angel One credentials.")
        self.smart = SmartConnect(api_key=self.api_key)
        otp = pyotp.TOTP(self.totp_secret).now()
        data = self.smart.generateSession(self.client_id, self.password, otp)
        if not data or not data.get("status"):
            raise RuntimeError(f"Angel One login failed: {data}")
        self.connected = True
        return True

    def get_candles(self, symboltoken, exchange="NSE", interval="ONE_DAY",
                    fromdate=None, todate=None):
        todate = todate or datetime.today()
        fromdate = fromdate or (todate - timedelta(days=1200))
        params = {
            "exchange": exchange,
            "symboltoken": str(symboltoken),
            "interval": interval,
            "fromdate": fromdate.strftime("%Y-%m-%d %H:%M"),
            "todate": todate.strftime("%Y-%m-%d %H:%M"),
        }
        resp = self.smart.getCandleData(params)
        rows = resp.get("data", []) if isinstance(resp, dict) else []
        if not rows:
            raise RuntimeError("No candle data returned.")
        df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
        df["ts"] = pd.to_datetime(df["ts"])
        return df.set_index("ts")[["open", "high", "low", "close", "volume"]].astype(float)

    def ltp(self, tradingsymbol, symboltoken, exchange="NSE"):
        resp = self.smart.ltpData(exchange, tradingsymbol, str(symboltoken))
        return float(resp["data"]["ltp"])

    def search_token(self, symbol, exchange="NSE"):
        """Resolve an Angel One symbol token using SmartAPI searchScrip.

        Users type normal symbols like SBIN or RELIANCE. Angel needs numeric
        symbol tokens. This method searches the broker instrument master and
        prefers exact equity matches such as SBIN-EQ.
        """
        if self.smart is None:
            raise RuntimeError("Angel One session not connected.")

        query = symbol.upper().replace("-EQ", "")
        resp = self.smart.searchScrip(exchange, query)
        rows = resp.get("data", []) if isinstance(resp, dict) else []
        if not rows:
            raise RuntimeError(f"No Angel token found for {symbol}.")

        def score(row):
            ts = str(row.get("tradingsymbol", "")).upper()
            name = str(row.get("name", "")).upper()
            token = str(row.get("symboltoken", ""))
            exact_eq = ts == f"{query}-EQ"
            exact_name = name == query or ts == query
            cash_like = ts.endswith("-EQ")
            return (exact_eq, exact_name, cash_like, -len(ts), token)

        best = sorted(rows, key=score, reverse=True)[0]
        token = best.get("symboltoken")
        if token is None:
            raise RuntimeError(f"Angel search result had no token for {symbol}.")
        return str(token)


@dataclass
class MarketData:
    prices: dict                # {ticker: OHLCV DataFrame}
    fundamentals: dict          # {ticker: {...}}
    sentiment: dict             # {ticker: {score, headlines}}
    source: str = "synthetic"

    @property
    def tickers(self):
        return [t for t in self.prices if t != BENCHMARK]

    def close_matrix(self, tickers=None):
        tickers = tickers or self.tickers
        cols = {t: self.prices[t]["close"] for t in tickers}
        return pd.DataFrame(cols).dropna()

    def returns_matrix(self, tickers=None):
        return self.close_matrix(tickers).pct_change().dropna()

    def benchmark_returns(self):
        return self.prices[BENCHMARK]["close"].pct_change().dropna()


class DataLoader:
    """Unified loader: live Angel One if configured, else synthetic demo."""

    def __init__(self, use_live=False, credentials=None, seed=42):
        self.requested_live = use_live
        self.use_live = use_live and HAS_SMARTAPI
        self.credentials = credentials or {}
        self.synth = SyntheticDataGenerator(seed=seed)
        self.last_live_error = None

    def load(self, tickers=None, n_days=750, token_map=None, log=print):
        tickers = tickers or DEFAULT_UNIVERSE
        self.last_live_error = None
        if self.use_live:
            try:
                return self._load_live(tickers, n_days, token_map or {}, log)
            except Exception as e:
                self.last_live_error = str(e)
                log(f"[DataLoader] Live load failed ({e}); using synthetic demo.")
        elif self.requested_live and not HAS_SMARTAPI:
            self.last_live_error = ("SmartApi/pyotp not installed in this Python "
                                     "environment (pip install smartapi-python pyotp).")
        prices = self.synth.generate(tickers, n_days=n_days)
        return MarketData(
            prices=prices,
            fundamentals=self.synth.fundamentals(tickers),
            sentiment=self.synth.sentiment(tickers),
            source="synthetic",
        )

    def _resolve_token(self, client, symbol, user_token_map, cache, log):
        sym = symbol.upper().replace("-EQ", "")
        if sym in cache:
            return cache[sym]
        token = (user_token_map or {}).get(sym) or DEFAULT_TOKEN_MAP.get(sym)
        if token:
            cache[sym] = str(token)
            return cache[sym]
        token = client.search_token(sym)
        cache[sym] = str(token)
        log(f"[DataLoader] Auto-resolved {sym} -> token {token}.")
        return cache[sym]

    def _load_live(self, tickers, n_days, token_map, log):
        client = AngelOneClient(**self.credentials)
        client.connect()
        log(f"[DataLoader] Angel One session established. Runtime: {SMARTAPI_RUNTIME_DIR}")
        prices = {}
        token_cache = {}
        requested = list(dict.fromkeys(tickers + [BENCHMARK]))
        for tk in requested:
            tok = self._resolve_token(client, tk, token_map, token_cache, log)
            prices[tk] = client.get_candles(tok)
            log(f"[DataLoader] Loaded {tk} using token {tok}.")
            time.sleep(0.35)  # be gentle with rate limits
        missing = [t for t in tickers if t not in prices]
        if missing:
            raise RuntimeError(f"No live prices loaded for: {', '.join(missing)}")
        if BENCHMARK not in prices:
            log("[DataLoader] NIFTY unavailable; synthetic benchmark will be used.")
            prices[BENCHMARK] = self.synth.generate([BENCHMARK], n_days=n_days)[BENCHMARK]
        # Fundamentals/sentiment aren't in the candle API; use synthetic stand-ins.
        return MarketData(
            prices=prices,
            fundamentals=self.synth.fundamentals(tickers),
            sentiment=self.synth.sentiment(tickers),
            source="angelone",
        )


# =========================================================================== #
#  STEP 2 — FEATURE ENGINEERING                                               #
# =========================================================================== #
def rsi(close, period=14):
    delta = close.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = safe_div(up.values, down.values, default=0.0)
    return pd.Series(100 - 100 / (1 + rs), index=close.index)


def macd(close, fast=12, slow=26, signal=9):
    ema_f = close.ewm(span=fast, adjust=False).mean()
    ema_s = close.ewm(span=slow, adjust=False).mean()
    line = ema_f - ema_s
    sig = line.ewm(span=signal, adjust=False).mean()
    return line, sig, line - sig


def rolling_drawdown(close, window=63):
    roll_max = close.rolling(window, min_periods=1).max()
    return close / roll_max - 1.0


class FeatureEngineer:
    """Turn raw OHLCV + fundamentals + sentiment into a per-(date,stock) matrix."""

    FEATURES = [
        "ret_1d", "ret_1w", "ret_1m", "vol_21d", "momentum_3m", "rsi_14",
        "macd_hist", "ma_ratio", "vol_chg", "beta_63", "drawdown_63",
        "pe", "pb", "roe", "de", "sentiment",
    ]

    def build(self, md: MarketData, tickers=None):
        tickers = tickers or md.tickers
        bench_ret = md.benchmark_returns()
        frames = []
        for tk in tickers:
            px = md.prices[tk]
            close, vol = px["close"], px["volume"]
            r1 = close.pct_change()
            f = pd.DataFrame(index=close.index)
            f["ret_1d"] = r1
            f["ret_1w"] = close.pct_change(5)
            f["ret_1m"] = close.pct_change(MONTH_DAYS)
            f["vol_21d"] = r1.rolling(21).std()
            f["momentum_3m"] = close.pct_change(63)
            f["rsi_14"] = rsi(close) / 100.0
            _, _, hist = macd(close)
            f["macd_hist"] = hist / close  # normalise
            ma20 = close.rolling(20).mean()
            ma50 = close.rolling(50).mean()
            f["ma_ratio"] = safe_div(ma20.values, ma50.values, 1.0) - 1.0
            f["vol_chg"] = vol.pct_change(5)
            # rolling beta vs benchmark
            aligned = pd.concat([r1, bench_ret], axis=1, join="inner").dropna()
            aligned.columns = ["s", "m"]
            cov = aligned["s"].rolling(63).cov(aligned["m"])
            var = aligned["m"].rolling(63).var()
            beta = safe_div(cov.values, var.values, 1.0)
            f["beta_63"] = pd.Series(beta, index=aligned.index).reindex(f.index)
            f["drawdown_63"] = rolling_drawdown(close, 63)
            # fundamentals (static per stock in demo) + sentiment
            fu = md.fundamentals.get(tk, {})
            f["pe"] = fu.get("pe", np.nan)
            f["pb"] = fu.get("pb", np.nan)
            f["roe"] = fu.get("roe", np.nan) / 100.0
            f["de"] = fu.get("de", np.nan)
            f["sentiment"] = md.sentiment.get(tk, {}).get("score", 0.0)
            f["stock"] = tk
            frames.append(f)
        feat = pd.concat(frames).reset_index()

        # -------------------------------------------------------------
        # Robust date-column fix
        # -------------------------------------------------------------
        # Pandas uses the index name after reset_index(). If the price
        # index is named "datetime", "timestamp", "Date", etc., the
        # column will NOT be called "index". Later we merge on "date",
        # so force the first reset_index column to a standard name.
        if "date" not in feat.columns:
            first_col = feat.columns[0]
            if first_col not in ["stock"] + self.FEATURES:
                feat = feat.rename(columns={first_col: "date"})

        # Final safety: create a date column from the index if somehow
        # it is still missing. This prevents KeyError: 'date'.
        if "date" not in feat.columns:
            feat["date"] = feat.index

        feat["date"] = pd.to_datetime(feat["date"], errors="coerce")
        feat = feat.replace([np.inf, -np.inf], np.nan)
        return feat


# =========================================================================== #
#  STEP 3 — TARGET VARIABLE (next 1-month return)                             #
# =========================================================================== #
def build_targets(md: MarketData, horizon=MONTH_DAYS, tickers=None):
    tickers = tickers or md.tickers
    rows = []
    for tk in tickers:
        close = md.prices[tk]["close"]
        fwd = close.shift(-horizon) / close - 1.0
        tmp = pd.DataFrame({
            "date": pd.to_datetime(close.index, errors="coerce"),
            "stock": tk,
            "target": fwd.values
        })
        rows.append(tmp)
    out = pd.concat(rows).reset_index(drop=True)
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    return out


def assemble_dataset(md: MarketData, horizon=MONTH_DAYS, tickers=None):
    """Return a clean (features + target) panel keyed by (date, stock)."""
    fe = FeatureEngineer()
    feat = fe.build(md, tickers)
    tgt = build_targets(md, horizon, tickers)

    # Defensive validation so GUI shows a useful error instead of a pandas
    # traceback when date construction fails.
    for name, df in [("features", feat), ("targets", tgt)]:
        missing = [c for c in ["date", "stock"] if c not in df.columns]
        if missing:
            raise ValueError(
                f"Feature engineering failed: {name} table missing {missing}. "
                f"Columns found: {list(df.columns)}"
            )

    feat["date"] = pd.to_datetime(feat["date"], errors="coerce")
    tgt["date"] = pd.to_datetime(tgt["date"], errors="coerce")
    feat = feat.dropna(subset=["date", "stock"])
    tgt = tgt.dropna(subset=["date", "stock"])

    data = feat.merge(tgt, on=["date", "stock"], how="left")
    data = data.sort_values(["date", "stock"]).reset_index(drop=True)
    return data, fe.FEATURES


# =========================================================================== #
#  STEP 4 — ML MODELS (unified interface)                                     #
# =========================================================================== #
class BaseReturnModel:
    """Common interface: fit(X, y) / predict(X). X is a 2-D float array."""
    name = "base"
    needs_sequence = False

    def __init__(self, **kw):
        self.kw = kw
        self.model = None
        self.scaler = StandardScaler()
        self.feature_importances_ = None

    def _prep(self, X, fit=False):
        X = np.asarray(X, dtype=float)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        return self.scaler.fit_transform(X) if fit else self.scaler.transform(X)

    def fit(self, X, y):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError


class RandomForestModel(BaseReturnModel):
    name = "Random Forest"

    def fit(self, X, y):
        Xs = self._prep(X, fit=True)
        self.model = RandomForestRegressor(
            n_estimators=self.kw.get("n_estimators", 300),
            max_depth=self.kw.get("max_depth", 6),
            min_samples_leaf=self.kw.get("min_samples_leaf", 20),
            n_jobs=-1, random_state=42,
        )
        self.model.fit(Xs, np.asarray(y, float))
        self.feature_importances_ = self.model.feature_importances_
        return self

    def predict(self, X):
        return self.model.predict(self._prep(X))


class XGBoostModel(BaseReturnModel):
    """XGBoost if available, else sklearn HistGradientBoosting surrogate."""
    name = "XGBoost" if HAS_XGB else "Gradient Boosting (XGB fallback)"

    def fit(self, X, y):
        Xs = self._prep(X, fit=True)
        y = np.asarray(y, float)
        if HAS_XGB:
            self.model = xgb.XGBRegressor(
                n_estimators=self.kw.get("n_estimators", 400),
                max_depth=self.kw.get("max_depth", 4),
                learning_rate=self.kw.get("learning_rate", 0.03),
                subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                n_jobs=-1, random_state=42,
            )
            self.model.fit(Xs, y)
            self.feature_importances_ = self.model.feature_importances_
        else:
            self.model = HistGradientBoostingRegressor(
                max_depth=self.kw.get("max_depth", 4),
                learning_rate=self.kw.get("learning_rate", 0.05),
                max_iter=self.kw.get("n_estimators", 400),
                l2_regularization=1.0, random_state=42,
            )
            self.model.fit(Xs, y)
            # HGB has no native importances; approximate via permutation-lite.
            self.feature_importances_ = self._perm_importance(Xs, y)
        return self

    def _perm_importance(self, Xs, y):
        base = np.mean((self.model.predict(Xs) - y) ** 2)
        imp = np.zeros(Xs.shape[1])
        rng = np.random.default_rng(0)
        for j in range(Xs.shape[1]):
            Xp = Xs.copy()
            Xp[:, j] = rng.permutation(Xp[:, j])
            imp[j] = max(0.0, np.mean((self.model.predict(Xp) - y) ** 2) - base)
        s = imp.sum()
        return imp / s if s > 0 else np.ones_like(imp) / len(imp)

    def predict(self, X):
        return self.model.predict(self._prep(X))


class _TorchLSTMNet(nn.Module if HAS_TORCH else object):
    """Small LSTM regressor for sequence expected-return forecasting."""
    def __init__(self, n_features, hidden=48, dropout=0.15):
        if not HAS_TORCH:
            return
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden,
                            num_layers=2, batch_first=True, dropout=dropout)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, 24), nn.ReLU(), nn.Dropout(dropout), nn.Linear(24, 1)
        )
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


class _TorchTransformerNet(nn.Module if HAS_TORCH else object):
    """Small Transformer encoder regressor for sequence expected-return forecasting."""
    def __init__(self, n_features, d_model=48, nhead=4, layers=2, dropout=0.15, max_len=128):
        if not HAS_TORCH:
            return
        super().__init__()
        self.proj = nn.Linear(n_features, d_model)
        self.pos = nn.Parameter(torch.zeros(1, max_len, d_model))
        enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                         dim_feedforward=96, dropout=dropout,
                                         batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(enc, num_layers=layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, 24), nn.ReLU(), nn.Dropout(dropout), nn.Linear(24, 1)
        )
    def forward(self, x):
        z = self.proj(x)
        z = z + self.pos[:, :z.shape[1], :]
        z = self.encoder(z)
        return self.head(z[:, -1, :]).squeeze(-1)


class _SeqSurrogate(BaseReturnModel):
    """Real PyTorch sequence models.

    The model consumes a rolling window shaped (samples, lookback_days, features).
    LSTM and Transformer are trained on true time-ordered sequences, not shuffled rows.
    No sklearn fallback is allowed, because that creates misleading results.
    """
    needs_sequence = True
    torch_kind = "lstm"

    def __init__(self, **kw):
        super().__init__(**kw)
        self.lookback = int(self.kw.get("lookback", 20))
        self.seq_scaler = StandardScaler()
        self.device = None

    def _prep_seq(self, Xseq, fit=False):
        Xseq = np.asarray(Xseq, dtype=float)
        Xseq = np.nan_to_num(Xseq, nan=0.0, posinf=0.0, neginf=0.0)
        n, t, f = Xseq.shape
        flat = Xseq.reshape(n * t, f)
        flat = self.seq_scaler.fit_transform(flat) if fit else self.seq_scaler.transform(flat)
        return flat.reshape(n, t, f).astype(np.float32)

    def fit_seq(self, Xseq, y):
        y = np.asarray(y, dtype=np.float32)
        Xs = self._prep_seq(Xseq, fit=True)
        require_torch_for_deep(self.name)
        torch.manual_seed(42)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_features = Xs.shape[2]
        if self.torch_kind == "transformer":
            self.model = _TorchTransformerNet(n_features).to(self.device)
        else:
            self.model = _TorchLSTMNet(n_features).to(self.device)
        opt = torch.optim.AdamW(self.model.parameters(), lr=1e-3, weight_decay=1e-4)
        loss_fn = nn.SmoothL1Loss()
        X_t = torch.tensor(Xs, dtype=torch.float32, device=self.device)
        y_t = torch.tensor(y, dtype=torch.float32, device=self.device)
        n = len(y)
        batch = min(64, max(8, n))
        epochs = int(self.kw.get("epochs", 12))
        self.model.train()
        for _ in range(epochs):
            perm = torch.randperm(n, device=self.device)
            for i in range(0, n, batch):
                idx = perm[i:i+batch]
                opt.zero_grad()
                pred = self.model(X_t[idx])
                loss = loss_fn(pred, y_t[idx])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()
        return self

    def predict_seq(self, Xseq):
        Xs = self._prep_seq(Xseq, fit=False)
        require_torch_for_deep(self.name)
        self.model.eval()
        with torch.no_grad():
            X_t = torch.tensor(Xs, dtype=torch.float32, device=self.device)
            return self.model(X_t).detach().cpu().numpy().astype(float)

    def fit(self, X, y):
        # Defensive flat path: treat each row as a one-step sequence.
        X = np.asarray(X, dtype=float)
        return self.fit_seq(X.reshape(X.shape[0], 1, X.shape[1]), y)

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return self.predict_seq(X.reshape(X.shape[0], 1, X.shape[1]))


class LSTMModel(_SeqSurrogate):
    name = "Deep LSTM"
    torch_kind = "lstm"


class TransformerModel(_SeqSurrogate):
    name = "Transformer Encoder"
    torch_kind = "transformer"


MODEL_REGISTRY = {
    "Random Forest": RandomForestModel,
    "XGBoost": XGBoostModel,
    "LSTM": LSTMModel,
    "Transformer": TransformerModel,
}


# =========================================================================== #
#  STEP 5 — VALIDATION (time-series split, rank-aware metrics)                #
# =========================================================================== #
def rmse(y, yhat):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(yhat)) ** 2)))


def mae(y, yhat):
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(yhat))))


def directional_accuracy(y, yhat):
    y, yhat = np.asarray(y), np.asarray(yhat)
    return float(np.mean(np.sign(y) == np.sign(yhat)))


def information_coefficient(y, yhat):
    """Spearman rank correlation between forecast and realised return."""
    y, yhat = np.asarray(y), np.asarray(yhat)
    if len(y) < 3:
        return 0.0
    ry = pd.Series(y).rank().values
    rp = pd.Series(yhat).rank().values
    if np.std(ry) == 0 or np.std(rp) == 0:
        return 0.0
    return float(np.corrcoef(ry, rp)[0, 1])


def top_decile_return(y, yhat):
    y, yhat = np.asarray(y), np.asarray(yhat)
    if len(y) < 10:
        return float(np.mean(y))
    k = max(1, len(y) // 10)
    top = np.argsort(yhat)[-k:]
    return float(np.mean(y[top]))


def hit_ratio(y, yhat):
    """Fraction of picks above the cross-sectional median that outperformed."""
    y, yhat = np.asarray(y), np.asarray(yhat)
    if len(y) < 4:
        return 0.5
    picked = yhat >= np.median(yhat)
    if picked.sum() == 0:
        return 0.5
    return float(np.mean(y[picked] >= np.median(y)))


def evaluate_predictions(y, yhat):
    return dict(
        rmse=rmse(y, yhat), mae=mae(y, yhat),
        dir_acc=directional_accuracy(y, yhat),
        ic=information_coefficient(y, yhat),
        top_decile=top_decile_return(y, yhat),
        hit_ratio=hit_ratio(y, yhat),
    )


def time_series_folds(dates, n_folds=3, min_train_frac=0.4):
    """Yield (train_mask, test_mask) expanding-window folds by calendar date."""
    uniq = np.array(sorted(pd.unique(dates)))
    n = len(uniq)
    start = int(n * min_train_frac)
    if start >= n - 1:
        start = max(1, n // 2)
    bounds = np.linspace(start, n, n_folds + 1).astype(int)
    dates = np.asarray(dates)
    for i in range(n_folds):
        tr_end = bounds[i]
        te_end = bounds[i + 1]
        if te_end <= tr_end:
            continue
        train_dates = set(uniq[:tr_end])
        test_dates = set(uniq[tr_end:te_end])
        yield (np.array([d in train_dates for d in dates]),
               np.array([d in test_dates for d in dates]))


# =========================================================================== #
#  STEP 6 — ML EXPECTED-RETURN ENGINE                                         #
# =========================================================================== #
@dataclass
class TrainResult:
    model_name: str
    cv_metrics: dict
    feature_importances: dict
    forecasts: dict          # {stock: monthly expected return}
    error_bands: dict        # {stock: +/- band on the forecast}
    trained_model: object = None
    features: list = field(default_factory=list)


class ExpectedReturnEngine:
    """Trains a chosen ML model, validates with time-series CV, and produces a
    monthly forecast per stock (plus an uncertainty band for robust optimisation).
    """

    def __init__(self, features):
        self.features = features
        self.lookback = 20

    def _xy(self, data):
        d = data.dropna(subset=self.features + ["target"])
        return d, d[self.features].values, d["target"].values

    def _seq_xy(self, data, lookback=None):
        """Build true rolling windows for LSTM/Transformer.

        Output:
          Xseq: (samples, lookback, features)
          y: target at the window end date
          end_dates/stocks: metadata used for time-series folds and latest forecasts
        """
        lookback = int(lookback or self.lookback)
        d = data.dropna(subset=self.features + ["target", "date", "stock"]).copy()
        d["date"] = pd.to_datetime(d["date"], errors="coerce")
        d = d.dropna(subset=["date"]).sort_values(["stock", "date"])
        Xs, ys, dates, stocks = [], [], [], []
        for tk, g in d.groupby("stock"):
            g = g.sort_values("date")
            arr = g[self.features].values.astype(float)
            target = g["target"].values.astype(float)
            dt = g["date"].values
            if len(g) < lookback + 1:
                continue
            for i in range(lookback - 1, len(g)):
                if not np.isfinite(target[i]):
                    continue
                Xs.append(arr[i - lookback + 1:i + 1])
                ys.append(target[i])
                dates.append(dt[i])
                stocks.append(tk)
        if not Xs:
            return (np.empty((0, lookback, len(self.features))), np.array([]),
                    np.array([]), np.array([]))
        return np.asarray(Xs, float), np.asarray(ys, float), np.asarray(dates), np.asarray(stocks)

    def _permutation_importance(self, model, X, y, repeats=3):
        """Model-agnostic feature importance for 2-D models."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        base_pred = np.asarray(model.predict(X), dtype=float)
        base = np.mean((base_pred - y) ** 2)
        rng = np.random.default_rng(42)
        imp = np.zeros(X.shape[1], dtype=float)
        for j in range(X.shape[1]):
            scores = []
            for _ in range(repeats):
                Xp = X.copy()
                Xp[:, j] = rng.permutation(Xp[:, j])
                pred = np.asarray(model.predict(Xp), dtype=float)
                scores.append(max(0.0, np.mean((pred - y) ** 2) - base))
            imp[j] = float(np.mean(scores))
        total = imp.sum()
        if total <= 0 or not np.isfinite(total):
            return np.ones(X.shape[1], dtype=float) / max(1, X.shape[1])
        return imp / total

    def _permutation_importance_seq(self, model, Xseq, y, repeats=2):
        """Permutation importance for sequence models.

        A whole feature channel is shuffled across samples but kept across its
        lookback window, so the importance is reported for the original 16 features.
        """
        Xseq = np.asarray(Xseq, dtype=float)
        y = np.asarray(y, dtype=float)
        base = np.mean((np.asarray(model.predict_seq(Xseq)) - y) ** 2)
        rng = np.random.default_rng(123)
        imp = np.zeros(Xseq.shape[2], dtype=float)
        for j in range(Xseq.shape[2]):
            scores = []
            for _ in range(repeats):
                Xp = Xseq.copy()
                order = rng.permutation(Xp.shape[0])
                Xp[:, :, j] = Xp[order, :, j]
                pred = np.asarray(model.predict_seq(Xp), dtype=float)
                scores.append(max(0.0, np.mean((pred - y) ** 2) - base))
            imp[j] = float(np.mean(scores))
        total = imp.sum()
        if total <= 0 or not np.isfinite(total):
            return np.ones(Xseq.shape[2], dtype=float) / max(1, Xseq.shape[2])
        return imp / total

    def cross_validate(self, data, model_name, n_folds=3, log=print):
        proto = MODEL_REGISTRY[model_name]()
        agg = []
        if getattr(proto, "needs_sequence", False):
            Xseq, y, dates, _stocks = self._seq_xy(data, lookback=self.lookback)
            if len(y) == 0:
                return dict(rmse=float("nan"), mae=float("nan"), dir_acc=float("nan"),
                            ic=float("nan"), top_decile=float("nan"), hit_ratio=float("nan"))
            for tr, te in time_series_folds(dates, n_folds=n_folds):
                if tr.sum() < 30 or te.sum() < 5:
                    continue
                m = MODEL_REGISTRY[model_name]()
                m.fit_seq(Xseq[tr], y[tr])
                pred = m.predict_seq(Xseq[te])
                agg.append(evaluate_predictions(y[te], pred))
        else:
            d, _, _ = self._xy(data)
            for tr, te in time_series_folds(d["date"].values, n_folds=n_folds):
                if tr.sum() < 30 or te.sum() < 5:
                    continue
                m = MODEL_REGISTRY[model_name]()
                m.fit(d[self.features].values[tr], d["target"].values[tr])
                pred = m.predict(d[self.features].values[te])
                agg.append(evaluate_predictions(d["target"].values[te], pred))
        if not agg:
            return dict(rmse=float("nan"), mae=float("nan"), dir_acc=float("nan"),
                        ic=float("nan"), top_decile=float("nan"), hit_ratio=float("nan"))
        return {k: float(np.mean([a[k] for a in agg])) for k in agg[0]}

    def quick_leaderboard(self, data, n_folds=3):
        """Small model comparison table for the GUI/report."""
        rows = []
        for name in ["Random Forest", "XGBoost"]:
            try:
                cv = self.cross_validate(data, name, n_folds=n_folds, log=lambda *_: None)
                rows.append((name, cv.get("ic", 0.0), cv.get("dir_acc", 0.0),
                             cv.get("hit_ratio", 0.0), cv.get("rmse", float("nan"))))
            except Exception:
                rows.append((name, float("nan"), float("nan"), float("nan"), float("nan")))
        rows.sort(key=lambda r: ((-999 if math.isnan(r[1]) else r[1]),
                                 (-999 if math.isnan(r[3]) else r[3])), reverse=True)
        return rows

    def train_and_forecast(self, data, model_name, n_folds=3, log=print, leaderboard=False):
        cv = self.cross_validate(data, model_name, n_folds=n_folds, log=log)
        proto = MODEL_REGISTRY[model_name]()
        forecasts, bands = {}, {}
        resid_std = cv.get("rmse", 0.03)
        resid_std = 0.03 if (resid_std is None or math.isnan(resid_std)) else resid_std

        if getattr(proto, "needs_sequence", False):
            Xseq, y, _dates, _stocks = self._seq_xy(data, lookback=self.lookback)
            model = MODEL_REGISTRY[model_name]()
            model.fit_seq(Xseq, y)
            log(f"[Return] Trained {model.name} on {len(y)} rolling windows.")

            # Forecast from each stock's latest lookback window.
            latest = data.dropna(subset=self.features).copy()
            latest["date"] = pd.to_datetime(latest["date"], errors="coerce")
            latest = latest.dropna(subset=["date"]).sort_values(["stock", "date"])
            for tk, grp in latest.groupby("stock"):
                g = grp.sort_values("date")
                if len(g) < self.lookback:
                    continue
                Xlast = g.tail(self.lookback)[self.features].values.astype(float)[None, :, :]
                forecasts[tk] = float(model.predict_seq(Xlast)[0])
                bands[tk] = float(resid_std)
            arr = self._permutation_importance_seq(model, Xseq, y)
        else:
            d, X, y = self._xy(data)
            model = MODEL_REGISTRY[model_name]()
            model.fit(X, y)
            log(f"[Return] Trained {model.name} on {len(d)} rows.")
            latest = data.dropna(subset=self.features).sort_values("date")
            for tk, grp in latest.groupby("stock"):
                row = grp.iloc[[-1]][self.features].values
                forecasts[tk] = float(model.predict(row)[0])
                bands[tk] = float(resid_std)
            native_imp = getattr(model, "feature_importances_", None)
            if native_imp is not None:
                arr = np.asarray(native_imp, dtype=float)
                if arr.sum() > 0:
                    arr = arr / arr.sum()
            else:
                arr = self._permutation_importance(model, X, y)

        imp = {f: float(v) for f, v in zip(self.features, arr)}
        result = TrainResult(
            model_name=model.name, cv_metrics=cv, feature_importances=imp,
            forecasts=forecasts, error_bands=bands, trained_model=model,
            features=self.features,
        )
        if leaderboard:
            result.cv_metrics["leaderboard"] = self.quick_leaderboard(data, n_folds=n_folds)
        return result


# =========================================================================== #
#  STEP 7 — COMBINE WITH OTHER RETURN MODELS (CAPM / Factor / BL / Ensemble)  #
# =========================================================================== #
def capm_expected_returns(md: MarketData, tickers, horizon=MONTH_DAYS):
    """CAPM monthly expected returns: rf + beta * (E[rm] - rf)."""
    bench = md.benchmark_returns()
    mkt_monthly = float(bench.mean() * horizon)
    rf_monthly = RISK_FREE_DAILY * horizon
    mrp = mkt_monthly - rf_monthly
    rets = md.returns_matrix(tickers)
    out = {}
    for tk in tickers:
        aligned = pd.concat([rets[tk], bench], axis=1, join="inner").dropna()
        aligned.columns = ["s", "m"]
        var = aligned["m"].var()
        beta = aligned["s"].cov(aligned["m"]) / var if var > 0 else 1.0
        out[tk] = rf_monthly + beta * mrp
    return out


def factor_expected_returns(data, features, tickers, horizon=MONTH_DAYS):
    """Lightweight cross-sectional factor tilt (value/quality/momentum proxy).

    A hook for the HELM Fama-French / Carhart engine; here it applies a simple
    z-scored linear tilt so the ensemble has a non-ML component to blend.
    """
    latest = data.dropna(subset=["momentum_3m", "pe", "roe"]).sort_values("date")
    rows = latest.groupby("stock").tail(1).set_index("stock")
    if rows.empty:
        return {tk: 0.0 for tk in tickers}

    def z(s):
        s = s.astype(float)
        return (s - s.mean()) / (s.std() + 1e-9)

    tilt = (0.5 * z(rows["momentum_3m"])
            - 0.3 * z(rows["pe"])
            + 0.4 * z(rows["roe"]))
    base = RISK_FREE_DAILY * horizon
    # scale tilt to a plausible monthly spread (~+/-3%)
    scaled = base + 0.015 * tilt
    return {tk: float(scaled.get(tk, base)) for tk in tickers}


def black_litterman_returns(md: MarketData, tickers, ml_forecasts,
                            tau=0.05, horizon=MONTH_DAYS):
    """Compact Black-Litterman: market-implied prior blended with ML 'views'.

    Prior pi = delta * Sigma * w_mkt (reverse optimisation from cap weights).
    Views P=I, Q=ml_forecasts, with per-view uncertainty Omega=diag(tau*Sigma).
    """
    rets = md.returns_matrix(tickers)
    Sigma = rets.cov().values * horizon
    n = len(tickers)
    mcaps = np.array([md.fundamentals.get(t, {}).get("market_cap", 1.0) for t in tickers])
    w_mkt = mcaps / mcaps.sum()
    delta = 2.5  # risk-aversion
    pi = delta * Sigma @ w_mkt
    P = np.eye(n)
    Q = np.array([ml_forecasts.get(t, pi[i]) for i, t in enumerate(tickers)])
    Omega = np.diag(np.diag(tau * Sigma) + 1e-8)
    tS = tau * Sigma
    try:
        inv = np.linalg.inv(np.linalg.inv(tS) + P.T @ np.linalg.inv(Omega) @ P)
        post = inv @ (np.linalg.inv(tS) @ pi + P.T @ np.linalg.inv(Omega) @ Q)
    except np.linalg.LinAlgError:
        post = pi
    return {t: float(post[i]) for i, t in enumerate(tickers)}


def ensemble_returns(components: dict, weights: dict):
    """components: {name: {stock: ret}}; weights: {name: w}. Weights renormalised."""
    names = [n for n in components if weights.get(n, 0) > 0]
    wsum = sum(weights[n] for n in names) or 1.0
    stocks = set().union(*[set(components[n]) for n in names]) if names else set()
    out = {}
    for s in stocks:
        out[s] = sum(weights[n] / wsum * components[n].get(s, 0.0) for n in names)
    return out


# =========================================================================== #
#  STEP 8 — RISK (COVARIANCE) MODELS                                          #
# =========================================================================== #
class CovarianceEngine:
    """Estimate the (daily) covariance matrix Sigma via several methods.

    Mirrors the standalone HELM covariance engine: Historical, EWMA (RiskMetrics
    lambda=0.94), Ledoit-Wolf shrinkage, DCC-GARCH (arch fallback -> DCC-lite),
    and a single-factor (market) covariance.
    """

    METHODS = ["Historical", "EWMA", "Ledoit-Wolf", "DCC-GARCH", "Factor"]

    def __init__(self, returns: pd.DataFrame, benchmark: pd.Series = None):
        self.R = returns.dropna()
        self.tickers = list(self.R.columns)
        self.bench = benchmark

    def estimate(self, method="Ledoit-Wolf"):
        m = {
            "Historical": self._historical,
            "EWMA": self._ewma,
            "Ledoit-Wolf": self._ledoit_wolf,
            "DCC-GARCH": self._dcc_garch,
            "Factor": self._factor,
        }[method]
        Sigma = m()
        return self._psd_fix(pd.DataFrame(Sigma, index=self.tickers, columns=self.tickers))

    # --- individual estimators ------------------------------------------- #
    def _historical(self):
        return self.R.cov().values

    def _ewma(self, lam=0.94):
        X = self.R.values - self.R.values.mean(0)
        n, k = X.shape
        S = np.cov(X, rowvar=False)
        for t in range(n):
            x = X[t][:, None]
            S = lam * S + (1 - lam) * (x @ x.T)
        return S

    def _ledoit_wolf(self):
        lw = LedoitWolf().fit(self.R.values)
        return lw.covariance_

    def _dcc_garch(self):
        if HAS_ARCH and self.R.shape[1] <= 12:
            try:
                return self._dcc_arch()
            except Exception:
                pass
        return self._dcc_lite()

    def _dcc_arch(self):
        std_resid = np.zeros_like(self.R.values)
        cond_vol = np.zeros(self.R.shape[1])
        for j, c in enumerate(self.R.columns):
            am = arch_model(self.R[c].values * 100, vol="Garch", p=1, q=1, dist="normal")
            res = am.fit(disp="off")
            sigma = res.conditional_volatility / 100.0
            std_resid[:, j] = (self.R[c].values) / (sigma + 1e-12)
            cond_vol[j] = sigma[-1]
        Q_bar = np.cov(std_resid, rowvar=False)
        a, b = 0.02, 0.95
        Q = Q_bar.copy()
        for t in range(len(std_resid)):
            z = std_resid[t][:, None]
            Q = (1 - a - b) * Q_bar + a * (z @ z.T) + b * Q
        d = np.sqrt(np.diag(Q))
        Rt = Q / np.outer(d, d)
        D = np.diag(cond_vol)
        return D @ Rt @ D

    def _dcc_lite(self):
        """EWMA univariate vols + EWMA correlation (a robust DCC surrogate)."""
        lam_v, lam_c = 0.94, 0.97
        X = self.R.values
        n, k = X.shape
        var = X.var(0)
        std_resid = np.zeros_like(X)
        vols = np.sqrt(var).copy()
        for t in range(n):
            var = lam_v * var + (1 - lam_v) * X[t] ** 2
            vols = np.sqrt(var)
            std_resid[t] = X[t] / (vols + 1e-12)
        Rt = np.corrcoef(std_resid, rowvar=False)
        C = Rt.copy()
        for t in range(n):
            z = std_resid[t][:, None]
            C = lam_c * C + (1 - lam_c) * (z @ z.T)
        d = np.sqrt(np.diag(C))
        Rt = C / np.outer(d, d)
        D = np.diag(vols)
        return D @ Rt @ D

    def _factor(self):
        """Single-factor (market) covariance: beta*beta'*var_m + diag(idio)."""
        if self.bench is None:
            return self._historical()
        betas, idio = [], []
        aligned = pd.concat([self.R, self.bench.rename("m")], axis=1, join="inner").dropna()
        var_m = aligned["m"].var()
        for c in self.tickers:
            b = aligned[c].cov(aligned["m"]) / var_m if var_m > 0 else 1.0
            resid = aligned[c] - b * aligned["m"]
            betas.append(b)
            idio.append(resid.var())
        betas = np.array(betas)
        return np.outer(betas, betas) * var_m + np.diag(idio)

    @staticmethod
    def _psd_fix(cov: pd.DataFrame):
        vals, vecs = np.linalg.eigh(cov.values)
        vals = np.clip(vals, 1e-10, None)
        fixed = vecs @ np.diag(vals) @ vecs.T
        return pd.DataFrame((fixed + fixed.T) / 2, index=cov.index, columns=cov.columns)


# =========================================================================== #
#  STEP 9 & 10 — OPTIMIZER ENGINE + CONSTRAINTS                               #
# =========================================================================== #
@dataclass
class OptConstraints:
    max_weight: float = 0.30
    min_weight: float = 0.0
    allow_short: bool = False
    sector_cap: float = 0.35
    turnover_limit: float = None       # e.g. 0.20; None = unconstrained
    txn_cost_bps: float = 5.0          # per unit turnover, in bps
    cash_floor: float = 0.0            # min cash fraction (0..1)


@dataclass
class PortfolioResult:
    weights: dict                      # {stock: w}  (may include "Cash")
    exp_return_monthly: float
    volatility_monthly: float
    sharpe: float
    concentration: float               # Herfindahl
    objective: str
    diagnostics: dict = field(default_factory=dict)


class Optimizer:
    OBJECTIVES = ["Max Sharpe", "Min Variance", "Markowitz",
                  "Risk Parity", "CVaR", "Robust"]

    def __init__(self, mu_monthly: dict, Sigma_daily: pd.DataFrame,
                 constraints: OptConstraints = None, prev_weights: dict = None,
                 scenario_returns: np.ndarray = None):
        self.tickers = list(Sigma_daily.columns)
        self.mu = np.array([mu_monthly.get(t, 0.0) for t in self.tickers])
        self.Sigma_m = Sigma_daily.values * MONTH_DAYS   # scale daily->monthly
        self.cons = constraints or OptConstraints()
        self.prev = np.array([(prev_weights or {}).get(t, 0.0) for t in self.tickers])
        self.scen = scenario_returns  # for CVaR: (paths x assets) monthly returns

    # ---- helpers -------------------------------------------------------- #
    def _bounds(self):
        lo = 0.0 if not self.cons.allow_short else -self.cons.max_weight
        lo = max(lo, self.cons.min_weight) if not self.cons.allow_short else lo
        return [(lo, self.cons.max_weight)] * len(self.tickers)

    def _base_constraints(self):
        invest = 1.0 - self.cons.cash_floor
        cons = [{"type": "eq", "fun": lambda w: np.sum(w) - invest}]
        # sector caps
        sectors = {}
        for i, t in enumerate(self.tickers):
            sectors.setdefault(SECTOR_MAP.get(t, "Other"), []).append(i)
        for idxs in sectors.values():
            cons.append({"type": "ineq",
                         "fun": (lambda w, ix=idxs: self.cons.sector_cap - np.sum(w[ix]))})
        # turnover
        if self.cons.turnover_limit is not None:
            cons.append({"type": "ineq",
                         "fun": lambda w: self.cons.turnover_limit - np.sum(np.abs(w - self.prev))})
        return cons

    def _txn_penalty(self, w):
        return (self.cons.txn_cost_bps / 1e4) * np.sum(np.abs(w - self.prev))

    def _port_stats(self, w):
        r = float(self.mu @ w) - self._txn_penalty(w)
        v = float(np.sqrt(max(w @ self.Sigma_m @ w, 1e-12)))
        rf_m = RISK_FREE_DAILY * MONTH_DAYS
        sharpe = (r - rf_m) / v if v > 0 else 0.0
        return r, v, sharpe

    # ---- objective functions -------------------------------------------- #
    def _neg_sharpe(self, w):
        _, _, s = self._port_stats(w)
        return -s

    def _variance(self, w):
        return float(w @ self.Sigma_m @ w)

    def _markowitz(self, w, risk_aversion=3.0):
        r, v, _ = self._port_stats(w)
        return -(r - 0.5 * risk_aversion * v ** 2)

    def _risk_parity(self, w):
        w = np.maximum(w, 1e-8)
        port_var = w @ self.Sigma_m @ w
        mrc = self.Sigma_m @ w
        rc = w * mrc
        target = port_var / len(w)
        return float(np.sum((rc - target) ** 2))

    def _cvar(self, w, alpha=0.95):
        if self.scen is None:
            return self._variance(w)  # fallback
        pnl = self.scen @ w                        # monthly portfolio returns
        losses = -pnl
        var = np.quantile(losses, alpha)
        cvar = losses[losses >= var].mean() if np.any(losses >= var) else var
        return float(cvar)

    def _robust(self, w, error_bands=None, kappa=1.0):
        """Max-Sharpe on uncertainty-penalised returns: mu_adj = mu - kappa*err."""
        eb = self._err
        mu_adj = self.mu - kappa * eb
        r = float(mu_adj @ w) - self._txn_penalty(w)
        v = float(np.sqrt(max(w @ self.Sigma_m @ w, 1e-12)))
        rf_m = RISK_FREE_DAILY * MONTH_DAYS
        return -((r - rf_m) / v if v > 0 else 0.0)

    # ---- driver --------------------------------------------------------- #
    def optimize(self, objective="Max Sharpe", error_bands=None, robust_kappa=1.0):
        n = len(self.tickers)
        invest = 1.0 - self.cons.cash_floor
        w0 = np.full(n, invest / n)
        self._err = np.array([(error_bands or {}).get(t, 0.02) for t in self.tickers])

        fun = {
            "Max Sharpe": self._neg_sharpe,
            "Min Variance": self._variance,
            "Markowitz": self._markowitz,
            "Risk Parity": self._risk_parity,
            "CVaR": self._cvar,
            "Robust": lambda w: self._robust(w, kappa=robust_kappa),
        }[objective]

        res = minimize(fun, w0, method="SLSQP", bounds=self._bounds(),
                       constraints=self._base_constraints(),
                       options={"maxiter": 500, "ftol": 1e-9})
        w = res.x if res.success else w0
        w = np.clip(w, 0 if not self.cons.allow_short else -1, None)
        # renormalise investable part
        s = w.sum()
        if s > 0:
            w = w * (invest / s)

        r, v, sharpe = self._port_stats(w)
        weights = {t: float(wi) for t, wi in zip(self.tickers, w)}
        cash = 1.0 - sum(weights.values())
        if cash > 1e-4:
            weights["Cash"] = float(cash)
        hhi = float(np.sum(w ** 2))
        return PortfolioResult(
            weights=weights, exp_return_monthly=r, volatility_monthly=v,
            sharpe=sharpe, concentration=hhi, objective=objective,
            diagnostics=dict(converged=bool(res.success), message=str(res.message)),
        )


# =========================================================================== #
#  STEP 11 — MONTE CARLO RISK SIMULATOR                                       #
# =========================================================================== #
@dataclass
class MonteCarloResult:
    horizon_days: int
    paths: int
    terminal_wealth: np.ndarray
    portfolio_paths: np.ndarray        # (paths x steps) wealth index (subsample)
    expected_wealth: float
    pct5: float
    pct95: float
    var_95: float
    cvar_95: float
    max_drawdown_med: float
    prob_loss: float

    def summary(self):
        return {
            "Expected wealth (x)": round(self.expected_wealth, 4),
            "Worst 5% (x)": round(self.pct5, 4),
            "Best 5% (x)": round(self.pct95, 4),
            "VaR 95% (loss)": to_pct(self.var_95),
            "CVaR 95% (loss)": to_pct(self.cvar_95),
            "Median max drawdown": to_pct(self.max_drawdown_med),
            "Prob. of loss": to_pct(self.prob_loss),
        }


class MonteCarloSimulator:
    """Correlated GBM Monte Carlo on the optimised portfolio weights."""

    def __init__(self, mu_monthly: dict, Sigma_daily: pd.DataFrame, seed=7):
        self.tickers = list(Sigma_daily.columns)
        self.mu_d = np.array([mu_monthly.get(t, 0.0) for t in self.tickers]) / MONTH_DAYS
        self.Sigma_d = Sigma_daily.values
        self.rng = np.random.default_rng(seed)

    def run(self, weights: dict, horizon_days=MONTH_DAYS, paths=10000):
        w = np.array([weights.get(t, 0.0) for t in self.tickers])
        cash = weights.get("Cash", 0.0)
        try:
            L = np.linalg.cholesky(self.Sigma_d + 1e-10 * np.eye(len(self.tickers)))
        except np.linalg.LinAlgError:
            vals, vecs = np.linalg.eigh(self.Sigma_d)
            L = vecs @ np.diag(np.sqrt(np.clip(vals, 1e-12, None)))

        n_assets = len(self.tickers)
        wealth = np.ones(paths)
        keep_steps = min(horizon_days, 63)
        subsample = min(paths, 400)
        port_paths = np.ones((subsample, keep_steps + 1))
        drift = self.mu_d - 0.5 * np.diag(self.Sigma_d)

        for step in range(horizon_days):
            z = self.rng.standard_normal((paths, n_assets))
            shocks = z @ L.T
            asset_ret = drift + shocks                    # log-returns
            simple = np.expm1(asset_ret)                  # -> simple returns
            port_ret = simple @ w + cash * RISK_FREE_DAILY
            wealth *= (1.0 + port_ret)
            if step < keep_steps:
                port_paths[:, step + 1] = wealth[:subsample]

        term = wealth
        # drawdown per subsampled path
        run_max = np.maximum.accumulate(port_paths, axis=1)
        dd = (port_paths / run_max - 1.0).min(axis=1)
        losses = 1.0 - term
        var95 = float(np.quantile(losses, 0.95))
        cvar95 = float(losses[losses >= var95].mean()) if np.any(losses >= var95) else var95
        return MonteCarloResult(
            horizon_days=horizon_days, paths=paths, terminal_wealth=term,
            portfolio_paths=port_paths, expected_wealth=float(term.mean()),
            pct5=float(np.quantile(term, 0.05)), pct95=float(np.quantile(term, 0.95)),
            var_95=var95, cvar_95=cvar95,
            max_drawdown_med=float(np.median(dd)),
            prob_loss=float(np.mean(term < 1.0)),
        )


# =========================================================================== #
#  STEP 12 — BACKTESTING (walk-forward)                                       #
# =========================================================================== #
@dataclass
class BacktestResult:
    dates: list
    equity: list
    bench_equity: list
    monthly_returns: list
    turnover: list
    metrics: dict


@dataclass
class StockModelResearchResult:
    rows: list
    best_by_stock: dict


class StockModelResearch:
    """Walk-forward model research for every stock separately.

    Each stock is evaluated only on future folds that were not used for training.
    The output ranks Random Forest, XGBoost, LSTM and Transformer independently
    for each stock, so the best model can differ from one stock to another.
    """

    def __init__(self, md: MarketData, features):
        self.md = md
        self.features = features

    @staticmethod
    def _score(metrics):
        """Balanced model-selection score; higher is better.

        IC and directional accuracy are rewarded, while large RMSE is penalised.
        The score is only for ranking models on the same stock/horizon.
        """
        ic = float(metrics.get("ic", 0.0) or 0.0)
        da = float(metrics.get("dir_acc", 0.5) or 0.5)
        hit = float(metrics.get("hit_ratio", 0.5) or 0.5)
        rm = float(metrics.get("rmse", 1.0) or 1.0)
        return 2.0 * ic + 1.0 * (da - 0.5) + 0.5 * (hit - 0.5) - 0.25 * rm

    def run(self, model_names=None, n_folds=3, min_rows=120, log=print, progress=None):
        data, _ = assemble_dataset(self.md, tickers=self.md.tickers)
        model_names = model_names or list(MODEL_REGISTRY.keys())
        jobs = [(tk, model) for tk in self.md.tickers for model in model_names]
        rows = []
        eng = ExpectedReturnEngine(self.features)

        for k, (tk, model_name) in enumerate(jobs):
            stock_data = data[data["stock"] == tk].copy()
            usable = stock_data.dropna(subset=self.features + ["target"])
            row = {"stock": tk, "model": model_name, "status": "OK"}
            try:
                if len(usable) < min_rows:
                    raise RuntimeError(f"only {len(usable)} usable rows")
                cv = eng.cross_validate(stock_data, model_name, n_folds=n_folds, log=lambda *_: None)
                if not np.isfinite(cv.get("rmse", np.nan)):
                    raise RuntimeError("no valid walk-forward folds")
                row.update(cv)
                row["score"] = self._score(cv)
            except Exception as exc:
                row.update(dict(rmse=np.nan, mae=np.nan, dir_acc=np.nan, ic=np.nan,
                                top_decile=np.nan, hit_ratio=np.nan, score=-np.inf))
                row["status"] = str(exc)
                log(f"[Research] {tk} / {model_name}: {exc}")
            rows.append(row)
            if progress:
                progress((k + 1) / max(1, len(jobs)))

        best = {}
        for tk in self.md.tickers:
            valid = [r for r in rows if r["stock"] == tk and np.isfinite(r.get("score", -np.inf))]
            valid.sort(key=lambda r: r.get("score", -np.inf), reverse=True)
            if valid:
                best[tk] = valid[0]["model"]
                for rank, r in enumerate(valid, 1):
                    r["rank"] = rank
                    r["best"] = (rank == 1)
        return StockModelResearchResult(rows=rows, best_by_stock=best)


class Backtester:
    """Walk-forward: each rebalance, train on past-only data, forecast, optimise,
    hold one month, record P/L. Compares against a NIFTY buy-and-hold benchmark.
    """

    def __init__(self, md: MarketData, features, model_name="Random Forest",
                 cov_method="Ledoit-Wolf", objective="Max Sharpe",
                 constraints: OptConstraints = None, return_source="ML"):
        self.md = md
        self.features = features
        self.model_name = model_name
        self.cov_method = cov_method
        self.objective = objective
        self.cons = constraints or OptConstraints()
        self.return_source = return_source

    def run(self, rebal_step=MONTH_DAYS, min_train=180, log=print, progress=None):
        data, _ = assemble_dataset(self.md, tickers=self.md.tickers)
        close = self.md.close_matrix()
        bench = self.md.prices[BENCHMARK]["close"]
        dates = close.index
        idxs = list(range(min_train, len(dates) - rebal_step, rebal_step))
        if not idxs:
            raise RuntimeError("Not enough history to backtest.")

        equity, bench_eq, mret, turn, out_dates = [1.0], [1.0], [], [], []
        prev_w = {}
        eng = ExpectedReturnEngine(self.features)

        for n, i in enumerate(idxs):
            asof = dates[i]
            hist = data[data["date"] <= asof]
            if hist.dropna(subset=self.features + ["target"]).shape[0] < 40:
                continue
            # --- expected returns (train on past only) ---
            if self.return_source == "CAPM":
                mu = capm_expected_returns(self._slice_md(asof), self.md.tickers)
                err = {t: 0.02 for t in self.md.tickers}
            else:
                tr = eng.train_and_forecast(hist, self.model_name, n_folds=2, log=lambda *_: None)
                mu, err = tr.forecasts, tr.error_bands
            # --- covariance (past only) ---
            hret = close.loc[:asof].pct_change().dropna().tail(250)
            cov = CovarianceEngine(hret, bench.loc[:asof].pct_change().dropna()).estimate(self.cov_method)
            # --- optimise ---
            opt = Optimizer(mu, cov, self.cons, prev_weights=prev_w)
            port = opt.optimize(self.objective, error_bands=err)
            w = port.weights
            turn.append(sum(abs(w.get(t, 0) - prev_w.get(t, 0)) for t in self.md.tickers))
            prev_w = w

            # --- realise next-month P/L ---
            j = min(i + rebal_step, len(dates) - 1)
            fwd = close.iloc[j] / close.iloc[i] - 1.0
            txn = (self.cons.txn_cost_bps / 1e4) * turn[-1]
            port_ret = sum(w.get(t, 0) * fwd[t] for t in self.md.tickers) \
                + w.get("Cash", 0) * RISK_FREE_DAILY * rebal_step - txn
            b_ret = bench.iloc[j] / bench.iloc[i] - 1.0
            equity.append(equity[-1] * (1 + port_ret))
            bench_eq.append(bench_eq[-1] * (1 + b_ret))
            mret.append(port_ret)
            out_dates.append(asof)
            if progress:
                progress((n + 1) / len(idxs))

        metrics = self._metrics(mret, equity, bench_eq, turn)
        return BacktestResult(out_dates, equity[1:], bench_eq[1:], mret, turn, metrics)

    def _slice_md(self, asof):
        sub = MarketData(
            prices={t: df.loc[:asof] for t, df in self.md.prices.items()},
            fundamentals=self.md.fundamentals, sentiment=self.md.sentiment,
            source=self.md.source)
        return sub

    def _metrics(self, mret, equity, bench_eq, turn):
        mret = np.array(mret)
        if len(mret) == 0:
            return {}
        periods_per_year = TRADING_DAYS / MONTH_DAYS
        total = equity[-1] / equity[0] - 1
        years = len(mret) / periods_per_year
        cagr = (equity[-1] / equity[0]) ** (1 / max(years, 1e-9)) - 1
        vol = mret.std() * math.sqrt(periods_per_year)
        rf_m = RISK_FREE_DAILY * MONTH_DAYS
        sharpe = (mret.mean() - rf_m) / (mret.std() + 1e-12) * math.sqrt(periods_per_year)
        downside = mret[mret < 0].std()
        sortino = (mret.mean() - rf_m) / (downside + 1e-12) * math.sqrt(periods_per_year)
        eq = np.array(equity)
        mdd = float((eq / np.maximum.accumulate(eq) - 1).min())
        b_total = bench_eq[-1] / bench_eq[0] - 1
        return dict(
            total_return=total, cagr=cagr, volatility=vol, sharpe=sharpe,
            sortino=sortino, max_drawdown=mdd, win_rate=float(np.mean(mret > 0)),
            avg_turnover=float(np.mean(turn)) if turn else 0.0,
            bench_total_return=b_total, excess_vs_bench=total - b_total,
        )


# =========================================================================== #
#  STEP 17 — REPORT + PIPELINE ORCHESTRATOR                                   #
# =========================================================================== #
@dataclass
class PipelineOutput:
    market: MarketData
    dataset: pd.DataFrame
    features: list
    return_components: dict
    mu_final: dict
    error_bands: dict
    train_result: TrainResult
    cov_method: str
    Sigma: pd.DataFrame
    portfolio: PortfolioResult
    montecarlo: MonteCarloResult


class Pipeline:
    """Single entry point wiring the full workflow:

        data -> features -> return models (+ensemble) -> covariance
             -> optimizer -> portfolio -> Monte Carlo.
    """

    def __init__(self, md: MarketData):
        self.md = md

    def run(self, model_name="Random Forest", cov_method="Ledoit-Wolf",
            objective="Max Sharpe", return_source="ML",
            ensemble_weights=None, constraints: OptConstraints = None,
            mc_paths=10000, mc_horizon=MONTH_DAYS, robust_kappa=1.0, log=print):
        tickers = self.md.tickers
        data, features = assemble_dataset(self.md, tickers=tickers)
        log(f"[Pipeline] Dataset: {data.shape[0]} rows x {len(features)} features.")

        # --- expected returns ---
        eng = ExpectedReturnEngine(features)
        tr = eng.train_and_forecast(data, model_name, log=log)
        components = {
            "ML Forecast": tr.forecasts,
            "CAPM": capm_expected_returns(self.md, tickers),
            "Factor Model": factor_expected_returns(data, features, tickers),
        }
        components["Black-Litterman"] = black_litterman_returns(
            self.md, tickers, tr.forecasts)

        if return_source == "Ensemble":
            w = ensemble_weights or {"Factor Model": 0.3, "Black-Litterman": 0.3,
                                     "ML Forecast": 0.4}
            mu_final = ensemble_returns(components, w)
        else:
            mu_final = components.get(return_source, tr.forecasts)
        err = tr.error_bands
        log(f"[Pipeline] Return source: {return_source}.")

        # --- covariance ---
        rets = self.md.returns_matrix(tickers)
        cov = CovarianceEngine(rets, self.md.benchmark_returns()).estimate(cov_method)
        log(f"[Pipeline] Covariance: {cov_method}.")

        # --- CVaR scenarios (if needed) ---
        scen = None
        if objective == "CVaR":
            mc0 = MonteCarloSimulator(mu_final, cov)
            z = mc0.rng.standard_normal((4000, len(tickers)))
            try:
                L = np.linalg.cholesky(cov.values * MONTH_DAYS + 1e-10 * np.eye(len(tickers)))
            except np.linalg.LinAlgError:
                vals, vecs = np.linalg.eigh(cov.values * MONTH_DAYS)
                L = vecs @ np.diag(np.sqrt(np.clip(vals, 1e-12, None)))
            mu_vec = np.array([mu_final.get(t, 0) for t in tickers])
            scen = mu_vec + z @ L.T

        # --- optimise ---
        opt = Optimizer(mu_final, cov, constraints or OptConstraints(),
                        scenario_returns=scen)
        port = opt.optimize(objective, error_bands=err, robust_kappa=robust_kappa)
        log(f"[Pipeline] Optimised ({objective}). Sharpe={port.sharpe:.2f}")

        # --- Monte Carlo ---
        mc = MonteCarloSimulator(mu_final, cov).run(
            port.weights, horizon_days=mc_horizon, paths=mc_paths)
        log(f"[Pipeline] Monte Carlo: {mc_paths} paths, "
            f"prob-loss {to_pct(mc.prob_loss)}, VaR95 {to_pct(mc.var_95)}")

        return PipelineOutput(
            market=self.md, dataset=data, features=features,
            return_components=components, mu_final=mu_final, error_bands=err,
            train_result=tr, cov_method=cov_method, Sigma=cov,
            portfolio=port, montecarlo=mc,
        )


def build_report(out: PipelineOutput, model_name, return_source, objective) -> str:
    p, tr, mc = out.portfolio, out.train_result, out.montecarlo
    L = []
    A = L.append
    A("=" * 70)
    A(f"  {APP_NAME}")
    A(f"  Final Risk & Allocation Report — {datetime.today():%Y-%m-%d %H:%M}")
    A("=" * 70)
    A(f"\nData source        : {out.market.source}")
    A(f"Universe           : {', '.join(out.market.tickers)}")
    A(f"Return model       : {tr.model_name}  (source: {return_source})")
    A(f"Covariance method  : {out.cov_method}")
    A(f"Optimizer objective: {objective}")

    A("\n--- Model validation (time-series CV) ---")
    cv = tr.cv_metrics
    A(f"  RMSE               : {cv.get('rmse', float('nan')):.4f}")
    A(f"  MAE                : {cv.get('mae', float('nan')):.4f}")
    A(f"  Directional acc.   : {to_pct(cv.get('dir_acc', 0))}")
    A(f"  Information coeff. : {cv.get('ic', 0):.3f}")
    A(f"  Top-decile return  : {to_pct(cv.get('top_decile', 0))}")
    A(f"  Hit ratio          : {to_pct(cv.get('hit_ratio', 0))}")

    A("\n--- Expected monthly returns (final) ---")
    for t in out.market.tickers:
        band = out.error_bands.get(t, 0)
        A(f"  {t:<10} {to_pct(out.mu_final.get(t,0)):>8}   +/- {to_pct(band)}")

    A("\n--- Portfolio allocation ---")
    for t, w in sorted(p.weights.items(), key=lambda kv: -kv[1]):
        if w > 1e-4:
            A(f"  {t:<10} {to_pct(w):>8}")
    A(f"\n  Expected return (1m): {to_pct(p.exp_return_monthly)}")
    A(f"  Volatility (1m)     : {to_pct(p.volatility_monthly)}")
    A(f"  Sharpe (monthly)    : {p.sharpe:.2f}")
    A(f"  Concentration (HHI) : {p.concentration:.3f}")

    A("\n--- Monte Carlo risk (post-purchase) ---")
    for k, v in mc.summary().items():
        A(f"  {k:<22}: {v}")

    A("\n--- Warnings ---")
    warns = []
    if cv.get("ic", 0) < 0.02:
        warns.append("Low information coefficient — forecasts weakly predictive.")
    if p.concentration > 0.4:
        warns.append("High concentration — portfolio dominated by few names.")
    if mc.prob_loss > 0.45:
        warns.append("High probability of loss over the horizon.")
    if out.market.source == "synthetic":
        warns.append("Synthetic demo data — not investable; wire live NSE feed.")
    if not warns:
        warns.append("None flagged.")
    for w in warns:
        A(f"  * {w}")
    A("\n" + "=" * 70)
    A("  Educational tool. Not investment advice.")
    A("=" * 70)
    return "\n".join(L)


# =========================================================================== #
#  HEADLESS SELF-TEST (CI) + DEMO RUNNER                                       #
# =========================================================================== #
def run_selftest(verbose=True):
    def ok(cond, msg):
        status = "PASS" if cond else "FAIL"
        if verbose:
            print(f"  [{status}] {msg}")
        return bool(cond)

    results = []
    print("HELM ML Portfolio — self-test")
    print("-" * 50)
    print(f"  xgboost={HAS_XGB} torch={HAS_TORCH} arch={HAS_ARCH} smartapi={HAS_SMARTAPI}")

    # 1. Data
    md = DataLoader(seed=1).load(DEFAULT_UNIVERSE, n_days=600)
    results.append(ok(len(md.tickers) == 5, "data loader returns 5 tickers + NIFTY"))
    results.append(ok(BENCHMARK in md.prices, "benchmark present"))
    results.append(ok((md.prices["SBIN"]["close"] > 0).all(), "prices strictly positive"))

    # 2. Features + target
    data, feats = assemble_dataset(md)
    results.append(ok(len(feats) == 16, f"feature count == 16 (got {len(feats)})"))
    results.append(ok(data["target"].notna().sum() > 100, "targets computed"))
    r = rsi(md.prices["SBIN"]["close"]).dropna()
    results.append(ok(((r >= 0) & (r <= 100)).all(), "RSI within [0,100]"))

    # 3. Model train + metrics sanity
    eng = ExpectedReturnEngine(feats)
    tr = eng.train_and_forecast(data, "Random Forest", n_folds=2, log=lambda *_: None)
    results.append(ok(len(tr.forecasts) == 5, "RF forecast for each stock"))
    results.append(ok(all(abs(v) < 1.0 for v in tr.forecasts.values()),
                      "forecasts in sane (<100% monthly) range"))
    # metric identities
    y = np.array([0.01, -0.02, 0.03, -0.01, 0.04])
    results.append(ok(abs(rmse(y, y)) < 1e-12, "rmse(y,y)==0"))
    results.append(ok(directional_accuracy(y, y) == 1.0, "dir-acc(y,y)==1"))
    results.append(ok(abs(information_coefficient(y, y) - 1.0) < 1e-9, "IC(y,y)==1"))

    # 4. XGB surrogate + sequence surrogate run
    xr = eng.train_and_forecast(data, "XGBoost", n_folds=2, log=lambda *_: None)
    results.append(ok(len(xr.forecasts) == 5, "XGBoost/GB forecast works"))
    if HAS_TORCH:
        lr = eng.train_and_forecast(data, "LSTM", n_folds=2, log=lambda *_: None)
        results.append(ok(len(lr.forecasts) == 5, "Deep LSTM forecast works"))
    else:
        try:
            eng.train_and_forecast(data, "LSTM", n_folds=2, log=lambda *_: None)
            results.append(ok(False, "LSTM refuses to fake fallback when PyTorch missing"))
        except RuntimeError:
            results.append(ok(True, "LSTM refuses to fake fallback when PyTorch missing"))

    # 5. Covariance: PSD + symmetry for all methods
    rets = md.returns_matrix()
    for method in CovarianceEngine.METHODS:
        cov = CovarianceEngine(rets, md.benchmark_returns()).estimate(method)
        vals = np.linalg.eigvalsh(cov.values)
        sym = np.allclose(cov.values, cov.values.T, atol=1e-8)
        results.append(ok(vals.min() > -1e-8 and sym, f"cov {method}: PSD & symmetric"))

    # 6. Return models
    capm = capm_expected_returns(md, md.tickers)
    results.append(ok(len(capm) == 5, "CAPM returns computed"))
    bl = black_litterman_returns(md, md.tickers, tr.forecasts)
    results.append(ok(len(bl) == 5 and all(np.isfinite(list(bl.values()))),
                      "Black-Litterman finite"))
    ens = ensemble_returns(
        {"a": {"SBIN": 0.02}, "b": {"SBIN": 0.04}}, {"a": 0.5, "b": 0.5})
    results.append(ok(abs(ens["SBIN"] - 0.03) < 1e-12, "ensemble 50/50 == mean"))

    # 7. Optimizer: constraints respected
    cov = CovarianceEngine(rets, md.benchmark_returns()).estimate("Ledoit-Wolf")
    cons = OptConstraints(max_weight=0.30, sector_cap=0.35, cash_floor=0.05)
    for obj in Optimizer.OBJECTIVES:
        opt = Optimizer(tr.forecasts, cov, cons)
        scen = None
        if obj == "CVaR":
            scen = np.random.default_rng(0).standard_normal((2000, 5)) * 0.05
            opt = Optimizer(tr.forecasts, cov, cons, scenario_returns=scen)
        port = opt.optimize(obj, error_bands=tr.error_bands)
        wsum = sum(v for k, v in port.weights.items())
        maxw = max(v for k, v in port.weights.items() if k != "Cash")
        long_ok = all(v >= -1e-6 for v in port.weights.values())
        results.append(ok(abs(wsum - 1.0) < 1e-4 and maxw <= 0.30 + 1e-3 and long_ok,
                          f"optimizer {obj}: budget+max-weight+long-only"))

    # 8. Monte Carlo
    port = Optimizer(tr.forecasts, cov, cons).optimize("Max Sharpe", error_bands=tr.error_bands)
    mc = MonteCarloSimulator(tr.forecasts, cov).run(port.weights, paths=2000)
    results.append(ok(0.0 <= mc.prob_loss <= 1.0, "MC prob_loss in [0,1]"))
    results.append(ok(mc.pct5 <= mc.expected_wealth <= mc.pct95, "MC pct5 <= mean <= pct95"))
    results.append(ok(mc.max_drawdown_med <= 0.0, "MC drawdown non-positive"))

    # 9. Backtest end-to-end
    bt = Backtester(md, feats, "Random Forest", "EWMA", "Max Sharpe",
                    OptConstraints(cash_floor=0.05)).run(log=lambda *_: None)
    results.append(ok(len(bt.equity) > 0 and np.isfinite(bt.equity[-1]),
                      "backtest produces finite equity curve"))
    results.append(ok("sharpe" in bt.metrics, "backtest metrics computed"))

    # 10. Full pipeline + report
    out = Pipeline(md).run(model_name="Random Forest", cov_method="EWMA",
                           objective="Max Sharpe", return_source="Ensemble",
                           mc_paths=2000, log=lambda *_: None)
    rep = build_report(out, "Random Forest", "Ensemble", "Max Sharpe")
    results.append(ok("Portfolio allocation" in rep and len(rep) > 500,
                      "report renders end-to-end"))

    print("-" * 50)
    passed = sum(results)
    print(f"  {passed}/{len(results)} checks passed.")
    return passed == len(results)


def run_demo():
    print("Running end-to-end demo (synthetic NSE data)...\n")
    md = DataLoader(seed=7).load(DEFAULT_UNIVERSE, n_days=750)
    out = Pipeline(md).run(
        model_name="XGBoost", cov_method="Ledoit-Wolf", objective="Robust",
        return_source="Ensemble", mc_paths=10000, log=print)
    print("\n" + build_report(out, "XGBoost", "Ensemble", "Robust"))


# =========================================================================== #
#  ENTRY POINT                                                                 #
# =========================================================================== #
def main():
    parser = argparse.ArgumentParser(description=APP_NAME)
    parser.add_argument("--selftest", "--test", action="store_true",
                        dest="selftest", help="run headless numeric self-tests")
    parser.add_argument("--demo", action="store_true",
                        help="run a headless end-to-end demo")
    args = parser.parse_args()

    if args.selftest:
        sys.exit(0 if run_selftest() else 1)
    if args.demo:
        run_demo()
        return
    # GUI
    try:
        launch_gui()
    except Exception as e:
        print(f"GUI unavailable ({e}). Try --selftest or --demo for headless use.")
        traceback.print_exc()


# =========================================================================== #
#  GUI LAYER (Tkinter) — native canvas charts, light theme, threaded jobs      #
# =========================================================================== #
def launch_gui():
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox

    # ------------------------------------------------------------------ #
    #  Native canvas chart widget                                         #
    # ------------------------------------------------------------------ #
    class Chart(tk.Canvas):
        PAD_L, PAD_R, PAD_T, PAD_B = 56, 20, 34, 40

        def __init__(self, master, height=260, **kw):
            super().__init__(master, height=height, bg=CARD, highlightthickness=0, **kw)
            self._draw_fn = None
            self.bind("<Configure>", lambda e: self._redraw())

        def _redraw(self):
            self.delete("all")
            if self._draw_fn:
                try:
                    self._draw_fn()
                except Exception:
                    pass

        def _area(self):
            w = self.winfo_width() or 600
            h = self.winfo_height() or 260
            return (self.PAD_L, self.PAD_T, w - self.PAD_R, h - self.PAD_B)

        def _title(self, text):
            self.create_text(self.PAD_L, 16, text=text, anchor="w",
                             font=(FONT, 11, "bold"), fill=INK)

        def _grid_y(self, x0, y0, x1, y1, vmin, vmax, fmt=lambda v: f"{v:.2f}"):
            for i in range(5):
                y = y1 - (y1 - y0) * i / 4
                v = vmin + (vmax - vmin) * i / 4
                self.create_line(x0, y, x1, y, fill=GRID)
                self.create_text(x0 - 6, y, text=fmt(v), anchor="e",
                                 font=(FONT, 8), fill=MUTED)

        # ---- chart types ---- #
        def lines(self, series, labels, colors, title="", yfmt=lambda v: f"{v:.2f}"):
            def draw():
                self._title(title)
                x0, y0, x1, y1 = self._area()
                allv = np.concatenate([np.asarray(s[1], float) for s in series if len(s[1])])
                if len(allv) == 0:
                    return
                vmin, vmax = float(np.min(allv)), float(np.max(allv))
                if vmax - vmin < 1e-9:
                    vmax += 1
                self._grid_y(x0, y0, x1, y1, vmin, vmax, yfmt)
                for (xs, ys), col in zip(series, colors):
                    ys = np.asarray(ys, float)
                    n = len(ys)
                    if n < 2:
                        continue
                    pts = []
                    for i in range(n):
                        px = x0 + (x1 - x0) * i / (n - 1)
                        py = y1 - (y1 - y0) * (ys[i] - vmin) / (vmax - vmin)
                        pts += [px, py]
                    self.create_line(*pts, fill=col, width=2, smooth=True)
                # legend
                lx = x0 + 8
                for lab, col in zip(labels, colors):
                    self.create_line(lx, y0 + 6, lx + 16, y0 + 6, fill=col, width=3)
                    self.create_text(lx + 20, y0 + 6, text=lab, anchor="w",
                                     font=(FONT, 9), fill=INK)
                    lx += 24 + len(lab) * 7
            self._draw_fn = draw
            self._redraw()

        def bars(self, labels, values, title="", colors=None, yfmt=to_pct):
            def draw():
                self._title(title)
                x0, y0, x1, y1 = self._area()
                vals = np.asarray(values, float)
                if len(vals) == 0:
                    return
                vmax = max(vals.max(), 0.0)
                vmin = min(vals.min(), 0.0)
                if vmax - vmin < 1e-9:
                    vmax += 1
                self._grid_y(x0, y0, x1, y1, vmin, vmax, yfmt)
                zero_y = y1 - (y1 - y0) * (0 - vmin) / (vmax - vmin)
                n = len(vals)
                bw = (x1 - x0) / n * 0.62
                for i, v in enumerate(vals):
                    cx = x0 + (x1 - x0) * (i + 0.5) / n
                    vy = y1 - (y1 - y0) * (v - vmin) / (vmax - vmin)
                    col = colors[i] if colors else (GREEN if v >= 0 else RED)
                    self.create_rectangle(cx - bw / 2, min(vy, zero_y),
                                          cx + bw / 2, max(vy, zero_y),
                                          fill=col, outline="")
                    self.create_text(cx, y1 + 12, text=str(labels[i]), anchor="n",
                                     font=(FONT, 8), fill=MUTED)
            self._draw_fn = draw
            self._redraw()

        def hist(self, data, title="", bins=40, color=ACCENT):
            def draw():
                self._title(title)
                x0, y0, x1, y1 = self._area()
                d = np.asarray(data, float)
                if len(d) == 0:
                    return
                counts, edges = np.histogram(d, bins=bins)
                cmax = counts.max() or 1
                self._grid_y(x0, y0, x1, y1, 0, cmax, lambda v: f"{int(v)}")
                n = len(counts)
                bw = (x1 - x0) / n
                for i, c in enumerate(counts):
                    bx = x0 + i * bw
                    by = y1 - (y1 - y0) * c / cmax
                    self.create_rectangle(bx, by, bx + bw * 0.92, y1,
                                          fill=color, outline="")
                # mean & VaR lines
                mean = d.mean()
                for val, col, lab in [(mean, INK, "mean"),
                                      (np.quantile(d, 0.05), RED, "5%")]:
                    fx = x0 + (x1 - x0) * (val - edges[0]) / (edges[-1] - edges[0] + 1e-9)
                    self.create_line(fx, y0, fx, y1, fill=col, dash=(4, 3))
                    self.create_text(fx, y0 - 2, text=lab, anchor="s",
                                     font=(FONT, 8), fill=col)
            self._draw_fn = draw
            self._redraw()

        def fan(self, paths, title=""):
            def draw():
                self._title(title)
                x0, y0, x1, y1 = self._area()
                P = np.asarray(paths, float)
                if P.ndim != 2 or P.shape[0] == 0:
                    return
                steps = P.shape[1]
                vmin, vmax = float(P.min()), float(P.max())
                if vmax - vmin < 1e-9:
                    vmax += 0.1
                self._grid_y(x0, y0, x1, y1, vmin, vmax, lambda v: f"{v:.2f}x")
                # a light sample of individual paths
                for k in range(0, min(P.shape[0], 120), 2):
                    pts = []
                    for s in range(steps):
                        px = x0 + (x1 - x0) * s / (steps - 1)
                        py = y1 - (y1 - y0) * (P[k, s] - vmin) / (vmax - vmin)
                        pts += [px, py]
                    self.create_line(*pts, fill="#cbd5e1", width=1)
                # median path
                med = np.median(P, axis=0)
                pts = []
                for s in range(steps):
                    px = x0 + (x1 - x0) * s / (steps - 1)
                    py = y1 - (y1 - y0) * (med[s] - vmin) / (vmax - vmin)
                    pts += [px, py]
                self.create_line(*pts, fill=ACCENT, width=2.5)
                # break-even line at 1.0
                if vmin <= 1.0 <= vmax:
                    by = y1 - (y1 - y0) * (1.0 - vmin) / (vmax - vmin)
                    self.create_line(x0, by, x1, by, fill=RED, dash=(5, 3))
            self._draw_fn = draw
            self._redraw()

        def heatmap(self, matrix, labels, title=""):
            def draw():
                self._title(title)
                x0, y0, x1, y1 = self._area()
                M = np.asarray(matrix, float)
                n = M.shape[0]
                if n == 0:
                    return
                cw = (x1 - x0) / n
                ch = (y1 - y0) / n
                for i in range(n):
                    for j in range(n):
                        v = M[i, j]
                        # blue(neg) - white(0) - teal(pos)
                        t = clamp((v + 1) / 2, 0, 1)
                        r = int(255 * (1 - t) + 13 * t)
                        g = int(255 * (1 - t) + 148 * t)
                        b = int(255 * (1 - t) + 136 * t)
                        col = f"#{r:02x}{g:02x}{b:02x}"
                        rx, ry = x0 + j * cw, y0 + i * ch
                        self.create_rectangle(rx, ry, rx + cw, ry + ch, fill=col, outline=CARD)
                        self.create_text(rx + cw / 2, ry + ch / 2, text=f"{v:.2f}",
                                         font=(FONT, 8), fill=INK if abs(v) < 0.6 else "white")
                    self.create_text(x0 - 4, y0 + i * ch + ch / 2, text=labels[i],
                                     anchor="e", font=(FONT, 8), fill=MUTED)
                    self.create_text(x0 + i * cw + cw / 2, y1 + 6, text=labels[i],
                                     anchor="n", font=(FONT, 8), fill=MUTED)
            self._draw_fn = draw
            self._redraw()

        def message(self, text):
            self._draw_fn = lambda: self.create_text(
                (self.winfo_width() or 600) / 2, (self.winfo_height() or 260) / 2,
                text=text, font=(FONT, 10), fill=MUTED)
            self._redraw()

    # ------------------------------------------------------------------ #
    #  Main application                                                   #
    # ------------------------------------------------------------------ #
    class App(tk.Tk):
        def __init__(self):
            super().__init__()
            self.title(f"{APP_NAME}  v{VERSION}")
            self.geometry("1200x820")
            self.configure(bg=BG)
            self.minsize(1040, 700)

            # shared state
            self.md = None
            self.dataset = None
            self.features = None
            self.train_result = None
            self.cov = None
            self.cov_method = None
            self.mu_final = None
            self.error_bands = {}
            self.return_components = {}
            self.portfolio = None
            self.mc_result = None
            self.backtest_result = None
            self.auto_research_result = None

            self.q = queue.Queue()
            self._style()
            self._build_header()
            self._build_tabs()
            self._set_status("Ready. Load data to begin (synthetic demo works offline).")
            self.after(80, self._poll)

        # ---- threading plumbing ---- #
        def _poll(self):
            try:
                while True:
                    fn = self.q.get_nowait()
                    fn()
            except queue.Empty:
                pass
            self.after(80, self._poll)

        def _submit(self, worker, on_success, busy="Working…", btn=None):
            self._set_status(busy)
            if btn is not None:
                btn.config(state="disabled")

            def run():
                try:
                    res = worker()
                    self.q.put(lambda: self._finish(on_success, res, btn))
                except Exception as e:
                    tb = traceback.format_exc()
                    self.q.put(lambda e=e, tb=tb: self._error(e, tb, btn))
            threading.Thread(target=run, daemon=True).start()

        def _finish(self, on_success, res, btn):
            try:
                on_success(res)
                self._set_status("Done.")
            finally:
                if btn is not None:
                    btn.config(state="normal")

        def _error(self, e, tb, btn):
            self._set_status(f"Error: {e}")
            if btn is not None:
                btn.config(state="normal")
            messagebox.showerror("HELM", f"{e}\n\n{tb[-800:]}")

        def _tlog(self, msg):
            self.q.put(lambda m=msg: self._append_log(m))

        # ---- styling ---- #
        def _style(self):
            st = ttk.Style(self)
            st.theme_use("clam")
            st.configure(".", background=BG, foreground=INK, font=(FONT, 10))
            st.configure("TFrame", background=BG)
            st.configure("Card.TFrame", background=CARD, relief="flat")
            st.configure("TLabel", background=BG, foreground=INK, font=(FONT, 10))
            st.configure("Card.TLabel", background=CARD, foreground=INK)
            st.configure("Muted.TLabel", background=CARD, foreground=MUTED, font=(FONT, 9))
            st.configure("H.TLabel", background=CARD, foreground=INK, font=(FONT, 12, "bold"))
            st.configure("TButton", font=(FONT, 10), padding=6)
            st.configure("Accent.TButton", background=ACCENT, foreground="white",
                         font=(FONT, 10, "bold"), padding=7, borderwidth=0)
            st.map("Accent.TButton",
                   background=[("active", ACCENT_DK), ("disabled", "#94a3b8")])
            st.configure("TNotebook", background=BG, borderwidth=0)
            st.configure("TNotebook.Tab", font=(FONT, 9), padding=(12, 7),
                         background="#e2e8f0", foreground=INK)
            st.map("TNotebook.Tab", background=[("selected", CARD)],
                   foreground=[("selected", ACCENT_DK)])
            st.configure("TEntry", fieldbackground="white", padding=4)
            st.configure("TCombobox", fieldbackground="white", padding=4)
            st.configure("Treeview", background="white", fieldbackground="white",
                         font=(FONT, 9), rowheight=22)
            st.configure("Treeview.Heading", font=(FONT, 9, "bold"),
                         background="#e2e8f0", foreground=INK)

        def _build_header(self):
            bar = tk.Frame(self, bg=ACCENT, height=52)
            bar.pack(fill="x", side="top")
            bar.pack_propagate(False)
            tk.Label(bar, text="HELM FINSERV", bg=ACCENT, fg="white",
                     font=(FONT, 15, "bold")).pack(side="left", padx=(18, 8))
            tk.Label(bar, text="ML Portfolio Construction System", bg=ACCENT,
                     fg="#ccfbf1", font=(FONT, 11)).pack(side="left")
            self.status = tk.Label(bar, text="", bg=ACCENT, fg="white", font=(FONT, 9))
            self.status.pack(side="right", padx=16)

        def _set_status(self, txt):
            try:
                self.status.config(text=txt)
            except Exception:
                pass

        def _build_tabs(self):
            self.nb = ttk.Notebook(self)
            self.nb.pack(fill="both", expand=True, padx=10, pady=10)
            self._tab_data()
            self._tab_features()
            self._tab_training()
            self._tab_forecast()
            self._tab_covariance()
            self._tab_optimizer()
            self._tab_portfolio()
            self._tab_montecarlo()
            self._tab_backtest()
            self._tab_report()
            self._tab_auto_research()

        # ---- small UI helpers ---- #
        def _card(self, parent):
            c = tk.Frame(parent, bg=CARD, bd=0, highlightbackground=GRID,
                         highlightthickness=1)
            return c

        def _labeled(self, parent, label, default, width=14):
            row = tk.Frame(parent, bg=CARD)
            row.pack(fill="x", pady=3, padx=10)
            tk.Label(row, text=label, bg=CARD, fg=MUTED, font=(FONT, 9),
                     width=16, anchor="w").pack(side="left")
            var = tk.StringVar(value=str(default))
            ttk.Entry(row, textvariable=var, width=width).pack(side="left", fill="x", expand=True)
            return var

        def _combo(self, parent, label, values, default):
            row = tk.Frame(parent, bg=CARD)
            row.pack(fill="x", pady=3, padx=10)
            tk.Label(row, text=label, bg=CARD, fg=MUTED, font=(FONT, 9),
                     width=16, anchor="w").pack(side="left")
            var = tk.StringVar(value=default)
            ttk.Combobox(row, textvariable=var, values=values, state="readonly",
                         width=20).pack(side="left", fill="x", expand=True)
            return var

        def _tree(self, parent, cols, height=8):
            tv = ttk.Treeview(parent, columns=cols, show="headings", height=height)
            for c in cols:
                tv.heading(c, text=c)
                tv.column(c, width=110, anchor="center")
            return tv

        def _fill_tree(self, tv, rows):
            tv.delete(*tv.get_children())
            for r in rows:
                tv.insert("", "end", values=r)

        # ================= TAB 1 — DATA LOADER ===================== #
        def _tab_data(self):
            tab = ttk.Frame(self.nb)
            self.nb.add(tab, text="1 · Data Loader")
            left = self._card(tab)
            left.pack(side="left", fill="y", padx=10, pady=10)
            tk.Label(left, text="Data Loader", bg=CARD, font=(FONT, 12, "bold"),
                     fg=INK).pack(anchor="w", padx=10, pady=(10, 4))
            self.v_universe = self._labeled(left, "Universe", ",".join(DEFAULT_UNIVERSE), 22)
            self.v_days = self._labeled(left, "History (days)", 750)
            self.v_source = self._combo(left, "Source",
                                        ["Synthetic (demo)", "Angel One (live)"],
                                        "Synthetic (demo)")
            tk.Label(left, text="Angel One credentials (live only)", bg=CARD,
                     fg=MUTED, font=(FONT, 8, "italic")).pack(anchor="w", padx=10, pady=(8, 0))
            self.v_api = self._labeled(left, "API key", os.getenv("ANGEL_API_KEY", ""))
            self.v_cid = self._labeled(left, "Client ID", os.getenv("ANGEL_CLIENT_ID", ""))
            self.v_pwd = self._labeled(left, "Password/PIN", "")
            self.v_totp = self._labeled(left, "TOTP secret", "")
            b = ttk.Button(left, text="Load Data", style="Accent.TButton",
                           command=self._do_load)
            b.pack(fill="x", padx=10, pady=12)
            self.btn_load = b

            right = self._card(tab)
            right.pack(side="left", fill="both", expand=True, padx=(0, 10), pady=10)
            self.data_chart = Chart(right, height=300)
            self.data_chart.pack(fill="both", expand=True, padx=8, pady=8)
            self.data_chart.message("Load data to view normalised price history.")
            self.data_tree = self._tree(right,
                ["Stock", "Last", "1M %", "3M %", "Ann.Vol", "Sector"], height=7)
            self.data_tree.pack(fill="x", padx=8, pady=(0, 8))

        def _do_load(self):
            tickers = [t.strip().upper() for t in self.v_universe.get().split(",") if t.strip()]
            days = int(float(self.v_days.get()))
            live = self.v_source.get().startswith("Angel")
            creds = {}
            if live:
                creds = dict(api_key=self.v_api.get(), client_id=self.v_cid.get(),
                             password=self.v_pwd.get(), totp_secret=self.v_totp.get())

            self._loader = None

            def work():
                loader = DataLoader(use_live=live, credentials=creds)
                self._loader = loader
                return loader.load(tickers, n_days=days, log=self._tlog)
            self._submit(work, self._on_loaded, "Loading data…", self.btn_load)

        def _on_loaded(self, md):
            self.md = md
            self.dataset = None
            if self._loader is not None and self._loader.requested_live and md.source == "synthetic":
                reason = self._loader.last_live_error or "unknown reason"
                messagebox.showwarning(
                    "HELM — Falling back to synthetic data",
                    f"Angel One (live) was requested but could not be used, so "
                    f"synthetic demo data is being shown instead.\n\nReason: {reason}")
            close = md.close_matrix()
            norm = close / close.iloc[0]
            self.data_chart.lines(
                [(norm.index, norm[c].values) for c in norm.columns],
                list(norm.columns),
                [ACCENT, "#6366f1", "#f59e0b", "#ec4899", "#0ea5e9", "#10b981"][:len(norm.columns)],
                title=f"Normalised prices  ·  source: {md.source}",
                yfmt=lambda v: f"{v:.2f}x")
            rets = md.returns_matrix()
            rows = []
            for t in md.tickers:
                c = md.prices[t]["close"]
                rows.append([t, f"{c.iloc[-1]:.2f}", to_pct(c.iloc[-1]/c.iloc[-22]-1),
                             to_pct(c.iloc[-1]/c.iloc[-64]-1),
                             to_pct(annualize_vol(rets[t].std())),
                             SECTOR_MAP.get(t, "—")])
            self._fill_tree(self.data_tree, rows)
            self._set_status(f"Loaded {len(md.tickers)} stocks ({md.source}).")

        # ================= TAB 2 — FEATURES ======================== #
        def _tab_features(self):
            tab = ttk.Frame(self.nb)
            self.nb.add(tab, text="2 · Features")
            top = self._card(tab)
            top.pack(fill="x", padx=10, pady=(10, 6))
            tk.Label(top, text="Feature Engineering", bg=CARD, font=(FONT, 12, "bold"),
                     fg=INK).pack(side="left", padx=10, pady=8)
            self.btn_feat = ttk.Button(top, text="Build Features", style="Accent.TButton",
                                       command=self._do_features)
            self.btn_feat.pack(side="right", padx=10, pady=8)
            tk.Label(top, text="17 ML columns: 16 features + target. Use horizontal scroll to view all.",
                     bg=CARD, fg=MUTED, font=(FONT, 9)).pack(side="left", padx=6)
            body = self._card(tab)
            body.pack(fill="both", expand=True, padx=10, pady=(0, 10))

            # Show ALL engineered features, not only the first six.
            # Columns displayed = date + stock + 16 features + target = full ML table.
            self.feature_display_cols = ["date", "stock"] + list(FeatureEngineer.FEATURES) + ["target"]

            tree_frame = tk.Frame(body, bg=CARD)
            tree_frame.pack(fill="both", expand=True, padx=8, pady=8)

            self.feat_tree = ttk.Treeview(
                tree_frame,
                columns=self.feature_display_cols,
                show="headings",
                height=18
            )

            vsb = ttk.Scrollbar(tree_frame, orient="vertical", command=self.feat_tree.yview)
            hsb = ttk.Scrollbar(tree_frame, orient="horizontal", command=self.feat_tree.xview)
            self.feat_tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

            self.feat_tree.grid(row=0, column=0, sticky="nsew")
            vsb.grid(row=0, column=1, sticky="ns")
            hsb.grid(row=1, column=0, sticky="ew")
            tree_frame.grid_rowconfigure(0, weight=1)
            tree_frame.grid_columnconfigure(0, weight=1)

            for c in self.feature_display_cols:
                self.feat_tree.heading(c, text=c)
                width = 95
                if c in ("date", "stock"):
                    width = 105
                elif c in ("macd_hist", "momentum_3m", "drawdown_63", "sentiment"):
                    width = 120
                self.feat_tree.column(c, width=width, minwidth=width, anchor="center", stretch=False)

        def _do_features(self):
            if self.md is None:
                messagebox.showinfo("HELM", "Load data first (Tab 1).")
                return

            def work():
                data, feats = assemble_dataset(self.md)
                return data, feats
            self._submit(work, self._on_features, "Building features…", self.btn_feat)

        def _on_features(self, res):
            self.dataset, self.features = res
            show = self.dataset.dropna(subset=self.features).tail(250)
            cols = getattr(self, "feature_display_cols", ["date", "stock"] + self.features + ["target"])

            rows = []
            for _, r in show.iterrows():
                row = []
                for c in cols:
                    if c == "date":
                        row.append(str(r.get("date", ""))[:10])
                    elif c == "stock":
                        row.append(r.get("stock", ""))
                    else:
                        val = r.get(c, np.nan)
                        if pd.isna(val):
                            row.append("—")
                        else:
                            row.append(f"{float(val):.4f}")
                rows.append(row)

            self._fill_tree(self.feat_tree, rows)
            self._set_status(
                f"Built {len(self.features)} features + target · "
                f"displaying {len(cols)} columns · {len(self.dataset)} rows."
            )

        # ================= TAB 3 — ML TRAINING ===================== #
        def _tab_training(self):
            tab = ttk.Frame(self.nb)
            self.nb.add(tab, text="3 · ML Training")
            left = self._card(tab)
            left.pack(side="left", fill="y", padx=10, pady=10)
            tk.Label(left, text="ML Model Training", bg=CARD, font=(FONT, 12, "bold"),
                     fg=INK).pack(anchor="w", padx=10, pady=(10, 4))
            self.v_model = self._combo(left, "Model", list(MODEL_REGISTRY.keys()), "Random Forest")
            self.v_folds = self._labeled(left, "CV folds", 3)
            self.v_horizon = self._labeled(left, "Target horizon (d)", MONTH_DAYS)
            tk.Label(left, text="Time-series CV (no shuffle) — expanding window.",
                     bg=CARD, fg=MUTED, font=(FONT, 8, "italic")).pack(anchor="w", padx=10, pady=4)
            self.btn_train = ttk.Button(left, text="Train + Validate", style="Accent.TButton",
                                        command=self._do_train)
            self.btn_train.pack(fill="x", padx=10, pady=10)
            self.train_metrics = tk.Text(left, height=9, width=30, bg="#f8fafc",
                                         relief="flat", font=(FONT, 9), fg=INK)
            self.train_metrics.pack(fill="x", padx=10, pady=(0, 10))

            right = self._card(tab)
            right.pack(side="left", fill="both", expand=True, padx=(0, 10), pady=10)
            self.imp_chart = Chart(right, height=340)
            self.imp_chart.pack(fill="both", expand=True, padx=8, pady=8)
            self.imp_chart.message("Train a model to see feature importance.")

        def _do_train(self):
            if self.md is None:
                messagebox.showinfo("HELM", "Load data first (Tab 1).")
                return
            model = self.v_model.get()
            if model in ("LSTM", "Transformer") and not HAS_TORCH:
                messagebox.showerror("PyTorch missing",
                    "LSTM and Transformer require PyTorch.\n\n"
                    "Run this in Command Prompt using the same Python:\n"
                    "python -m pip install --upgrade pip\n"
                    "python -m pip install torch torchvision torchaudio\n\n"
                    "Verify with:\n"
                    "python -c \"import torch; print(torch.__version__)\"\n\n"
                    f"Import error: {TORCH_IMPORT_ERROR or 'torch not installed'}")
                self._set_status("PyTorch missing — deep model not trained.")
                return
            folds = int(float(self.v_folds.get()))
            horizon = int(float(self.v_horizon.get()))
            if horizon < 2:
                messagebox.showinfo("HELM", "Target horizon must be at least 2 trading days.")
                return

            def work():
                # Rebuild target using the selected horizon so the GUI field is real.
                # Example: horizon=21 means predict next 21 trading-day return.
                data, feats = assemble_dataset(self.md, horizon=horizon)
                self.dataset, self.features = data, feats
                eng = ExpectedReturnEngine(feats)
                tr = eng.train_and_forecast(data, model, n_folds=folds, log=self._tlog, leaderboard=True)
                tr.cv_metrics["target_horizon"] = horizon
                return tr
            self._submit(work, self._on_trained, f"Training {model} for {horizon}d target…", self.btn_train)

        def _on_trained(self, tr):
            self.train_result = tr
            cv = tr.cv_metrics
            self.train_metrics.delete("1.0", "end")
            self.train_metrics.insert("end",
                f"Model: {tr.model_name}\n"
                f"Target horizon  {cv.get('target_horizon', MONTH_DAYS)} trading days\n\n"
                f"RMSE            {cv.get('rmse', float('nan')):.4f}\n"
                f"MAE             {cv.get('mae', float('nan')):.4f}\n"
                f"Directional acc {to_pct(cv.get('dir_acc', 0))}\n"
                f"Info. coeff.    {cv.get('ic', 0):.3f}\n"
                f"Top-decile ret  {to_pct(cv.get('top_decile', 0))}\n"
                f"Hit ratio       {to_pct(cv.get('hit_ratio', 0))}\n")
            lb = cv.get("leaderboard") or []
            if lb:
                self.train_metrics.insert("end", "\nLeaderboard (IC / Hit)\n")
                for name, ic, da, hit, r in lb[:4]:
                    self.train_metrics.insert("end", f"{name[:12]:12s} {ic:+.3f} / {hit:.1%}\n")
            items = sorted(tr.feature_importances.items(), key=lambda kv: -kv[1])
            labs = [k for k, _ in items]
            vals = [v for _, v in items]
            title = "Feature importance"
            if tr.model_name in ("Deep LSTM", "Transformer Encoder"):
                title = "Permutation feature importance"
            self.imp_chart.bars(labs, vals, title=title,
                                colors=[ACCENT] * len(labs),
                                yfmt=lambda v: f"{v:.2f}")
            self._set_status(f"Trained {tr.model_name}. IC={cv.get('ic',0):.3f}")

        # ================= TAB 4 — FORECASTS ======================= #
        def _tab_forecast(self):
            tab = ttk.Frame(self.nb)
            self.nb.add(tab, text="4 · Forecasts")
            top = self._card(tab)
            top.pack(fill="x", padx=10, pady=(10, 6))
            tk.Label(top, text="Expected Return Forecasts", bg=CARD,
                     font=(FONT, 12, "bold"), fg=INK).pack(side="left", padx=10, pady=8)
            ttk.Button(top, text="Refresh", command=self._on_forecast).pack(side="right", padx=10)
            mid = self._card(tab)
            mid.pack(fill="both", expand=True, padx=10, pady=(0, 10))
            self.fc_tree = self._tree(mid,
                ["Stock", "ML Forecast", "Confidence", "Error Band"], height=7)
            self.fc_tree.pack(fill="x", padx=8, pady=8)
            self.fc_chart = Chart(mid, height=260)
            self.fc_chart.pack(fill="both", expand=True, padx=8, pady=8)
            self.fc_chart.message("Train a model (Tab 3) to view forecasts.")

        def _on_forecast(self, *_):
            if self.train_result is None:
                messagebox.showinfo("HELM", "Train a model first (Tab 3).")
                return
            tr = self.train_result
            rows, labs, vals = [], [], []
            for t, f in sorted(tr.forecasts.items(), key=lambda kv: -kv[1]):
                band = tr.error_bands.get(t, 0)
                conf = "High" if band < 0.03 else ("Medium" if band < 0.06 else "Low")
                rows.append([t, to_pct(f), conf, "±" + to_pct(band)])
                labs.append(t)
                vals.append(f)
            self._fill_tree(self.fc_tree, rows)
            self.fc_chart.bars(labs, vals, title="ML expected monthly return")
            self._set_status("Forecasts updated.")

        # ================= TAB 5 — COVARIANCE ====================== #
        def _tab_covariance(self):
            tab = ttk.Frame(self.nb)
            self.nb.add(tab, text="5 · Covariance")
            left = self._card(tab)
            left.pack(side="left", fill="y", padx=10, pady=10)
            tk.Label(left, text="Covariance Engine", bg=CARD, font=(FONT, 12, "bold"),
                     fg=INK).pack(anchor="w", padx=10, pady=(10, 4))
            self.v_cov = self._combo(left, "Method", CovarianceEngine.METHODS, "Ledoit-Wolf")
            self.btn_cov = ttk.Button(left, text="Estimate Σ", style="Accent.TButton",
                                      command=self._do_cov)
            self.btn_cov.pack(fill="x", padx=10, pady=10)
            self.cov_text = tk.Text(left, height=10, width=28, bg="#f8fafc",
                                    relief="flat", font=(FONT, 9), fg=INK)
            self.cov_text.pack(fill="x", padx=10, pady=(0, 10))
            right = self._card(tab)
            right.pack(side="left", fill="both", expand=True, padx=(0, 10), pady=10)
            self.cov_chart = Chart(right, height=360)
            self.cov_chart.pack(fill="both", expand=True, padx=8, pady=8)
            self.cov_chart.message("Estimate Σ to view the correlation heatmap.")

        def _do_cov(self):
            if self.md is None:
                messagebox.showinfo("HELM", "Load data first (Tab 1).")
                return
            method = self.v_cov.get()

            def work():
                rets = self.md.returns_matrix()
                return method, CovarianceEngine(rets, self.md.benchmark_returns()).estimate(method)
            self._submit(work, self._on_cov, f"Estimating {method}…", self.btn_cov)

        def _on_cov(self, res):
            method, cov = res
            self.cov, self.cov_method = cov, method
            d = np.sqrt(np.diag(cov.values))
            corr = cov.values / np.outer(d, d)
            self.cov_chart.heatmap(corr, list(cov.columns),
                                   title=f"Correlation ({method})")
            self.cov_text.delete("1.0", "end")
            self.cov_text.insert("end", f"Method: {method}\n\nAnnualised vol:\n")
            for t, v in zip(cov.columns, d):
                self.cov_text.insert("end", f"  {t:<10} {to_pct(annualize_vol(v))}\n")
            self._set_status(f"Σ estimated ({method}).")

        # ================= TAB 6 — OPTIMIZER ======================= #
        def _tab_optimizer(self):
            tab = ttk.Frame(self.nb)
            self.nb.add(tab, text="6 · Optimizer")
            left = self._card(tab)
            left.pack(side="left", fill="y", padx=10, pady=10)
            tk.Label(left, text="Optimizer Engine", bg=CARD, font=(FONT, 12, "bold"),
                     fg=INK).pack(anchor="w", padx=10, pady=(10, 4))
            self.v_retsrc = self._combo(left, "Return source",
                ["ML Forecast", "CAPM", "Factor Model", "Black-Litterman", "Ensemble"],
                "Ensemble")
            self.v_riskmodel = self._combo(left, "Risk model", CovarianceEngine.METHODS, "Ledoit-Wolf")
            self.v_obj = self._combo(left, "Objective", Optimizer.OBJECTIVES, "Max Sharpe")
            self.v_maxw = self._labeled(left, "Max weight", 0.30)
            self.v_sector = self._labeled(left, "Sector cap", 0.35)
            self.v_turn = self._labeled(left, "Turnover limit", "")
            self.v_cash = self._labeled(left, "Cash floor", 0.05)
            self.v_kappa = self._labeled(left, "Robust κ", 1.0)
            self.btn_opt = ttk.Button(left, text="Run Optimizer", style="Accent.TButton",
                                      command=self._do_opt)
            self.btn_opt.pack(fill="x", padx=10, pady=10)
            ttk.Button(left, text="▶ Run FULL pipeline",
                       command=self._do_pipeline).pack(fill="x", padx=10, pady=(0, 6))
            self.btn_all = ttk.Button(
                left, text="Run ALL combinations + rank",
                command=self._do_all_combinations)
            self.btn_all.pack(fill="x", padx=10, pady=(0, 10))
            right = self._card(tab)
            right.pack(side="left", fill="both", expand=True, padx=(0, 10), pady=10)
            self.opt_stats = tk.Text(right, height=6, bg="#f8fafc", relief="flat",
                                     font=(FONT, 10), fg=INK)
            self.opt_stats.pack(fill="x", padx=8, pady=8)
            self.opt_chart = Chart(right, height=280)
            self.opt_chart.pack(fill="both", expand=True, padx=8, pady=8)
            self.opt_chart.message("Run the optimizer to see allocation.")

        def _constraints(self):
            turn = self.v_turn.get().strip()
            return OptConstraints(
                max_weight=float(self.v_maxw.get()),
                sector_cap=float(self.v_sector.get()),
                turnover_limit=float(turn) if turn else None,
                cash_floor=float(self.v_cash.get()))

        def _return_source_vec(self):
            """Assemble mu + error bands for the chosen return source."""
            tickers = self.md.tickers
            src = self.v_retsrc.get()
            comps = {}
            if self.train_result is not None:
                comps["ML Forecast"] = self.train_result.forecasts
                self.error_bands = self.train_result.error_bands
            comps["CAPM"] = capm_expected_returns(self.md, tickers)
            if self.dataset is not None:
                comps["Factor Model"] = factor_expected_returns(self.dataset, self.features, tickers)
            if "ML Forecast" in comps:
                comps["Black-Litterman"] = black_litterman_returns(self.md, tickers, comps["ML Forecast"])
            self.return_components = comps
            if src == "Ensemble":
                w = {"Factor Model": 0.3, "Black-Litterman": 0.3, "ML Forecast": 0.4}
                mu = ensemble_returns(comps, {k: v for k, v in w.items() if k in comps})
            else:
                mu = comps.get(src) or comps.get("CAPM")
            if not self.error_bands:
                self.error_bands = {t: 0.03 for t in tickers}
            return mu

        def _do_opt(self):
            if self.md is None:
                messagebox.showinfo("HELM", "Load data first (Tab 1).")
                return
            method = self.v_riskmodel.get()
            obj = self.v_obj.get()
            cons = self._constraints()
            kappa = float(self.v_kappa.get())

            def work():
                mu = self._return_source_vec()
                rets = self.md.returns_matrix()
                cov = CovarianceEngine(rets, self.md.benchmark_returns()).estimate(method)
                scen = None
                if obj == "CVaR":
                    sim = MonteCarloSimulator(mu, cov)
                    z = sim.rng.standard_normal((4000, len(self.md.tickers)))
                    try:
                        L = np.linalg.cholesky(cov.values * MONTH_DAYS + 1e-10*np.eye(len(self.md.tickers)))
                    except np.linalg.LinAlgError:
                        vals, vecs = np.linalg.eigh(cov.values * MONTH_DAYS)
                        L = vecs @ np.diag(np.sqrt(np.clip(vals, 1e-12, None)))
                    scen = np.array([mu.get(t, 0) for t in self.md.tickers]) + z @ L.T
                opt = Optimizer(mu, cov, cons, scenario_returns=scen)
                port = opt.optimize(obj, error_bands=self.error_bands, robust_kappa=kappa)
                return mu, cov, method, port
            self._submit(work, self._on_opt, f"Optimising ({obj})…", self.btn_opt)

        def _on_opt(self, res):
            mu, cov, method, port = res
            self.mu_final, self.cov, self.cov_method, self.portfolio = mu, cov, method, port
            self.opt_stats.delete("1.0", "end")
            self.opt_stats.insert("end",
                f"Objective: {port.objective}   ·   Return src: {self.v_retsrc.get()}   ·   Risk: {method}\n"
                f"Exp. return (1m): {to_pct(port.exp_return_monthly)}    "
                f"Volatility (1m): {to_pct(port.volatility_monthly)}\n"
                f"Sharpe (monthly): {port.sharpe:.2f}    "
                f"Concentration (HHI): {port.concentration:.3f}\n")
            self._draw_weights(self.opt_chart, port)
            self._refresh_portfolio_tab(port)
            self._set_status(f"Optimised ({port.objective}). Sharpe {port.sharpe:.2f}")

        def _do_all_combinations(self):
            """Run every return-source × covariance-model × objective combination.

            This is a current-snapshot research sweep, not an out-of-sample proof.
            It ranks feasible portfolios on return, risk, Sharpe, concentration and
            a transparent composite score. Use Tab 9 walk-forward backtesting to
            validate shortlisted combinations before treating one as the winner.
            """
            if self.md is None:
                messagebox.showinfo("HELM", "Load data first (Tab 1).")
                return

            try:
                cons = self._constraints()
                kappa = float(self.v_kappa.get())
            except Exception as exc:
                messagebox.showerror("HELM", f"Invalid optimizer input: {exc}")
                return

            model_name = self.v_model.get() if hasattr(self, "v_model") else "Random Forest"
            self._set_status("Preparing all-combination research sweep…")

            def work():
                # Build features and one ML forecast set if not already available.
                if self.dataset is None or self.features is None:
                    self.dataset, self.features = assemble_dataset(self.md)
                if self.train_result is None:
                    eng = ExpectedReturnEngine(self.features)
                    self.train_result = eng.train_and_forecast(
                        self.dataset, model_name, n_folds=3, log=self._tlog)

                tickers = self.md.tickers
                comps = {
                    "ML Forecast": self.train_result.forecasts,
                    "CAPM": capm_expected_returns(self.md, tickers),
                    "Factor Model": factor_expected_returns(
                        self.dataset, self.features, tickers),
                }
                comps["Black-Litterman"] = black_litterman_returns(
                    self.md, tickers, comps["ML Forecast"])
                comps["Ensemble"] = ensemble_returns(
                    comps,
                    {"ML Forecast": 0.40, "Factor Model": 0.30,
                     "Black-Litterman": 0.30},
                )
                error_bands = self.train_result.error_bands or {t: 0.03 for t in tickers}
                rets = self.md.returns_matrix()
                bench = self.md.benchmark_returns()
                covariances = {
                    method: CovarianceEngine(rets, bench).estimate(method)
                    for method in CovarianceEngine.METHODS
                }

                rows = []
                total = len(comps) * len(covariances) * len(Optimizer.OBJECTIVES)
                done = 0
                for src, mu in comps.items():
                    for risk, cov in covariances.items():
                        for obj in Optimizer.OBJECTIVES:
                            done += 1
                            try:
                                scen = None
                                if obj == "CVaR":
                                    rng = np.random.default_rng(20260710)
                                    z = rng.standard_normal((4000, len(tickers)))
                                    cov_m = cov.values * MONTH_DAYS
                                    try:
                                        L = np.linalg.cholesky(
                                            cov_m + 1e-10 * np.eye(len(tickers)))
                                    except np.linalg.LinAlgError:
                                        vals, vecs = np.linalg.eigh(cov_m)
                                        L = vecs @ np.diag(
                                            np.sqrt(np.clip(vals, 1e-12, None)))
                                    mu_vec = np.array([mu.get(t, 0.0) for t in tickers])
                                    scen = mu_vec + z @ L.T

                                port = Optimizer(
                                    mu, cov, cons, scenario_returns=scen
                                ).optimize(
                                    obj, error_bands=error_bands,
                                    robust_kappa=kappa)
                                rows.append({
                                    "Return source": src,
                                    "Risk model": risk,
                                    "Objective": obj,
                                    "Return": float(port.exp_return_monthly),
                                    "Volatility": float(port.volatility_monthly),
                                    "Sharpe": float(port.sharpe),
                                    "Concentration": float(port.concentration),
                                    "Converged": bool(port.diagnostics.get("converged", False)),
                                    "Weights": dict(port.weights),
                                    "Portfolio": port,
                                    "Mu": mu,
                                    "Cov": cov,
                                })
                            except Exception as exc:
                                rows.append({
                                    "Return source": src, "Risk model": risk,
                                    "Objective": obj, "Return": np.nan,
                                    "Volatility": np.nan, "Sharpe": np.nan,
                                    "Concentration": np.nan, "Converged": False,
                                    "Error": str(exc), "Weights": {},
                                })
                            if done % 5 == 0 or done == total:
                                self.q.put(lambda d=done, t=total: self._set_status(
                                    f"Combination sweep: {d}/{t}"))

                df = pd.DataFrame(rows)
                valid = df[["Return", "Volatility", "Sharpe", "Concentration"]].replace(
                    [np.inf, -np.inf], np.nan).notna().all(axis=1)
                df["Score"] = np.nan
                if valid.any():
                    v = df.loc[valid]
                    def zscore(col):
                        x = v[col].astype(float)
                        sd = float(x.std(ddof=0))
                        return (x - float(x.mean())) / (sd if sd > 1e-12 else 1.0)
                    # Reward risk-adjusted return and return; penalise risk and concentration.
                    df.loc[valid, "Score"] = (
                        1.00 * zscore("Sharpe")
                        + 0.50 * zscore("Return")
                        - 0.50 * zscore("Volatility")
                        - 0.25 * zscore("Concentration")
                    )
                    # Failed SLSQP solutions remain visible but cannot rank first.
                    df.loc[valid & ~df["Converged"], "Score"] -= 2.0
                return df.sort_values(
                    ["Score", "Sharpe", "Return"], ascending=False,
                    na_position="last").reset_index(drop=True)

            self._submit(work, self._on_all_combinations,
                         "Running all portfolio combinations…", self.btn_all)

        def _on_all_combinations(self, df):
            self.research_results = df
            self._set_status(f"Combination sweep complete: {len(df)} portfolios tested.")
            self._show_combination_results(df)

        def _show_combination_results(self, df):
            win = tk.Toplevel(self)
            win.title("HELM — All Combination Research Results")
            win.geometry("1240x720")
            win.configure(bg=BG)

            top = tk.Frame(win, bg=CARD, highlightbackground=GRID,
                           highlightthickness=1)
            top.pack(fill="x", padx=10, pady=10)
            tk.Label(top, text="All return × risk × objective combinations",
                     bg=CARD, fg=INK, font=(FONT, 12, "bold")).pack(
                         side="left", padx=10, pady=9)
            tk.Label(top,
                     text="Composite score = Sharpe + 0.5 Return − 0.5 Risk − 0.25 Concentration",
                     bg=CARD, fg=MUTED, font=(FONT, 9)).pack(
                         side="left", padx=12)

            sort_var = tk.StringVar(value="Composite Score")
            ttk.Combobox(top, textvariable=sort_var, state="readonly", width=20,
                         values=["Composite Score", "Sharpe", "Return",
                                 "Lowest Risk", "Lowest Concentration"]).pack(
                                     side="right", padx=10, pady=8)

            body = tk.Frame(win, bg=CARD, highlightbackground=GRID,
                            highlightthickness=1)
            body.pack(fill="both", expand=True, padx=10, pady=(0, 10))
            cols = ("Rank", "Return source", "Risk model", "Objective",
                    "Return", "Volatility", "Sharpe", "Concentration",
                    "Score", "Converged")
            tree = ttk.Treeview(body, columns=cols, show="headings", height=22)
            widths = {"Rank":55, "Return source":135, "Risk model":110,
                      "Objective":105, "Return":85, "Volatility":85,
                      "Sharpe":75, "Concentration":100, "Score":75,
                      "Converged":80}
            for c in cols:
                tree.heading(c, text=c)
                tree.column(c, width=widths.get(c, 100), anchor="center")
            sy = ttk.Scrollbar(body, orient="vertical", command=tree.yview)
            sx = ttk.Scrollbar(body, orient="horizontal", command=tree.xview)
            tree.configure(yscrollcommand=sy.set, xscrollcommand=sx.set)
            tree.grid(row=0, column=0, sticky="nsew")
            sy.grid(row=0, column=1, sticky="ns")
            sx.grid(row=1, column=0, sticky="ew")
            body.rowconfigure(0, weight=1)
            body.columnconfigure(0, weight=1)

            note = tk.Label(
                win, bg=BG, fg=MUTED, anchor="w", justify="left",
                text=("This ranks portfolios using the current forecast and covariance snapshot. "
                      "It does not prove future superiority. Shortlist the top combinations and "
                      "confirm them with the walk-forward Backtest tab."),
                font=(FONT, 9))
            note.pack(fill="x", padx=14, pady=(0, 8))

            def sorted_frame():
                mode = sort_var.get()
                if mode == "Sharpe":
                    return df.sort_values("Sharpe", ascending=False, na_position="last")
                if mode == "Return":
                    return df.sort_values("Return", ascending=False, na_position="last")
                if mode == "Lowest Risk":
                    return df.sort_values("Volatility", ascending=True, na_position="last")
                if mode == "Lowest Concentration":
                    return df.sort_values("Concentration", ascending=True, na_position="last")
                return df.sort_values("Score", ascending=False, na_position="last")

            def refresh(*_):
                for item in tree.get_children():
                    tree.delete(item)
                view = sorted_frame().reset_index(drop=True)
                for i, row in view.iterrows():
                    tree.insert("", "end", iid=str(i), values=(
                        i + 1, row.get("Return source", ""), row.get("Risk model", ""),
                        row.get("Objective", ""), to_pct(row.get("Return", np.nan)),
                        to_pct(row.get("Volatility", np.nan)),
                        f"{row.get('Sharpe', np.nan):.3f}",
                        f"{row.get('Concentration', np.nan):.3f}",
                        f"{row.get('Score', np.nan):.3f}",
                        "Yes" if row.get("Converged", False) else "No"))
                tree._view_df = view

            def use_selected(_event=None):
                sel = tree.selection()
                if not sel:
                    return
                row = tree._view_df.iloc[int(sel[0])]
                if not row.get("Converged", False) or "Portfolio" not in row:
                    return
                self.v_retsrc.set(row["Return source"])
                self.v_riskmodel.set(row["Risk model"])
                self.v_obj.set(row["Objective"])
                self._on_opt((row["Mu"], row["Cov"], row["Risk model"], row["Portfolio"]))
                self._set_status("Selected ranked combination loaded into Optimizer.")

            sort_var.trace_add("write", refresh)
            tree.bind("<Double-1>", use_selected)
            refresh()

        def _draw_weights(self, chart, port):
            items = [(k, v) for k, v in port.weights.items() if v > 1e-4]
            items.sort(key=lambda kv: -kv[1])
            labs = [k for k, _ in items]
            vals = [v for _, v in items]
            cols = [ACCENT if k != "Cash" else "#94a3b8" for k in labs]
            chart.bars(labs, vals, title="Portfolio weights", colors=cols)

        # ================= TAB 7 — PORTFOLIO ======================= #
        def _tab_portfolio(self):
            tab = ttk.Frame(self.nb)
            self.nb.add(tab, text="7 · Portfolio")
            top = self._card(tab)
            top.pack(fill="x", padx=10, pady=(10, 6))
            tk.Label(top, text="Portfolio Allocation", bg=CARD,
                     font=(FONT, 12, "bold"), fg=INK).pack(side="left", padx=10, pady=8)
            body = self._card(tab)
            body.pack(fill="both", expand=True, padx=10, pady=(0, 10))
            self.pf_tree = self._tree(body, ["Asset", "Weight", "Sector"], height=8)
            self.pf_tree.pack(fill="x", padx=8, pady=8)
            self.pf_chart = Chart(body, height=280)
            self.pf_chart.pack(fill="both", expand=True, padx=8, pady=8)
            self.pf_chart.message("Run the optimizer (Tab 6) to populate.")

        def _refresh_portfolio_tab(self, port):
            rows = []
            for k, v in sorted(port.weights.items(), key=lambda kv: -kv[1]):
                if v > 1e-4:
                    rows.append([k, to_pct(v), SECTOR_MAP.get(k, "Cash" if k == "Cash" else "—")])
            self._fill_tree(self.pf_tree, rows)
            self._draw_weights(self.pf_chart, port)

        # ================= TAB 8 — MONTE CARLO ===================== #
        def _tab_montecarlo(self):
            tab = ttk.Frame(self.nb)
            self.nb.add(tab, text="8 · Monte Carlo")
            left = self._card(tab)
            left.pack(side="left", fill="y", padx=10, pady=10)
            tk.Label(left, text="Monte Carlo Simulator", bg=CARD,
                     font=(FONT, 12, "bold"), fg=INK).pack(anchor="w", padx=10, pady=(10, 4))
            self.v_paths = self._labeled(left, "Paths", 10000)
            self.v_mc_h = self._labeled(left, "Horizon (days)", MONTH_DAYS)
            self.btn_mc = ttk.Button(left, text="Run Simulation", style="Accent.TButton",
                                     command=self._do_mc)
            self.btn_mc.pack(fill="x", padx=10, pady=10)
            self.mc_text = tk.Text(left, height=10, width=28, bg="#f8fafc",
                                   relief="flat", font=(FONT, 9), fg=INK)
            self.mc_text.pack(fill="x", padx=10, pady=(0, 10))
            right = self._card(tab)
            right.pack(side="left", fill="both", expand=True, padx=(0, 10), pady=10)
            self.mc_fan = Chart(right, height=220)
            self.mc_fan.pack(fill="both", expand=True, padx=8, pady=(8, 4))
            self.mc_hist = Chart(right, height=200)
            self.mc_hist.pack(fill="both", expand=True, padx=8, pady=(4, 8))
            self.mc_fan.message("Run the optimizer, then simulate.")

        def _do_mc(self):
            if self.portfolio is None or self.cov is None or self.mu_final is None:
                messagebox.showinfo("HELM", "Run the optimizer first (Tab 6).")
                return
            paths = int(float(self.v_paths.get()))
            horizon = int(float(self.v_mc_h.get()))

            def work():
                return MonteCarloSimulator(self.mu_final, self.cov).run(
                    self.portfolio.weights, horizon_days=horizon, paths=paths)
            self._submit(work, self._on_mc, f"Simulating {paths} paths…", self.btn_mc)

        def _on_mc(self, mc):
            self.mc_result = mc
            self.mc_fan.fan(mc.portfolio_paths, title="Wealth paths (subsample) & median")
            self.mc_hist.hist(mc.terminal_wealth, title="Terminal wealth distribution")
            self.mc_text.delete("1.0", "end")
            for k, v in mc.summary().items():
                self.mc_text.insert("end", f"{k:<20}: {v}\n")
            self._set_status(f"Monte Carlo done. Prob-loss {to_pct(mc.prob_loss)}")

        # ================= TAB 9 — BACKTEST ======================== #
        def _tab_backtest(self):
            tab = ttk.Frame(self.nb)
            self.nb.add(tab, text="9 · Backtest")
            left = self._card(tab)
            left.pack(side="left", fill="y", padx=10, pady=10)
            tk.Label(left, text="Walk-forward Backtest", bg=CARD,
                     font=(FONT, 12, "bold"), fg=INK).pack(anchor="w", padx=10, pady=(10, 4))
            self.v_bt_model = self._combo(left, "Model", list(MODEL_REGISTRY.keys()), "Random Forest")
            self.v_bt_ret = self._combo(left, "Return source", ["ML", "CAPM"], "ML")
            self.v_bt_cov = self._combo(left, "Risk model", CovarianceEngine.METHODS, "EWMA")
            self.v_bt_obj = self._combo(left, "Objective", Optimizer.OBJECTIVES, "Max Sharpe")
            self.v_bt_min = self._labeled(left, "Min train (d)", 180)
            self.v_bt_folds = self._labeled(left, "Research CV folds", 3)
            self.btn_bt = ttk.Button(left, text="Run Portfolio Backtest", style="Accent.TButton",
                                     command=self._do_bt)
            self.btn_bt.pack(fill="x", padx=10, pady=(10, 4))
            self.btn_stock_research = ttk.Button(
                left, text="Run Per-Stock Model Research", style="Accent.TButton",
                command=self._do_stock_research)
            self.btn_stock_research.pack(fill="x", padx=10, pady=(0, 8))
            self.bt_prog = ttk.Progressbar(left, mode="determinate", maximum=1.0)
            self.bt_prog.pack(fill="x", padx=10, pady=(0, 8))
            self.bt_text = tk.Text(left, height=12, width=30, bg="#f8fafc",
                                   relief="flat", font=(FONT, 9), fg=INK)
            self.bt_text.pack(fill="x", padx=10, pady=(0, 10))
            right = self._card(tab)
            right.pack(side="left", fill="both", expand=True, padx=(0, 10), pady=10)
            self.bt_view = ttk.Notebook(right)
            self.bt_view.pack(fill="both", expand=True, padx=8, pady=8)

            chart_tab = ttk.Frame(self.bt_view)
            research_tab = ttk.Frame(self.bt_view)
            self.bt_view.add(chart_tab, text="Portfolio Walk-Forward")
            self.bt_view.add(research_tab, text="Per-Stock Model Research")

            self.bt_chart = Chart(chart_tab, height=360)
            self.bt_chart.pack(fill="both", expand=True, padx=4, pady=4)
            self.bt_chart.message("Run a portfolio walk-forward backtest vs NIFTY.")

            cols = ("stock", "rank", "model", "rmse", "mae", "direction",
                    "ic", "hit", "topdecile", "score", "status")
            self.bt_research_tree = ttk.Treeview(research_tab, columns=cols, show="headings")
            headings = {
                "stock":"Stock", "rank":"Rank", "model":"Model", "rmse":"RMSE",
                "mae":"MAE", "direction":"Direction", "ic":"IC", "hit":"Hit ratio",
                "topdecile":"Top-decile", "score":"Score", "status":"Status"}
            widths = {"stock":90, "rank":50, "model":125, "rmse":75, "mae":75,
                      "direction":85, "ic":65, "hit":80, "topdecile":90,
                      "score":70, "status":180}
            for c in cols:
                self.bt_research_tree.heading(c, text=headings[c])
                self.bt_research_tree.column(c, width=widths[c], anchor="center")
            ysb = ttk.Scrollbar(research_tab, orient="vertical", command=self.bt_research_tree.yview)
            xsb = ttk.Scrollbar(research_tab, orient="horizontal", command=self.bt_research_tree.xview)
            self.bt_research_tree.configure(yscrollcommand=ysb.set, xscrollcommand=xsb.set)
            self.bt_research_tree.grid(row=0, column=0, sticky="nsew")
            ysb.grid(row=0, column=1, sticky="ns")
            xsb.grid(row=1, column=0, sticky="ew")
            research_tab.rowconfigure(0, weight=1)
            research_tab.columnconfigure(0, weight=1)

        def _do_bt(self):
            if self.md is None:
                messagebox.showinfo("HELM", "Load data first (Tab 1).")
                return
            if self.dataset is None:
                self.dataset, self.features = assemble_dataset(self.md)
            cfg = dict(model=self.v_bt_model.get(), ret=self.v_bt_ret.get(),
                       cov=self.v_bt_cov.get(), obj=self.v_bt_obj.get(),
                       mintrain=int(float(self.v_bt_min.get())))
            self.bt_prog["value"] = 0

            def prog(p):
                self.q.put(lambda p=p: self.bt_prog.config(value=p))

            def work():
                bt = Backtester(self.md, self.features, cfg["model"], cfg["cov"],
                                cfg["obj"], OptConstraints(cash_floor=0.05),
                                return_source=cfg["ret"])
                return bt.run(min_train=cfg["mintrain"], log=self._tlog, progress=prog)
            self._submit(work, self._on_bt, "Backtesting (walk-forward)…", self.btn_bt)

        def _do_stock_research(self):
            if self.md is None:
                messagebox.showinfo("HELM", "Load data first (Tab 1).")
                return
            if self.dataset is None:
                self.dataset, self.features = assemble_dataset(self.md)
            try:
                folds = max(2, int(float(self.v_bt_folds.get())))
                min_rows = max(80, int(float(self.v_bt_min.get())))
            except Exception:
                messagebox.showerror("HELM", "Research CV folds and Min train must be numbers.")
                return
            self.bt_prog["value"] = 0
            for item in self.bt_research_tree.get_children():
                self.bt_research_tree.delete(item)
            self.bt_view.select(1)

            def prog(p):
                self.q.put(lambda p=p: self.bt_prog.config(value=p))

            def work():
                lab = StockModelResearch(self.md, self.features)
                return lab.run(n_folds=folds, min_rows=min_rows,
                               log=self._tlog, progress=prog)
            self._submit(work, self._on_stock_research,
                         "Running per-stock walk-forward model research…",
                         self.btn_stock_research)

        def _on_stock_research(self, result):
            self.stock_model_research = result
            self.bt_prog["value"] = 1.0
            for item in self.bt_research_tree.get_children():
                self.bt_research_tree.delete(item)
            ordered = sorted(result.rows, key=lambda r: (r.get("stock", ""), r.get("rank", 999)))
            for r in ordered:
                ok = np.isfinite(r.get("rmse", np.nan))
                vals = (
                    r.get("stock", ""), r.get("rank", "—"),
                    ("★ " if r.get("best") else "") + r.get("model", ""),
                    f"{r.get('rmse', np.nan):.4f}" if ok else "—",
                    f"{r.get('mae', np.nan):.4f}" if ok else "—",
                    to_pct(r.get("dir_acc", 0)) if ok else "—",
                    f"{r.get('ic', 0):.3f}" if ok else "—",
                    to_pct(r.get("hit_ratio", 0)) if ok else "—",
                    to_pct(r.get("top_decile", 0)) if ok else "—",
                    f"{r.get('score', 0):.3f}" if ok else "—",
                    r.get("status", "OK"))
                self.bt_research_tree.insert("", "end", values=vals)
            winners = ", ".join(f"{s}: {m}" for s, m in result.best_by_stock.items())
            self.bt_text.delete("1.0", "end")
            self.bt_text.insert("end", "BEST WALK-FORWARD MODEL PER STOCK\n\n" +
                                (winners or "No model produced valid folds.") +
                                "\n\nSelection uses out-of-sample IC, direction, hit ratio and RMSE.\n")
            self._set_status(f"Per-stock research done for {len(result.best_by_stock)} stocks.")

        def _on_bt(self, bt):
            self.backtest_result = bt
            self.bt_prog["value"] = 1.0
            self.bt_chart.lines(
                [(range(len(bt.equity)), bt.equity), (range(len(bt.bench_equity)), bt.bench_equity)],
                ["Strategy", "NIFTY"], [ACCENT, "#94a3b8"],
                title="Equity curve (walk-forward)", yfmt=lambda v: f"{v:.2f}x")
            m = bt.metrics
            self.bt_text.delete("1.0", "end")
            self.bt_text.insert("end",
                f"CAGR          {to_pct(m.get('cagr',0))}\n"
                f"Total return  {to_pct(m.get('total_return',0))}\n"
                f"Volatility    {to_pct(m.get('volatility',0))}\n"
                f"Sharpe        {m.get('sharpe',0):.2f}\n"
                f"Sortino       {m.get('sortino',0):.2f}\n"
                f"Max drawdown  {to_pct(m.get('max_drawdown',0))}\n"
                f"Win rate      {to_pct(m.get('win_rate',0))}\n"
                f"Avg turnover  {to_pct(m.get('avg_turnover',0))}\n"
                f"NIFTY return  {to_pct(m.get('bench_total_return',0))}\n"
                f"Excess        {to_pct(m.get('excess_vs_bench',0))}\n")
            self._set_status(f"Backtest done. Sharpe {m.get('sharpe',0):.2f}")

        # ================= TAB 10 — REPORT ========================= #
        def _tab_report(self):
            tab = ttk.Frame(self.nb)
            self.nb.add(tab, text="10 · Report")
            top = self._card(tab)
            top.pack(fill="x", padx=10, pady=(10, 6))
            tk.Label(top, text="Final Risk Report", bg=CARD,
                     font=(FONT, 12, "bold"), fg=INK).pack(side="left", padx=10, pady=8)
            ttk.Button(top, text="Generate", style="Accent.TButton",
                       command=self._do_report).pack(side="right", padx=6, pady=8)
            ttk.Button(top, text="Export…", command=self._export_report).pack(side="right", padx=6, pady=8)
            body = self._card(tab)
            body.pack(fill="both", expand=True, padx=10, pady=(0, 6))
            self.report_text = tk.Text(body, bg="white", relief="flat",
                                       font=("Consolas", 9), fg=INK, wrap="word")
            self.report_text.pack(fill="both", expand=True, padx=8, pady=8)
            # log console
            logc = self._card(tab)
            logc.pack(fill="x", padx=10, pady=(0, 10))
            tk.Label(logc, text="Log", bg=CARD, fg=MUTED, font=(FONT, 8)).pack(anchor="w", padx=8)
            self.log_text = tk.Text(logc, height=5, bg="#0f172a", fg="#5eead4",
                                    relief="flat", font=("Consolas", 8))
            self.log_text.pack(fill="x", padx=8, pady=(0, 8))

        # ================= TAB 11 — AUTOMATED RESEARCH ============== #
        def _tab_auto_research(self):
            tab = ttk.Frame(self.nb)
            self.nb.add(tab, text="11 · Auto Research")

            top = self._card(tab)
            top.pack(fill="x", padx=10, pady=(10, 6))
            tk.Label(top, text="Automated Model & Portfolio Tournament", bg=CARD,
                     font=(FONT, 12, "bold"), fg=INK).pack(side="left", padx=10, pady=8)
            tk.Label(top,
                     text="All models × return sources × risk models × objectives",
                     bg=CARD, fg=MUTED, font=(FONT, 9)).pack(side="left", padx=12)
            self.v_research_mode = tk.StringVar(value="Full")
            ttk.Combobox(top, textvariable=self.v_research_mode,
                         values=["Full", "Fast"], state="readonly", width=9).pack(
                             side="right", padx=(4, 10), pady=8)
            self.btn_research = ttk.Button(top, text="Run Research", style="Accent.TButton",
                                           command=self._do_auto_research)
            self.btn_research.pack(side="right", padx=6, pady=8)

            summary = self._card(tab)
            summary.pack(fill="x", padx=10, pady=(0, 6))
            self.research_summary = tk.StringVar(value=(
                "Load data, choose Full or Fast, then run. Full mode evaluates every available "
                "combination using purged walk-forward testing."))
            tk.Label(summary, textvariable=self.research_summary, bg=CARD, fg=INK,
                     justify="left", anchor="w", wraplength=1080,
                     font=(FONT, 9)).pack(fill="x", padx=10, pady=8)

            body = ttk.Notebook(tab)
            body.pack(fill="both", expand=True, padx=10, pady=(0, 6))
            self.research_trees = {}
            specs = {
                "Models": ["rank", "model", "score", "rmse", "mae", "dir_acc", "ic", "hit_ratio"],
                "Covariance": ["rank", "method", "score", "frobenius_error", "variance_mae", "windows"],
                "Combinations": ["rank", "model", "return_source", "risk_model", "objective",
                                 "score", "sharpe", "cagr", "max_drawdown", "avg_turnover"],
                "Per Stock": ["stock", "rank_for_stock", "model", "score", "rmse", "dir_acc", "ic"],
            }
            for title, columns in specs.items():
                frame = ttk.Frame(body)
                body.add(frame, text=title)
                tree = ttk.Treeview(frame, columns=columns, show="headings")
                ybar = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
                xbar = ttk.Scrollbar(frame, orient="horizontal", command=tree.xview)
                tree.configure(yscrollcommand=ybar.set, xscrollcommand=xbar.set)
                for col in columns:
                    tree.heading(col, text=col.replace("_", " ").title())
                    tree.column(col, width=120 if col not in ("model", "return_source") else 145,
                                anchor="center", stretch=True)
                tree.grid(row=0, column=0, sticky="nsew")
                ybar.grid(row=0, column=1, sticky="ns")
                xbar.grid(row=1, column=0, sticky="ew")
                frame.rowconfigure(0, weight=1)
                frame.columnconfigure(0, weight=1)
                self.research_trees[title] = (tree, columns)

            logc = self._card(tab)
            logc.pack(fill="x", padx=10, pady=(0, 10))
            tk.Label(logc, text="Research progress", bg=CARD, fg=MUTED,
                     font=(FONT, 8)).pack(anchor="w", padx=8)
            self.research_log = tk.Text(logc, height=6, bg="#0f172a", fg="#5eead4",
                                        relief="flat", font=("Consolas", 8))
            self.research_log.pack(fill="x", padx=8, pady=(0, 8))

        def _append_research_log(self, msg):
            try:
                self.research_log.insert("end", f"{msg}\n")
                self.research_log.see("end")
            except Exception:
                pass

        def _do_auto_research(self):
            if self.md is None:
                messagebox.showinfo("HELM", "Load data first (Tab 1).")
                return
            mode = self.v_research_mode.get().strip().lower()
            cons = self._constraints()
            cfg = ResearchConfig(mode=mode)
            cfg.max_weight = cons.max_weight
            cfg.sector_cap = cons.sector_cap
            cfg.cash_floor = cons.cash_floor
            cfg.transaction_cost_bps = cons.txn_cost_bps
            self.research_log.delete("1.0", "end")
            self.research_summary.set(
                f"Running {mode.upper()} research. The interface remains responsive; "
                "deep models in Full mode may take considerable time.")

            def research_log(msg):
                self.q.put(lambda m=msg: self._append_research_log(m))

            def work():
                result = AutoResearchEngine(self.md, cfg, log=research_log).run()
                save_result(result, Path("research_output_gui"))
                return result

            self._submit(work, self._on_auto_research,
                         f"Running {mode} automated research…", self.btn_research)

        @staticmethod
        def _research_value(value, column):
            if value is None or (isinstance(value, float) and not np.isfinite(value)):
                return "—"
            if column in ("cagr", "max_drawdown", "avg_turnover", "dir_acc", "hit_ratio"):
                return to_pct(value)
            if isinstance(value, (float, np.floating)):
                return f"{value:.4f}"
            return value

        def _fill_research_tree(self, title, frame, limit=200):
            tree, columns = self.research_trees[title]
            for item in tree.get_children():
                tree.delete(item)
            if frame is None or frame.empty:
                return
            for _, row in frame.head(limit).iterrows():
                tree.insert("", "end", values=[
                    self._research_value(row.get(c), c) for c in columns])

        def _on_auto_research(self, result):
            self.auto_research_result = result
            self._fill_research_tree("Models", result.model_leaderboard)
            self._fill_research_tree("Covariance", result.covariance_leaderboard)
            self._fill_research_tree("Combinations", result.combination_leaderboard)
            self._fill_research_tree("Per Stock", result.per_stock_leaderboard)
            w = result.winner
            self.research_summary.set(
                f"WINNER — Model: {w.get('model')} | Return: {w.get('return_source')} | "
                f"Risk: {w.get('risk_model')} | Objective: {w.get('objective')} | "
                f"OOS score: {w.get('score', 0):.3f} | Sharpe: {w.get('sharpe', 0):.2f}. "
                "Leaderboards and report were saved in research_output_gui.")

            out = result.final_pipeline
            if out is not None:
                self.md = out.market
                self.dataset, self.features = out.dataset, out.features
                self.train_result = out.train_result
                self.return_components = out.return_components
                self.mu_final, self.error_bands = out.mu_final, out.error_bands
                self.cov, self.cov_method = out.Sigma, out.cov_method
                self.portfolio, self.mc_result = out.portfolio, out.montecarlo
                if hasattr(self, "v_model"):
                    self.v_model.set(w.get("model", self.v_model.get()))
                self.v_retsrc.set(w.get("return_source", self.v_retsrc.get()))
                self.v_riskmodel.set(w.get("risk_model", self.v_riskmodel.get()))
                self.v_obj.set(w.get("objective", self.v_obj.get()))
                self._on_opt((out.mu_final, out.Sigma, out.cov_method, out.portfolio))
                self._on_mc(out.montecarlo)
                self._on_forecast()
            self.report_text.delete("1.0", "end")
            self.report_text.insert("end", result.final_report)
            self._set_status("Automated research complete. Winner loaded into the portfolio tabs.")

        def _append_log(self, msg):
            try:
                self.log_text.insert("end", f"{msg}\n")
                self.log_text.see("end")
            except Exception:
                pass

        def _do_report(self):
            if self.portfolio is None:
                messagebox.showinfo("HELM", "Run the optimizer first (Tab 6).")
                return
            out = PipelineOutput(
                market=self.md, dataset=self.dataset, features=self.features,
                return_components=self.return_components, mu_final=self.mu_final,
                error_bands=self.error_bands,
                train_result=self.train_result or TrainResult("(none)", {}, {}, {}, {}),
                cov_method=self.cov_method or "—", Sigma=self.cov,
                portfolio=self.portfolio,
                montecarlo=self.mc_result or MonteCarloSimulator(
                    self.mu_final, self.cov).run(self.portfolio.weights, paths=3000))
            rep = build_report(out, self.v_model.get() if hasattr(self, "v_model") else "—",
                               self.v_retsrc.get(), self.portfolio.objective)
            self.report_text.delete("1.0", "end")
            self.report_text.insert("end", rep)
            self._set_status("Report generated.")

        def _export_report(self):
            txt = self.report_text.get("1.0", "end").strip()
            if not txt:
                messagebox.showinfo("HELM", "Generate the report first.")
                return
            path = filedialog.asksaveasfilename(
                defaultextension=".txt", filetypes=[("Text", "*.txt")],
                initialfile=f"HELM_report_{datetime.today():%Y%m%d}.txt")
            if path:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(txt)
                self._set_status(f"Saved: {path}")

        # ---- full pipeline convenience ---- #
        def _do_pipeline(self):
            if self.md is None:
                messagebox.showinfo("HELM", "Load data first (Tab 1).")
                return
            cons = self._constraints()
            cfg = dict(model=(self.v_model.get() if hasattr(self, "v_model") else "Random Forest"),
                       cov=self.v_riskmodel.get(), obj=self.v_obj.get(),
                       src=self.v_retsrc.get(), kappa=float(self.v_kappa.get()),
                       paths=int(float(self.v_paths.get())) if hasattr(self, "v_paths") else 10000)

            def work():
                return Pipeline(self.md).run(
                    model_name=cfg["model"], cov_method=cfg["cov"], objective=cfg["obj"],
                    return_source=cfg["src"], constraints=cons, mc_paths=cfg["paths"],
                    robust_kappa=cfg["kappa"], log=self._tlog)
            self._submit(work, self._on_pipeline, "Running full pipeline…", self.btn_opt)

        def _on_pipeline(self, out):
            self.md = out.market
            self.dataset, self.features = out.dataset, out.features
            self.train_result = out.train_result
            self.return_components = out.return_components
            self.mu_final, self.error_bands = out.mu_final, out.error_bands
            self.cov, self.cov_method = out.Sigma, out.cov_method
            self.portfolio, self.mc_result = out.portfolio, out.montecarlo
            self._on_opt((out.mu_final, out.Sigma, out.cov_method, out.portfolio))
            self._on_mc(out.montecarlo)
            self._on_forecast()
            rep = build_report(out, self.v_model.get(), self.v_retsrc.get(),
                               out.portfolio.objective)
            self.report_text.delete("1.0", "end")
            self.report_text.insert("end", rep)
            self._set_status("Full pipeline complete. See Report tab.")

    app = App()
    app.mainloop()


"""Leakage-safe automated research tournament for HELM FINSERV.

The original desktop application exposes each research stage independently.
This module coordinates those stages without selecting a winner on the final
test observations.  Inner expanding-window CV chooses/calibrates models; outer
walk-forward windows compare complete portfolio configurations.
"""

import argparse
import json
import math
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd



# The automated controller originally lived in a separate module.  Point its
# namespace alias at this module so the merged file remains fully standalone.
hc = sys.modules[__name__]

RETURN_SOURCES = ["ML", "CAPM", "Factor Model", "Black-Litterman", "Ensemble"]


@dataclass
class ResearchConfig:
    mode: str = "full"
    cv_folds: int = 3
    target_horizon: int = hc.MONTH_DAYS
    min_train_days: int = 252
    rebalance_days: int = hc.MONTH_DAYS
    covariance_lookback: int = 252
    transaction_cost_bps: float = 5.0
    max_weight: float = 0.30
    sector_cap: float = 0.35
    cash_floor: float = 0.05
    monte_carlo_paths: int = 10_000
    monte_carlo_horizon: int = hc.MONTH_DAYS
    seed: int = 42
    models: list[str] = field(default_factory=lambda: list(hc.MODEL_REGISTRY))
    return_sources: list[str] = field(default_factory=lambda: RETURN_SOURCES.copy())
    covariance_methods: list[str] = field(default_factory=lambda: hc.CovarianceEngine.METHODS.copy())
    objectives: list[str] = field(default_factory=lambda: hc.Optimizer.OBJECTIVES.copy())

    def __post_init__(self):
        if self.mode == "fast":
            self.models = [m for m in ["Random Forest", "XGBoost"] if m in hc.MODEL_REGISTRY]
            self.covariance_methods = [m for m in ["EWMA", "Ledoit-Wolf"]
                                       if m in hc.CovarianceEngine.METHODS]
            self.objectives = [m for m in ["Max Sharpe", "Min Variance", "Robust"]
                               if m in hc.Optimizer.OBJECTIVES]


@dataclass
class ResearchResult:
    generated_at: str
    config: dict
    model_leaderboard: pd.DataFrame
    covariance_leaderboard: pd.DataFrame
    combination_leaderboard: pd.DataFrame
    per_stock_leaderboard: pd.DataFrame
    failures: pd.DataFrame
    winner: dict
    final_pipeline: hc.PipelineOutput | None
    final_report: str


def _finite(x, default=0.0):
    try:
        x = float(x)
        return x if np.isfinite(x) else default
    except Exception:
        return default


def _rank_score(df: pd.DataFrame, higher: Iterable[str], lower: Iterable[str]) -> pd.Series:
    """Average percentile rank.  This avoids mixing unlike metric units."""
    parts = []
    for col in higher:
        if col in df:
            parts.append(pd.to_numeric(df[col], errors="coerce").rank(pct=True))
    for col in lower:
        if col in df:
            parts.append((-pd.to_numeric(df[col], errors="coerce")).rank(pct=True))
    if not parts:
        return pd.Series(0.0, index=df.index)
    return pd.concat(parts, axis=1).fillna(0.0).mean(axis=1)


def _slice_market(md: hc.MarketData, asof) -> hc.MarketData:
    return hc.MarketData(
        prices={t: df.loc[:asof].copy() for t, df in md.prices.items()},
        fundamentals=md.fundamentals,
        sentiment=md.sentiment,
        source=md.source,
    )


def _metrics(daily: np.ndarray, turnover: list[float], benchmark: np.ndarray) -> dict:
    daily = np.asarray(daily, float)
    benchmark = np.asarray(benchmark, float)
    if len(daily) == 0:
        return {}
    wealth = np.cumprod(1.0 + daily)
    years = len(daily) / hc.TRADING_DAYS
    cagr = wealth[-1] ** (1 / max(years, 1e-9)) - 1
    vol = daily.std(ddof=1) * math.sqrt(hc.TRADING_DAYS) if len(daily) > 1 else 0.0
    excess = daily - hc.RISK_FREE_DAILY
    sharpe = excess.mean() / (daily.std(ddof=1) + 1e-12) * math.sqrt(hc.TRADING_DAYS)
    downside = daily[daily < 0]
    sortino = excess.mean() / (downside.std(ddof=1) + 1e-12) * math.sqrt(hc.TRADING_DAYS)
    drawdown = wealth / np.maximum.accumulate(wealth) - 1.0
    losses = -daily
    var95 = float(np.quantile(losses, .95))
    cvar95 = float(losses[losses >= var95].mean()) if np.any(losses >= var95) else var95
    bench_total = float(np.prod(1.0 + benchmark) - 1.0) if len(benchmark) else 0.0
    return {
        "cagr": float(cagr), "volatility": float(vol), "sharpe": float(sharpe),
        "sortino": float(sortino), "max_drawdown": float(drawdown.min()),
        "cvar95_daily": cvar95, "total_return": float(wealth[-1] - 1.0),
        "benchmark_total": bench_total,
        "excess_vs_benchmark": float(wealth[-1] - 1.0 - bench_total),
        "win_rate": float(np.mean(daily > 0)),
        "avg_turnover": float(np.mean(turnover)) if turnover else 0.0,
    }


class AutoResearchEngine:
    def __init__(self, market: hc.MarketData, config: ResearchConfig | None = None,
                 log: Callable[[str], None] = print):
        self.md = market
        self.cfg = config or ResearchConfig()
        self.log = log
        self.dataset, self.features = hc.assemble_dataset(market, tickers=market.tickers)
        self.failures: list[dict] = []
        self.constraints = hc.OptConstraints(
            max_weight=self.cfg.max_weight, sector_cap=self.cfg.sector_cap,
            cash_floor=self.cfg.cash_floor, txn_cost_bps=self.cfg.transaction_cost_bps)

    def _failure(self, stage, name, exc):
        self.failures.append({"stage": stage, "name": name,
                              "error": f"{type(exc).__name__}: {exc}"})
        self.log(f"[skip] {stage} / {name}: {exc}")

    def model_tournament(self) -> tuple[pd.DataFrame, dict[str, hc.TrainResult]]:
        engine = hc.ExpectedReturnEngine(self.features)
        rows, trained = [], {}
        for name in self.cfg.models:
            self.log(f"[models] validating {name}")
            try:
                result = engine.train_and_forecast(
                    self.dataset, name, n_folds=self.cfg.cv_folds, log=lambda *_: None)
                trained[name] = result
                rows.append({"model": name, **result.cv_metrics})
            except Exception as exc:
                self._failure("model", name, exc)
        df = pd.DataFrame(rows)
        if not df.empty:
            df["score"] = _rank_score(df, ["ic", "dir_acc", "hit_ratio", "top_decile"],
                                      ["rmse", "mae"])
            df = df.sort_values(["score", "ic"], ascending=False).reset_index(drop=True)
            df.insert(0, "rank", np.arange(1, len(df) + 1))
        return df, trained

    def per_stock_tournament(self) -> pd.DataFrame:
        rows = []
        for ticker in self.md.tickers:
            part = self.dataset[self.dataset.stock == ticker]
            engine = hc.ExpectedReturnEngine(self.features)
            for name in self.cfg.models:
                try:
                    cv = engine.cross_validate(part, name, n_folds=self.cfg.cv_folds,
                                               log=lambda *_: None)
                    rows.append({"stock": ticker, "model": name, **cv})
                except Exception as exc:
                    self._failure("per-stock model", f"{ticker}/{name}", exc)
        df = pd.DataFrame(rows)
        if not df.empty:
            df["score"] = _rank_score(df, ["ic", "dir_acc", "hit_ratio"], ["rmse", "mae"])
            df["rank_for_stock"] = df.groupby("stock")["score"].rank(ascending=False,
                                                                       method="first").astype(int)
            df = df.sort_values(["stock", "rank_for_stock"]).reset_index(drop=True)
        return df

    def covariance_tournament(self) -> pd.DataFrame:
        returns = self.md.returns_matrix(self.md.tickers).dropna()
        benchmark = self.md.benchmark_returns()
        n = len(returns)
        starts = list(range(self.cfg.min_train_days, n - self.cfg.rebalance_days,
                            self.cfg.rebalance_days))
        rows = []
        for method in self.cfg.covariance_methods:
            losses, var_errors = [], []
            try:
                for i in starts:
                    hist = returns.iloc[max(0, i-self.cfg.covariance_lookback):i]
                    future = returns.iloc[i:i+self.cfg.rebalance_days]
                    sigma = hc.CovarianceEngine(hist, benchmark.reindex(hist.index)).estimate(method)
                    realised = future.cov().reindex(index=sigma.index, columns=sigma.columns).fillna(0.0)
                    losses.append(float(np.linalg.norm(sigma.values-realised.values, ord="fro")))
                    var_errors.append(float(np.mean(np.abs(np.diag(sigma)-np.diag(realised)))))
                rows.append({"method": method, "frobenius_error": np.mean(losses),
                             "variance_mae": np.mean(var_errors), "windows": len(losses)})
            except Exception as exc:
                self._failure("covariance", method, exc)
        df = pd.DataFrame(rows)
        if not df.empty:
            df["score"] = _rank_score(df, [], ["frobenius_error", "variance_mae"])
            df = df.sort_values("score", ascending=False).reset_index(drop=True)
            df.insert(0, "rank", np.arange(1, len(df)+1))
        return df

    def _return_components(self, sliced_md, hist_data, forecasts, model_name):
        tickers = sliced_md.tickers
        ml = forecasts[model_name].forecasts
        capm = hc.capm_expected_returns(sliced_md, tickers)
        factor = hc.factor_expected_returns(hist_data, self.features, tickers)
        bl = hc.black_litterman_returns(sliced_md, tickers, ml)
        ensemble = hc.ensemble_returns(
            {"ML": ml, "Factor": factor, "BL": bl},
            {"ML": .4, "Factor": .3, "BL": .3})
        return {"ML": ml, "CAPM": capm, "Factor Model": factor,
                "Black-Litterman": bl, "Ensemble": ensemble}

    def walk_forward_tournament(self) -> pd.DataFrame:
        close = self.md.close_matrix().dropna()
        benchmark = self.md.prices[hc.BENCHMARK]["close"].reindex(close.index).ffill()
        dates = close.index
        cutoffs = list(range(self.cfg.min_train_days, len(dates)-self.cfg.rebalance_days,
                             self.cfg.rebalance_days))
        if not cutoffs:
            raise RuntimeError("Not enough observations for an outer walk-forward test")

        keys = [(m, s, c, o) for m in self.cfg.models for s in self.cfg.return_sources
                for c in self.cfg.covariance_methods for o in self.cfg.objectives]
        state = {k: {"returns": [], "benchmark": [], "turnover": [], "prev": {},
                     "converged": [], "hhi": []} for k in keys}
        engine = hc.ExpectedReturnEngine(self.features)

        for wi, i in enumerate(cutoffs, 1):
            asof, end = dates[i], dates[min(i+self.cfg.rebalance_days, len(dates)-1)]
            # Purge the label horizon: no training row may use a target reaching past as-of.
            purge_i = max(0, i-self.cfg.target_horizon)
            purge_date = dates[purge_i]
            hist_data = self.dataset[self.dataset.date <= purge_date]
            sliced_md = _slice_market(self.md, asof)
            self.log(f"[walk-forward] {wi}/{len(cutoffs)} as-of {asof.date()}")

            forecasts = {}
            for model in self.cfg.models:
                try:
                    forecasts[model] = engine.train_and_forecast(
                        hist_data, model, n_folds=2, log=lambda *_: None)
                except Exception as exc:
                    self._failure("walk-forward model", f"{asof}/{model}", exc)

            hist_returns = close.loc[:asof].pct_change().dropna().tail(self.cfg.covariance_lookback)
            covariances = {}
            for method in self.cfg.covariance_methods:
                try:
                    covariances[method] = hc.CovarianceEngine(
                        hist_returns, sliced_md.benchmark_returns()).estimate(method)
                except Exception as exc:
                    self._failure("walk-forward covariance", f"{asof}/{method}", exc)

            components = {m: self._return_components(sliced_md, hist_data, forecasts, m)
                          for m in forecasts}
            period_prices = close.loc[asof:end]
            daily_assets = period_prices.pct_change().dropna()
            daily_bench = benchmark.loc[period_prices.index].pct_change().dropna()

            for key in keys:
                model, source, cov_method, objective = key
                if model not in components or cov_method not in covariances:
                    continue
                rec = state[key]
                try:
                    mu = components[model][source]
                    cov = covariances[cov_method]
                    scen = None
                    if objective == "CVaR":
                        rng = np.random.default_rng(self.cfg.seed + wi)
                        scen = rng.multivariate_normal(
                            [mu.get(t, 0.0) for t in cov.columns],
                            cov.values*hc.MONTH_DAYS, size=2000)
                    opt = hc.Optimizer(mu, cov, self.constraints, prev_weights=rec["prev"],
                                      scenario_returns=scen)
                    port = opt.optimize(objective,
                                        error_bands=forecasts[model].error_bands)
                    w = port.weights
                    turnover = sum(abs(w.get(t, 0)-rec["prev"].get(t, 0))
                                   for t in self.md.tickers)
                    r = daily_assets.mul(pd.Series(w), axis=1).sum(axis=1)
                    r += w.get("Cash", 0.0)*hc.RISK_FREE_DAILY
                    if len(r):
                        r.iloc[0] -= self.cfg.transaction_cost_bps/1e4*turnover
                    rec["returns"].extend(r.tolist())
                    rec["benchmark"].extend(daily_bench.reindex(r.index).fillna(0.0).tolist())
                    rec["turnover"].append(turnover)
                    rec["prev"] = w
                    rec["converged"].append(bool(port.diagnostics.get("converged")))
                    rec["hhi"].append(port.concentration)
                except Exception as exc:
                    self._failure("combination", "/".join(key), exc)

        rows = []
        for key, rec in state.items():
            if not rec["returns"]:
                continue
            met = _metrics(np.asarray(rec["returns"]), rec["turnover"],
                           np.asarray(rec["benchmark"]))
            rows.append({"model": key[0], "return_source": key[1],
                         "risk_model": key[2], "objective": key[3], **met,
                         "mean_concentration": float(np.mean(rec["hhi"])),
                         "convergence_rate": float(np.mean(rec["converged"])),
                         "windows": len(rec["turnover"])})
        df = pd.DataFrame(rows)
        if not df.empty:
            df["score"] = _rank_score(
                df, ["sharpe", "sortino", "cagr", "excess_vs_benchmark",
                     "win_rate", "convergence_rate"],
                ["volatility", "cvar95_daily", "avg_turnover", "mean_concentration"])
            df = df.sort_values(["score", "sharpe"], ascending=False).reset_index(drop=True)
            df.insert(0, "rank", np.arange(1, len(df)+1))
        return df

    def run(self) -> ResearchResult:
        model_board, trained = self.model_tournament()
        per_stock = self.per_stock_tournament()
        covariance_board = self.covariance_tournament()
        combinations = self.walk_forward_tournament()
        if combinations.empty:
            raise RuntimeError("No portfolio combination completed successfully")
        winner = combinations.iloc[0].to_dict()
        self.log(f"[winner] {winner['model']} / {winner['return_source']} / "
                 f"{winner['risk_model']} / {winner['objective']}")
        final = hc.Pipeline(self.md).run(
            model_name=winner["model"], cov_method=winner["risk_model"],
            objective=winner["objective"], return_source=winner["return_source"],
            constraints=self.constraints, mc_paths=self.cfg.monte_carlo_paths,
            mc_horizon=self.cfg.monte_carlo_horizon, log=self.log)
        report = self._report(model_board, covariance_board, combinations,
                              per_stock, winner, final)
        return ResearchResult(
            generated_at=datetime.utcnow().isoformat()+"Z", config=asdict(self.cfg),
            model_leaderboard=model_board, covariance_leaderboard=covariance_board,
            combination_leaderboard=combinations, per_stock_leaderboard=per_stock,
            failures=pd.DataFrame(self.failures), winner=winner,
            final_pipeline=final, final_report=report)

    def _report(self, models, covs, combos, per_stock, winner, final):
        lines = ["HELM FINSERV — AUTOMATED RESEARCH REPORT", "="*72,
                 f"Generated: {datetime.utcnow():%Y-%m-%d %H:%M} UTC",
                 "Method: inner expanding-window validation + purged outer walk-forward",
                 "Winner is selected only from outer out-of-sample results.", "",
                 "WINNING CONFIGURATION"]
        for k in ["model", "return_source", "risk_model", "objective", "score",
                  "sharpe", "sortino", "cagr", "max_drawdown", "avg_turnover"]:
            lines.append(f"  {k:<22}: {winner.get(k)}")
        lines += ["", "TOP MODELS", models.head(10).to_string(index=False), "",
                  "TOP COVARIANCE METHODS", covs.head(10).to_string(index=False), "",
                  "TOP PORTFOLIO COMBINATIONS", combos.head(20).to_string(index=False), "",
                  "BEST MODEL PER STOCK",
                  per_stock[per_stock.get("rank_for_stock", 0) == 1].to_string(index=False)
                  if not per_stock.empty else "No results", "",
                  hc.build_report(final, winner["model"], winner["return_source"],
                                  winner["objective"]), "",
                  "IMPORTANT: Research output, not investment advice. Results may be unstable;",
                  "use live-quality data and a final untouched holdout before deployment."]
        return "\n".join(lines)


def save_result(result: ResearchResult, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    result.model_leaderboard.to_csv(output_dir/"model_leaderboard.csv", index=False)
    result.covariance_leaderboard.to_csv(output_dir/"covariance_leaderboard.csv", index=False)
    result.combination_leaderboard.to_csv(output_dir/"combination_leaderboard.csv", index=False)
    result.per_stock_leaderboard.to_csv(output_dir/"per_stock_leaderboard.csv", index=False)
    result.failures.to_csv(output_dir/"failures.csv", index=False)
    (output_dir/"report.txt").write_text(result.final_report, encoding="utf-8")
    (output_dir/"winner.json").write_text(json.dumps(result.winner, indent=2, default=str),
                                           encoding="utf-8")


def research_main():
    parser = argparse.ArgumentParser(description="HELM automated portfolio research")
    parser.add_argument("--mode", choices=["fast", "full"], default="full")
    parser.add_argument("--history", type=int, default=750)
    parser.add_argument("--output", default="research_output")
    args = parser.parse_args()
    md = hc.DataLoader(seed=42).load(hc.DEFAULT_UNIVERSE, n_days=args.history)
    result = AutoResearchEngine(md, ResearchConfig(mode=args.mode)).run()
    save_result(result, Path(args.output))
    print(result.final_report)



# Unified single-file entry point
hc = sys.modules[__name__]

def unified_main():
    parser = argparse.ArgumentParser(description=APP_NAME)
    parser.add_argument("--selftest", "--test", action="store_true")
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--research", action="store_true", help="run automated model/portfolio tournament")
    parser.add_argument("--mode", choices=["fast", "full"], default="full")
    parser.add_argument("--history", type=int, default=750)
    parser.add_argument("--output", default="research_output")
    args = parser.parse_args()
    if args.selftest:
        raise SystemExit(0 if run_selftest() else 1)
    if args.demo:
        run_demo(); return
    if args.research:
        md = DataLoader(seed=42).load(DEFAULT_UNIVERSE, n_days=args.history)
        result = AutoResearchEngine(md, ResearchConfig(mode=args.mode)).run()
        save_result(result, Path(args.output))
        print(result.final_report)
        return
    launch_gui()

if __name__ == "__main__":
    unified_main()
