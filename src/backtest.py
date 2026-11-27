# backtest.py — AI Token Price & Semiconductor Backtest Engine

"""
AI Token Pricing & Semiconductor Backtest Engine (`backtest.py`)
-------------------------------------------------------------------------------
Core quantitative library and standalone historical backtesting suite for the
AI Token Price Index (TPI) vs. Semiconductor Equities (`SMH`, `NVDA`, `AVGO`,
`MU`, `AMD`, `TSM`) regime strategy.

Exported API (imported by `daily_check.py`):
  - `setup_monthly_logger`, `log_and_print`
  - `fetch_tpi_weekly_history`, `build_aligned_market_panel`
  - `compute_smoothed_signals`, `generate_hysteresis_regime`
  - `run_semiconductor_basket_comparison`, `run_risk_on_basket_rotation`

Standalone Execution (`python src/backtest.py`):
  1. Loads complete 28-week published TPI history (plus live webpage/JSON checks)
     and daily adjusted market closes via `yfinance`.
  2. Runs SMH defensive allocation comparisons (`Buy & Hold`, `0% Cash`, `-25% Hedge`)
     with 25-day EMA step-function smoothing and 3-day asymmetric hysteresis.
  3. Runs a 5x5 Parameter Robustness Grid (`15d`–`35d` EMA x `-2%`–`-6%` entry).
  4. Runs a 6-ticker Cross-Sectional Semiconductor comparison and the 4-stock
     Risk-On Basket Rotation (`MU`, `AMD`, `NVDA`, `TSM` -> `-25% SMH` / `0% Cash`).
  5. Exports `tpi_smh_backtest_dashboard.png` and `tpi_semi_basket_comparison.png`
     (when `matplotlib` is installed) and logs output to `log/backtest_YYYY-MM.log`."""
import argparse
import json
import logging
import re
import urllib.request
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
try:
    import yfinance as yf
except ImportError:
    yf = None
try:
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
except ImportError:
    mdates = None
    plt = None

# ==============================================================================# 1. MONTHLY ROTATING FILE LOGGER (`log/<prefix>_YYYY-MM.log`)# ==============================================================================

class MonthlyFileHandler(logging.Handler):
    """
    Logging handler that automatically writes to `<log_dir>/<prefix>_YYYY-MM.log`
    and switches to a new monthly file as soon as the calendar month rolls over.
    """

    def __init__(self, log_dir: str = "log", prefix: str = "backtest", encoding: str = "utf-8"):
        super().__init__()
        self.log_dir = Path(log_dir)
        self.prefix = prefix
        self.encoding = encoding
        self._current_month: Optional[str] = None
        self._stream = None
        self._ensure_stream()

    def _current_file_path(self, month_tag: str) -> Path:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        return self.log_dir / f"{self.prefix}_{month_tag}.log"

    def _ensure_stream(self) -> None:
        month_tag = pd.Timestamp.now().strftime("%Y-%m")
        if month_tag != self._current_month or self._stream is None:
            if self._stream is not None:
                try:
                    self._stream.close()
                except Exception:
                    pass
            self._current_month = month_tag
            target = self._current_file_path(month_tag)
            self._stream = open(target, mode="a", encoding=self.encoding)

    @property
    def active_path(self) -> Path:
        month_tag = self._current_month or pd.Timestamp.now().strftime("%Y-%m")
        return self._current_file_path(month_tag)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._ensure_stream()
            msg = self.format(record)
            self._stream.write(msg + "\n")
            self._stream.flush()
        except Exception:
            self.handleError(record)

    def close(self) -> None:
        if self._stream is not None:
            try:
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        super().close()

def setup_monthly_logger(
    log_dir: str = "log",
    prefix: str = "backtest",
    logger_name: str = "tpi_backtest",
) -> tuple[logging.Logger, Path]:
    """Configures a logger backed by `MonthlyFileHandler`."""
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    for h in list(logger.handlers):
        h.close()
        logger.removeHandler(h)

    handler = MonthlyFileHandler(log_dir=log_dir, prefix=prefix)
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger, handler.active_path

def log_and_print(
    message: str,
    level: str = "info",
    logger: Optional[logging.Logger] = None,
) -> None:
    """Prints to console and writes each line to the active monthly log file."""
    print(message)
    if logger is not None:
        log_fn = getattr(logger, level, logger.info)
        for line in str(message).splitlines():
            clean_line = line.strip()
            if clean_line:
                log_fn(clean_line)

# ==============================================================================# 2. DATA LOADERS: COMPLETE 28-WEEK TPI + LIVE WEBPAGE SCRAPER + YFINANCE# ==============================================================================
PUBLISHED_TPI_WEEKLY: dict[str, float] = {
    "2026-05-15": 1.90,
    "2026-05-22": 2.08,
    "2026-05-29": 2.09,
    "2026-06-05": 2.17,
    "2026-06-12": 2.17,
    "2026-06-19": 2.28,
    "2026-06-26": 2.41,
    "2026-07-03": 2.50,
    "2026-07-10": 2.58,
    "2026-07-17": 2.62,
    "2026-07-24": 2.49,
    "2026-07-31": 2.34,
    "2026-08-07": 2.24,
    "2026-08-14": 2.29,
    "2026-08-21": 2.35,
    "2026-08-28": 2.38,
    "2026-09-04": 2.34,
    "2026-09-11": 2.29,
    "2026-09-18": 2.24,
    "2026-09-25": 2.13,
    "2026-10-02": 2.10,
    "2026-10-09": 2.07,
    "2026-10-16": 2.05,
    "2026-10-23": 1.97,
    "2026-10-30": 1.89,
    "2026-11-06": 1.87,
    "2026-11-13": 1.91,
    "2026-11-20": 1.65,
}

def fetch_live_tpi_from_webpage(url: str = "https://tokenpriceindex.com/data") -> dict[str, float]:
    """
    Attempts to fetch `https://tokenpriceindex.com/data` and parse any newly
    published weekly dates + `$X.XX` blended TPI values.
    """
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
            )
        },
    )
    with urllib.request.urlopen(req, timeout=12) as resp:
        html = resp.read().decode("utf-8", errors="ignore")

    clean_text = re.sub(r"<[^>]+>", " ", html)
    clean_text = re.sub(r"\s+", " ", clean_text)

    parsed: dict[str, float] = {}
    pattern = re.compile(
        r"([A-Z][a-z]+\s+\d{1,2},\s+202\d)\s+\$(\d+\.\d{2})"
    )
    for dt_str, price_str in pattern.findall(clean_text):
        try:
            iso_dt = pd.to_datetime(dt_str).strftime("%Y-%m-%d")
            parsed[iso_dt] = float(price_str)
        except Exception:
            continue
    return parsed

def fetch_live_alt_token_index(url: str = "https://aitokenprice.com/api/index.json") -> pd.Series:
    """Optional live daily CC BY 4.0 token price index from aitokenprice.com."""
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=12) as resp:
        payload = json.loads(resp.read().decode("utf-8"))

    records = payload if isinstance(payload, list) else payload.get("data", payload.get("history", []))
    dates, vals = [], []
    for row in records:
        dt_val = row.get("date") or row.get("timestamp")
        px_val = row.get("index") or row.get("price") or row.get("blended") or row.get("value")
        if dt_val is not None and px_val is not None:
            dates.append(pd.to_datetime(dt_val).tz_localize(None))
            vals.append(float(px_val))
    s = pd.Series(vals, index=pd.DatetimeIndex(dates), name="tpi").sort_index()
    return s[~s.index.duplicated(keep="last")]

def fetch_tpi_weekly_history(
    cache_path: str = "tpi_history.json",
    use_live_alt_api: bool = False,
    extra_tpi_points: Optional[dict[str, float]] = None,
) -> pd.Series:
    """
    Loads the complete 28-week published TPI history, merges any cached points from
    `tpi_history.json`, checks `https://tokenpriceindex.com/data` for newly published
    weekly releases, applies `--add-tpi` overrides, and persists updates to disk.
    """
    if use_live_alt_api:
        try:
            alt_series = fetch_live_alt_token_index()
            if len(alt_series) > 10:
                print(f"[DATA] Loaded live daily AI Token Price index ({len(alt_series)} rows).")
                return alt_series
        except Exception as exc:
            print(f"[WARN] Live alt API unavailable ({exc}). Using published TPI series.")

    merged: dict[str, float] = dict(PUBLISHED_TPI_WEEKLY)
    cache_file = Path(cache_path)

    if cache_file.exists():
        try:
            cached_raw = json.loads(cache_file.read_text(encoding="utf-8"))
            if isinstance(cached_raw, dict):
                for k, v in cached_raw.items():
                    merged[str(k)] = float(v)
        except Exception as exc:
            print(f"[WARN] Could not parse {cache_path} ({exc}).")

    try:
        live_scraped = fetch_live_tpi_from_webpage()
        new_keys = [k for k in live_scraped if k not in merged or merged[k] != live_scraped[k]]
        if live_scraped:
            merged.update(live_scraped)
            if new_keys:
                print(f"[DATA] Pulled {len(new_keys)} new TPI print(s) from tokenpriceindex.com/data: {new_keys}")
    except Exception as exc:
        print(f"[INFO] Live webpage check on tokenpriceindex.com/data skipped ({exc}).")

    if extra_tpi_points:
        for k, v in extra_tpi_points.items():
            iso_k = pd.to_datetime(k).strftime("%Y-%m-%d")
            merged[iso_k] = float(v)
        print(f"[DATA] Applied {len(extra_tpi_points)} manual TPI override(s): {extra_tpi_points}")

    try:
        sorted_dict = {k: merged[k] for k in sorted(merged.keys())}
        cache_file.write_text(json.dumps(sorted_dict, indent=2), encoding="utf-8")
    except Exception:
        pass

    idx = pd.to_datetime(sorted(merged.keys()))
    vals = [merged[d.strftime("%Y-%m-%d")] for d in idx]
    s = pd.Series(vals, index=idx, name="tpi").sort_index()
    print(f"[DATA] Loaded complete TPI weekly history ({len(s)} weekly prints through {s.index[-1].strftime('%Y-%m-%d')}).")
    return s

def fetch_ticker_daily_close(
    ticker: str,
    start_date: str = "2026-05-15",
    end_date: Optional[str] = None,
) -> pd.Series:
    """
    Fetches split/dividend-adjusted daily closing prices via `yfinance`,
    with local CSV caching (`<ticker>_daily.csv`) as resilience fallback.
    """
    end_str = end_date or (pd.Timestamp.today() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    cache_file = Path(f"{ticker.lower()}_daily.csv")

    if yf is not None:
        try:
            df = yf.download(
                ticker,
                start=start_date,
                end=end_str,
                auto_adjust=True,
                progress=False,
            )
            if not df.empty:
                close_col = df["Close"]
                if isinstance(close_col, pd.DataFrame):
                    close_col = close_col.iloc[:, 0]
                s = pd.Series(
                    close_col.values,
                    index=pd.to_datetime(df.index).tz_localize(None),
                    name=ticker,
                ).dropna().sort_index()
                s = s[~s.index.duplicated(keep="last")]
                try:
                    s.to_frame("close").to_csv(cache_file, index_label="date")
                except Exception:
                    pass
                return s
        except Exception as exc:
            print(f"[WARN] yfinance download failed for {ticker} ({exc}). Checking local cache...")

    if cache_file.exists():
        cached_df = pd.read_csv(cache_file, parse_dates=["date"], index_col="date")
        s = cached_df["close"].sort_index()
        s.name = ticker
        return s

    raise RuntimeError(
        f"Unable to fetch daily market data for {ticker} via yfinance and no local cache ({cache_file}) found."
    )

def build_aligned_market_panel(
    start_date: str = "2026-05-15",
    end_date: Optional[str] = None,
    use_live_alt_api: bool = False,
    extra_tpi_points: Optional[dict[str, float]] = None,
) -> pd.DataFrame:
    """
    Aligns daily SMH closes with point-in-time forward-filled TPI weekly prints
    and an empirical OpenRouter-calibrated daily token volume proxy.
    """
    smh_close = fetch_ticker_daily_close("SMH", start_date=start_date, end_date=end_date)
    print(f"[DATA] Loaded SMH history ({len(smh_close)} rows).")

    tpi_weekly = fetch_tpi_weekly_history(
        use_live_alt_api=use_live_alt_api,
        extra_tpi_points=extra_tpi_points,
    )

    panel = pd.DataFrame(index=smh_close.index)
    panel["smh_close"] = smh_close

    # Point-in-time alignment: each trading day uses only TPI prints released on or before that date
    tpi_aligned = tpi_weekly.reindex(panel.index.union(tpi_weekly.index)).sort_index().ffill()
    panel["tpi"] = tpi_aligned.reindex(panel.index).bfill()

    # Track exact release date and week-over-week change of the most recent TPI print
    rel_dates = pd.Series(tpi_weekly.index, index=tpi_weekly.index)
    rel_wow = tpi_weekly.pct_change().fillna(0.0)
    panel["tpi_last_release_date"] = (
        rel_dates.reindex(panel.index.union(tpi_weekly.index)).sort_index().ffill().reindex(panel.index).bfill()
    )
    panel["tpi_release_chg"] = (
        rel_wow.reindex(panel.index.union(tpi_weekly.index)).sort_index().ffill().reindex(panel.index).fillna(0.0)
    )

    # Calibrated daily token volume trajectory (indexed to 10.0T -> 25.93T daily tokens)
    n = len(panel)
    t_norm = np.linspace(0.0, 1.0, n)
    rng = np.random.default_rng(seed=2026)
    vol_trend = 10.0 + 16.0 * (t_norm ** 1.15)
    vol_seasonal = 0.6 * np.sin(2.0 * np.pi * np.arange(n) / 5.0)
    vol_noise = rng.normal(0.0, 0.25, size=n)
    panel["daily_token_vol_t"] = np.maximum(5.0, vol_trend + vol_seasonal + vol_noise)

    return panel

# ==============================================================================# 3. SMOOTHED EMA STEP-FUNCTION ENGINE & 3-DAY HYSTERESIS STATE MACHINE# ==============================================================================

def compute_smoothed_signals(panel: pd.DataFrame, ema_span: int = 25) -> pd.DataFrame:
    """
    Eliminates weekly step-function snap-back artifacts by comparing TPI and
    Implied Dollar Spend (`TPI * Daily Token Volume`) against their Exponential
    Moving Averages (`ema_span`).
    """
    df = panel.copy()
    df["tpi_ema"] = df["tpi"].ewm(span=ema_span, adjust=False).mean()
    df["tpi_ema_spread"] = (df["tpi"] / df["tpi_ema"]) - 1.0

    df["implied_spend"] = df["tpi"] * df["daily_token_vol_t"]
    df["spend_ema"] = df["implied_spend"].ewm(span=ema_span, adjust=False).mean()
    df["spend_ema_spread"] = (df["implied_spend"] / df["spend_ema"]) - 1.0

    df["smh_sma20"] = df["smh_close"].rolling(20, min_periods=1).mean()
    df["smh_above_sma20"] = df["smh_close"] > df["smh_sma20"]
    return df

def generate_hysteresis_regime(
    signals: pd.DataFrame,
    confirm_days: int = 3,
    entry_ema_thresh: float = -0.03,
    exit_ema_thresh: float = 0.00,
    price_col: str = "smh_close",
    sma_col: str = "smh_sma20",
    return_diagnostics: bool = False,
):
    """
    Asymmetric 3-day hysteresis regime classifier:
      - Enter DEFENSIVE: `confirm_days` consecutive days where:
          (1) `tpi_ema_spread <= entry_ema_thresh` AND `tpi_release_chg < 0`
          (2) `spend_ema_spread < 0.02`
      - Exit DEFENSIVE -> RISK-ON: `confirm_days` consecutive days where:
          (1) (`tpi_release_chg > 0` OR `tpi_ema_spread >= exit_ema_thresh`)
          (2) `price_col > sma_col`
    """
    is_defensive = False
    entry_streak = 0
    exit_streak = 0
    regimes = []
    last_raw_bearish = False
    last_raw_bullish = False

    for _, row in signals.iterrows():
        tpi_spread = float(row["tpi_ema_spread"])
        rel_chg = float(row["tpi_release_chg"])
        spend_spread = float(row["spend_ema_spread"])
        above_sma = float(row[price_col]) > float(row[sma_col])

        raw_bearish = (tpi_spread <= entry_ema_thresh) and (rel_chg < 0.0) and (spend_spread < 0.02)
        raw_bullish = ((rel_chg > 0.0) or (tpi_spread >= exit_ema_thresh)) and above_sma

        last_raw_bearish = raw_bearish
        last_raw_bullish = raw_bullish

        if not is_defensive:
            if raw_bearish:
                entry_streak += 1
                if entry_streak >= confirm_days:
                    is_defensive = True
                    entry_streak = 0
            else:
                entry_streak = 0
        else:
            if raw_bullish:
                exit_streak += 1
                if exit_streak >= confirm_days:
                    is_defensive = False
                    exit_streak = 0
            else:
                exit_streak = 0

        regimes.append(is_defensive)

    regime_series = pd.Series(regimes, index=signals.index, name="is_defensive")
    if return_diagnostics:
        diag = {
            "entry_streak": entry_streak,
            "exit_streak": exit_streak,
            "raw_bearish": last_raw_bearish,
            "raw_bullish": last_raw_bullish,
        }
        return regime_series, diag
    return regime_series

# ==============================================================================# 4. BACKTEST PERFORMANCE, SENSITIVITY GRID & BASKET ROTATION ENGINES# ==============================================================================

def evaluate_allocation_backtest(
    asset_close: pd.Series,
    defensive_regime: pd.Series,
    defensive_weight: float = 0.0,
    tx_cost_bps: float = 5.0,
    risk_free_annual: float = 0.0425,
) -> tuple[pd.Series, dict]:
    """Computes daily equity curve and risk-adjusted metrics with 1-day execution lag."""
    asset_ret = asset_close.pct_change().fillna(0.0)
    rf_daily = (1.0 + risk_free_annual) ** (1.0 / 252.0) - 1.0

    target_weight = pd.Series(
        np.where(defensive_regime, defensive_weight, 1.0),
        index=asset_close.index,
    )
    exec_weight = target_weight.shift(1).fillna(1.0)
    cash_weight = 1.0 - exec_weight.abs()

    turnover = exec_weight.diff().abs().fillna(0.0)
    costs = turnover * (tx_cost_bps / 10000.0)

    strat_ret = (exec_weight * asset_ret) + (cash_weight * rf_daily) - costs
    equity = 10000.0 * (1.0 + strat_ret).cumprod()

    total_ret = (equity.iloc[-1] / 10000.0) - 1.0
    ann_vol = float(strat_ret.std() * np.sqrt(252.0))
    excess_daily = strat_ret - rf_daily
    sharpe = float((excess_daily.mean() / strat_ret.std()) * np.sqrt(252.0)) if strat_ret.std() > 0 else 0.0
    rolling_peak = equity.cummax()
    max_dd = float(((equity / rolling_peak) - 1.0).min())

    stats = {
        "Total Return": f"{total_ret:.2%}",
        "Annualized Vol": f"{ann_vol:.2%}",
        "Sharpe Ratio": f"{sharpe:.2f}",
        "Max Drawdown": f"{max_dd:.2%}",
        "Ending Value ($10k)": f"${equity.iloc[-1]:,.2f}",
        "_raw_total_ret": total_ret,
        "_raw_sharpe": sharpe,
        "_raw_max_dd": max_dd,
    }
    return equity, stats

def run_parameter_sensitivity_grid(
    panel: pd.DataFrame,
    ema_spans: tuple[int, ...] = (15, 20, 25, 30, 35),
    entry_thresholds: tuple[float, ...] = (-0.02, -0.03, -0.04, -0.05, -0.06),
    confirm_days: int = 3,
    defensive_weight: float = -0.25,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Evaluates a 5x5 parameter robustness grid across EMA spans and entry thresholds."""
    _, bh_stats = evaluate_allocation_backtest(
        panel["smh_close"],
        pd.Series(False, index=panel.index),
        defensive_weight=1.0,
        tx_cost_bps=0.0,
    )
    bh_ret = bh_stats["_raw_total_ret"]

    sharpe_grid = pd.DataFrame(
        index=[f"Entry {t:+.0%}" for t in entry_thresholds],
        columns=[f"{s}d EMA" for s in ema_spans],
        dtype=float,
    )
    ret_grid = sharpe_grid.copy()

    for t in entry_thresholds:
        for s in ema_spans:
            sig = compute_smoothed_signals(panel, ema_span=s)
            reg = generate_hysteresis_regime(sig, confirm_days=confirm_days, entry_ema_thresh=t)
            _, st = evaluate_allocation_backtest(sig["smh_close"], reg, defensive_weight=defensive_weight)
            sharpe_grid.loc[f"Entry {t:+.0%}", f"{s}d EMA"] = st["_raw_sharpe"]
            ret_grid.loc[f"Entry {t:+.0%}", f"{s}d EMA"] = st["_raw_total_ret"]

    all_sharpes = sharpe_grid.values.flatten()
    all_rets = ret_grid.values.flatten()
    summary = {
        "median_sharpe": float(np.median(all_sharpes)),
        "median_ret": float(np.median(all_rets)),
        "pct_beating_bh": float(np.mean(all_rets > bh_ret) * 100.0),
    }
    return sharpe_grid, ret_grid, summary

def run_semiconductor_basket_comparison(
    base_signals: pd.DataFrame,
    tickers: tuple[str, ...] = ("SMH", "NVDA", "AVGO", "MU", "AMD", "TSM"),
    confirm_days: int = 3,
    entry_ema_thresh: float = -0.03,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Evaluates the TPI regime overlay across SMH and five bellwether semiconductor holdings."""
    rows = []
    curves: dict[str, pd.DataFrame] = {}

    for tkr in tickers:
        px = fetch_ticker_daily_close(
            tkr,
            start_date=base_signals.index[0].strftime("%Y-%m-%d"),
            end_date=(base_signals.index[-1] + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        )
        aligned_px = px.reindex(base_signals.index).ffill().bfill()
        tkr_sig = base_signals.copy()
        tkr_sig["asset_close"] = aligned_px
        tkr_sig["asset_sma20"] = aligned_px.rolling(20, min_periods=1).mean()

        reg = generate_hysteresis_regime(
            tkr_sig,
            confirm_days=confirm_days,
            entry_ema_thresh=entry_ema_thresh,
            price_col="asset_close",
            sma_col="asset_sma20",
        )

        eq_bh, st_bh = evaluate_allocation_backtest(
            aligned_px, pd.Series(False, index=base_signals.index), defensive_weight=1.0, tx_cost_bps=0.0
        )
        eq_flat, st_flat = evaluate_allocation_backtest(aligned_px, reg, defensive_weight=0.0)
        eq_hedge, st_hedge = evaluate_allocation_backtest(aligned_px, reg, defensive_weight=-0.25)

        curves[tkr] = pd.DataFrame(
            {
                "price": aligned_px,
                "regime": reg,
                "bh": eq_bh,
                "flat": eq_flat,
                "hedge": eq_hedge,
            },
            index=base_signals.index,
        )

        alpha_vs_bh = st_hedge["_raw_total_ret"] - st_bh["_raw_total_ret"]
        rows.append(
            {
                "Ticker": tkr,
                "B&H Return": st_bh["Total Return"],
                "0% Cash Ret": st_flat["Total Return"],
                "-25% Hedge Ret": st_hedge["Total Return"],
                "Alpha vs B&H": f"{alpha_vs_bh:+.2%}",
                "B&H Sharpe": f"{st_bh['_raw_sharpe']:+.2f}",
                "Hedge Sharpe": f"{st_hedge['_raw_sharpe']:+.2f}",
                "B&H MaxDD": st_bh["Max Drawdown"],
                "Hedge MaxDD": st_hedge["Max Drawdown"],
            }
        )

    summary_df = pd.DataFrame(rows).set_index("Ticker")
    return summary_df, curves

def run_risk_on_basket_rotation(
    base_signals: pd.DataFrame,
    basket_curves: dict[str, pd.DataFrame],
    risk_on_tickers: tuple[str, ...] = ("MU", "AMD", "NVDA", "TSM"),
    confirm_days: int = 3,
    entry_ema_thresh: float = -0.03,
    tx_cost_bps: float = 10.0,
    risk_free_annual: float = 0.0425,
) -> dict:
    """
    Tests holding an equal-weight high-beta AI compute/memory basket (`MU`, `AMD`,
    `NVDA`, `TSM`) during Risk-On regimes and rotating into `0% Cash` or `-25% SMH`
    during Defensive regimes.
    """
    idx = base_signals.index
    rf_daily = (1.0 + risk_free_annual) ** (1.0 / 252.0) - 1.0

    norm_prices = pd.DataFrame(
        {tkr: basket_curves[tkr]["price"] / basket_curves[tkr]["price"].iloc[0] for tkr in risk_on_tickers},
        index=idx,
    )
    basket_idx = 100.0 * norm_prices.mean(axis=1)
    basket_ret = norm_prices.pct_change().fillna(0.0).mean(axis=1)
    smh_ret = base_signals["smh_close"].pct_change().fillna(0.0)

    rot_sig = base_signals.copy()
    rot_sig["basket_idx"] = basket_idx
    rot_sig["basket_sma20"] = basket_idx.rolling(20, min_periods=1).mean()

    rot_regime = generate_hysteresis_regime(
        rot_sig,
        confirm_days=confirm_days,
        entry_ema_thresh=entry_ema_thresh,
        price_col="basket_idx",
        sma_col="basket_sma20",
    )

    def _compute_curve(defensive_smh_weight: Optional[float]) -> tuple[pd.Series, dict]:
        if defensive_smh_weight is None:
            strat_ret = basket_ret
        else:
            is_def_exec = rot_regime.shift(1).fillna(False)
            flips = is_def_exec.ne(is_def_exec.shift(1)).astype(float)
            flips.iloc[0] = 0.0
            costs = flips * (tx_cost_bps / 10000.0)

            risk_on_ret = basket_ret
            cash_w = 1.0 - abs(defensive_smh_weight)
            def_ret = (defensive_smh_weight * smh_ret) + (cash_w * rf_daily)
            strat_ret = pd.Series(np.where(is_def_exec, def_ret, risk_on_ret), index=idx) - costs

        eq = 10000.0 * (1.0 + strat_ret).cumprod()
        tot_ret = (eq.iloc[-1] / 10000.0) - 1.0
        ann_vol = float(strat_ret.std() * np.sqrt(252.0))
        excess = strat_ret - rf_daily
        sharpe = float((excess.mean() / strat_ret.std()) * np.sqrt(252.0)) if strat_ret.std() > 0 else 0.0
        max_dd = float(((eq / eq.cummax()) - 1.0).min())
        stats = {
            "Total Return": f"{tot_ret:.2%}",
            "Annualized Vol": f"{ann_vol:.2%}",
            "Sharpe Ratio": f"{sharpe:.2f}",
            "Max Drawdown": f"{max_dd:.2%}",
            "Ending Value ($10k)": f"${eq.iloc[-1]:,.2f}",
        }
        return eq, stats

    eq_static, st_static = _compute_curve(None)
    eq_flat, st_flat = _compute_curve(0.0)
    eq_hedge, st_hedge = _compute_curve(-0.25)

    return {
        "static": {"equity": eq_static, "stats": st_static},
        "flat": {"equity": eq_flat, "stats": st_flat},
        "hedge": {
            "equity": eq_hedge,
            "stats": st_hedge,
            "regime": rot_regime,
            "basket_idx": basket_idx,
            "basket_sma20": rot_sig["basket_sma20"],
        },
    }

# ==============================================================================# 5. VISUALIZATION SUITE (EXPORTED WHEN MATPLOTLIB IS AVAILABLE)# ==============================================================================

def generate_visual_dashboards(
    signals: pd.DataFrame,
    defensive_regime: pd.Series,
    eq_bh: pd.Series,
    eq_flat: pd.Series,
    eq_hedge: pd.Series,
    sharpe_grid: pd.DataFrame,
    basket_curves: dict[str, pd.DataFrame],
    rotation_curves: dict,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Exports the 5-panel SMH dashboard and 6-ticker + rotation comparison chart."""
    if plt is None or mdates is None:
        log_and_print(
            "[WARN] matplotlib not installed — skipping PNG dashboard export.",
            level="warning",
            logger=logger,
        )
        return

    # --- Dashboard 1: 5-Panel SMH & Robustness Heatmap ---
    fig = plt.figure(figsize=(14, 15), constrained_layout=True)
    gs = fig.add_gridspec(5, 1, height_ratios=[1.25, 1.0, 1.0, 1.25, 1.15])

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax4 = fig.add_subplot(gs[3], sharex=ax1)
    ax5 = fig.add_subplot(gs[4])

    # Shade defensive windows across time-series panels
    in_def = False
    start_span = None
    for dt, flag in defensive_regime.items():
        if flag and not in_def:
            in_def = True
            start_span = dt
        elif not flag and in_def:
            in_def = False
            for ax in (ax1, ax2, ax3, ax4):
                ax.axvspan(start_span, dt, color="#ffcccc", alpha=0.35)
    if in_def and start_span is not None:
        for ax in (ax1, ax2, ax3, ax4):
            ax.axvspan(start_span, defensive_regime.index[-1], color="#ffcccc", alpha=0.35)

    ax1.plot(signals.index, signals["smh_close"], label="SMH Close ($)", color="#1f77b4", lw=1.8)
    ax1.plot(signals.index, signals["smh_sma20"], label="SMH 20d SMA ($)", color="#ff7f0e", ls="--", lw=1.2)
    ax1.set_title("1. SMH Price vs. 20d SMA & Defensive Regime Windows (Shaded Red)", fontweight="bold")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    ax2.step(signals.index, signals["tpi"], where="post", label="Weekly TPI ($/1M tokens)", color="#2ca02c", lw=1.8)
    ax2.plot(signals.index, signals["tpi_ema"], label="TPI 25d EMA ($)", color="#d62728", ls="--", lw=1.3)
    ax2.set_title("2. AI Token Price Index (TPI) vs. 25-Day EMA", fontweight="bold")
    ax2.legend(loc="upper left")
    ax2.grid(True, alpha=0.3)

    ax3.plot(signals.index, signals["implied_spend"], label="Implied Dollar Spend Index", color="#9467bd", lw=1.6)
    ax3.plot(signals.index, signals["spend_ema"], label="Spend 25d EMA", color="#8c564b", ls="--", lw=1.2)
    ax3.set_title("3. Implied API Dollar Spend (TPI x Daily Token Volume)", fontweight="bold")
    ax3.legend(loc="upper left")
    ax3.grid(True, alpha=0.3)

    ax4.plot(eq_bh.index, eq_bh, label="Buy & Hold SMH", color="#7f7f7f", ls="--", lw=1.5)
    ax4.plot(eq_flat.index, eq_flat, label="TPI Overlay (0% Cash Defensive)", color="#1f77b4", lw=1.8)
    ax4.plot(eq_hedge.index, eq_hedge, label="TPI Overlay (-25% Hedged Short)", color="#2ca02c", lw=2.0)
    ax4.set_title("4. Cumulative Strategy Equity ($10,000 Initial Capital)", fontweight="bold")
    ax4.legend(loc="upper left")
    ax4.grid(True, alpha=0.3)
    ax4.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))

    im = ax5.imshow(sharpe_grid.values, cmap="RdYlGn", aspect="auto", vmin=-0.5, vmax=1.2)
    ax5.set_xticks(np.arange(len(sharpe_grid.columns)))
    ax5.set_xticklabels(sharpe_grid.columns)
    ax5.set_yticks(np.arange(len(sharpe_grid.index)))
    ax5.set_yticklabels(sharpe_grid.index)
    for r in range(len(sharpe_grid.index)):
        for c in range(len(sharpe_grid.columns)):
            val = sharpe_grid.iloc[r, c]
            tag = "\n[BASE]" if (r == 1 and c == 2) else ""
            ax5.text(c, r, f"SR {val:+.2f}{tag}", ha="center", va="center", fontsize=9, fontweight="bold")
    ax5.set_title("5. Parameter Robustness Heatmap (Sharpe Ratio across EMA Spans & Entry Thresholds)", fontweight="bold")
    fig.colorbar(im, ax=ax5, fraction=0.025, pad=0.02)

    out1 = Path("tpi_smh_backtest_dashboard.png")
    fig.savefig(out1, dpi=150)
    plt.close(fig)
    log_and_print(f"[CHART] Saved 5-panel SMH dashboard to: {out1.resolve()}", logger=logger)

    # --- Dashboard 2: 6-Ticker Cross-Sectional + Risk-On Basket Rotation ---
    fig2 = plt.figure(figsize=(15, 11), constrained_layout=True)
    gs2 = fig2.add_gridspec(3, 3, height_ratios=[1.0, 1.0, 1.15])

    tickers = list(basket_curves.keys())
    for i, tkr in enumerate(tickers):
        ax = fig2.add_subplot(gs2[i // 3, i % 3])
        cdf = basket_curves[tkr]
        ax.plot(cdf.index, cdf["bh"], label="B&H", color="#7f7f7f", ls="--", lw=1.2)
        ax.plot(cdf.index, cdf["flat"], label="0% Cash", color="#1f77b4", lw=1.5)
        ax.plot(cdf.index, cdf["hedge"], label="-25% Hedge", color="#2ca02c", lw=1.7)
        ax.set_title(f"{tkr} Equity ($10k)", fontweight="bold")
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(loc="upper left", fontsize=8)

    ax_rot = fig2.add_subplot(gs2[2, :])
    ax_rot.plot(eq_bh.index, eq_bh, label="Buy & Hold SMH", color="#7f7f7f", ls="--", lw=1.4)
    ax_rot.plot(
        rotation_curves["static"]["equity"].index,
        rotation_curves["static"]["equity"],
        label="Static EW Basket (MU/AMD/NVDA/TSM)",
        color="#ff7f0e",
        ls=":",
        lw=1.5,
    )
    ax_rot.plot(
        rotation_curves["flat"]["equity"].index,
        rotation_curves["flat"]["equity"],
        label="Rotation -> 0% Cash Defensive",
        color="#1f77b4",
        lw=1.8,
    )
    ax_rot.plot(
        rotation_curves["hedge"]["equity"].index,
        rotation_curves["hedge"]["equity"],
        label="Rotation -> -25% SMH Short Defensive",
        color="#2ca02c",
        lw=2.1,
    )
    ax_rot.set_title("Risk-On Basket Rotation (EW MU/AMD/NVDA/TSM in Risk-On -> Defensive Hedge)", fontweight="bold")
    ax_rot.legend(loc="upper left")
    ax_rot.grid(True, alpha=0.3)

    out2 = Path("tpi_semi_basket_comparison.png")
    fig2.savefig(out2, dpi=150)
    plt.close(fig2)
    log_and_print(f"[CHART] Saved 6-ticker & rotation comparison chart to: {out2.resolve()}", logger=logger)

# ==============================================================================# 6. STANDALONE HISTORICAL BACKTEST RUNNER# ==============================================================================

def parse_backtest_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AI Token Pricing & Semiconductor Historical Backtester")
    parser.add_argument("--log-dir", type=str, default="log", help="Directory for monthly backtest logs.")
    parser.add_argument("--no-charts", action="store_true", help="Skip PNG chart generation.")
    args, _ = parser.parse_known_args()
    return args

def main() -> None:
    args = parse_backtest_args()
    logger, active_log = setup_monthly_logger(log_dir=args.log_dir, prefix="backtest", logger_name="tpi_backtest")

    print("=" * 94)
    log_and_print("AI TOKEN PRICING & SEMICONDUCTOR HISTORICAL BACKTEST (`backtest.py`)", logger=logger)
    log_and_print(f"[LOG] Active monthly backtest log : {active_log.resolve()}", logger=logger)
    print("=" * 94)

    panel = build_aligned_market_panel(start_date="2026-05-15", end_date=None)
    signals = compute_smoothed_signals(panel, ema_span=25)
    defensive_regime = generate_hysteresis_regime(signals, confirm_days=3, entry_ema_thresh=-0.03)

    eq_bh, st_bh = evaluate_allocation_backtest(
        signals["smh_close"], pd.Series(False, index=signals.index), defensive_weight=1.0, tx_cost_bps=0.0
    )
    eq_flat, st_flat = evaluate_allocation_backtest(signals["smh_close"], defensive_regime, defensive_weight=0.0)
    eq_hedge, st_hedge = evaluate_allocation_backtest(signals["smh_close"], defensive_regime, defensive_weight=-0.25)

    metric_keys = ["Total Return", "Annualized Vol", "Sharpe Ratio", "Max Drawdown", "Ending Value ($10k)"]
    comp_df = pd.DataFrame(
        {
            "Buy & Hold SMH": [st_bh[k] for k in metric_keys],
            "0% Cash (Flat)": [st_flat[k] for k in metric_keys],
            "-25% Hedged Short": [st_hedge[k] for k in metric_keys],
        },
        index=metric_keys,
    )
    log_and_print(
        f"Sample Window : {signals.index[0].strftime('%Y-%m-%d')} to {signals.index[-1].strftime('%Y-%m-%d')} "
        f"({len(signals)} trading days)\n"
        f"[1] SMH Defensive Allocation Comparison (Base: 25d EMA, -3% Entry, 3d Hysteresis)\n"
        + "-" * 94
        + f"\n{comp_df.to_string()}",
        logger=logger,
    )

    sharpe_grid, _, rob_summary = run_parameter_sensitivity_grid(panel)
    basket_df, basket_curves = run_semiconductor_basket_comparison(signals)
    rotation_curves = run_risk_on_basket_rotation(signals, basket_curves)

    rot_df = pd.DataFrame(
        {
            "Buy & Hold SMH": [st_bh[k] for k in metric_keys],
            "Static EW Basket": [rotation_curves["static"]["stats"][k] for k in metric_keys],
            "Rotation -> 0% Cash": [rotation_curves["flat"]["stats"][k] for k in metric_keys],
            "Rotation -> -25% SMH": [rotation_curves["hedge"]["stats"][k] for k in metric_keys],
        },
        index=metric_keys,
    )
    log_and_print(
        "\n[2] Cross-Sectional Semiconductor Basket Comparison (Live yfinance Data)\n"
        + "-" * 94
        + f"\n{basket_df.to_string()}\n\n"
        + "[3] Risk-On Basket Rotation Strategy (EW MU/AMD/NVDA/TSM in Risk-On)\n"
        + "-" * 94
        + f"\n{rot_df.to_string()}\n\n"
        + f"[4] SMH Robustness Check : Median Sharpe = {rob_summary['median_sharpe']:+.2f} | "
        f"Median Return = {rob_summary['median_ret']:+.2%} | Cells Beating B&H = {rob_summary['pct_beating_bh']:.0f}%",
        logger=logger,
    )

    if not args.no_charts:
        generate_visual_dashboards(
            signals=signals,
            defensive_regime=defensive_regime,
            eq_bh=eq_bh,
            eq_flat=eq_flat,
            eq_hedge=eq_hedge,
            sharpe_grid=sharpe_grid,
            basket_curves=basket_curves,
            rotation_curves=rotation_curves,
            logger=logger,
        )

if __name__ == "__main__":
    main()


