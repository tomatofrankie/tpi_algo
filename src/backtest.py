# daily_check.py — Daily & Weekly AI Token Signal Monitor

"""
AI Token Pricing & Semiconductor Daily/Weekly Signal Monitor (`daily_check.py`)
-------------------------------------------------------------------------------
Lightweight operational monitoring script separated from `backtest.py`.
Imports the core market/TPI data loaders, EMA step-function smoother, and 3-day
hysteresis regime engine directly from `backtest.py` (with path auto-resolution
whether invoked from the project root or `src/`).

Features:
  1. Automatic Live Data Checks: Pulls latest daily closes via `yfinance` for
     `SMH` and the Risk-On basket (`MU`, `AMD`, `NVDA`, `TSM`) and queries
     `https://tokenpriceindex.com/data` for any new weekly TPI prints.
  2. Fast Execution: Skips historical parameter sensitivity grids and PNG chart
     rendering so daily/Friday checks complete in seconds.
  3. Actionable Regime Readout: Displays current regime (`RISK-ON` vs `DEFENSIVE`),
     days in active regime, TPI vs. 25-day EMA spread, hysteresis confirmation
     streak, exact next-Friday TPI flip price levels, and target portfolio weights.
  4. Monthly Execution & CSV Logging:
     - Runtime logs   -> `log/daily_check_YYYY-MM.log` (auto-rolls every new month)
     - CSV snapshots  -> `log/signal_monitor_YYYY-MM.csv` (deduplicated by market date
                         and TPI release date)
  5. Outbound Telegram Notifications: Sends formatted regime & allocation summaries
     via Telegram Bot API (`TELEGRAM_BOT_TOKEN` & `TELEGRAM_CHAT_ID`), supporting
     `--telegram-mode always` (default), `--telegram-mode flip-only`, or `off`.

Usage:
  python daily_check.py
  python daily_check.py --add-tpi 2026-11-27:1.72
  python daily_check.py --telegram-mode flip-only"""
import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path
from typing import Optional
import pandas as pd

def _ensure_import_paths() -> None:
    """Ensures sibling `backtest.py` is importable whether run from root, `src/`, or an IDE."""
    candidates = [Path.cwd(), Path.cwd() / "src"]
    if "__file__" in globals():
        candidates.insert(0, Path(__file__).resolve().parent)
    for folder in candidates:
        folder_str = str(folder)
        if folder.exists() and folder_str not in sys.path:
            sys.path.insert(0, folder_str)


_ensure_import_paths()
from backtest import (  # noqa: E402
    build_aligned_market_panel,
    compute_smoothed_signals,
    generate_hysteresis_regime,
    log_and_print,
    run_risk_on_basket_rotation,
    run_semiconductor_basket_comparison,
    setup_monthly_logger,
)

# ==============================================================================# 1. SNAPSHOT BUILDER, TELEGRAM NOTIFIER & MONTHLY CSV SIGNAL LOGGER# ==============================================================================

def build_live_signal_snapshot(
    signals: pd.DataFrame,
    defensive_regime: pd.Series,
    diag: dict,
    rotation_curves: dict,
    ema_span: int = 25,
    confirm_days: int = 3,
    entry_ema_thresh: float = -0.03,
    exit_ema_thresh: float = 0.00,
) -> dict:
    """
    Computes a structured dictionary of current regime diagnostics, next-Friday
    TPI trigger levels, and target portfolio allocations.
    """
    latest = signals.iloc[-1]
    latest_dt = signals.index[-1]
    last_tpi_rel = pd.to_datetime(latest["tpi_last_release_date"])

    is_def_now = bool(defensive_regime.iloc[-1])
    prev_is_def = bool(defensive_regime.iloc[-2]) if len(defensive_regime) >= 2 else is_def_now
    flipped_today = is_def_now != prev_is_def
    regime_label = "DEFENSIVE" if is_def_now else "RISK-ON"

    days_in_regime = 0
    for val in reversed(defensive_regime.values):
        if bool(val) == is_def_now:
            days_in_regime += 1
        else:
            break

    transitions = defensive_regime[defensive_regime != defensive_regime.shift(1)]
    last_flip_dt = transitions.index[-1] if not transitions.empty else signals.index[0]

    alpha = 2.0 / (ema_span + 1.0)
    prev_ema = float(latest["tpi_ema"])
    curr_tpi = float(latest["tpi"])

    def solve_tpi_for_spread(target_spread: float) -> float:
        mult = 1.0 + target_spread
        denom = 1.0 - (mult * alpha)
        return (mult * (1.0 - alpha) * prev_ema) / denom

    tpi_entry_trigger = solve_tpi_for_spread(entry_ema_thresh)
    tpi_exit_trigger = solve_tpi_for_spread(exit_ema_thresh)

    rot_df = rotation_curves["hedge"]
    basket_px = float(rot_df["basket_idx"].iloc[-1])
    basket_sma = float(rot_df["basket_sma20"].iloc[-1])
    smh_px = float(latest["smh_close"])
    smh_sma = float(latest["smh_sma20"])

    return {
        "run_timestamp_utc": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ"),
        "market_close_date": latest_dt.strftime("%Y-%m-%d"),
        "last_tpi_release_date": last_tpi_rel.strftime("%Y-%m-%d"),
        "active_regime": regime_label,
        "flipped_today": flipped_today,
        "days_in_regime": days_in_regime,
        "last_transition_date": last_flip_dt.strftime("%Y-%m-%d"),
        "tpi_price": round(curr_tpi, 4),
        "tpi_release_wow_pct": round(float(latest["tpi_release_chg"]) * 100, 2),
        "tpi_ema25": round(prev_ema, 4),
        "tpi_ema_spread_pct": round(float(latest["tpi_ema_spread"]) * 100, 2),
        "implied_spend_idx": round(float(latest["implied_spend"]), 2),
        "spend_ema25": round(float(latest["spend_ema"]), 2),
        "spend_ema_spread_pct": round(float(latest["spend_ema_spread"]) * 100, 2),
        "smh_close": round(smh_px, 2),
        "smh_sma20": round(smh_sma, 2),
        "smh_above_sma20": bool(smh_px > smh_sma),
        "basket_idx": round(basket_px, 2),
        "basket_sma20": round(basket_sma, 2),
        "basket_above_sma20": bool(basket_px > basket_sma),
        "entry_streak": f"{diag['entry_streak']}/{confirm_days}",
        "exit_streak": f"{diag['exit_streak']}/{confirm_days}",
        "next_friday_tpi_entry_trigger": round(tpi_entry_trigger, 4),
        "next_friday_tpi_exit_trigger": round(tpi_exit_trigger, 4),
        "target_smh_flat_alloc": "0% SMH / 100% Cash" if is_def_now else "100% Long SMH",
        "target_smh_hedged_alloc": "-25% Short SMH / 100% Cash" if is_def_now else "100% Long SMH",
        "target_rotation_alloc": (
            "0% Basket -> -25% Short SMH (or 100% Cash)"
            if is_def_now
            else "25% MU, 25% AMD, 25% NVDA, 25% TSM"
        ),
    }

def format_telegram_message(snapshot: dict, ema_span: int = 25) -> str:
    """Builds a clean, mobile-friendly Telegram message from the daily snapshot."""
    is_def = snapshot["active_regime"] == "DEFENSIVE"
    status_badge = "DEFENSIVE (RISK-OFF)" if is_def else "RISK-ON (BULLISH)"
    header = (
        f"🚨 *REGIME FLIP DETECTED: {status_badge}*"
        if snapshot["flipped_today"]
        else f"📊 *AI Token & Semi Signal: {status_badge}*"
    )

    if is_def:
        streak_line = f"Exit Streak: `{snapshot['exit_streak']}` | Re-Enter TPI: `>${snapshot['tpi_price']:.2f}` or `>=${snapshot['next_friday_tpi_exit_trigger']:.2f}`"
    else:
        streak_line = f"Entry Streak: `{snapshot['entry_streak']}` | Defensive TPI: `<=${snapshot['next_friday_tpi_entry_trigger']:.2f}`"

    lines = [
        header,
        f"Market Close: `{snapshot['market_close_date']}` | Last TPI: `{snapshot['last_tpi_release_date']}`",
        f"Regime Duration: `{snapshot['days_in_regime']}d` (since `{snapshot['last_transition_date']}`)",
        "",
        "*Indicators:*",
        f"• TPI: `${snapshot['tpi_price']:.2f}` ({snapshot['tpi_release_wow_pct']:+.1f}% WoW) vs {ema_span}d EMA `${snapshot['tpi_ema25']:.2f}` (`{snapshot['tpi_ema_spread_pct']:+.2f}%`)",
        f"• Dollar Spend: `{snapshot['implied_spend_idx']:.2f}` vs EMA `{snapshot['spend_ema25']:.2f}` (`{snapshot['spend_ema_spread_pct']:+.2f}%`)",
        f"• SMH: `${snapshot['smh_close']:.2f}` vs 20d SMA `${snapshot['smh_sma20']:.2f}` ({'ABOVE' if snapshot['smh_above_sma20'] else 'BELOW'})",
        f"• EW Basket: `{snapshot['basket_idx']:.2f}` vs 20d SMA `{snapshot['basket_sma20']:.2f}` ({'ABOVE' if snapshot['basket_above_sma20'] else 'BELOW'})",
        f"• {streak_line}",
        "",
        "*Target Allocation (Next Open):*",
        f"• Rotation: `{snapshot['target_rotation_alloc']}`",
        f"• SMH Hedged: `{snapshot['target_smh_hedged_alloc']}`",
    ]
    return "\n".join(lines)

def send_telegram_alert(
    snapshot: dict,
    logger,
    mode: str = "always",
    ema_span: int = 25,
) -> bool:
    """
    Sends an outbound Telegram notification using standard-library `urllib` when
    `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` are set in environment variables.
    Modes:
      - `always`    : Send summary on every run (default)
      - `flip-only` : Send message only when `snapshot["flipped_today"]` is True
      - `off`       : Disable Telegram alerts
    """
    if mode == "off":
        log_and_print("[TELEGRAM] Notifications disabled (--telegram-mode off).", logger=logger)
        return False

    if mode == "flip-only" and not snapshot["flipped_today"]:
        log_and_print(
            "[TELEGRAM] No regime flip today — skipping alert (--telegram-mode flip-only).",
            logger=logger,
        )
        return False

    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    if not bot_token or not chat_id:
        log_and_print(
            "[TELEGRAM] TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set — skipping notification.",
            logger=logger,
        )
        return False

    message_text = format_telegram_message(snapshot, ema_span=ema_span)
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = json.dumps(
        {
            "chat_id": chat_id,
            "text": message_text,
            "parse_mode": "Markdown",
        }
    ).encode("utf-8")

    try:
        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            resp_body = json.loads(resp.read().decode("utf-8"))
        if resp_body.get("ok"):
            log_and_print("[TELEGRAM] Sent notification successfully.", logger=logger)
            return True
        log_and_print(
            f"[WARN] Telegram API returned non-ok response: {resp_body}",
            level="warning",
            logger=logger,
        )
        return False
    except Exception as exc:
        log_and_print(
            f"[WARN] Failed to send Telegram notification ({exc}).",
            level="warning",
            logger=logger,
        )
        return False

def log_weekly_signal_to_csv(
    snapshot: dict,
    logger,
    log_dir: str = "log",
    custom_csv_path: Optional[str] = None,
) -> Path:
    """
    Appends the live signal snapshot to a monthly CSV file inside `log/`
    (e.g. `log/signal_monitor_2026-11.csv`), automatically creating a new
    monthly file whenever a new calendar month begins.
    Deduplicates by `(market_close_date, last_tpi_release_date, tpi_price)`.
    """
    if custom_csv_path:
        csv_path = Path(custom_csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        month_tag = pd.Timestamp.now().strftime("%Y-%m")
        folder = Path(log_dir)
        folder.mkdir(parents=True, exist_ok=True)
        csv_path = folder / f"signal_monitor_{month_tag}.csv"

    new_row_df = pd.DataFrame([snapshot])

    if csv_path.exists():
        existing_df = pd.read_csv(csv_path)
        combined = pd.concat([existing_df, new_row_df], ignore_index=True)
        combined = combined.drop_duplicates(
            subset=["market_close_date", "last_tpi_release_date", "tpi_price"],
            keep="last",
        )
    else:
        combined = new_row_df

    combined.to_csv(csv_path, index=False)
    log_and_print(
        f"[LOG] Saved monthly CSV snapshot to: {csv_path.resolve()} (Rows in month file: {len(combined)})",
        logger=logger,
    )
    return csv_path

def print_live_weekly_signal_monitor(
    snapshot: dict,
    diag: dict,
    logger,
    ema_span: int = 25,
    confirm_days: int = 3,
    entry_ema_thresh: float = -0.03,
) -> None:
    """
    Prints an actionable daily/Friday monitoring dashboard summarizing the snapshot
    and writes the full status summary into the monthly log file.
    """
    is_def_now = snapshot["active_regime"] == "DEFENSIVE"
    regime_display = "DEFENSIVE (RISK-OFF)" if is_def_now else "RISK-ON (BULLISH)"

    if snapshot["flipped_today"]:
        print("!" * 94)
        log_and_print(
            f"[ALERT] REGIME FLIP DETECTED TODAY ({snapshot['market_close_date']}): "
            f"NEW REGIME -> {regime_display}",
            level="warning",
            logger=logger,
        )
        print("!" * 94)

    log_and_print("\nDAILY / WEEKLY SIGNAL MONITOR READOUT", logger=logger)
    print("=" * 94)
    log_and_print(
        f"    As-Of Market Close   : {snapshot['market_close_date']} | "
        f"Last Weekly TPI Print : {snapshot['last_tpi_release_date']}",
        logger=logger,
    )
    log_and_print(
        f"    Active Signal Regime : {regime_display} "
        f"(Active for {snapshot['days_in_regime']} trading days since {snapshot['last_transition_date']})",
        logger=logger,
    )
    print("-" * 94)
    log_and_print("    1. Core Indicator Readings:", logger=logger)
    log_and_print(
        f"       • Weekly TPI Price        : ${snapshot['tpi_price']:.2f} / 1M tokens "
        f"(Last Release WoW: {snapshot['tpi_release_wow_pct']:+.1f}%)",
        logger=logger,
    )
    log_and_print(
        f"       • TPI {ema_span}-Day EMA Baseline : ${snapshot['tpi_ema25']:.2f} / 1M tokens "
        f"(Current Spread: {snapshot['tpi_ema_spread_pct']:+.2f}%)",
        logger=logger,
    )
    log_and_print(
        f"       • Implied Dollar Spend    : {snapshot['implied_spend_idx']:.2f} vs. {ema_span}d EMA "
        f"{snapshot['spend_ema25']:.2f} (Spread: {snapshot['spend_ema_spread_pct']:+.2f}%)",
        logger=logger,
    )
    log_and_print(
        f"       • SMH Trend Confirmation  : ${snapshot['smh_close']:.2f} vs. 20d SMA "
        f"${snapshot['smh_sma20']:.2f} ({'ABOVE' if snapshot['smh_above_sma20'] else 'BELOW'} SMA)",
        logger=logger,
    )
    log_and_print(
        f"       • EW Basket Confirmation  : {snapshot['basket_idx']:.2f} vs. 20d SMA "
        f"{snapshot['basket_sma20']:.2f} ({'ABOVE' if snapshot['basket_above_sma20'] else 'BELOW'} SMA)",
        logger=logger,
    )
    print("-" * 94)
    log_and_print("    2. Hysteresis & Next Friday Flip Thresholds:", logger=logger)
    if is_def_now:
        log_and_print(
            f"       • Exit Confirmation Streak: {snapshot['exit_streak']} consecutive bullish days "
            f"(Raw Bullish Today: {'YES' if diag['raw_bullish'] else 'NO'})",
            logger=logger,
        )
        log_and_print(
            f"       • Conditions to Re-Enter RISK-ON:\n"
            f"           (a) Next Friday TPI prints > ${snapshot['tpi_price']:.2f} (positive WoW print) "
            f"OR >= ${snapshot['next_friday_tpi_exit_trigger']:.2f} (reclaims {ema_span}d EMA), AND\n"
            f"           (b) Asset closes above its 20d SMA (SMH > ${snapshot['smh_sma20']:.2f} / "
            f"Basket > {snapshot['basket_sma20']:.2f}) for {confirm_days} straight days.",
            logger=logger,
        )
    else:
        log_and_print(
            f"       • Entry Confirmation Streak: {snapshot['entry_streak']} consecutive bearish days "
            f"(Raw Bearish Today: {'YES' if diag['raw_bearish'] else 'NO'})",
            logger=logger,
        )
        log_and_print(
            f"       • Conditions to Trigger DEFENSIVE:\n"
            f"           (a) Next Friday TPI prints < ${snapshot['tpi_price']:.2f} (negative WoW print) "
            f"AND <= ${snapshot['next_friday_tpi_entry_trigger']:.2f} ({entry_ema_thresh:+.0%} below {ema_span}d EMA), AND\n"
            f"           (b) Implied Dollar Spend spread stays < +2.0% for {confirm_days} straight days.",
            logger=logger,
        )
    print("-" * 94)
    log_and_print("    3. Target Portfolio Allocations (Next Trading Open):", logger=logger)
    log_and_print(f"       • SMH Overlay (Flat)      : {snapshot['target_smh_flat_alloc']}", logger=logger)
    log_and_print(f"       • SMH Overlay (Hedged)    : {snapshot['target_smh_hedged_alloc']}", logger=logger)
    log_and_print(f"       • Risk-On Basket Rotation : {snapshot['target_rotation_alloc']}", logger=logger)
    print("=" * 94)

# ==============================================================================# 2. CLI ARGUMENT PARSER & DAILY CHECK PIPELINE# ==============================================================================

def parse_daily_check_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Daily & Weekly AI Token Pricing Signal Monitor (`daily_check.py`)"
    )
    parser.add_argument(
        "--add-tpi",
        nargs="*",
        default=[],
        help="Append one or more new weekly TPI prints as YYYY-MM-DD:PRICE (e.g., --add-tpi 2026-11-27:1.72).",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="log",
        help="Directory for monthly execution logs and monthly CSV snapshots (default: log).",
    )
    parser.add_argument(
        "--log-path",
        type=str,
        default=None,
        help="Optional explicit CSV path override (defaults to log/signal_monitor_YYYY-MM.csv).",
    )
    parser.add_argument(
        "--telegram-mode",
        choices=["always", "flip-only", "off"],
        default="always",
        help="Telegram notification trigger mode: 'always' (default), 'flip-only', or 'off'.",
    )
    parser.add_argument(
        "--use-live-alt-api",
        action="store_true",
        help="Use live daily CC BY 4.0 feed from aitokenprice.com instead of the weekly TPI table.",
    )
    args, _ = parser.parse_known_args()
    return args

def main() -> None:
    args = parse_daily_check_args()
    logger, active_log_file = setup_monthly_logger(
        log_dir=args.log_dir,
        prefix="daily_check",
        logger_name="tpi_daily_check",
    )

    extra_tpi: dict[str, float] = {}
    for item in args.add_tpi:
        if ":" in item:
            dt_part, val_part = item.split(":", 1)
            extra_tpi[dt_part.strip()] = float(val_part.strip())

    ema_span = 25
    confirm_days = 3
    entry_ema_thresh = -0.03
    exit_ema_thresh = 0.00

    print("=" * 94)
    log_and_print("AI TOKEN PRICING & SEMICONDUCTOR DAILY SIGNAL CHECK (`daily_check.py`)", logger=logger)
    log_and_print(f"[LOG] Active monthly check log   : {active_log_file.resolve()}", logger=logger)
    print("=" * 94)

    try:
        panel = build_aligned_market_panel(
            start_date="2026-05-15",
            end_date=None,
            use_live_alt_api=args.use_live_alt_api,
            extra_tpi_points=extra_tpi,
        )
        signals = compute_smoothed_signals(panel, ema_span=ema_span)
        defensive_regime, smh_diag = generate_hysteresis_regime(
            signals,
            confirm_days=confirm_days,
            entry_ema_thresh=entry_ema_thresh,
            exit_ema_thresh=exit_ema_thresh,
            return_diagnostics=True,
        )

        # Compute Risk-On Basket (MU/AMD/NVDA/TSM) 20d SMA for live trend confirmation
        basket_tickers = ("SMH", "NVDA", "AVGO", "MU", "AMD", "TSM")
        _, basket_curves = run_semiconductor_basket_comparison(
            base_signals=signals,
            tickers=basket_tickers,
            confirm_days=confirm_days,
            entry_ema_thresh=entry_ema_thresh,
        )
        rotation_curves = run_risk_on_basket_rotation(
            base_signals=signals,
            basket_curves=basket_curves,
            risk_on_tickers=("MU", "AMD", "NVDA", "TSM"),
            confirm_days=confirm_days,
            entry_ema_thresh=entry_ema_thresh,
            tx_cost_bps=10.0,
        )

        snapshot = build_live_signal_snapshot(
            signals=signals,
            defensive_regime=defensive_regime,
            diag=smh_diag,
            rotation_curves=rotation_curves,
            ema_span=ema_span,
            confirm_days=confirm_days,
            entry_ema_thresh=entry_ema_thresh,
            exit_ema_thresh=exit_ema_thresh,
        )
        print_live_weekly_signal_monitor(
            snapshot=snapshot,
            diag=smh_diag,
            logger=logger,
            ema_span=ema_span,
            confirm_days=confirm_days,
            entry_ema_thresh=entry_ema_thresh,
        )
        log_weekly_signal_to_csv(
            snapshot=snapshot,
            logger=logger,
            log_dir=args.log_dir,
            custom_csv_path=args.log_path,
        )
        send_telegram_alert(
            snapshot=snapshot,
            logger=logger,
            mode=args.telegram_mode,
            ema_span=ema_span,
        )
        logger.info("Daily check completed successfully.")
    except Exception as exc:
        logger.exception(f"Unhandled exception during daily check: {exc}")
        raise

if __name__ == "__main__":
    main()


