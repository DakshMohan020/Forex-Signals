"""
run_scanner.py  —  Forex Swing Trading System v4.4
ENHANCED VERSION: With Google Sheets webhook integration for auto-logging

Usage:
    python run_scanner.py              # run once immediately
    python run_scanner.py --schedule   # run on H4 bar-close schedule

Schedule: checks at 12:05, 16:05, 20:05 UTC every day (5 min after bar close)
to ensure the completed bar is available from yfinance.

GOOGLE SHEETS INTEGRATION:
- Signals automatically POST to your Google Sheet
- Use Google Apps Script webhook (setup in AUTOMATION_GUIDE.md)
- Signals auto-fill into trading journal
"""

import argparse
import json
import os
import smtplib
import time
import urllib.request
from datetime import datetime, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ── Local modules ─────────────────────────────────────────────────────────────
from fetch_data import fetch_all
from signals    import check_signals, print_signal, PAIRS

# ── Notification config — loaded from .env ───────────────────────────────────
# All notification methods are optional. Set ENABLE_* to True to activate.

# Google Sheets Webhook (highly recommended for auto-logging)
# Steps to set up:
# 1. Create a Google Apps Script webhook (see AUTOMATION_GUIDE.md)
# 2. Add to .env: GOOGLE_SHEETS_WEBHOOK_URL=your_url_here
ENABLE_GOOGLE_SHEETS = True  # Set to True once you create the webhook
GOOGLE_SHEETS_WEBHOOK_URL = os.getenv("GOOGLE_SHEETS_WEBHOOK_URL", "")

# Email (Gmail recommended — use an App Password, not your main password)
# Add to .env: EMAIL_PASSWORD=your_app_password_here
ENABLE_EMAIL    = True
EMAIL_FROM      = "dakshmohan180@gmail.com"
EMAIL_PASSWORD  = os.getenv("EMAIL_PASSWORD", "")   # From .env
EMAIL_TO        = ["dakshmohan180@gmail.com"]   # can be multiple recipients
EMAIL_SMTP_HOST = "smtp.gmail.com"
EMAIL_SMTP_PORT = 587

# Telegram (create a bot via @BotFather, then get your chat ID)
# Add to .env: TELEGRAM_TOKEN=your_token and TELEGRAM_CHAT_ID=your_chat_id
ENABLE_TELEGRAM = False
TELEGRAM_TOKEN  = os.getenv("TELEGRAM_TOKEN", "your_bot_token_here")      # from @BotFather
TELEGRAM_CHAT_ID= os.getenv("TELEGRAM_CHAT_ID", "your_chat_id_here")        # from @userinfobot

# Signal log file (backup to JSON)
SIGNAL_LOG      = Path("outputs/signal_log.json")

# Open positions (update manually as you enter/exit trades)
# e.g. OPEN_POSITIONS = ["GBPUSD"] if you have a GBP/USD trade open
OPEN_POSITIONS  = []


# ── Google Sheets Webhook ─────────────────────────────────────────────────────
def send_to_google_sheets(alerts: list[dict]):
    """
    Send signals to Google Sheets via webhook.
    The Google Apps Script webhook will auto-fill them into your trading journal.
    """
    if not ENABLE_GOOGLE_SHEETS or not alerts:
        return
    
    try:
        for alert in alerts:
            # Format data for Google Sheets
            payload = {
                "pair": alert["pair"],
                "direction": alert["direction"],
                "date": alert["signal_bar"].split()[0],  # YYYY-MM-DD
                "time": alert["signal_bar"].split()[1],  # HH:MM
                "entry_price": alert["entry_approx"],
                "stop_loss": alert["stop_loss"],
                "tp1": alert["tp1"],
                "tp2": alert["tp2"],
                "stop_pips": alert["stop_dist_pips"],
                "risk_usd": alert["risk_usd"],
                "size_units": alert["size_units"],
                "adx": alert["adx"],
                "rsi": alert["rsi"],
                "atr": alert["atr"],
                "corr_adjusted": alert["corr_adjusted"],
            }
            
            data = json.dumps(payload).encode()
            req = urllib.request.Request(
                GOOGLE_SHEETS_WEBHOOK_URL,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            
            response = urllib.request.urlopen(req, timeout=10)
            result = response.read().decode()
            print(f"  ✅ Google Sheets webhook sent for {alert['pair']} {alert['direction']}")
            
    except Exception as e:
        print(f"  ❌ Google Sheets webhook failed: {e}")
        print(f"     Check your webhook URL: {GOOGLE_SHEETS_WEBHOOK_URL}")


# ── Email ─────────────────────────────────────────────────────────────────────
def _email_body(alert: dict) -> str:
    corr = "  ⚠ Corr-adjusted (50% size)" if alert["corr_adjusted"] else ""
    return f"""
FOREX SIGNAL ALERT — {alert['pair']} {alert['direction']}
{'='*50}
Signal bar : {alert['signal_bar']}
Checked at : {alert['checked_at']}

ENTRY PLAN
  Entry (approx) : {alert['entry_approx']}  (next bar open)
  Stop loss      : {alert['stop_loss']}  ({alert['stop_dist_pips']} pips)
  TP1  (1R, 50%) : {alert['tp1']}
  TP2  (3R, 50%) : {alert['tp2']}

SIZING
  Risk    : ${alert['risk_usd']:.2f}{corr}
  Size    : {alert['size_units']:,.0f} units

INDICATORS
  ADX : {alert['adx']}   RSI : {alert['rsi']}   ATR : {alert['atr']}

Remember:
  - Enter at NEXT bar open, not at signal bar close
  - Set TP1 order for 50% of position
  - Set TP2 order for remaining 50%
  - Move stop to breakeven after TP1 hits
  - Close manually after 40 bars if neither TP nor stop hit
{'='*50}
"""


def send_email(alerts: list[dict]):
    if not ENABLE_EMAIL or not alerts:
        return
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = (
            f"[Forex Signal] {', '.join(a['pair']+' '+a['direction'] for a in alerts)}"
        )
        msg["From"] = EMAIL_FROM
        msg["To"]   = ", ".join(EMAIL_TO)

        body = "\n\n".join(_email_body(a) for a in alerts)
        msg.attach(MIMEText(body, "plain"))

        with smtplib.SMTP(EMAIL_SMTP_HOST, EMAIL_SMTP_PORT) as server:
            server.ehlo()
            server.starttls()
            server.login(EMAIL_FROM, EMAIL_PASSWORD)
            server.sendmail(EMAIL_FROM, EMAIL_TO, msg.as_string())
        print(f"  ✅ Email sent to {EMAIL_TO}")
    except Exception as e:
        print(f"  ❌ Email failed: {e}")


# ── Telegram ──────────────────────────────────────────────────────────────────
def send_telegram(alerts: list[dict]):
    if not ENABLE_TELEGRAM or not alerts:
        return
    try:
        for alert in alerts:
            corr = " ⚠ Corr-adjusted (50% size)" if alert["corr_adjusted"] else ""
            text = (
                f"🔔 *{alert['pair']} {alert['direction']}*\n"
                f"Signal: `{alert['signal_bar']}`\n\n"
                f"Entry ≈ `{alert['entry_approx']}`\n"
                f"Stop:   `{alert['stop_loss']}` ({alert['stop_dist_pips']} pips)\n"
                f"TP1:    `{alert['tp1']}`\n"
                f"TP2:    `{alert['tp2']}`\n\n"
                f"Risk: ${alert['risk_usd']:.2f}{corr}\n"
                f"Size: {alert['size_units']:,.0f} units\n\n"
                f"ADX {alert['adx']}  RSI {alert['rsi']}  ATR {alert['atr']}"
            )
            url  = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
            data = json.dumps({
                "chat_id": TELEGRAM_CHAT_ID,
                "text":    text,
                "parse_mode": "Markdown",
            }).encode()
            req  = urllib.request.Request(
                url, data=data,
                headers={"Content-Type": "application/json"}
            )
            urllib.request.urlopen(req, timeout=10)
        print(f"  ✅ Telegram message sent")
    except Exception as e:
        print(f"  ❌ Telegram failed: {e}")


# ── Signal log ────────────────────────────────────────────────────────────────
def log_signals(alerts: list[dict]):
    """Backup signals to local JSON file."""
    SIGNAL_LOG.parent.mkdir(exist_ok=True)
    existing = []
    if SIGNAL_LOG.exists():
        try:
            existing = json.loads(SIGNAL_LOG.read_text())
        except Exception:
            existing = []
    existing.extend(alerts)
    SIGNAL_LOG.write_text(json.dumps(existing, indent=2))
    print(f"  📝 Signal logged to {SIGNAL_LOG}")


# ── Single scan ───────────────────────────────────────────────────────────────
def run_once(open_positions: list[str] | None = None):
    now = datetime.now(timezone.utc)
    print(f"\n{'='*52}")
    print(f"  Forex Signal Scanner  v4.4")
    print(f"  {now.strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"{'='*52}")

    # 1. Refresh data
    print("\n  [1/3] Fetching latest data...")
    try:
        fetch_all()
    except Exception as e:
        print(f"  ❌ Data fetch failed: {e}")
        return []

    # 2. Check signals
    print("\n  [2/3] Checking signals...")
    try:
        alerts = check_signals(open_positions=open_positions or OPEN_POSITIONS)
    except Exception as e:
        print(f"  ❌ Signal check failed: {e}")
        return []

    if not alerts:
        print("  No signals on current bar.\n")
        return []

    # 3. Send notifications
    print(f"\n  [3/3] {len(alerts)} signal(s) found — sending alerts...")
    for a in alerts:
        print_signal(a)

    # Send via all enabled channels
    send_to_google_sheets(alerts)
    send_email(alerts)
    send_telegram(alerts)
    log_signals(alerts)

    print("\n  ✅ Alert cycle complete.\n")
    return alerts


# ── Scheduled runner ──────────────────────────────────────────────────────────
# Check times: 5 minutes after each H4 bar close during London-NY overlap
# 12:05 UTC = 17:35 IST (London open)
# 16:05 UTC = 21:35 IST (London close / US open)
# 20:05 UTC = 01:35 IST next day (US close)
CHECK_TIMES_UTC = [(12, 5), (16, 5), (20, 5)]


def already_checked_this_bar(last_check_time: datetime | None,
                              now: datetime) -> bool:
    """True if we already ran a scan within the last 3 hours (same H4 bar)."""
    if last_check_time is None:
        return False
    return (now - last_check_time).total_seconds() < 3 * 3600


def run_scheduled():
    print("\nForex Signal Scanner — Scheduled Mode")
    print(f"Checking at {CHECK_TIMES_UTC} UTC daily")
    print("UTC times convert to IST:")
    for h, m in CHECK_TIMES_UTC:
        ist_hour = (h + 5 + (m + 30) // 60) % 24
        ist_min = (m + 30) % 60
        print(f"  {h:02d}:{m:02d} UTC = {ist_hour:02d}:{ist_min:02d} IST")
    print("\nPress Ctrl+C to stop.\n")

    last_check = None

    while True:
        now = datetime.now(timezone.utc)
        current_hm = (now.hour, now.minute)

        should_run = any(
            h == now.hour and now.minute >= m and now.minute < m + 10
            for h, m in CHECK_TIMES_UTC
        )

        if should_run and not already_checked_this_bar(last_check, now):
            run_once()
            last_check = now

        # Sleep 60 seconds between polls
        time.sleep(60)


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Forex signal scanner")
    parser.add_argument("--schedule", action="store_true",
                        help="Run on H4 bar-close schedule (daemon mode)")
    parser.add_argument("--open", nargs="+", metavar="PAIR",
                        help="Pairs with open trades, e.g. --open GBPUSD")
    args = parser.parse_args()

    if args.schedule:
        run_scheduled()
    else:
        run_once(open_positions=args.open)