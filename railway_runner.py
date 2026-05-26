"""
railway_runner.py — One-time signal check for Railway scheduled jobs
Run this instead of run_scanner.py --schedule
"""

from run_scanner import run_once

if __name__ == "__main__":
    run_once()