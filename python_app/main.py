"""
Refactored main.py that uses BacktestManager and PerformanceAnalyzer.
Drop this into python_app/ and run from there:

cd BACKTESTING_V1/python_app
python main.py
"""

import argparse
import os
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import ccxt
import ctypes
import time
import struct

from backtest_manager import BacktestManager
from performance import PerformanceAnalyzer
from strategies import STRATEGIES

import yfinance as yf

# --------------------------
# C++ Engine Bindings
# --------------------------
class NormalizedRow(ctypes.Structure):
    _fields_ = [
        ("timestamp", ctypes.c_uint64),
        ("price", ctypes.c_double),
        ("size", ctypes.c_double),
        ("symbol", ctypes.c_char * 8),
    ]

class HighPerformanceEngine:
    def __init__(self, lib_path):
        if not lib_path:
            # Try default location
            lib_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../cpp_engine/backtester.dll"))
        
        self.lib = ctypes.CDLL(lib_path)
        
        # Define argtypes
        self.lib.engine_create.restype = ctypes.c_void_p
        self.lib.engine_destroy.argtypes = [ctypes.c_void_p]
        self.lib.engine_ingest.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self.lib.engine_get_data_count.argtypes = [ctypes.c_void_p]
        self.lib.engine_get_data_count.restype = ctypes.c_int
        self.lib.engine_get_data.argtypes = [ctypes.c_void_p, ctypes.POINTER(NormalizedRow), ctypes.c_int]

        self.engine_ptr = self.lib.engine_create()
        print("High-Performance Engine initialized.")

    def close(self):
        if self.engine_ptr:
            self.lib.engine_destroy(self.engine_ptr)
            self.engine_ptr = None
            print("Engine destroyed.")

    def run_ingestion_demo(self, count=160000):
        """
        Simulates ingesting 'count' packets and saving to Parquet.
        """
        print(f"Ingesting {count} packets via C++ Event Loop...")
        start_time = time.time()
        
        # Trigger ingestion in C++
        self.lib.engine_ingest(self.engine_ptr, count)
        
        # Retrieve data
        num_rows = self.lib.engine_get_data_count(self.engine_ptr)
        print(f"Retrieved {num_rows} normalized rows from engine.")
        
        if num_rows > 0:
            buffer = (NormalizedRow * num_rows)()
            self.lib.engine_get_data(self.engine_ptr, buffer, num_rows)
            
            # Convert to DataFrame
            data = []
            for i in range(num_rows):
                row = buffer[i]
                data.append({
                    "timestamp": row.timestamp,
                    "price": row.price,
                    "size": row.size,
                    "symbol": row.symbol.decode('utf-8')
                })
            
            df = pd.DataFrame(data)
            
            # Save to Parquet
            output_file = "network_analysis.parquet"
            df.to_parquet(output_file)
            end_time = time.time()
            
            print(f"Serialized {len(df)} rows to {output_file}")
            print(f"Total time: {end_time - start_time:.4f}s")
            print(f"Throughput: {num_rows / (end_time - start_time):.0f} events/sec")

# --------------------------
# Helpers for strategy mapping
# --------------------------

def load_config(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main(cfg_path: str = "config_example.yaml", lib_path: str = None, show_plot: bool = True, mode: str = "backtest"):
    # -----------------------------
    # Mode Selection
    # -----------------------------
    if mode == "ingest":
        engine = HighPerformanceEngine(lib_path)
        try:
            # Simulate 1 second of high-frequency data
            engine.run_ingestion_demo(count=160000)
        finally:
            engine.close()
        return

    # -----------------------------
    # 1. Load config and parameters
    # -----------------------------
    cfg = load_config(cfg_path)

    ticker = cfg.get("ticker", "AAPL")
    start = cfg.get("start", "2015-01-01")
    end = cfg.get("end", None)
    initial_cash = float(cfg.get("initial_cash", 10000.0))

    print(f"Loading data for {ticker} from {start} -> {end or 'now'}")

    # --------------------------------
    # 2. Download historical price data
    # --------------------------------


    def load_market_data(ticker: str, start: str, end: str = None) -> pd.DataFrame:
        """
        Loads market data automatically:
        - Uses Yahoo Finance for stock tickers
        - Uses KuCoin (via ccxt) for crypto pairs like BTC/USDT or ETH/USDT
        Returns a standardized OHLCV DataFrame.
        """
        if "/" not in ticker and "-" not in ticker:
            # Stock symbol - use Yahoo Finance
            import yfinance as yf
            print(f"Loading data for {ticker} from {start} -> {end or 'now'}")
            df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
            df.index = pd.to_datetime(df.index)
            return df

        # Otherwise use KuCoin
        print(f"Fetching crypto data from KuCoin for {ticker}...")
        exchange = ccxt.kucoin()
        symbol = ticker.replace("/", "-").upper()  # normalize symbol

        # KuCoin supports limited candles per request (1500 max)
        timeframe = "1d"  # 1 day candles
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=1500)
        if not ohlcv:
            raise ValueError(f"No data fetched for {ticker} from KuCoin")

        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        return df

    
    
    df = load_market_data(ticker, start, end)


    if df.empty:
        raise SystemExit(f"No data downloaded for ticker '{ticker}'")

    # --------------------------------
    # 3. Normalize column names
    # --------------------------------
    if isinstance(df.columns, pd.MultiIndex):
        # Flatten MultiIndex like ('Close', 'AAPL') -> 'close_aapl'
        df.columns = ['_'.join([str(c).strip().lower() for c in col if c]) for col in df.columns]
    else:
        df.columns = [str(c).strip().lower() for c in df.columns]

    # --------------------------------
    # 4. Identify and rename 'close' column
    # --------------------------------
    close_candidates = [c for c in df.columns if 'close' in c]
    if not close_candidates:
        raise ValueError(f"No close-like column found. Available columns: {df.columns}")

    # Use first candidate as 'close' (covers cases like 'adj close', 'close_aapl', etc.)
    if 'close' not in df.columns:
        df.rename(columns={close_candidates[0]: 'close'}, inplace=True)

    # --------------------------------
    # 5. Auto-fill missing OHLCV columns for safety
    # --------------------------------
    required_cols = ["open", "high", "low", "close", "volume"]
    for col in required_cols:
        if col not in df.columns:
            if col == "volume":
                df[col] = 0  # assume zero volume if unavailable
            else:
                df[col] = df["close"]  # replicate close price for OHLC

    # --------------------------------
    # 6. Choose and generate strategy signals
    # --------------------------------
    strat_cfg = cfg.get("strategy", {})
    strat_name = strat_cfg.get("name", "moving_average_crossover")
    strat_params = strat_cfg.get("params", {})

    strategy_fn = STRATEGIES.get(strat_name)
    if strategy_fn is None:
        raise ValueError(f"Unknown strategy: {strat_name}")

    print(f"Generating signals with strategy: {strat_name} params={strat_params}")
    signals = strategy_fn(df, **strat_params) if strat_params else strategy_fn(df)

    # --------------------------------
    # 7. Run C++ backtester
    # --------------------------------
    manager = BacktestManager(lib_path=lib_path, verbose=True)
    # Ensure numeric types and aligned indexes
    numeric_cols = ["open", "high", "low", "close", "volume"]
    for col in numeric_cols:
        if col not in df.columns:
            df[col] = df["close"]
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(method="ffill").fillna(method="bfill")

    # Align signals and price data
    signals = signals.reindex(df.index).fillna(0).astype(int)

    # Sort by timestamp just in case KuCoin returned unordered data
    df = df.sort_index()

    portfolio = manager.run_backtest(df, signals, initial_cash=initial_cash)

    # --------------------------------
    # 8. Analyze performance
    # --------------------------------
    analyzer = PerformanceAnalyzer(portfolio, price_series=df["close"])
    metrics = analyzer.summary()

    print("\n===== Performance Summary =====")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    # --------------------------------
    # 9. Plot results
    # --------------------------------
    if show_plot:
        plt.figure(figsize=(14, 9))

        # Portfolio value
        ax1 = plt.subplot(2, 1, 1)
        portfolio.plot(ax=ax1, label="Portfolio Value")
        ax1.set_title(f"Backtest Performance on {ticker}")
        ax1.legend()

        # Price + trade markers
        ax2 = plt.subplot(2, 1, 2, sharex=ax1)
        df["close"].plot(ax=ax2, label=f"{ticker} Close Price")
        ax2.plot(df.loc[signals == 1].index, df["close"][signals == 1], "^", markersize=9, label="Buy Signal")
        ax2.plot(df.loc[signals == -1].index, df["close"][signals == -1], "v", markersize=9, label="Sell Signal")
        ax2.legend()

        plt.tight_layout()
        plt.savefig("backtest_results.png")
        print("Plot saved to backtest_results.png")
        # plt.show() # blocking call, disable for automation


    return portfolio, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config_example.yaml")
    parser.add_argument("--lib", default=None, help="Path to backtester .so or .dll (optional)")
    parser.add_argument("--no-plot", action="store_true", help="Disable plotting")
    parser.add_argument("--mode", default="backtest", choices=["backtest", "ingest"], help="Mode: backtest or ingest (demo)")
    args = parser.parse_args()

    main(cfg_path=args.config, lib_path=args.lib, show_plot=not args.no_plot, mode=args.mode)
