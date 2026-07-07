# indicators.py

import pandas as pd
import numpy as np

REQUIRED_PRICE_COLUMNS = ["open", "high", "low", "close"]


def _standardize_columns(df):
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def _validate_price_columns(df):
    missing = [col for col in REQUIRED_PRICE_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required OHLC columns: {missing}")


def _rsi(close, length=14):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def _sma(series, length):
    return series.rolling(window=length).mean()


def _atr(high, low, close, length=14):
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()


def _macd(close, fast=12, slow=26, signal=9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line


def _bb_width(close, length=20, std=2.0):
    mid = close.rolling(window=length).mean()
    dev = close.rolling(window=length).std()
    upper = mid + std * dev
    lower = mid - std * dev
    return (upper - lower) / mid * 100


def load_raw(csv_path):
    df = pd.read_csv(csv_path)
    df = _standardize_columns(df)
    _validate_price_columns(df)

    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    elif "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    elif "gmt time" in df.columns:
        df["gmt time"] = pd.to_datetime(
            df["gmt time"], format="%d.%m.%Y %H:%M:%S.%f", errors="coerce"
        )

    return df


def add_indicators(df, indicators):
    df = df.copy()

    if "rsi" in indicators:
        df["rsi"] = _rsi(df["close"], length=14)

    if "ma20" in indicators:
        df["ma20"] = _sma(df["close"], length=20)

    if "ma50" in indicators:
        df["ma50"] = _sma(df["close"], length=50)

    if "atr" in indicators:
        df["atr"] = _atr(df["high"], df["low"], df["close"], length=14)

    if "ma20_slope" in indicators:
        if "ma20" not in df.columns:
            df["ma20"] = _sma(df["close"], length=20)
        df["ma20_slope"] = df["ma20"].diff()

    if "macd" in indicators:
        macd_line, signal_line = _macd(df["close"], fast=12, slow=26, signal=9)
        df["macd"] = macd_line
        df["macd_signal"] = signal_line

    if "bb_width" in indicators:
        df["bb_width"] = _bb_width(df["close"], length=20, std=2.0)

    feature_cols = []
    if "rsi" in indicators:
        feature_cols.append("rsi")
    if "ma20" in indicators:
        feature_cols.append("ma20")
    if "ma50" in indicators:
        feature_cols.append("ma50")
    if "atr" in indicators:
        feature_cols.append("atr")
    if "ma20_slope" in indicators:
        feature_cols.append("ma20_slope")
    if "macd" in indicators:
        feature_cols.extend(["macd", "macd_signal"])
    if "bb_width" in indicators:
        feature_cols.append("bb_width")

    df = df.dropna().reset_index(drop=True)

    return df, feature_cols


def load_and_preprocess_data(csv_path, indicators):
    df = load_raw(csv_path)
    return add_indicators(df, indicators)