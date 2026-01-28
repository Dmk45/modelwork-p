"""
LLM Stock Analyzer - Data Parsing Pipeline
------------------------------------------
- Downloads Yahoo Finance price data
- Creates next-day return & direction labels
- Loads news text
- Aligns text with market data
- Outputs a PyTorch-ready Dataset & DataLoader
"""

import yfinance as yf
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from datetime import timedelta
from pathlib import Path

BASE_DIR = Path(__file__).parent
NEWS_CSV = BASE_DIR / "news.csv"


# -------------------------------
# CONFIG
# -------------------------------
TICKER = "AAPL"
START_DATE = "2019-01-01"
END_DATE = "2025-01-01"
NEWS_CSV = "news.csv"   # must contain: date,text
MODEL_NAME = "ProsusAI/finbert"
MAX_LEN = 256
BATCH_SIZE = 16


# -------------------------------
# STEP 1: DOWNLOAD YAHOO FINANCE DATA
# -------------------------------
def load_price_data(ticker, start, end):
    prices = yf.download(ticker, start=start, end=end, progress=False)
    prices.reset_index(inplace=True)

    # next-day return
    prices["return_1d"] = prices["Close"].pct_change().shift(-1)

    # binary direction label
    prices["label"] = (prices["return_1d"] > 0).astype(int)

    prices.dropna(inplace=True)
    return prices


# -------------------------------
# STEP 2: LOAD NEWS DATA
# -------------------------------
def load_news_data(path):
    news = pd.read_csv(path, parse_dates=["date"])
    news.sort_values("date", inplace=True)
    return news


# -------------------------------
# STEP 3: ALIGN TEXT WITH PRICE DATA
# (news → next trading day movement)
# -------------------------------
def align_news_prices(news, prices):
    prices = prices.copy()
    prices["Date"] = pd.to_datetime(prices["Date"])

    merged = pd.merge_asof(
        news.sort_values("date"),
        prices.sort_values("Date"),
        left_on="date",
        right_on="Date",
        direction="forward"
    )

    merged = merged[["date", "text", "return_1d", "label"]]
    merged.dropna(inplace=True)
    return merged.reset_index(drop=True)


# -------------------------------
# STEP 4: PYTORCH DATASET
# -------------------------------
class FinancialTextDataset(Dataset):
    def __init__(self, dataframe, tokenizer_name, max_len):
        self.df = dataframe
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = self.df.loc[idx, "text"]
        label = self.df.loc[idx, "label"]
        ret = self.df.loc[idx, "return_1d"]

        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
            "return": torch.tensor(ret, dtype=torch.float)
        }


# -------------------------------
# STEP 5: MAIN
# -------------------------------
def main():
    print("Loading price data...")
    prices = load_price_data(TICKER, START_DATE, END_DATE)

    print("Loading news data...")
    news = load_news_data(NEWS_CSV)

    print("Aligning text with market data...")
    df = align_news_prices(news, prices)

    print(f"Final dataset size: {len(df)} samples")

    dataset = FinancialTextDataset(
        df,
        tokenizer_name=MODEL_NAME,
        max_len=MAX_LEN
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    # sanity check
    batch = next(iter(loader))
    print("Batch keys:", batch.keys())
    print("input_ids shape:", batch["input_ids"].shape)
    print("labels shape:", batch["label"].shape)


if __name__ == "__main__":
    main()
