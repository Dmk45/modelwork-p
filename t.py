from pathlib import Path
import pandas as pd


BASE_DIR = Path(__file__).parent
NEWS_CSV = BASE_DIR / "news.csv"

news = pd.read_csv(NEWS_CSV, parse_dates=["date"])

print("News file exists:", NEWS_CSV.exists())
