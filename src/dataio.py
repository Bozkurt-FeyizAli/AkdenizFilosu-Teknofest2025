# -*- coding: utf-8 -*-
"""
dataio.py
Veri okuma/temel hazırlık fonksiyonları.
Bu v0'da pandas  pyarrow ile okuyoruz. Gerekirse polars/duckdb'ye geçeriz.
"""

from pathlib import Path
import pandas as pd
from .utils import reduce_memory_df

DATA_DIR = Path("./data")

def read_parquet(rel_path: str, columns=None):
    """data/ altından parquet okur; isteğe bağlı kolon seçimi ile hızlı okuma."""
    p = DATA_DIR / rel_path
    if not p.exists():
        raise FileNotFoundError(f"Missing file: {p}")
    df = pd.read_parquet(p, columns=columns)
    return df

def load_train_sessions():
    cols = [
        "ts_hour","search_term_normalized","clicked","ordered",
        "added_to_cart","added_to_fav","user_id_hashed","content_id_hashed","session_id"
    ]
    df = read_parquet("train_sessions.parquet", columns=cols)
    df["ts_hour"] = pd.to_datetime(df["ts_hour"], utc=True).dt.tz_localize(None)
    return reduce_memory_df(df)

def load_test_sessions():
    cols = ["ts_hour","search_term_normalized","user_id_hashed","content_id_hashed","session_id"]
    df = read_parquet("test_sessions.parquet", columns=cols)
    df["ts_hour"] = pd.to_datetime(df["ts_hour"], utc=True).dt.tz_localize(None)
    return reduce_memory_df(df)


def load_content_top_terms():
    cols = ["date","search_term_normalized","total_search_impression","total_search_click","content_id_hashed"]
    df = read_parquet("content/top_terms_log.parquet", columns=cols)
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None)
    return reduce_memory_df(df)

def load_content_sitewide():
    cols = ["date","total_click","total_cart","total_fav","total_order","content_id_hashed"]
    df = read_parquet("content/sitewide_log.parquet", columns=cols)
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None)
    return reduce_memory_df(df)

def load_content_meta():
    cols = [
        "level1_category_name","level2_category_name","leaf_category_name","attribute_type_count",
        "total_attribute_option_count","merchant_count","filterable_label_count",
        "content_creation_date","cv_tags","content_id_hashed"
    ]
    df = read_parquet("content/metadata.parquet", columns=cols)
    df["content_creation_date"] = (
        pd.to_datetime(df["content_creation_date"], errors="coerce", utc=True)
       .dt.tz_localize(None)
    )
    return reduce_memory_df(df)


# --- sample submission index ---
def load_sample_submission_session_ids(path: str = "data/sample_submission.csv"):
    """
    Kaggle'daki sample_submission.csv dosyasından session_id listesini okur.
    Bu listeyi submission satır sayısı ve sırası için “tek doğru kaynak” olarak kullanırız.
    """
    import pandas as pd
    df = pd.read_csv(path)
    if "session_id" not in df.columns:
        raise ValueError(f"'{path}' içinde 'session_id' kolonu yok!")
    return df["session_id"].astype(str).tolist()
