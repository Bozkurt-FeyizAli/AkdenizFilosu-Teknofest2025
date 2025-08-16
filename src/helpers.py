import numpy as np
import pandas as pd
import duckdb

def safe_div(a, b):
    a = a.astype(float); b = b.astype(float)
    out = np.zeros_like(a, dtype=float)
    m = b != 0
    out[m] = a[m] / b[m]
    return out

def read_parquet(path):
    return pd.read_parquet(path)

def duck(q, **kwargs):
    """DuckDB sorgu yardımcıcısı. kwargs ile temp tablo geçebilirsin."""
    con = duckdb.connect()
    try:
        for k, v in kwargs.items():
            con.register(k, v)
        df = con.execute(q).fetch_df()
    finally:
        con.close()
    return df

def reduce_cats(df, cols):
    for c in cols:
        df[c] = df[c].astype("category")
    return df

def groupwise_rank(df, by, cols, method="dense", prefix="rank"):
    g = df.groupby(by, observed=True)
    for c in cols:
        if c in df.columns:
            df[f"{prefix}_{c}_in_{by}"] = g[c].rank(method)
    return df

def merge_safe(left, right, on, how="left"):
    inter = list(set(left.columns) & set(right.columns)) - set(on if isinstance(on, list) else [on])
    if inter:
        right = right.rename(columns={c: f"{c}__r" for c in inter})
    return left.merge(right, on=on, how=how)

# === v5+ EKLEMELER ===
def decayed_avg_parquet(path, key_col, date_col, cols, half_life_days=30):
    """
    Exponential-decay ortalaması: w = 0.5 ** (age_days / half_life_days)
    DuckDB ile tek seferde hesaplar.
    """
    from pathlib import Path
    path = Path(path).as_posix()
    sel = ", ".join([key_col, date_col] + cols)
    agg_expr = ", ".join([
        f"""sum({c} * POWER(0.5, (julianday(max_date) - julianday({date_col}))/{half_life_days}))
             / NULLIF(sum(POWER(0.5, (julianday(max_date) - julianday({date_col}))/{half_life_days})),0)
             AS {c}_decay""" for c in cols
    ])
    q = f"""
    WITH src AS (
      SELECT {sel}
      FROM read_parquet('{path}')
    ),
    mx AS (
      SELECT {key_col}, MAX({date_col}) AS max_date FROM src GROUP BY {key_col}
    ),
    j AS (
      SELECT s.*, m.max_date
      FROM src s LEFT JOIN mx m USING({key_col})
    )
    SELECT {key_col}, {agg_expr}
    FROM j
    GROUP BY {key_col}
    """
    return duck(q)

def tfidf_cosine(a_series, b_series, min_df=3, ngram=(1,2)):
    from sklearn.feature_extraction.text import TfidfVectorizer
    import numpy as np
    s = a_series.fillna("").astype(str).str.lower()
    t = b_series.fillna("").astype(str).str.lower()
    corpus = list(s) + list(t)
    vec = TfidfVectorizer(min_df=min_df, ngram_range=ngram)
    X = vec.fit_transform(corpus)
    Q = X[:len(s)]
    T = X[len(s):]
    num = (Q.multiply(T)).sum(axis=1).A.ravel()
    den = (np.sqrt(Q.multiply(Q).sum(axis=1).A.ravel()) *
           np.sqrt(T.multiply(T).sum(axis=1).A.ravel()))
    return num / (den + 1e-9)

def set_global_seed(seed=42):
    import random, numpy as np
    random.seed(seed); np.random.seed(seed)

def decayed_avg_parquet(path, key_col, date_col, cols, half_life_days=30):
    """
    Exponential-decay ortalaması (DuckDB uyumlu):
      weight = 0.5 ** ( date_diff('day', <date>, max_date) / half_life_days )
    """
    from pathlib import Path
    path = Path(path).as_posix()
    sel = ", ".join([key_col, date_col] + cols)

    # DuckDB: integer bölmeyi önlemek için 1.0 * ... / half_life_days
    w_expr = f"(POWER(0.5, (1.0 * date_diff('day', {date_col}, max_date) / {float(half_life_days)})))"

    agg_expr = ", ".join([
        f"sum({c} * {w_expr}) / NULLIF(sum({w_expr}), 0) AS {c}_decay"
        for c in cols
    ])

    q = f"""
    WITH src AS (
      SELECT {sel}
      FROM read_parquet('{path}')
    ),
    mx AS (
      SELECT {key_col}, MAX({date_col}) AS max_date
      FROM src
      GROUP BY {key_col}
    ),
    j AS (
      SELECT s.*, m.max_date
      FROM src s
      LEFT JOIN mx m USING({key_col})
    )
    SELECT {key_col}, {agg_expr}
    FROM j
    GROUP BY {key_col}
    """
    return duck(q)


def tfidf_cosine(a_series, b_series, min_df=3, ngram=(1,2)):
    from sklearn.feature_extraction.text import TfidfVectorizer
    import numpy as np
    s = a_series.fillna("").astype(str).str.lower()
    t = b_series.fillna("").astype(str).str.lower()
    corpus = list(s) + list(t)
    vec = TfidfVectorizer(min_df=min_df, ngram_range=ngram)
    X = vec.fit_transform(corpus)
    Q = X[:len(s)]
    T = X[len(s):]
    num = (Q.multiply(T)).sum(axis=1).A.ravel()
    den = (np.sqrt(Q.multiply(Q).sum(axis=1).A.ravel()) *
           np.sqrt(T.multiply(T).sum(axis=1).A.ravel()))
    return num / (den + 1e-9)
