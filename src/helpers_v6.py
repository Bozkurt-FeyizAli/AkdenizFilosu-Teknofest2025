# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import duckdb
from typing import List, Tuple, Iterable, Optional
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

# ---------- genel yardımcılar ----------
def safe_div(a: pd.Series, b: pd.Series, eps: float = 1e-9) -> np.ndarray:
    a = a.astype(float); b = b.astype(float)
    out = np.zeros_like(a, dtype=float)
    m = np.abs(b) > eps
    out[m] = a[m] / b[m]
    return out

def reduce_mem_usage(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="float")
    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")
    return df

def reduce_cats(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = df[c].astype("category")
    return df

def duck(q: str, **tables) -> pd.DataFrame:
    """DuckDB sorgusu. tables={'name': dataframe} ile register edilir."""
    con = duckdb.connect()
    try:
        for k, v in tables.items():
            con.register(k, v)
        return con.execute(q).fetch_df()
    finally:
        con.close()


# ---------- grup içi işlemler ----------
def groupwise_rank(df: pd.DataFrame, by: str, cols: Iterable[str], prefix: str = "gr") -> pd.DataFrame:
    g = df.groupby(by, observed=True)
    for c in cols:
        if c in df.columns:
            df[f"{prefix}_{c}"] = g[c].rank("dense").astype("float32")
    return df


# ---------- zaman-saygılı fold bölücü (v6) ----------
def time_group_folds(df: pd.DataFrame, n_splits: int = 5,
                     sess_col: str = "session_id",
                     ts_col: str = "ts_hour") -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Oturumları (session_id) ilk görülme zamanına göre sıralayıp bloklara böler.
    Her fold 'geleceği' valid verir; sızıntıyı azaltır.
    """
    ss = df.groupby(sess_col)[ts_col].min().sort_values().index.values
    chunks = np.array_split(ss, n_splits)
    folds = []
    for i in range(n_splits):
        val_sess = set(chunks[i])
        tr_idx = df.index[~df[sess_col].isin(val_sess)].values
        va_idx = df.index[df[sess_col].isin(val_sess)].values
        folds.append((tr_idx, va_idx))
    return folds


# ---------- v6.1 için hazır yardımcılar (şimdilik opsiyonel) ----------
def token_overlap(term_s: pd.Series, tags_s: pd.Series) -> np.ndarray:
    """
    Basit term↔cv_tags örtüşmesi: ortak token / term_token_count.
    TF-IDF gelene kadar hızlı sinyal.
    """
    t = term_s.fillna("").str.lower().str.replace(r"[^\w]+", " ", regex=True).str.split()
    g = tags_s.fillna("").str.lower().str.replace(r"[^\w]+", " ", regex=True).str.split()
    inter = [len(set(a) & set(b)) for a, b in zip(t, g)]
    denom = [len(set(a)) + 1e-6 for a in t]
    return np.array(inter, dtype=float) / np.array(denom, dtype=float)


def tfidf_cosine_sparse(a_texts: pd.Series, b_texts: pd.Series,
                        max_features: int = 50_000,
                        ngram_range: Tuple[int, int] = (1, 2)) -> np.ndarray:
    """
    v6.1 için: term ↔ cv_tags TF-IDF uzayında kosinüs.
    Aşırı büyük olmaması için max_features kullanılır.
    NOT: Kaggle'da numpy.matrix ile çarpım sorununu önlemek için .A1 kullanılır.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    A = a_texts.fillna("").astype(str).values
    B = b_texts.fillna("").astype(str).values
    vec = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range, min_df=3)
    X = vec.fit_transform(np.concatenate([A, B], axis=0))
    Xa = X[: len(A)]; Xb = X[len(A):]

    # satır bazlı diyagonal dot → cosine (1D array olarak hesapla)
    numer = (Xa.multiply(Xb)).sum(axis=1).A1
    norm_a = np.sqrt(Xa.multiply(Xa).sum(axis=1)).A1
    norm_b = np.sqrt(Xb.multiply(Xb).sum(axis=1)).A1
    sim = numer / (norm_a * norm_b + 1e-9)
    return sim


def exp_time_decay(values: pd.Series, timestamps: pd.Series, ref_time: Optional[pd.Timestamp] = None,
                   half_life_days: float = 7.0) -> float:
    """
    v6.1 için: zaman ağırlıklı ortalama (λ-decay). Tek grubun agregatında kullan.
    weights = 0.5 ** (age_days / half_life)
    """
    if ref_time is None:
        ref_time = pd.to_datetime(timestamps).max()
    age_days = (pd.to_datetime(ref_time) - pd.to_datetime(timestamps)).dt.total_seconds() / 86400.0
    w = np.power(0.5, age_days / max(half_life_days, 1e-6))
    v = pd.to_numeric(values, errors="coerce").fillna(0.0).values
    return float(np.sum(v * w) / (np.sum(w) + 1e-9))


def covisitation_pairs(df_user_item_time: pd.DataFrame,
                       user_col: str = "user_id_hashed",
                       item_col: str = "content_id_hashed",
                       time_col: str = "ts_hour",
                       max_window: int = 30) -> pd.DataFrame:
    """
    v6.1 için: basit co-visitation sayacı (aynı kullanıcının yakın zamanda birlikte gördüğü ürün çiftleri).
    Dönen: pair, count. (Daha sonra recall-rank sinyali çıkarılır.)
    """
    df = df_user_item_time[[user_col, item_col, time_col]].dropna().copy()
    df[time_col] = pd.to_datetime(df[time_col])
    # aynı kullanıcı için yakın zaman pencere eşleşmesi (kabaca)
    df = df.sort_values([user_col, time_col])
    pairs = []
    for _, g in df.groupby(user_col):
        items = g[item_col].tolist()
        # küçük pencere: O(W^2) — dikkat: büyük datada DuckDB ile yapılmalı
        w = min(len(items), max_window)
        for i in range(len(items)):
            for j in range(i + 1, min(len(items), i + w)):
                a, b = items[i], items[j]
                if a == b: continue
                if a < b: pairs.append((a, b))
                else:     pairs.append((b, a))
    if not pairs:
        return pd.DataFrame(columns=[f"{item_col}_A", f"{item_col}_B", "cnt"])
    out = pd.DataFrame(pairs, columns=[f"{item_col}_A", f"{item_col}_B"])
    out = out.value_counts().reset_index(name="cnt")
    return out



# =============== TF-IDF / BM25 ===================
def build_tfidf_similarity(df, query_col="search_term", item_col="cv_tags", max_features=50000):
    """
    Returns cosine similarity between query and cv_tags using TF-IDF.
    """
    vectorizer = TfidfVectorizer(max_features=max_features)
    tfidf_items = vectorizer.fit_transform(df[item_col].astype(str).fillna(""))
    tfidf_queries = vectorizer.transform(df[query_col].astype(str).fillna(""))
    sims = cosine_similarity(tfidf_queries, tfidf_items).diagonal()
    df["tfidf_sim"] = sims
    return df

# =============== Time Decay Aggregates ===================
def time_decay_weight(ts, current_ts, half_life=7*24*3600):
    """
    λ-decay: w = exp(-Δt / τ), τ = half_life/ln(2)
    """
    tau = half_life / np.log(2)
    delta = (current_ts - ts).dt.total_seconds()
    return np.exp(-delta / tau)

def aggregate_with_decay(log_df, group_cols, value_col, ts_col, half_life=7*24*3600):
    """
    Weighted mean with λ-decay
    """
    current_ts = log_df[ts_col].max()
    log_df["w"] = time_decay_weight(log_df[ts_col], current_ts, half_life)
    agg = (log_df.groupby(group_cols)
           .apply(lambda g: np.average(g[value_col], weights=g["w"]))
           .reset_index()
           .rename(columns={0: f"{value_col}_decay"}))
    return agg

# =============== Cross Features ===================
def build_cross_counts(df, user_col="user_id", cat_col="category_id", term_col="search_term"):
    """
    User × Category and User × Term counts
    """
    user_cat = df.groupby([user_col, cat_col]).size().reset_index(name="user_cat_count")
    user_term = df.groupby([user_col, term_col]).size().reset_index(name="user_term_count")
    return user_cat, user_term

# =============== Co-visitation ===================
def build_covisitation(log_df, user_col="user_id", item_col="item_id", topk=50):
    """
    Item-Item co-visitation counts.
    """
    pairs = (log_df.merge(log_df, on=user_col)
             .query("item_id_x != item_id_y")
             .groupby(["item_id_x", "item_id_y"])
             .size()
             .reset_index(name="covis_count"))
    # normalize
    pairs["covis_score"] = pairs["covis_count"] / pairs.groupby("item_id_x")["covis_count"].transform("max")
    return pairs

# ==== v6.1: PRICE / REVIEW / DISCOUNT özellikleri ====
def add_price_review_features(df: pd.DataFrame, data_dir: str) -> pd.DataFrame:
    pr = pd.read_parquet(Path(data_dir) / "content/price_rate_review_data.parquet",
                         columns=["content_id_hashed","update_date",
                                  "original_price","selling_price","discounted_price",
                                  "content_review_count","content_rate_avg"])
    # En güncel kaydı al
    pr = pr.sort_values(["content_id_hashed","update_date"]).groupby("content_id_hashed", as_index=False).tail(1)

    pr["discount_rate"] = (pr["original_price"] - pr["selling_price"]) / (pr["original_price"].replace(0, np.nan))
    pr["discount_rate"] = pr["discount_rate"].clip(lower=-1, upper=1).fillna(0).astype("float32")

    out = df.merge(pr.drop(columns=["update_date"]), on="content_id_hashed", how="left")

    # Kategori medyanına göre göreceli fiyat ve z-score
    for lvl in ["leaf_category_name","level2_category_name","level1_category_name"]:
        if lvl in out.columns:
            med = out.groupby(lvl)["selling_price"].transform("median")
            mad = out.groupby(lvl)["selling_price"].transform("std")
            out[f"rel_price_{lvl.split('_')[0]}"] = (out["selling_price"] / (med.replace(0, np.nan))).fillna(1.0).astype("float32")
            out[f"z_price_{lvl.split('_')[0]}"] = ((out["selling_price"] - med) / (mad.replace(0, np.nan))).fillna(0.0).astype("float32")

    # Review & rating kalitesi
    out["rating_quality"] = (out["content_rate_avg"].fillna(0) * np.log1p(out["content_review_count"].fillna(0))).astype("float32")

    # İndirim oranını da session içi rank’la besle
    if "session_id" in out.columns:
        out["r_discount_rate"] = out.groupby("session_id", observed=True)["discount_rate"].rank("dense").astype("float32")

    return out


# ==== v6.1: USER×TERM ve USER×CATEGORY özellikleri ====
def add_user_term_category_features(tr: pd.DataFrame, te: pd.DataFrame, data_dir: str):
    # user/top_terms_log → user-term CTR
    ut = pd.read_parquet(Path(data_dir)/"user/top_terms_log.parquet",
                         columns=["ts_hour","user_id_hashed","search_term_normalized",
                                  "total_search_impression","total_search_click"])
    ut["ts_hour"] = pd.to_datetime(ut["ts_hour"])
    grp = ut.groupby(["user_id_hashed","search_term_normalized"], observed=True).agg(
        uterm_imp=("total_search_impression","mean"),
        uterm_clk=("total_search_click","mean")
    ).reset_index()
    grp["user_term_ctr"] = (grp["uterm_clk"] / (grp["uterm_imp"] + 1e-9)).astype("float32")
    keep_ut = grp[["user_id_hashed","search_term_normalized","user_term_ctr"]]

    # user/fashion_sitewide_log → user-category tercih oranları
    uf = pd.read_parquet(Path(data_dir)/"user/fashion_sitewide_log.parquet",
                         columns=["ts_hour","user_id_hashed","content_id_hashed","total_click"])
    # kategori map'ini train’den çek (te’de olmayan content için de işlesin diye birleşik map üretelim)
    catmap = pd.concat([
        tr[["content_id_hashed","leaf_category_name"]],
        te[["content_id_hashed","leaf_category_name"]]
    ]).drop_duplicates()
    uf = uf.merge(catmap, on="content_id_hashed", how="left")
    ucat = uf.groupby(["user_id_hashed","leaf_category_name"], observed=True)["total_click"].mean().reset_index()
    usum = ucat.groupby("user_id_hashed")["total_click"].transform("sum")
    ucat["user_cat_pref"] = (ucat["total_click"] / (usum + 1e-9)).astype("float32")
    keep_uc = ucat[["user_id_hashed","leaf_category_name","user_cat_pref"]]

    tr = tr.merge(keep_ut, on=["user_id_hashed","search_term_normalized"], how="left") \
           .merge(keep_uc, on=["user_id_hashed","leaf_category_name"],   how="left")
    te = te.merge(keep_ut, on=["user_id_hashed","search_term_normalized"], how="left") \
           .merge(keep_uc, on=["user_id_hashed","leaf_category_name"],   how="left")

    tr[["user_term_ctr","user_cat_pref"]] = tr[["user_term_ctr","user_cat_pref"]].fillna(0).astype("float32")
    te[["user_term_ctr","user_cat_pref"]] = te[["user_term_ctr","user_cat_pref"]].fillna(0).astype("float32")
    return tr, te


# ==== v6.1: ZAMAN KALIPLARI (kullanıcı ve içerik) ====
def add_time_patterns(df: pd.DataFrame) -> pd.DataFrame:
    ts = pd.to_datetime(df["ts_hour"], errors="coerce")
    df["hour"] = ts.dt.hour.astype("int16")
    df["dow"]  = ts.dt.dayofweek.astype("int16")

    # Kullanıcı saat tercihi (basit): kullanıcının saat histogramına göre P(hour)
    uh = df.groupby(["user_id_hashed","hour"], observed=True).size().rename("u_hour_cnt").reset_index()
    us = uh.groupby("user_id_hashed")["u_hour_cnt"].transform("sum")
    uh["u_hour_pref"] = (uh["u_hour_cnt"] / (us + 1e-9)).astype("float32")
    df = df.merge(uh[["user_id_hashed","hour","u_hour_pref"]], on=["user_id_hashed","hour"], how="left")

    # İçerik saat sinyali: content için hour popülerliği
    ch = df.groupby(["content_id_hashed","hour"], observed=True).size().rename("c_hour_cnt").reset_index()
    cs = ch.groupby("content_id_hashed")["c_hour_cnt"].transform("sum")
    ch["c_hour_pref"] = (ch["c_hour_cnt"] / (cs + 1e-9)).astype("float32")
    df = df.merge(ch[["content_id_hashed","hour","c_hour_pref"]], on=["content_id_hashed","hour"], how="left")

    df[["u_hour_pref","c_hour_pref"]] = df[["u_hour_pref","c_hour_pref"]].fillna(0).astype("float32")
    return df

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_datetime64_any_dtype, is_timedelta64_dtype, is_bool_dtype


def reduce_mem_usage(df: pd.DataFrame, use_float16=True, verbose=True):
    start_mem = df.memory_usage(deep=True).sum() / 1024**2

    for col in df.columns:
        col_series = df[col]

        # Datetime/timedelta/bool/object → sıkıştırma deneme, sadece kategorikleştir
        if is_datetime64_any_dtype(col_series) or is_timedelta64_dtype(col_series) or not is_numeric_dtype(col_series):
            # İsteğe bağlı: az unique ise category yap
            if col_series.dtype == 'object':
                nunique = col_series.nunique(dropna=False)
                if nunique / max(1, len(col_series)) < 0.5:
                    df[col] = col_series.astype('category')
            continue

        if is_bool_dtype(col_series):
            df[col] = col_series.astype(np.uint8)
            continue

        c_min = col_series.min()
        c_max = col_series.max()

        if str(col_series.dtype).startswith('int'):
            if c_min >= 0:
                if c_max < 255:
                    df[col] = col_series.astype(np.uint8)
                elif c_max < 65535:
                    df[col] = col_series.astype(np.uint16)
                elif c_max < 4294967295:
                    df[col] = col_series.astype(np.uint32)
                else:
                    df[col] = col_series.astype(np.uint64)
            else:
                if np.iinfo(np.int8).min <= c_min and c_max <= np.iinfo(np.int8).max:
                    df[col] = col_series.astype(np.int8)
                elif np.iinfo(np.int16).min <= c_min and c_max <= np.iinfo(np.int16).max:
                    df[col] = col_series.astype(np.int16)
                elif np.iinfo(np.int32).min <= c_min and c_max <= np.iinfo(np.int32).max:
                    df[col] = col_series.astype(np.int32)
                else:
                    df[col] = col_series.astype(np.int64)
        else:
            # float kolonlar
            if use_float16 and (c_min > np.finfo(np.float16).min) and (c_max < np.finfo(np.float16).max):
                df[col] = col_series.astype(np.float16)
            elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                df[col] = col_series.astype(np.float32)
            else:
                df[col] = col_series.astype(np.float64)

    end_mem = df.memory_usage(deep=True).sum() / 1024**2
    if verbose:
        print(f"Mem: {start_mem:.2f} MB → {end_mem:.2f} MB ({100*(start_mem-end_mem)/max(1e-9,start_mem):.1f}% azalma)")
    return df