import pandas as pd
import numpy as np
from pathlib import Path
from ..helpers import duck, safe_div, decayed_avg_parquet, tfidf_cosine

# ---- content agg helpers ----
def _latest_per_content_price_rate(data_dir: Path):
    q = f"""
    SELECT
      content_id_hashed,
      max_by(discounted_price, update_date) AS discounted_price_last,
      max_by(original_price,   update_date) AS original_price_last,
      max_by(content_rate_avg, update_date) AS rate_avg_last,
      max_by(content_review_count, update_date) AS review_cnt_last
    FROM read_parquet('{(data_dir/'content/price_rate_review_data.parquet').as_posix()}')
    GROUP BY content_id_hashed
    """
    return duck(q)

def _agg_content_logs(data_dir: Path):
    q = f"""
    WITH search AS (
      SELECT content_id_hashed,
             avg(total_search_impression) AS c_total_search_imp_avg,
             avg(total_search_click)      AS c_total_search_clk_avg
      FROM read_parquet('{(data_dir/'content/search_log.parquet').as_posix()}')
      GROUP BY content_id_hashed
    ),
    site AS (
      SELECT content_id_hashed,
             avg(total_click) AS c_total_clk_avg,
             avg(total_cart)  AS c_total_cart_avg,
             avg(total_fav)   AS c_total_fav_avg,
             avg(total_order) AS c_total_order_avg
      FROM read_parquet('{(data_dir/'content/sitewide_log.parquet').as_posix()}')
      GROUP BY content_id_hashed
    )
    SELECT *
    FROM search
    FULL JOIN site USING (content_id_hashed)
    """
    return duck(q)

def _agg_term_logs(data_dir: Path):
    q = f"""
    SELECT
      search_term_normalized,
      avg(total_search_impression) AS term_imp_avg,
      avg(total_search_click)      AS term_clk_avg
    FROM read_parquet('{(data_dir/'term/search_log.parquet').as_posix()}')
    GROUP BY search_term_normalized
    """
    return duck(q)

def _agg_user_logs(data_dir: Path):
    q = f"""
    WITH u_search AS (
      SELECT user_id_hashed,
             avg(total_search_impression) AS u_term_imp_avg,
             avg(total_search_click)      AS u_term_clk_avg
      FROM read_parquet('{(data_dir/'user/search_log.parquet').as_posix()}')
      GROUP BY user_id_hashed
    ),
    u_site AS (
      SELECT user_id_hashed,
             avg(total_click) AS u_clk_avg,
             avg(total_cart)  AS u_cart_avg,
             avg(total_fav)   AS u_fav_avg,
             avg(total_order) AS u_order_avg
      FROM read_parquet('{(data_dir/'user/sitewide_log.parquet').as_posix()}')
      GROUP BY user_id_hashed
    )
    SELECT *
    FROM u_search
    FULL JOIN u_site USING (user_id_hashed)
    """
    return duck(q)

def _metadata_min(data_dir: Path):
    return pd.read_parquet(
        data_dir / "content/metadata.parquet",
        columns=[
            "content_id_hashed","level1_category_name","level2_category_name",
            "leaf_category_name","attribute_type_count","total_attribute_option_count",
            "merchant_count","filterable_label_count","content_creation_date","cv_tags"
        ]
    )

def _term_match_features(df_sessions: pd.DataFrame, df_meta_same_index: pd.DataFrame):
    sterm = df_sessions["search_term_normalized"].fillna("").astype(str).str.lower()
    tags  = df_sessions["cv_tags"].fillna("").astype(str).str.lower()

    def _tok(s): 
        return set([t for t in s.replace(",", " ").replace("|"," ").split() if t])

    term_tokens = sterm.map(_tok)
    tag_tokens  = tags.map(_tok)
    inter = [len(term_tokens.iloc[i] & tag_tokens.iloc[i]) for i in range(len(df_sessions))]
    denom = [len(term_tokens.iloc[i]) + 1e-6 for i in range(len(df_sessions))]
    df_sessions["q_cvtag_overlap"] = np.asarray(inter) / np.asarray(denom)
    return df_sessions

# ---- exponential-decay blocks ----
def _decay_blocks(data_dir: Path, half_life_days: int = 30):
    c_search = decayed_avg_parquet(
        data_dir / "content/search_log.parquet",
        key_col="content_id_hashed",
        date_col="date",
        cols=["total_search_impression","total_search_click"],
        half_life_days=half_life_days
    )
    c_site = decayed_avg_parquet(
        data_dir / "content/sitewide_log.parquet",
        key_col="content_id_hashed",
        date_col="date",
        cols=["total_click","total_cart","total_fav","total_order"],
        half_life_days=half_life_days
    )
    t_decay = decayed_avg_parquet(
        data_dir / "term/search_log.parquet",
        key_col="search_term_normalized",
        date_col="ts_hour",
        cols=["total_search_impression","total_search_click"],
        half_life_days=half_life_days
    )
    u_search = decayed_avg_parquet(
        data_dir / "user/search_log.parquet",
        key_col="user_id_hashed",
        date_col="ts_hour",
        cols=["total_search_impression","total_search_click"],
        half_life_days=half_life_days
    )
    u_site = decayed_avg_parquet(
        data_dir / "user/sitewide_log.parquet",
        key_col="user_id_hashed",
        date_col="ts_hour",
        cols=["total_click","total_cart","total_fav","total_order"],
        half_life_days=half_life_days
    )

    c_search = c_search.rename(columns={
        "total_search_impression_decay":"c_total_search_imp_decay",
        "total_search_click_decay":"c_total_search_clk_decay"
    })
    c_site = c_site.rename(columns={
        "total_click_decay":"c_total_clk_decay",
        "total_cart_decay":"c_total_cart_decay",
        "total_fav_decay":"c_total_fav_decay",
        "total_order_decay":"c_total_order_decay"
    })
    t_decay = t_decay.rename(columns={
        "total_search_impression_decay":"term_imp_decay",
        "total_search_click_decay":"term_clk_decay"
    })
    u_search = u_search.rename(columns={
        "total_search_impression_decay":"u_term_imp_decay",
        "total_search_click_decay":"u_term_clk_decay"
    })
    u_site = u_site.rename(columns={
        "total_click_decay":"u_clk_decay",
        "total_cart_decay":"u_cart_decay",
        "total_fav_decay":"u_fav_decay",
        "total_order_decay":"u_order_decay"
    })
    return c_search, c_site, t_decay, u_search, u_site

# Term-Content affinity (top terms per content)
def _term_content_affinity(data_dir: Path):
    # content/top_terms_log.parquet: (content_id_hashed, search_term_normalized, total_search_click/impression)
    q = f"""
    SELECT content_id_hashed, search_term_normalized,
           avg(total_search_impression) AS tc_imp_avg,
           avg(total_search_click)      AS tc_clk_avg
    FROM read_parquet('{(data_dir/'content/top_terms_log.parquet').as_posix()}')
    GROUP BY content_id_hashed, search_term_normalized
    """
    return duck(q)

# ---- MAIN FEATURE BUILDER ----
def build_features_v5(data_dir: str, is_train=True):
    data_dir = Path(data_dir)

    sessions = pd.read_parquet(
        data_dir / ("train_sessions.parquet" if is_train else "test_sessions.parquet")
    )
    prr   = _latest_per_content_price_rate(data_dir)
    clog  = _agg_content_logs(data_dir)
    meta  = _metadata_min(data_dir)
    tlog  = _agg_term_logs(data_dir)
    ulog  = _agg_user_logs(data_dir)
    c_search_d, c_site_d, t_decay, u_search_d, u_site_d = _decay_blocks(data_dir, half_life_days=30)

    df = (sessions.merge(meta, on="content_id_hashed", how="left")
                  .merge(prr,  on="content_id_hashed", how="left")
                  .merge(clog, on="content_id_hashed", how="left")
                  .merge(c_search_d, on="content_id_hashed", how="left")
                  .merge(c_site_d,   on="content_id_hashed", how="left")
                  .merge(tlog,  on="search_term_normalized", how="left")
                  .merge(t_decay, on="search_term_normalized", how="left")
                  .merge(ulog,  on="user_id_hashed", how="left")
                  .merge(u_search_d, on="user_id_hashed", how="left")
                  .merge(u_site_d,   on="user_id_hashed", how="left"))

    # Merge term-content affinity and compute CTR
    tc = _term_content_affinity(data_dir)
    df = df.merge(tc, on=["content_id_hashed","search_term_normalized"], how="left")
    df["tc_ctr"] = safe_div(df["tc_clk_avg"].fillna(0), df["tc_imp_avg"].fillna(0))

    # ratios
    df["c_search_ctr"]   = safe_div(df["c_total_search_clk_avg"].fillna(0), df["c_total_search_imp_avg"].fillna(0))
    df["term_ctr"]       = safe_div(df["term_clk_avg"].fillna(0), df["term_imp_avg"].fillna(0))
    df["u_term_ctr"]     = safe_div(df["u_term_clk_avg"].fillna(0), df["u_term_imp_avg"].fillna(0))
    df["c_search_ctr_d"] = safe_div(df["c_total_search_clk_decay"].fillna(0), df["c_total_search_imp_decay"].fillna(0))
    df["term_ctr_d"]     = safe_div(df["term_clk_decay"].fillna(0), df["term_imp_decay"].fillna(0))
    df["u_term_ctr_d"]   = safe_div(df["u_term_clk_decay"].fillna(0), df["u_term_imp_decay"].fillna(0))

    # Fiyat rekabeti
    df["price_discount_ratio"] = (df["discounted_price_last"] / (df["original_price_last"] + 1e-6)).clip(0, 10)
    df["log_discounted_price"] = np.log1p(df["discounted_price_last"].clip(lower=0))

    # Oturum içi z-score (göreli konum)
    for col in ["discounted_price_last","rate_avg_last","review_cnt_last","q_cvtag_tfidf_cos","c_search_ctr_d"]:
        if col in df.columns:
            g = df.groupby("session_id")[col]
            df[f"{col}_z_in_sess"] = (df[col] - g.transform("mean")) / (g.transform("std").replace(0, np.nan))

    # tf-idf cosine (term <-> cv_tags)
    try:
        df["q_cvtag_tfidf_cos"] = tfidf_cosine(df["search_term_normalized"], df["cv_tags"])
    except Exception:
        df["q_cvtag_tfidf_cos"] = np.nan

    # token-overlap (ek sinyal)
    df = _term_match_features(df, df)

    # Frequency encodings
    if {"search_term_normalized","leaf_category_name"}.issubset(df.columns):
        vc = df.groupby(["search_term_normalized","leaf_category_name"]).size().rename("freq_term_leaf")
        df = df.merge(vc.reset_index(), on=["search_term_normalized","leaf_category_name"], how="left")

    if {"user_id_hashed","leaf_category_name"}.issubset(df.columns):
        vc2 = df.groupby(["user_id_hashed","leaf_category_name"]).size().rename("freq_user_leaf")
        df = df.merge(vc2.reset_index(), on=["user_id_hashed","leaf_category_name"], how="left")

    cat_cols = ["level1_category_name","level2_category_name","leaf_category_name"]
    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].astype("category")

    keep_cols = [
        "session_id","user_id_hashed","content_id_hashed","search_term_normalized","ts_hour",
        "discounted_price_last","original_price_last","rate_avg_last","review_cnt_last",
        "c_total_search_imp_avg","c_total_search_clk_avg","c_total_clk_avg","c_total_cart_avg","c_total_fav_avg","c_total_order_avg",
        "c_total_search_imp_decay","c_total_search_clk_decay","c_total_clk_decay","c_total_cart_decay","c_total_fav_decay","c_total_order_decay",
        "term_imp_avg","term_clk_avg","term_imp_decay","term_clk_decay",
        "u_term_imp_avg","u_term_clk_avg","u_clk_avg","u_cart_avg","u_fav_avg","u_order_avg",
        "u_term_imp_decay","u_term_clk_decay","u_clk_decay","u_cart_decay","u_fav_decay","u_order_decay",
        "c_search_ctr","term_ctr","u_term_ctr","c_search_ctr_d","term_ctr_d","u_term_ctr_d",
        "price_discount_ratio","log_discounted_price",
        "discounted_price_last_z_in_sess","rate_avg_last_z_in_sess","review_cnt_last_z_in_sess",
        "q_cvtag_tfidf_cos_z_in_sess","c_search_ctr_d_z_in_sess",
        "tc_imp_avg","tc_clk_avg","tc_ctr",
        "freq_term_leaf","freq_user_leaf",
        "hour","dow","acc_age_days","q_cvtag_overlap","q_cvtag_tfidf_cos",
        "level1_category_name","level2_category_name","leaf_category_name",
    ]
    if "c_total_order_avg" in df.columns and "site_order_rate" not in df.columns:
        df["site_order_rate"] = df["c_total_order_avg"].fillna(0)
        keep_cols.append("site_order_rate")

    # Cold-start flags
    df["is_cold_content"] = (
        df[["c_total_search_imp_avg","c_total_clk_avg","c_total_order_avg"]].fillna(0).sum(axis=1) < 1e-6
    ).astype("int8")
    df["is_cold_term"] = df[["term_imp_avg","term_clk_avg"]].fillna(0).sum(axis=1).eq(0).astype("int8")
    df["is_cold_user"] = df[["u_clk_avg","u_cart_avg","u_fav_avg","u_order_avg"]].fillna(0).sum(axis=1).eq(0).astype("int8")

    keep_cols += ["is_cold_content","is_cold_term","is_cold_user"]

    if is_train:
        keep_cols += ["clicked","ordered"]

    keep_cols = [c for c in keep_cols if c in df.columns]
    df_out = df[keep_cols]

    # Label hazırlığı (train için 0/1/2)
    def make_labels(df_in: pd.DataFrame) -> pd.DataFrame:
        df_in = df_in.copy()
        df_in["label"] = 0
        if "clicked" in df_in.columns:
            df_in.loc[df_in["clicked"] == 1, "label"] = 1
        if "ordered" in df_in.columns:
            df_in.loc[df_in["ordered"] == 1, "label"] = 2
        return df_in

    if is_train:
        df_out = make_labels(df_out)
    return df_out
