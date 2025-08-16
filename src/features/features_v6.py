from pathlib import Path
import pandas as pd, numpy as np
from ..helpers import duck, safe_div, reduce_mem_usage

# ---------- CONTENT side aggregates ----------
def _content_latest_price_rate(data_dir: Path):
    q = f"""
    SELECT
      content_id_hashed,
      max_by(discounted_price, update_date) AS price_disc_last,
      max_by(original_price,   update_date) AS price_orig_last,
      max_by(content_rate_avg, update_date) AS rate_avg_last,
      max_by(content_review_count, update_date) AS review_cnt_last
    FROM read_parquet('{(data_dir/'content/price_rate_review_data.parquet').as_posix()}')
    GROUP BY content_id_hashed
    """
    return duck(q)

def _content_logs(data_dir: Path):
    q = f"""
    WITH s AS (
      SELECT content_id_hashed,
             avg(total_search_impression) AS c_imp_avg,
             avg(total_search_click)      AS c_clk_avg
      FROM read_parquet('{(data_dir/'content/search_log.parquet').as_posix()}')
      GROUP BY content_id_hashed
    ),
    w AS (
      SELECT content_id_hashed,
             avg(total_click) AS c_site_clk_avg,
             avg(total_cart)  AS c_site_cart_avg,
             avg(total_fav)   AS c_site_fav_avg,
             avg(total_order) AS c_site_ord_avg
      FROM read_parquet('{(data_dir/'content/sitewide_log.parquet').as_posix()}')
      GROUP BY content_id_hashed
    )
    SELECT * FROM s FULL JOIN w USING(content_id_hashed)
    """
    return duck(q)

def _content_metadata(data_dir: Path):
    cols = ["content_id_hashed","level1_category_name","level2_category_name",
            "leaf_category_name","attribute_type_count","total_attribute_option_count",
            "merchant_count","filterable_label_count","content_creation_date","cv_tags"]
    return pd.read_parquet(data_dir/"content/metadata.parquet", columns=cols)

# ---------- TERM / USER aggregates ----------
def _term_logs(data_dir: Path):
    q = f"""
    SELECT search_term_normalized,
           avg(total_search_impression) AS term_imp_avg,
           avg(total_search_click)      AS term_clk_avg
    FROM read_parquet('{(data_dir/'term/search_log.parquet').as_posix()}')
    GROUP BY search_term_normalized
    """
    return duck(q)

def _user_logs(data_dir: Path):
    q = f"""
    WITH u_s AS (
      SELECT user_id_hashed,
             avg(total_search_impression) AS u_imp_avg,
             avg(total_search_click)      AS u_clk_avg
      FROM read_parquet('{(data_dir/'user/search_log.parquet').as_posix()}')
      GROUP BY user_id_hashed
    ),
    u_w AS (
      SELECT user_id_hashed,
             avg(total_click) AS u_site_clk_avg,
             avg(total_cart)  AS u_site_cart_avg,
             avg(total_fav)   AS u_site_fav_avg,
             avg(total_order) AS u_site_ord_avg
      FROM read_parquet('{(data_dir/'user/sitewide_log.parquet').as_posix()}')
      GROUP BY user_id_hashed
    )
    SELECT * FROM u_s FULL JOIN u_w USING(user_id_hashed)
    """
    return duck(q)

# ---------- Retrieval-like features ----------
def _recall_features(df_sessions, data_dir: Path, topk_term=200, topk_global=200, topk_user=100):
    # TERM → topK content (content/top_terms_log)
    q_term = f"""
    WITH t AS (
      SELECT search_term_normalized, content_id_hashed,
             avg(total_search_click) AS score
      FROM read_parquet('{(data_dir/'content/top_terms_log.parquet').as_posix()}')
      GROUP BY 1,2
    ),
    r AS (
      SELECT *, ROW_NUMBER() OVER (PARTITION BY search_term_normalized ORDER BY score DESC) AS rk
      FROM t
    )
    SELECT search_term_normalized, content_id_hashed,
           MIN(rk) AS term_rank_in_pop
    FROM r WHERE rk <= {topk_term}
    GROUP BY 1,2
    """
    term_pop = duck(q_term)

    # GLOBAL popülerlik (sitewide total_click + total_order)
    q_glob = f"""
    SELECT content_id_hashed,
           avg(total_click) + 3*avg(total_order) AS gscore
    FROM read_parquet('{(data_dir/'content/sitewide_log.parquet').as_posix()}')
    GROUP BY 1
    """
    glob = duck(q_glob).sort_values("gscore", ascending=False)
    glob["global_rank_in_pop"] = np.arange(1, len(glob)+1)
    glob = glob[["content_id_hashed","global_rank_in_pop"]]
    glob_top = glob.head(topk_global)

    # USER yakın geçmiş popülerleri (user/fashion_sitewide_log)
    q_usr = f"""
    WITH u AS (
      SELECT user_id_hashed, content_id_hashed, avg(total_click)+3*avg(total_order) AS uscore
      FROM read_parquet('{(data_dir/'user/fashion_sitewide_log.parquet').as_posix()}')
      GROUP BY 1,2
    ),
    r AS (
      SELECT *, ROW_NUMBER() OVER (PARTITION BY user_id_hashed ORDER BY uscore DESC) AS rk
      FROM u
    )
    SELECT user_id_hashed, content_id_hashed, rk AS user_rank_in_hist
    FROM r WHERE rk <= {topk_user}
    """
    user_top = duck(q_usr)

    df = df_sessions.merge(term_pop, on=["search_term_normalized","content_id_hashed"], how="left") \
                    .merge(glob_top, on="content_id_hashed", how="left") \
                    .merge(user_top, on=["user_id_hashed","content_id_hashed"], how="left")

    # Boolean bayraklar ve eksik rütbeleri doldurma
    df["recall_term_hit"]   = df["term_rank_in_pop"].notna().astype("int8")
    df["recall_global_hit"] = df["global_rank_in_pop"].notna().astype("int8")
    df["recall_user_hit"]   = df["user_rank_in_hist"].notna().astype("int8")

    df["term_rank_in_pop"]   = df["term_rank_in_pop"].fillna(10_000)
    df["global_rank_in_pop"] = df["global_rank_in_pop"].fillna(10_000)
    df["user_rank_in_hist"]  = df["user_rank_in_hist"].fillna(10_000)
    return df

# ---------- Term ↔ cv_tags basit eşleşme ----------
def _token_overlap(term_s: pd.Series, tags_s: pd.Series):
    t = term_s.fillna("").str.lower().str.replace(r"[^\w]+"," ", regex=True).str.split()
    g = tags_s.fillna("").str.lower().str.replace(r"[^\w]+"," ", regex=True).str.split()
    inter = [len(set(a) & set(b)) for a,b in zip(t,g)]
    denom = [len(set(a)) + 1e-6 for a in t]
    return np.array(inter) / np.array(denom)

def build_features_v6(data_dir: str, is_train=True):
    data_dir = Path(data_dir)

    sessions = pd.read_parquet(
        data_dir / ("train_sessions.parquet" if is_train else "test_sessions.parquet")
    )

    meta = _content_metadata(data_dir)
    price = _content_latest_price_rate(data_dir)
    clog  = _content_logs(data_dir)
    tlog  = _term_logs(data_dir)
    ulog  = _user_logs(data_dir)

    df = sessions.merge(meta, on="content_id_hashed", how="left") \
                 .merge(price, on="content_id_hashed", how="left") \
                 .merge(clog,  on="content_id_hashed", how="left") \
                 .merge(tlog,  on="search_term_normalized", how="left") \
                 .merge(ulog,  on="user_id_hashed", how="left")

    # CTR/OR oranları
    df["content_search_ctr"] = safe_div(df["c_clk_avg"].fillna(0), df["c_imp_avg"].fillna(0))
    df["term_ctr"]           = safe_div(df["term_clk_avg"].fillna(0), df["term_imp_avg"].fillna(0))
    df["user_term_ctr"]      = safe_div(df["u_clk_avg"].fillna(0),   df["u_imp_avg"].fillna(0))
    df["site_order_rate"]    = df["c_site_ord_avg"].fillna(0)

    # zaman & yaş
    ts = pd.to_datetime(df["ts_hour"], errors="coerce")
    df["hour"] = ts.dt.hour.astype("int16"); df["dow"] = ts.dt.dayofweek.astype("int16")
    cc = pd.to_datetime(df["content_creation_date"], errors="coerce")
    df["content_age_days"] = (ts.dt.floor("D") - cc.dt.floor("D")).dt.days.clip(lower=0).astype("float32")

    # retrieval-benzeri sinyaller (hit/ rank)
    df = _recall_features(df, data_dir)

    # Göreceli fiyat (kategori medyanına göre)
    for lvl in ["level1_category_name","level2_category_name","leaf_category_name"]:
        if lvl in df.columns:
            med = df.groupby(lvl)["price_disc_last"].transform("median")
            df[f"rel_price_{lvl[-6:]}"] = safe_div(df["price_disc_last"].fillna(0), med.fillna(0))

    # session içi basit ranklar
    for c in ["price_disc_last","rate_avg_last","review_cnt_last",
              "content_search_ctr","term_ctr","user_term_ctr"]:
        if c in df.columns:
            df[f"r_{c}"] = df.groupby("session_id", observed=True)[c].rank("dense").astype("float32")

    # term ↔ cv_tags örtüşme
    df["term_cv_overlap"] = _token_overlap(df["search_term_normalized"], df["cv_tags"]).astype("float32")

    # kategorikleri category yap
    cats = [c for c in ["level1_category_name","level2_category_name","leaf_category_name"] if c in df.columns]
    for c in cats: df[c] = df[c].astype("category")

    keep = [
        "session_id","content_id_hashed","user_id_hashed","search_term_normalized","ts_hour",
        # num
        "price_disc_last","price_orig_last","rate_avg_last","review_cnt_last",
        "c_imp_avg","c_clk_avg","c_site_clk_avg","c_site_cart_avg","c_site_fav_avg","c_site_ord_avg",
        "term_imp_avg","term_clk_avg","u_imp_avg","u_clk_avg","u_site_clk_avg","u_site_cart_avg","u_site_fav_avg","u_site_ord_avg",
        "content_search_ctr","term_ctr","user_term_ctr","site_order_rate",
        "hour","dow","content_age_days","term_cv_overlap",
        "term_rank_in_pop","global_rank_in_pop","user_rank_in_hist",
        "recall_term_hit","recall_global_hit","recall_user_hit",
        "r_price_disc_last","r_rate_avg_last","r_review_cnt_last","r_content_search_ctr","r_term_ctr","r_user_term_ctr",
        "rel_price_1_name","rel_price_2_name","rel_price__name",  # level adı uzunluk hack; olmayanlar drop edilir
        # cats
        "level1_category_name","level2_category_name","leaf_category_name",
    ]
    if is_train: keep += ["clicked","ordered"]

    df = df[[c for c in keep if c in df.columns]]
    df = reduce_mem_usage(df)
    return df
