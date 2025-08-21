# -*- coding: utf-8 -*-
"""
one_run_v9.py — Tek dosyalık ranking pipeline (Trendyol Kaggle)

KULLANIM (örnek):
  # 1) LTR (LightGBM LambdaRank) eğit
  python3 one_run_v9.py --train_ltr --alpha 0.80

  # 2) Dörtlü ensemble ile submission üret (LTR+XGB+CatBoost+TA)
  python3 one_run_v9.py --infer_ensemble \
    --alpha 0.85 \
    --w_ltr 0.70 --w_xgb 0.15 --w_cb 0.15 --w_ta 0.00 \
    --out outputs/sub_ens.csv

  # Sadece TA baseline
  python3 one_run_v9.py --baseline_timeaware --out outputs/ta.csv

  # (Opsiyonel) XGBoost / CatBoost eğitim & infer
  python3 one_run_v9.py --train_xgb
  python3 one_run_v9.py --infer_xgb --out outputs/sub_xgb.csv
  python3 one_run_v9.py --train_cat
  python3 one_run_v9.py --infer_cat --out outputs/sub_cat.csv

NOTLAR:
- Kaggle no-internet: Sentence-Transformers indiremeyebilir. Varsayılan olarak kapalı:
  USE_TEXT_SIM=0 (ENV ile aç/kapat: USE_TEXT_SIM=1)
- Veri kökü: ./data (competition dataset'i buraya kopyalayın)
"""

import os, time, argparse, numpy as np, pandas as pd
import duckdb
import lightgbm as lgb
from sklearn.metrics import roc_auc_score


# XGBoost & CatBoost opsiyonel (yüklü değilse pipeline yine çalışır)
try:
    import xgboost as xgb
    from xgboost import XGBRanker
except Exception:
    xgb = None
    XGBRanker = None

try:
    from catboost import CatBoostRanker, Pool
except Exception:
    CatBoostRanker = None
    Pool = None

# ---------------------- genel ayarlar ----------------------
DATA_DIR   = "data"
MODELS_DIR = "models"
os.makedirs("outputs", exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

class timer:
    def __init__(self, msg): self.msg = msg
    def __enter__(self): self.t0=time.time(); print(f"[TIMER] {self.msg} ..."); return self
    def __exit__(self, *a): print(f"[TIMER] {self.msg} done in {time.time()-self.t0:.2f}s")

def set_seed(s=42): np.random.seed(s)

def reduce_memory_df(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        t = df[c].dtype
        if pd.api.types.is_integer_dtype(t):
            df[c] = pd.to_numeric(df[c], downcast="integer")
        elif pd.api.types.is_float_dtype(t):
            df[c] = pd.to_numeric(df[c], downcast="float")
    return df

def load_train_sessions():
    return pd.read_parquet(os.path.join(DATA_DIR, "train_sessions.parquet"))

def load_test_sessions():
    return pd.read_parquet(os.path.join(DATA_DIR, "test_sessions.parquet"))

def load_sample_submission_session_ids(path=os.path.join(DATA_DIR,"sample_submission.csv")):
    idx = pd.read_csv(path, usecols=["session_id"])["session_id"].astype(str)
    return idx

# ====================== METİN BENZERLİK ÖZELLİĞİ (güvenli-kapı) ======================
def generate_text_similarity_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    (Opsiyonel) Sentence-Transformers ile search_term vs cv_tags cosine similarity.
    ENV: USE_TEXT_SIM=1 ise dener, yoksa atlar.
    Kaggle no-internet'te otomatik atlanır.
    """
    with timer("Text Similarity Feature Generation"):
        if os.environ.get("USE_TEXT_SIM", "0") != "1":
            print("[INFO] Text similarity disabled by env."); 
            df["term_cv_similarity"] = 0.0
            return df

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            print("[WARN] sentence-transformers yok. Bu özellik atlanıyor.")
            df["term_cv_similarity"] = 0.0
            return df

        MODEL_NAME = 'paraphrase-multilingual-MiniLM-L12-v2'
        EMB_DIR = os.path.join(MODELS_DIR, "embeddings")
        os.makedirs(EMB_DIR, exist_ok=True)
        
        term_emb_path = os.path.join(EMB_DIR, "term_embeddings.npz")
        content_emb_path = os.path.join(EMB_DIR, "content_embeddings.npz")

        def _safe_load_or_none(path):
            try:
                if os.path.exists(path):
                    data = np.load(path, allow_pickle=True)
                    return data['keys'], data['vectors']
            except Exception:
                return None, None
            return None, None

        try:
            model = SentenceTransformer(MODEL_NAME)
        except Exception as e:
            print(f"[WARN] ST model yüklenemedi ({e}). Bu özellik atlanıyor.")
            df["term_cv_similarity"] = 0.0
            return df

        # --- Arama terimleri ---
        unique_terms, term_embeddings = _safe_load_or_none(term_emb_path)
        if unique_terms is None:
            print("Arama terimi embedding'leri oluşturuluyor...")
            s_tr = pd.read_parquet(os.path.join(DATA_DIR, "train_sessions.parquet"), columns=["search_term_normalized"])
            s_te = pd.read_parquet(os.path.join(DATA_DIR, "test_sessions.parquet"), columns=["search_term_normalized"])
            unique_terms = pd.concat([s_tr, s_te])["search_term_normalized"].dropna().unique()
            term_embeddings = model.encode(unique_terms, show_progress_bar=True, convert_to_numpy=True)
            np.savez_compressed(term_emb_path, keys=unique_terms, vectors=term_embeddings)
            del s_tr, s_te
        term_to_emb = {t: e for t, e in zip(unique_terms, term_embeddings)}

        # --- Ürün cv_tags ---
        unique_contents, content_embeddings = _safe_load_or_none(content_emb_path)
        if unique_contents is None:
            print("Ürün (cv_tags) embedding'leri oluşturuluyor...")
            meta = pd.read_parquet(os.path.join(DATA_DIR, "content/metadata.parquet"),
                                   columns=["content_id_hashed","cv_tags"])
            meta = meta.dropna(subset=["content_id_hashed","cv_tags"]).drop_duplicates("content_id_hashed").set_index("content_id_hashed")
            unique_contents = meta.index.to_numpy()
            tags_to_encode = meta.loc[unique_contents, "cv_tags"].astype(str).tolist()
            content_embeddings = model.encode(tags_to_encode, show_progress_bar=True, convert_to_numpy=True)
            np.savez_compressed(content_emb_path, keys=unique_contents, vectors=content_embeddings)
            del meta
        content_to_emb = {cid: e for cid, e in zip(unique_contents, content_embeddings)}

        print("Benzerlik skorları hesaplanıyor...")
        term_vecs = df["search_term_normalized"].map(term_to_emb).tolist()
        content_vecs = df["content_id_hashed"].map(content_to_emb).tolist()

        emb_dim = term_embeddings.shape[1]
        default_vec = np.zeros(emb_dim, dtype=np.float32)
        term_vecs_np = np.array([v if v is not None else default_vec for v in term_vecs])
        content_vecs_np = np.array([v if v is not None else default_vec for v in content_vecs])

        num = np.sum(term_vecs_np * content_vecs_np, axis=1)
        den = np.linalg.norm(term_vecs_np, axis=1) * np.linalg.norm(content_vecs_np, axis=1)
        sim = np.divide(num, den, out=np.zeros_like(num, dtype=float), where=den!=0)
        df["term_cv_similarity"] = sim.astype("float32")
    return df

# ====================== KULLANICI PROFİLİ (favori kategori + fiyat) ======================
def _load_price_last():
    price = pd.read_parquet(os.path.join(DATA_DIR,"content/price_rate_review_data.parquet"))
    # isim esnekliği
    if "update_date" not in price.columns:
        cands = [c for c in price.columns if "update_date" in c]
        if cands: price = price.rename(columns={cands[0]:"update_date"})
    cols = [c for c in ["content_id_hashed","selling_price","update_date"] if c in price.columns]
    price = price[cols].sort_values("update_date").drop_duplicates("content_id_hashed", keep="last")
    return price[["content_id_hashed","selling_price"]]

def generate_user_profile_features(df: pd.DataFrame) -> pd.DataFrame:
    with timer("User Profile Feature Generation"):
        profile_path = os.path.join(MODELS_DIR, "user_profiles.parquet")

        if os.path.exists(profile_path):
            user_profiles = pd.read_parquet(profile_path)
        else:
            print("Kullanıcı profilleri oluşturuluyor...")
            sessions = pd.read_parquet(os.path.join(DATA_DIR,"train_sessions.parquet"),
                                       columns=["user_id_hashed","content_id_hashed","ordered"])
            meta = pd.read_parquet(os.path.join(DATA_DIR,"content/metadata.parquet"),
                                   columns=["content_id_hashed","leaf_category_name"])
            price_last = _load_price_last()

            ordered_sessions = (sessions[sessions['ordered']==1]
                                .dropna(subset=['user_id_hashed'])
                                .merge(meta, on="content_id_hashed", how="left")
                                .merge(price_last, on="content_id_hashed", how="left"))

            fav_cat = (ordered_sessions.groupby(["user_id_hashed","leaf_category_name"])
                                      .size().reset_index(name="counts")
                                      .sort_values(["user_id_hashed","counts"], ascending=[True,False])
                                      .drop_duplicates("user_id_hashed")
                                      .rename(columns={"leaf_category_name":"user_fav_category"}))

            avg_price = (ordered_sessions.groupby("user_id_hashed")["selling_price"]
                                       .mean().reset_index()
                                       .rename(columns={"selling_price":"user_avg_order_price"}))

            user_profiles = fav_cat[["user_id_hashed","user_fav_category"]] \
                                .merge(avg_price, on="user_id_hashed", how="outer")
            user_profiles.to_parquet(profile_path)

        # DF'e kategori & fiyat da eklensin (yoksa)
        need_merge = []
        if 'leaf_category_name' not in df.columns: need_merge.append("leaf_category_name")
        if 'selling_price' not in df.columns: need_merge.append("selling_price")
        if need_merge:
            meta_cats = pd.read_parquet(os.path.join(DATA_DIR,"content/metadata.parquet"),
                                        columns=["content_id_hashed","leaf_category_name"])
            price_last = _load_price_last()
            df = (df.merge(meta_cats, on="content_id_hashed", how="left")
                    .merge(price_last, on="content_id_hashed", how="left"))

        df = df.merge(user_profiles, on="user_id_hashed", how="left")
        df["is_in_user_fav_category"] = (df["leaf_category_name"]==df["user_fav_category"]).astype("int8")
        df["price_vs_user_avg"] = (df["selling_price"].fillna(0) / (df["user_avg_order_price"].fillna(0)+1e-6)).astype("float32")
        df = df.drop(columns=["user_fav_category","user_avg_order_price","leaf_category_name"], errors="ignore")
    return df

# ---------------------- time-aware özellik inşası (DuckDB) ----------------------
def _mk_roll(win: int, alias_prefix: str, col: str, part_cols: str) -> str:
    return (f"SUM({col}) OVER (PARTITION BY {part_cols} ORDER BY d "
            f"RANGE BETWEEN INTERVAL {win} DAY PRECEDING AND CURRENT ROW) "
            f"AS {alias_prefix}_{win}d")

def assemble_timeaware_features(sessions: pd.DataFrame, windows=(7,30)) -> pd.DataFrame:
    print("[TA] DuckDB builder -> start")
    need = [c for c in [
        "ts_hour","search_term_normalized","content_id_hashed","session_id",
        "clicked","ordered","added_to_cart","added_to_fav","user_id_hashed"
    ] if c in sessions.columns]
    s = sessions[need].copy()
    s["ts_hour"] = pd.to_datetime(s["ts_hour"], utc=False)
    s["session_date"] = s["ts_hour"].dt.floor("D")

    con = duckdb.connect()
    try:
        n = max(1, min(os.cpu_count() or 4, 16))
        con.execute(f"PRAGMA threads={n};")
    except Exception:
        pass
    con.register("sessions_df", s)

    min_d, max_d = con.execute("SELECT MIN(session_date)::DATE, MAX(session_date)::DATE FROM sessions_df").fetchone()
    con.execute(f"CREATE OR REPLACE TEMP VIEW sess_bounds AS SELECT DATE '{min_d}' AS min_d, DATE '{max_d}' AS max_d")
    con.execute("CREATE OR REPLACE TEMP VIEW key_c AS SELECT DISTINCT content_id_hashed FROM sessions_df")
    con.execute("""
        CREATE OR REPLACE TEMP VIEW key_ct AS
        SELECT DISTINCT content_id_hashed, search_term_normalized FROM sessions_df
    """)
    con.execute("""
        CREATE OR REPLACE TEMP VIEW key_ut AS
        SELECT DISTINCT user_id_hashed, search_term_normalized
        FROM sessions_df WHERE user_id_hashed IS NOT NULL AND search_term_normalized IS NOT NULL
    """)
    con.execute("""
        CREATE OR REPLACE TEMP VIEW key_uc AS
        SELECT DISTINCT user_id_hashed, content_id_hashed
        FROM sessions_df WHERE user_id_hashed IS NOT NULL AND content_id_hashed IS NOT NULL
    """)
    con.execute("CREATE OR REPLACE TEMP VIEW key_u AS SELECT DISTINCT user_id_hashed FROM sessions_df WHERE user_id_hashed IS NOT NULL")

    # sitewide rolling (content)
    sw_roll_exprs=[]
    for w in windows:
        sw_roll_exprs += [_mk_roll(w,"click","total_click","content_id_hashed"),
                          _mk_roll(w,"order","total_order","content_id_hashed")]
    con.execute(f"""
        CREATE OR REPLACE TEMP VIEW sw_roll AS
        WITH sw AS (
            SELECT sw.content_id_hashed, CAST(sw.date AS DATE) AS d,
                   CAST(sw.total_click AS DOUBLE) AS total_click,
                   CAST(sw.total_order AS DOUBLE) AS total_order
            FROM read_parquet('{DATA_DIR}/content/sitewide_log.parquet') sw
            JOIN key_c USING (content_id_hashed)
            WHERE CAST(sw.date AS DATE) BETWEEN
              (SELECT min_d - INTERVAL 90 DAY FROM sess_bounds) AND (SELECT max_d FROM sess_bounds)
        )
        SELECT content_id_hashed, d, {", ".join(sw_roll_exprs)}
        FROM sw ORDER BY content_id_hashed, d
    """)

    # term×content rolling
    tt_roll_exprs=[]
    for w in windows:
        tt_roll_exprs += [_mk_roll(w,"imp","total_search_impression","content_id_hashed, search_term_normalized"),
                          _mk_roll(w,"clk","total_search_click","content_id_hashed, search_term_normalized")]
    con.execute(f"""
        CREATE OR REPLACE TEMP VIEW tt_roll AS
        WITH ctt AS (
            SELECT c.content_id_hashed, c.search_term_normalized,
                   CAST(c.date AS DATE) AS d,
                   CAST(c.total_search_impression AS DOUBLE) AS total_search_impression,
                   CAST(c.total_search_click AS DOUBLE) AS total_search_click
            FROM read_parquet('{DATA_DIR}/content/top_terms_log.parquet') c
            JOIN key_ct k
              ON k.content_id_hashed=c.content_id_hashed AND k.search_term_normalized=c.search_term_normalized
            WHERE CAST(c.date AS DATE) BETWEEN
              (SELECT min_d - INTERVAL 90 DAY FROM sess_bounds) AND (SELECT max_d FROM sess_bounds)
        )
        SELECT content_id_hashed, search_term_normalized, d, {", ".join(tt_roll_exprs)}
        FROM ctt ORDER BY content_id_hashed, search_term_normalized, d
    """)

    # price / rating son değer
    con.execute(f"""
        CREATE OR REPLACE TEMP VIEW prr AS
        SELECT p.content_id_hashed, CAST(p.update_date AS DATE) AS d,
               CAST(p.original_price AS DOUBLE) AS original_price,
               CAST(p.selling_price AS DOUBLE) AS selling_price,
               CAST(p.discounted_price AS DOUBLE) AS discounted_price,
               CAST(p.content_rate_avg AS DOUBLE) AS rate_avg,
               CAST(p.content_rate_count AS DOUBLE) AS rate_cnt
        FROM read_parquet('{DATA_DIR}/content/price_rate_review_data.parquet') p
        JOIN key_c USING (content_id_hashed)
        WHERE CAST(p.update_date AS DATE) BETWEEN
              (SELECT min_d - INTERVAL 90 DAY FROM sess_bounds) AND (SELECT max_d FROM sess_bounds)
    """)

    # user×term 30g
    con.execute(f"""
        CREATE OR REPLACE TEMP VIEW ut_roll AS
        WITH ut AS (
            SELECT u.user_id_hashed, u.search_term_normalized,
                   CAST(u.ts_hour AS DATE) AS d,
                   CAST(u.total_search_impression AS DOUBLE) AS imp,
                   CAST(u.total_search_click AS DOUBLE) AS clk
            FROM read_parquet('{DATA_DIR}/user/top_terms_log.parquet') u
            JOIN key_ut k ON k.user_id_hashed=u.user_id_hashed AND k.search_term_normalized=u.search_term_normalized
            WHERE CAST(u.ts_hour AS DATE) BETWEEN
                (SELECT min_d - INTERVAL 90 DAY FROM sess_bounds) AND (SELECT max_d FROM sess_bounds)
        )
        SELECT user_id_hashed, search_term_normalized, d,
               SUM(imp) OVER (PARTITION BY user_id_hashed, search_term_normalized
                              ORDER BY d RANGE BETWEEN INTERVAL 30 DAY PRECEDING AND CURRENT ROW) AS u_imp_30d,
               SUM(clk) OVER (PARTITION BY user_id_hashed, search_term_normalized
                              ORDER BY d RANGE BETWEEN INTERVAL 30 DAY PRECEDING AND CURRENT ROW) AS u_clk_30d
        FROM ut ORDER BY user_id_hashed, search_term_normalized, d
    """)

    # user sitewide rolling
    u_sw_exprs=[]
    for w in windows:
        u_sw_exprs += [
            _mk_roll(w,"u_sw_click","total_click","user_id_hashed"),
            _mk_roll(w,"u_sw_cart","total_cart","user_id_hashed"),
            _mk_roll(w,"u_sw_fav","total_fav","user_id_hashed"),
            _mk_roll(w,"u_sw_order","total_order","user_id_hashed"),
        ]
    con.execute(f"""
        CREATE OR REPLACE TEMP VIEW u_sw_roll AS
        WITH u AS (
            SELECT user_id_hashed, CAST(ts_hour AS DATE) AS d,
                   CAST(total_click AS DOUBLE) AS total_click,
                   CAST(total_cart  AS DOUBLE) AS total_cart,
                   CAST(total_fav   AS DOUBLE) AS total_fav,
                   CAST(total_order AS DOUBLE) AS total_order
            FROM read_parquet('{DATA_DIR}/user/sitewide_log.parquet')
            JOIN key_u USING (user_id_hashed)
            WHERE CAST(ts_hour AS DATE) BETWEEN
              (SELECT min_d - INTERVAL 90 DAY FROM sess_bounds) AND (SELECT max_d FROM sess_bounds)
        )
        SELECT user_id_hashed, d, {", ".join(u_sw_exprs)}
        FROM u ORDER BY user_id_hashed, d
    """)

    # user arama rolling
    u_search_exprs=[]
    for w in windows:
        u_search_exprs += [
            _mk_roll(w,"u_search_imp","total_search_impression","user_id_hashed"),
            _mk_roll(w,"u_search_clk","total_search_click","user_id_hashed"),
        ]
    con.execute(f"""
        CREATE OR REPLACE TEMP VIEW u_search_roll AS
        WITH u AS (
            SELECT user_id_hashed, CAST(ts_hour AS DATE) AS d,
                   CAST(total_search_impression AS DOUBLE) AS total_search_impression,
                   CAST(total_search_click      AS DOUBLE) AS total_search_click
            FROM read_parquet('{DATA_DIR}/user/search_log.parquet')
            JOIN key_u USING (user_id_hashed)
            WHERE CAST(ts_hour AS DATE) BETWEEN
              (SELECT min_d - INTERVAL 90 DAY FROM sess_bounds) AND (SELECT max_d FROM sess_bounds)
        )
        SELECT user_id_hashed, d, {", ".join(u_search_exprs)}
        FROM u ORDER BY user_id_hashed, d
    """)

    # content arama rolling
    c_search_exprs=[]
    for w in windows:
        c_search_exprs += [
            _mk_roll(w,"c_search_imp","total_search_impression","content_id_hashed"),
            _mk_roll(w,"c_search_clk","total_search_click","content_id_hashed"),
        ]
    con.execute(f"""
        CREATE OR REPLACE TEMP VIEW c_search_roll AS
        WITH c AS (
            SELECT content_id_hashed, CAST(date AS DATE) AS d,
                   CAST(total_search_impression AS DOUBLE) AS total_search_impression,
                   CAST(total_search_click      AS DOUBLE) AS total_search_click
            FROM read_parquet('{DATA_DIR}/content/search_log.parquet')
            JOIN key_c USING (content_id_hashed)
            WHERE CAST(date AS DATE) BETWEEN
              (SELECT min_d - INTERVAL 90 DAY FROM sess_bounds) AND (SELECT max_d FROM sess_bounds)
        )
        SELECT content_id_hashed, d, {", ".join(c_search_exprs)}
        FROM c ORDER BY content_id_hashed, d
    """)

    # term-only rolling
    t_search_exprs=[]
    for w in windows:
        t_search_exprs += [
            _mk_roll(w,"t_imp","total_search_impression","search_term_normalized"),
            _mk_roll(w,"t_clk","total_search_click","search_term_normalized"),
        ]
    con.execute(f"""
        CREATE OR REPLACE TEMP VIEW t_search_roll AS
        WITH t AS (
            SELECT search_term_normalized, CAST(ts_hour AS DATE) AS d,
                   CAST(total_search_impression AS DOUBLE) AS total_search_impression,
                   CAST(total_search_click      AS DOUBLE) AS total_search_click
            FROM read_parquet('{DATA_DIR}/term/search_log.parquet')
            WHERE CAST(ts_hour AS DATE) BETWEEN
              (SELECT min_d - INTERVAL 90 DAY FROM sess_bounds) AND (SELECT max_d FROM sess_bounds)
        )
        SELECT search_term_normalized, d, {", ".join(t_search_exprs)}
        FROM t ORDER BY search_term_normalized, d
    """)

    # user×content fashion logs
    con.execute(f"""
        CREATE OR REPLACE TEMP VIEW uf_roll AS
        WITH uf AS (
            SELECT user_id_hashed, content_id_hashed,
                   CAST(ts_hour AS DATE) AS d,
                   CAST(total_search_impression AS DOUBLE) AS imp,
                   CAST(total_search_click AS DOUBLE) AS clk
            FROM read_parquet('{DATA_DIR}/user/fashion_search_log.parquet')
            JOIN key_uc USING (user_id_hashed, content_id_hashed)
            WHERE CAST(ts_hour AS DATE) BETWEEN
              (SELECT min_d - INTERVAL 90 DAY FROM sess_bounds) AND (SELECT max_d FROM sess_bounds)
        )
        SELECT user_id_hashed, content_id_hashed, d,
               SUM(imp) OVER (PARTITION BY user_id_hashed, content_id_hashed
                              ORDER BY d RANGE BETWEEN INTERVAL 30 DAY PRECEDING AND CURRENT ROW) AS uf_imp_30d,
               SUM(clk) OVER (PARTITION BY user_id_hashed, content_id_hashed
                              ORDER BY d RANGE BETWEEN INTERVAL 30 DAY PRECEDING AND CURRENT ROW) AS uf_clk_30d
        FROM uf ORDER BY user_id_hashed, content_id_hashed, d
    """)

    # user×content fashion sitewide logs
    con.execute(f"""
        CREATE OR REPLACE TEMP VIEW ufs_roll AS
        WITH ufs AS (
            SELECT user_id_hashed, content_id_hashed,
                   CAST(ts_hour AS DATE) AS d,
                   CAST(total_click AS DOUBLE) AS total_click,
                   CAST(total_cart  AS DOUBLE) AS total_cart,
                   CAST(total_fav   AS DOUBLE) AS total_fav,
                   CAST(total_order AS DOUBLE) AS total_order
            FROM read_parquet('{DATA_DIR}/user/fashion_sitewide_log.parquet')
            JOIN key_uc USING (user_id_hashed, content_id_hashed)
            WHERE CAST(ts_hour AS DATE) BETWEEN
              (SELECT min_d - INTERVAL 90 DAY FROM sess_bounds) AND (SELECT max_d FROM sess_bounds)
        )
        SELECT user_id_hashed, content_id_hashed, d,
               SUM(total_click) OVER (PARTITION BY user_id_hashed, content_id_hashed
                                      ORDER BY d RANGE BETWEEN INTERVAL 30 DAY PRECEDING AND CURRENT ROW) AS uf_sw_click_30d,
               SUM(total_cart)  OVER (PARTITION BY user_id_hashed, content_id_hashed
                                      ORDER BY d RANGE BETWEEN INTERVAL 30 DAY PRECEDING AND CURRENT ROW) AS uf_sw_cart_30d,
               SUM(total_fav)   OVER (PARTITION BY user_id_hashed, content_id_hashed
                                      ORDER BY d RANGE BETWEEN INTERVAL 30 DAY PRECEDING AND CURRENT ROW) AS uf_sw_fav_30d,
               SUM(total_order) OVER (PARTITION BY user_id_hashed, content_id_hashed
                                      ORDER BY d RANGE BETWEEN INTERVAL 30 DAY PRECEDING AND CURRENT ROW) AS uf_sw_order_30d
        FROM ufs ORDER BY user_id_hashed, content_id_hashed, d
    """)

    # meta geniş
    con.execute(f"""
        CREATE OR REPLACE TEMP VIEW meta AS
        SELECT content_id_hashed,
               CAST(content_creation_date AS DATE) AS creation_date,
               CAST(attribute_type_count AS DOUBLE) AS attr_type_cnt,
               CAST(total_attribute_option_count AS DOUBLE) AS attr_opt_cnt,
               CAST(merchant_count AS DOUBLE) AS merchant_cnt,
               CAST(filterable_label_count AS DOUBLE) AS filter_label_cnt
        FROM read_parquet('{DATA_DIR}/content/metadata.parquet')
    """)

    # user metadata
    con.execute(f"""
        CREATE OR REPLACE TEMP VIEW user_meta AS
        SELECT user_id_hashed,
               CASE WHEN lower(user_gender)='female' THEN 1
                    WHEN lower(user_gender)='male' THEN 2
                    ELSE 0 END AS user_gender_code,
               CAST(user_birth_year AS DOUBLE) AS user_birth_year,
               CAST(user_tenure_in_days AS DOUBLE) AS user_tenure_days
        FROM read_parquet('{DATA_DIR}/user/metadata.parquet')
    """)

    # as-of join + oranlar
    rate_cols, ctr_cols, sel_sw, sel_tt = [], [], [], []
    for w in windows:
        sel_sw += [f"r.click_{w}d AS click_{w}d", f"r.order_{w}d AS order_{w}d"]
        sel_tt += [f"r.imp_{w}d AS imp_{w}d", f"r.clk_{w}d AS clk_{w}d"]
        rate_cols += [
            f"(COALESCE(swf.click_{w}d,0)+1.0)/(COALESCE(swf.click_{w}d,0)+1.0+3.0) AS click_rate_{w}d",
            f"(COALESCE(swf.order_{w}d,0)+1.0)/(COALESCE(swf.click_{w}d,0)+1.0+3.0) AS order_rate_{w}d",
        ]
        ctr_cols += [
            f"(COALESCE(ttf.clk_{w}d,0)+1.0)/(COALESCE(ttf.imp_{w}d,0)+1.0+3.0) AS tc_ctr_{w}d",
        ]

    sql = f"""
        WITH s AS (SELECT *, CAST(session_date AS DATE) AS sdate FROM sessions_df)
        SELECT
            s.*,
            {", ".join(rate_cols)},
            {", ".join(ctr_cols)},
            CASE
              WHEN prrf.original_price IS NOT NULL AND prrf.original_price > 0
              THEN (prrf.original_price - COALESCE(prrf.discounted_price, prrf.selling_price)) / prrf.original_price
              ELSE NULL
            END AS discount_pct,
            prrf.rate_avg AS rating_avg,
            LOG(1 + COALESCE(prrf.selling_price,0)) AS log_selling_price,
            LOG(1 + COALESCE(prrf.discounted_price,0)) AS log_discounted_price,
            LOG(1 + COALESCE(prrf.rate_cnt,0)) AS rating_log_cnt,
            (COALESCE(utf.u_clk_30d,0)+1.0)/(COALESCE(utf.u_imp_30d,0)+1.0+3.0) AS user_term_ctr_30d,
            {', '.join([f"(COALESCE(uswf.u_sw_cart_{w}d,0)+1.0)/(COALESCE(uswf.u_sw_click_{w}d,0)+1.0+3.0) AS user_sw_cart_rate_{w}d" for w in windows])},
            {', '.join([f"(COALESCE(uswf.u_sw_fav_{w}d,0)+1.0)/(COALESCE(uswf.u_sw_click_{w}d,0)+1.0+3.0) AS user_sw_fav_rate_{w}d" for w in windows])},
            {', '.join([f"(COALESCE(uswf.u_sw_order_{w}d,0)+1.0)/(COALESCE(uswf.u_sw_click_{w}d,0)+1.0+3.0) AS user_sw_order_rate_{w}d" for w in windows])},
            {', '.join([f"(COALESCE(usrf.u_search_clk_{w}d,0)+1.0)/(COALESCE(usrf.u_search_imp_{w}d,0)+1.0+3.0) AS user_search_ctr_{w}d" for w in windows])},
            {', '.join([f"(COALESCE(csr.c_search_clk_{w}d,0)+1.0)/(COALESCE(csr.c_search_imp_{w}d,0)+1.0+3.0) AS content_search_ctr_{w}d" for w in windows])},
            {', '.join([f"(COALESCE(tsr.t_clk_{w}d,0)+1.0)/(COALESCE(tsr.t_imp_{w}d,0)+1.0+3.0) AS term_ctr_{w}d" for w in windows])},
            (COALESCE(ufr.uf_clk_30d,0)+1.0)/(COALESCE(ufr.uf_imp_30d,0)+1.0+3.0) AS user_content_fashion_ctr_30d,
            (COALESCE(ufsr.uf_sw_order_30d,0)+1.0)/(COALESCE(ufsr.uf_sw_click_30d,0)+1.0+3.0) AS user_content_fashion_order_rate_30d,
            COALESCE(DATEDIFF('day', m.creation_date, s.sdate), -1) AS days_since_creation,
            m.attr_type_cnt, m.attr_opt_cnt, m.merchant_cnt, m.filter_label_cnt,
            um.user_gender_code, um.user_birth_year, um.user_tenure_days
        FROM s
        LEFT JOIN LATERAL (
            SELECT r.d, {', '.join(sel_sw)} FROM sw_roll r
            WHERE r.content_id_hashed = s.content_id_hashed AND r.d <= s.sdate
            ORDER BY r.d DESC LIMIT 1
        ) AS swf ON TRUE
        LEFT JOIN LATERAL (
            SELECT r.d, {', '.join(sel_tt)} FROM tt_roll r
            WHERE r.content_id_hashed = s.content_id_hashed
              AND r.search_term_normalized = s.search_term_normalized
              AND r.d <= s.sdate
            ORDER BY r.d DESC LIMIT 1
        ) AS ttf ON TRUE
        LEFT JOIN LATERAL (
            SELECT r.d, r.selling_price, r.discounted_price, r.original_price, r.rate_avg, r.rate_cnt
            FROM prr r WHERE r.content_id_hashed = s.content_id_hashed AND r.d <= s.sdate
            ORDER BY r.d DESC LIMIT 1
        ) AS prrf ON TRUE
        LEFT JOIN LATERAL (
            SELECT r.d, r.u_imp_30d, r.u_clk_30d
            FROM ut_roll r
            WHERE r.user_id_hashed = s.user_id_hashed
              AND r.search_term_normalized = s.search_term_normalized
              AND r.d <= s.sdate
            ORDER BY r.d DESC LIMIT 1
        ) AS utf ON TRUE
        LEFT JOIN LATERAL (
            SELECT r.d,
                   {', '.join([f'r.u_sw_click_{w}d AS u_sw_click_{w}d' for w in windows])},
                   {', '.join([f'r.u_sw_cart_{w}d  AS u_sw_cart_{w}d'  for w in windows])},
                   {', '.join([f'r.u_sw_fav_{w}d   AS u_sw_fav_{w}d'   for w in windows])},
                   {', '.join([f'r.u_sw_order_{w}d AS u_sw_order_{w}d' for w in windows])}
            FROM u_sw_roll r
            WHERE r.user_id_hashed = s.user_id_hashed AND r.d <= s.sdate
            ORDER BY r.d DESC LIMIT 1
        ) AS uswf ON TRUE
        LEFT JOIN LATERAL (
            SELECT r.d,
                   {', '.join([f'r.u_search_imp_{w}d AS u_search_imp_{w}d' for w in windows])},
                   {', '.join([f'r.u_search_clk_{w}d AS u_search_clk_{w}d' for w in windows])}
            FROM u_search_roll r
            WHERE r.user_id_hashed = s.user_id_hashed AND r.d <= s.sdate
            ORDER BY r.d DESC LIMIT 1
        ) AS usrf ON TRUE
        LEFT JOIN LATERAL (
            SELECT r.d,
                   {', '.join([f'r.c_search_imp_{w}d AS c_search_imp_{w}d' for w in windows])},
                   {', '.join([f'r.c_search_clk_{w}d AS c_search_clk_{w}d' for w in windows])}
            FROM c_search_roll r
            WHERE r.content_id_hashed = s.content_id_hashed AND r.d <= s.sdate
            ORDER BY r.d DESC LIMIT 1
        ) AS csr ON TRUE
        LEFT JOIN LATERAL (
            SELECT r.d,
                   {', '.join([f'r.t_imp_{w}d AS t_imp_{w}d' for w in windows])},
                   {', '.join([f'r.t_clk_{w}d AS t_clk_{w}d' for w in windows])}
            FROM t_search_roll r
            WHERE r.search_term_normalized = s.search_term_normalized AND r.d <= s.sdate
            ORDER BY r.d DESC LIMIT 1
        ) AS tsr ON TRUE
        LEFT JOIN LATERAL (
            SELECT r.d, r.uf_imp_30d, r.uf_clk_30d
            FROM uf_roll r
            WHERE r.user_id_hashed = s.user_id_hashed AND r.content_id_hashed = s.content_id_hashed AND r.d <= s.sdate
            ORDER BY r.d DESC LIMIT 1
        ) AS ufr ON TRUE
        LEFT JOIN LATERAL (
            SELECT r.d, r.uf_sw_click_30d, r.uf_sw_cart_30d, r.uf_sw_fav_30d, r.uf_sw_order_30d
            FROM ufs_roll r
            WHERE r.user_id_hashed = s.user_id_hashed AND r.content_id_hashed = s.content_id_hashed AND r.d <= s.sdate
            ORDER BY r.d DESC LIMIT 1
        ) AS ufsr ON TRUE
        LEFT JOIN meta m ON m.content_id_hashed = s.content_id_hashed
        LEFT JOIN user_meta um ON um.user_id_hashed = s.user_id_hashed
    """
    out = con.execute(sql).df()

    # (Opsiyonel) metin benzerliği
    out = generate_text_similarity_features(out)
    # Kullanıcı profil özellikleri
    out = generate_user_profile_features(out)

    # trend sinyalleri
    out["trend_order_rate_7v30"] = out.get("order_rate_7d", 0).fillna(0) - out.get("order_rate_30d", 0).fillna(0)
    out["trend_tc_ctr_7v30"]     = out.get("tc_ctr_7d", 0).fillna(0)       - out.get("tc_ctr_30d", 0).fillna(0)

    eps = 1e-4
    for a, b, name in [
        ("order_rate_7d","order_rate_30d","trend_order_7v30"),
        ("tc_ctr_7d","tc_ctr_30d","trend_tcctr_7v30"),
        ("click_7d","click_30d","trend_swclick_7v30"),
        ("order_7d","order_30d","trend_sworder_7v30"),
    ]:
        if a in out.columns and b in out.columns:
            out[name] = (out[a].fillna(0) + eps) / (out[b].fillna(0) + eps)

    if "discount_pct" in out.columns and "rating_avg" in out.columns:
        out["promo_x_rating"] = out["discount_pct"].clip(lower=0).fillna(0) * (out["rating_avg"].fillna(0) / 5.0)

    # oturum içi sinyaller
    out = add_in_session_features(out)

    out = reduce_memory_df(out)
    print("[TA] DuckDB builder -> done")
    return out

def add_in_session_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["session_id","ts_hour","content_id_hashed"]).reset_index(drop=True)
    df["sess_step_idx"] = df.groupby("session_id").cumcount().astype("int32")
    g_item = df.groupby(["session_id","content_id_hashed"], sort=False)
    df["item_occ_idx"] = g_item.cumcount().astype("int16")
    df["seen_before"]  = (df["item_occ_idx"] > 0).astype("int8")

    prev_step = g_item["sess_step_idx"].shift()
    df["steps_since_item_last_seen"] = (df["sess_step_idx"] - prev_step).fillna(1e6).clip(0, 1e6).astype("float32")
    prev_time = g_item["ts_hour"].shift()
    df["secs_since_item_last_seen"]  = (df["ts_hour"] - prev_time).dt.total_seconds().fillna(1e9).clip(0, 1e9).astype("float32")

    for lab in ["clicked","added_to_cart","added_to_fav","ordered"]:
        if lab in df.columns:
            cum = g_item[lab].cumsum()
            df[f"{lab}_before_item_sess"] = (cum - df[lab]).astype("int16")

    sess_start = df.groupby("session_id")["ts_hour"].transform("min")
    df["secs_since_session_start"] = (df["ts_hour"] - sess_start).dt.total_seconds().astype("float32")

    # oturum içi rütbeler
    if "log_discounted_price" in df.columns:
        df["cheap_rank_sess"] = 1.0 - df.groupby("session_id")["log_discounted_price"].rank(method="first", pct=True)
    if "discount_pct" in df.columns:
        df["discount_rank_sess"] = df.groupby("session_id")["discount_pct"].rank(method="first", ascending=False, pct=True)

    for base in ["order_rate_7d","tc_ctr_7d","discount_pct","rating_avg",
                 "log_discounted_price","trend_order_7v30","trend_tcctr_7v30"]:
        if base in df.columns:
            df[f"{base}_rank_sess"] = df.groupby("session_id")[base].rank(method="first", pct=True).astype("float32")

    # zaman & karşılaştırmalar
    df["hour_of_day"] = df["ts_hour"].dt.hour.astype("int8")
    df["day_of_week"] = df["ts_hour"].dt.dayofweek.astype("int8")

    if "selling_price" in df.columns:
        sess_mean_price = df.groupby("session_id")["selling_price"].transform("mean")
        df["price_vs_session_mean"] = (df["selling_price"] / (sess_mean_price + 1e-6)).astype("float32")

    if "discount_pct" in df.columns:
        df["is_highest_discount"] = (df["discount_pct"] == df.groupby("session_id")["discount_pct"].transform("max")).astype("int8")

    if "rating_avg" in df.columns:
        df["is_highest_rating"] = (df["rating_avg"] == df.groupby("session_id")["rating_avg"].transform("max")).astype("int8")

    return df

# ---------------------- TA baseline ----------------------
def score_timeaware_baseline(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["pred_click"] = 0.65*out.get("tc_ctr_7d", 0) + 0.35*out.get("tc_ctr_30d", 0)
    out["pred_order"] = 0.70*out.get("order_rate_7d", 0) + 0.30*out.get("order_rate_30d", 0)
    out["pred_order"] += 0.05*out.get("discount_pct", 0).clip(lower=0)
    out["pred_order"] += 0.02*out.get("rating_avg", 0).fillna(0)/5.0
    out["pred_click"] += 0.03*out.get("user_term_ctr_30d", 0)
    out["pred_final"] = 0.80*out["pred_order"] + 0.20*out["pred_click"]
    return out

def normalize_in_session(df: pd.DataFrame, score_col="pred_final") -> pd.DataFrame:
    out = df.copy()
    grp = out.groupby("session_id")[score_col]
    out[score_col] = (out[score_col] - grp.transform("min")) / (grp.transform("max") - grp.transform("min") + 1e-8)
    return out

# ---------------------- yardımcılar ----------------------
LABEL_COLS = {"clicked","ordered","added_to_cart","added_to_fav"}
ID_COLS    = {"session_id","content_id_hashed","ts_hour","session_date","search_term_normalized","user_id_hashed"}

def build_relevance(df: pd.DataFrame) -> pd.Series:
    o = df.get("ordered", 0).fillna(0).astype(int)
    c = df.get("clicked", 0).fillna(0).astype(int)
    a = df.get("added_to_cart", 0).fillna(0).astype(int)
    f = df.get("added_to_fav", 0).fillna(0).astype(int)
    return (4*o + 3*a + 2*f + 1*c).astype("int32")


def _feature_cols(df: pd.DataFrame) -> list:
    cols = []
    for c in df.columns:
        if c in LABEL_COLS or c in ID_COLS: continue
        if pd.api.types.is_numeric_dtype(df[c]): cols.append(c)
    return cols

def split_time_holdout(df: pd.DataFrame, holdout_days=7):
    print("[SPLIT] Using session-time 80/20 holdout...")
    session_times = df[["session_id","ts_hour"]].groupby("session_id")["ts_hour"].min().sort_values()
    num_sessions = len(session_times)
    split_idx = int(num_sessions * 0.80)
    train_session_ids = set(session_times.index[:split_idx])
    valid_session_ids = set(session_times.index[split_idx:])
    tr = df[df["session_id"].isin(train_session_ids)].copy()
    va = df[df["session_id"].isin(valid_session_ids)].copy()
    print(f"[SPLIT] Done. train={len(tr):,} rows ({len(train_session_ids):,} sessions) | valid={len(va):,} rows ({len(valid_session_ids):,} sessions)")
    if len(tr)==0 or len(va)==0: raise ValueError("Train/valid boş! Veri tutarlılığını kontrol edin.")
    return tr, va

# ---------------------- LTR (LightGBM LambdaRank) ----------------------
def _callbacks():
    return [lgb.early_stopping(stopping_rounds=300, verbose=True),
            lgb.log_evaluation(period=100)]

def _group_counts(df: pd.DataFrame) -> np.ndarray:
    return df.groupby("session_id").size().astype(int).values

def _sort_for_grouping(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(["session_id","ts_hour","content_id_hashed"]).reset_index(drop=True)

def train_ltr_models(tr: pd.DataFrame, va: pd.DataFrame, feat_cols: list):
    # *** KRİTİK: LightGBM grup sırası için session bloklarına göre sırala
    tr = _sort_for_grouping(tr)
    va = _sort_for_grouping(va)

    # CLICK modeli
    dtr_click = lgb.Dataset(
        tr[feat_cols], label=tr["clicked"].astype(int), group=_group_counts(tr)
    )
    dva_click = lgb.Dataset(
        va[feat_cols], label=va["clicked"].astype(int), group=_group_counts(va)
    )
    params_click = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "eval_at": [5, 10, 20, 100],
        "boosting": "gbdt",
        "learning_rate": 0.03,
        "num_leaves": 95,
        "max_depth": -1,
        "min_data_in_leaf": 60,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "verbose": -1,
        "seed": 42,
        # deterministik + thread kontrol
        "num_threads": os.cpu_count(),
        "deterministic": True,
        "force_row_wise": True,  # bazı sürümlerde determinism'e yardım eder
    }
    m_click = lgb.train(
        params_click, dtr_click, num_boost_round=6000,
        valid_sets=[dtr_click, dva_click], valid_names=["train", "valid"],
        callbacks=_callbacks()
    )

    # ORDER modeli (paramlar aynı)
    dtr_order = lgb.Dataset(
        tr[feat_cols], label=tr["ordered"].astype(int), group=_group_counts(tr)
    )
    dva_order = lgb.Dataset(
        va[feat_cols], label=va["ordered"].astype(int), group=_group_counts(va)
    )
    params_order = params_click.copy()
    m_order = lgb.train(
        params_order, dtr_order, num_boost_round=6000,
        valid_sets=[dtr_order, dva_order], valid_names=["train", "valid"],
        callbacks=_callbacks()
    )

    return m_click, m_order

def ensure_feature_columns(df: pd.DataFrame, feat_cols: list) -> pd.DataFrame:
    out = df.copy()
    for c in feat_cols:
        if c not in out.columns: out[c] = 0.0
    return out[feat_cols]

def predict_rank_lgb(df: pd.DataFrame, model: lgb.Booster) -> np.ndarray:
    feat_cols = list(model.feature_name())
    X = ensure_feature_columns(df, feat_cols)
    return model.predict(X, num_iteration=getattr(model, "best_iteration", None))

def save_lgb(model: lgb.Booster, path:str): model.save_model(path)
def load_lgb(path:str) -> lgb.Booster: return lgb.Booster(model_file=path)

# ---------------------- XGBoost Ranker (opsiyonel) ----------------------
def get_numeric_feature_cols(df: pd.DataFrame) -> list:
    cols = []
    for c in df.columns:
        if c in LABEL_COLS or c in ID_COLS: continue
        if pd.api.types.is_numeric_dtype(df[c]): cols.append(c)
    return cols

def to_float32(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns: out[c] = out[c].astype("float32")
        else: out[c] = np.float32(0.0)
    return out

def make_group_sizes(df: pd.DataFrame) -> np.ndarray:
    return df.groupby("session_id").size().values.astype(int)

def map_group_ids(df: pd.DataFrame) -> np.ndarray:
    sid = df["session_id"].astype("category").cat.codes.values
    return sid.astype(np.int32)

def run_train_xgb():
    global args
    assert XGBRanker is not None, "xgboost kurulu değil: pip install xgboost"
    print("[TIMER] XGBRanker train ...")
    train = load_train_sessions()
    feats = assemble_timeaware_features(train, windows=(7, 30, 60))
    feat_cols = get_numeric_feature_cols(feats)
    tr, va = split_time_holdout(feats, holdout_days=7)
    if args.use_rel if 'args' in globals() else False:
        y_tr = build_relevance(tr).values
        y_va = build_relevance(va).values
    else:
        y_tr = tr["ordered"].values
        y_va = va["ordered"].values

    X_tr = to_float32(tr, feat_cols)[feat_cols].values
    X_va = to_float32(va, feat_cols)[feat_cols].values
    group_tr, group_va = make_group_sizes(tr), make_group_sizes(va)
    ranker = XGBRanker(
        objective="rank:ndcg",
        eval_metric=["ndcg@5","ndcg@10","ndcg@100"],
        tree_method="hist",
        max_depth=8, n_estimators=2400, learning_rate=0.055,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.0, reg_lambda=1.0,
        n_jobs=os.cpu_count(),
        random_state=42
    )
    ranker.fit(X_tr, y_tr, group=group_tr.tolist(),
               eval_set=[(X_va, y_va)], eval_group=[group_va.tolist()],
               verbose=True, early_stopping_rounds=400)
    os.makedirs(MODELS_DIR, exist_ok=True)
    ranker.save_model(os.path.join(MODELS_DIR,"xgb_rank.json"))
    pd.Series(feat_cols).to_csv(os.path.join(MODELS_DIR,"xgb_rank_features.txt"), index=False, header=False)
    try:
        s = ranker.get_booster().get_score(importance_type="gain")
        imp = pd.DataFrame({"feature": list(s.keys()), "gain": list(s.values())})
        imp.sort_values("gain", ascending=False).to_csv(os.path.join("outputs","imp_xgb.csv"), index=False)
    except Exception as e:
        print(f"[WARN] XGB importance alınamadı: {e}")

    print("[XGB] model saved")
    print("[TIMER] XGBRanker train done")

def run_infer_xgb(out_path: str):
    assert xgb is not None, "xgboost kurulu değil: pip install xgboost"
    print("[TIMER] XGBRanker infer ...")
    test = load_test_sessions()
    feats_te = assemble_timeaware_features(test, windows=(7, 30, 60))
    feat_cols = pd.read_csv(os.path.join(MODELS_DIR,"xgb_rank_features.txt"), header=None).iloc[:,0].tolist()
    X_te = to_float32(ensure_feature_columns(feats_te, feat_cols), feat_cols).values
    model = XGBRanker(); model.load_model(os.path.join(MODELS_DIR,"xgb_rank.json"))
    preds = model.predict(X_te)
    out = feats_te[["session_id","content_id_hashed"]].copy()
    out["pred_final"] = preds.astype("float32")
    out = normalize_in_session(out, "pred_final")
    idx = load_sample_submission_session_ids(os.path.join(DATA_DIR, "sample_submission.csv"))
    make_submission(out, out_path, session_index=idx)
    print("[TIMER] XGBRanker infer done")

# ---------------------- CatBoost YetiRank (opsiyonel) ----------------------
def run_train_cat():
    global args
    assert CatBoostRanker is not None, "catboost kurulu değil: pip install catboost"
    print("[TIMER] CatBoost YetiRank train ...")
    train = load_train_sessions()
    feats = assemble_timeaware_features(train, windows=(7, 30, 60))
    feat_cols = get_numeric_feature_cols(feats)
    tr, va = split_time_holdout(feats, holdout_days=7)
    tr = _sort_for_grouping(tr); va = _sort_for_grouping(va)
    X_tr = to_float32(ensure_feature_columns(tr, feat_cols), feat_cols)
    X_va = to_float32(ensure_feature_columns(va, feat_cols), feat_cols)
    if args.use_rel:
        y_tr = build_relevance(tr).values
        y_va = build_relevance(va).values
    else:
        y_tr = tr["ordered"].values
        y_va = va["ordered"].values

    grp_tr, grp_va = tr["session_id"].astype(str).values, va["session_id"].astype(str).values
    train_pool = Pool(X_tr, label=y_tr, group_id=grp_tr)
    valid_pool = Pool(X_va, label=y_va, group_id=grp_va)
    model = CatBoostRanker(
        loss_function="YetiRank", eval_metric="NDCG:top=10",
        iterations=4000, learning_rate=0.05, depth=8, l2_leaf_reg=3.0,
        random_strength=1.0, bootstrap_type="Bayesian",
        od_type="Iter", od_wait=400, verbose=100,
        thread_count=os.cpu_count(),
        random_seed=42,
    )
    model.fit(train_pool, eval_set=valid_pool, use_best_model=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    model.save_model(os.path.join(MODELS_DIR,"cb_rank.cbm"))
    pd.Series(feat_cols).to_csv(os.path.join(MODELS_DIR,"cb_rank_features.txt"), index=False, header=False)
    try:
        imp_vals = model.get_feature_importance(type="FeatureImportance")
        imp = pd.DataFrame({"feature": feat_cols, "importance": imp_vals})
        imp.sort_values("importance", ascending=False).to_csv(os.path.join("outputs","imp_cb.csv"), index=False)
    except Exception as e:
        print(f"[WARN] CatBoost importance alınamadı: {e}")
    print("[CatBoost] model saved")
    print("[TIMER] CatBoost YetiRank train done")

def run_infer_cat(out_path: str):
    assert CatBoostRanker is not None, "catboost kurulu değil: pip install catboost"
    print("[TIMER] CatBoost YetiRank infer ...")
    test = load_test_sessions()
    feats_te = assemble_timeaware_features(test, windows=(7, 30, 60))
    df = _sort_for_grouping(feats_te)
    feat_cols = pd.read_csv(os.path.join(MODELS_DIR,"cb_rank_features.txt"), header=None).iloc[:,0].tolist()
    X_te = to_float32(ensure_feature_columns(df, feat_cols), feat_cols)
    grp_te = df["session_id"].astype(str).values
    model = CatBoostRanker(); model.load_model(os.path.join(MODELS_DIR,"cb_rank.cbm"))
    preds = model.predict(Pool(X_te, group_id=grp_te))
    out = df[["session_id","content_id_hashed"]].copy()
    out["pred_final"] = preds.astype("float32")
    out = normalize_in_session(out, "pred_final")
    idx = load_sample_submission_session_ids(os.path.join(DATA_DIR, "sample_submission.csv"))
    make_submission(out, out_path, session_index=idx)
    print("[TIMER] CatBoost YetiRank infer done")

# --- METRICS: MAP@K ve NDCG@K ---
def _dcg_at_k(rels, k=100):
    rels = np.asfarray(rels)[:k]
    if rels.size == 0: return 0.0
    discounts = 1.0 / np.log2(np.arange(2, rels.size + 2))
    return float(np.sum((2.0 ** rels - 1.0) * discounts))

def ndcg_grouped(df: pd.DataFrame, label_col="ordered", score_col="pred_final", k=100) -> float:
    vals = []
    for _, g in df.groupby("session_id", sort=False):
        g = g.sort_values(score_col, ascending=False)
        rels = g[label_col].values
        if (rels > 0).sum() == 0:  # hiç pozitif yoksa dahil etme
            continue
        dcg = _dcg_at_k(rels, k)
        ideal = np.sort(rels)[::-1]
        idcg = _dcg_at_k(ideal, k)
        if idcg > 0: vals.append(dcg / idcg)
    return float(np.mean(vals)) if vals else 0.0

def mapk_grouped(df: pd.DataFrame, label_col="ordered", score_col="pred_final", k=100) -> float:
    vals = []
    for _, g in df.groupby("session_id", sort=False):
        g = g.sort_values(score_col, ascending=False)
        rels = (g[label_col].values > 0).astype(int)
        if rels.sum() == 0: continue
        hits = 0; ap = 0.0; n = min(k, len(rels))
        for i in range(n):
            if rels[i]:
                hits += 1
                ap += hits / (i + 1)
        ap /= min(rels.sum(), k)
        vals.append(ap)
    return float(np.mean(vals)) if vals else 0.0

def group_auc_weighted(df: pd.DataFrame, score_col="pred_final",
                       w_order: float = 0.7, w_click: float = 0.3):
    """
    Yarışma kurgusuna paralel: her session için ayrı ayrı AUC hesaplar.
    - click AUC: clicked label'ı olan (0/1) ve her iki sınıfı da içeren oturumlar
    - order AUC: ordered label'ı olan ve her iki sınıfı da içeren oturumlar
    Sonra ortalamaları alır, w_order ve w_click ile ağırlıklar.
    """
    auc_clicks, auc_orders = [], []
    for _, g in df.groupby("session_id", sort=False):
        # CLICK
        if "clicked" in g and g["clicked"].nunique() > 1:
            try:
                auc_clicks.append(roc_auc_score(g["clicked"].values, g[score_col].values))
            except Exception:
                pass
        # ORDER
        if "ordered" in g and g["ordered"].nunique() > 1:
            try:
                auc_orders.append(roc_auc_score(g["ordered"].values, g[score_col].values))
            except Exception:
                pass
    click_auc = float(np.mean(auc_clicks)) if auc_clicks else 0.0
    order_auc = float(np.mean(auc_orders)) if auc_orders else 0.0
    final = w_order * order_auc + w_click * click_auc
    return final, click_auc, order_auc, len(auc_clicks), len(auc_orders)


# --- rank 0..1 (oturum içi) ---
def _rank01_in_session(df: pd.DataFrame, col: str) -> pd.Series:
    r = df.groupby("session_id")[col].rank(method="first", ascending=False)
    n = df.groupby("session_id")[col].transform("count")
    return (n - r) / (n - 1 + 1e-9)  # 1=best, 0=worst

# ---------------------- submission yardımcıları ----------------------
def make_submission(scored: pd.DataFrame, out_path: str,
                    session_index: pd.Series | None = None,
                    expected_sessions: int | None = None):
    key = ["session_id","content_id_hashed"]
    scored = scored[key + ["pred_final"]].drop_duplicates(key)
    sub = (scored.sort_values(["session_id","pred_final"], ascending=[True, False])
                 .groupby("session_id")["content_id_hashed"]
                 .apply(lambda x: " ".join(x.astype(str))).reset_index())
    sub.columns = ["session_id","prediction"]
    sub["session_id"] = sub["session_id"].astype(str)
    if session_index is not None:
        sub = session_index.to_frame().merge(sub, on="session_id", how="left").fillna({"prediction": ""})
    if expected_sessions is not None:
        assert sub.shape[0] == expected_sessions
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    sub.to_csv(out_path, index=False)
    print(f"[OK] Submission written -> {out_path} (rows={len(sub):,})")

def validate_submission(sub_path: str, sample_path: str):
    sub = pd.read_csv(sub_path)
    ss  = pd.read_csv(sample_path, usecols=["session_id"])
    ok_rows = (len(sub) == len(ss))
    merged = ss.merge(sub, on="session_id", how="left")
    missing = merged["prediction"].isna().sum()
    dup_sessions = sub["session_id"].duplicated().sum()
    print(f"[CHECK] row_count_match={ok_rows} | missing_sessions={missing} | dup_sessions={dup_sessions}")
    if not ok_rows or missing or dup_sessions:
        print("[CHECK] Örnek problemli satırlar:")
        print(merged[merged["prediction"].isna()].head())


# ---------------------- Runnner'lar ----------------------
def run_baseline_timeaware(out_path: str):
    set_seed(42)
    with timer("baseline_timeaware"):
        train = load_train_sessions()
        feats_tr = assemble_timeaware_features(train, windows=(7,30,60))
        scored_tr = score_timeaware_baseline(feats_tr)
        tr, va = split_time_holdout(scored_tr, holdout_days=7)
        va = normalize_in_session(va, "pred_final")
        print("[INFO] TA valid prepared.")

        test = load_test_sessions()
        feats_te = assemble_timeaware_features(test, windows=(7,30,60))
        scored_te = score_timeaware_baseline(feats_te)
        scored_te = normalize_in_session(scored_te, "pred_final")

        idx = load_sample_submission_session_ids()
        make_submission(scored_te[["session_id","content_id_hashed","pred_final"]],
                        out_path, session_index=idx)

def run_train_ltr(alpha: float):
    set_seed(42)
    with timer("LTR train"):
        train = load_train_sessions()
        feats = assemble_timeaware_features(train, windows=(7,30,60))
        feat_cols = _feature_cols(feats)
        tr, va = split_time_holdout(feats, holdout_days=7)
        m_click, m_order = train_ltr_models(tr, va, feat_cols)

        # alpha taramasını gerçek metrikle logla
        for a in [0.70,0.74,0.78,0.80,0.82,0.85]:
            tmp = va.copy()
            tmp["pred_click"] = predict_rank_lgb(tmp, m_click)
            tmp["pred_order"] = predict_rank_lgb(tmp, m_order)
            tmp["pred_final"] = (1-a)*tmp["pred_click"] + a*tmp["pred_order"]
            tmp = normalize_in_session(tmp, "pred_final")
            score = ndcg_grouped(tmp, label_col="ordered", score_col="pred_final", k=100)
            print(f"[ALPHA] a={a:.2f} NDCG@100={score:.6f}")

        save_lgb(m_click, os.path.join(MODELS_DIR,"lgb_click.txt"))
        save_lgb(m_order, os.path.join(MODELS_DIR,"lgb_order.txt"))
        print("[LTR] models saved.")

def run_infer_ltr(out_path: str, alpha: float):
    set_seed(42)
    with timer("LTR infer"):
        test = load_test_sessions()
        feats_te = assemble_timeaware_features(test, windows=(7,30,60))
        m_click = load_lgb(os.path.join(MODELS_DIR,"lgb_click.txt"))
        m_order = load_lgb(os.path.join(MODELS_DIR,"lgb_order.txt"))
        df = feats_te.copy()
        df["pred_click"] = predict_rank_lgb(df, m_click)
        df["pred_order"] = predict_rank_lgb(df, m_order)
        df["pred_final"] = (1-alpha)*df["pred_click"] + alpha*df["pred_order"]
        df = normalize_in_session(df, "pred_final")
        idx = load_sample_submission_session_ids()
        make_submission(df[["session_id","content_id_hashed","pred_final"]],
                        out_path, session_index=idx)

def run_infer_blend(out_path: str, alpha: float, beta: float):
    set_seed(42)
    with timer("Blend infer"):
        test = load_test_sessions()
        feats = assemble_timeaware_features(test, windows=(7,30,60))

        # LTR
        m_click = load_lgb(os.path.join(MODELS_DIR,"lgb_click.txt"))
        m_order = load_lgb(os.path.join(MODELS_DIR,"lgb_order.txt"))
        ltr = feats.copy()
        ltr["pred_click"] = predict_rank_lgb(ltr, m_click)
        ltr["pred_order"] = predict_rank_lgb(ltr, m_order)
        ltr["pred_final"] = (1-alpha)*ltr["pred_click"] + alpha*ltr["pred_order"]
        ltr = normalize_in_session(ltr, "pred_final")
        ltr = ltr.rename(columns={"pred_final":"pred_ltr"})[["session_id","content_id_hashed","pred_ltr"]]

        # TA
        ta = score_timeaware_baseline(feats)
        ta = normalize_in_session(ta, "pred_final")
        ta = ta.rename(columns={"pred_final":"pred_ta"})[["session_id","content_id_hashed","pred_ta"]]

        # blend
        df = ltr.merge(ta, on=["session_id","content_id_hashed"], how="inner")
        df["pred_final"] = (1-beta)*df["pred_ltr"] + beta*df["pred_ta"]
        df = normalize_in_session(df, "pred_final")

        idx = load_sample_submission_session_ids()
        make_submission(df[["session_id","content_id_hashed","pred_final"]],
                        out_path, session_index=idx)

def run_infer_ensemble(out_path: str,
                       alpha_ltr: float = 0.80,
                       w_ltr: float = 0.50,
                       w_xgb: float = 0.25,
                       w_cb:  float = 0.25,
                       w_ta:  float = 0.15):
    print("[TIMER] Ensemble infer ...")
    ws = np.array([w_ltr, w_xgb, w_cb, w_ta], dtype="float64")
    if ws.sum() <= 0: ws = np.array([1.0, 0.0, 0.0, 0.0])
    ws = ws / ws.sum()
    w_ltr, w_xgb, w_cb, w_ta = ws.tolist()

    test = load_test_sessions()
    feats_te = assemble_timeaware_features(test, windows=(7, 30, 60))
    feats_te = feats_te.sort_values(["session_id","ts_hour","content_id_hashed"]).reset_index(drop=True)
    key = ["session_id","content_id_hashed"]

    # LTR
    try:
        m_click = load_lgb(os.path.join(MODELS_DIR,"lgb_click.txt"))
        m_order = load_lgb(os.path.join(MODELS_DIR,"lgb_order.txt"))
        ltr = feats_te[key].copy()
        ltr["pred_click"] = predict_rank_lgb(feats_te, m_click)
        ltr["pred_order"] = predict_rank_lgb(feats_te, m_order)
        ltr["pred_ltr"] = (1.0 - alpha_ltr) * ltr["pred_click"] + alpha_ltr * ltr["pred_order"]
        ltr = normalize_in_session(ltr, "pred_ltr")[key+["pred_ltr"]]
    except Exception as e:
        print(f"[WARN] LTR yüklenemedi: {e}")
        ltr = feats_te[key].copy(); ltr["pred_ltr"] = 0.0

    # XGB
    try:
        xgb_feat_cols = pd.read_csv(os.path.join(MODELS_DIR,"xgb_rank_features.txt"), header=None).iloc[:,0].tolist()
        xgb_model = XGBRanker(); xgb_model.load_model(os.path.join(MODELS_DIR,"xgb_rank.json"))
        fe_mat = to_float32(ensure_feature_columns(feats_te, xgb_feat_cols), xgb_feat_cols)
        preds_xgb = xgb_model.predict(fe_mat.values)
        xgb_df = feats_te[key].copy()
        xgb_df["pred_xgb"] = preds_xgb.astype("float32")
        xgb_df = normalize_in_session(xgb_df, "pred_xgb")[key+["pred_xgb"]]
    except Exception as e:
        print(f"[WARN] XGB yüklenemedi: {e}")
        xgb_df = feats_te[key].copy(); xgb_df["pred_xgb"] = 0.0

    # CatBoost
    try:
        cb_feat_cols = pd.read_csv(os.path.join(MODELS_DIR,"cb_rank_features.txt"), header=None).iloc[:,0].tolist()
        X_cb = ensure_feature_columns(to_float32(feats_te, cb_feat_cols), cb_feat_cols)
        cb_model = CatBoostRanker(); cb_model.load_model(os.path.join(MODELS_DIR,"cb_rank.cbm"))
        grp_te = map_group_ids(feats_te)
        preds_cb = cb_model.predict(Pool(X_cb, group_id=grp_te))
        cb_df = feats_te[key].copy()
        cb_df["pred_cb"] = preds_cb.astype("float32")
        cb_df = normalize_in_session(cb_df, "pred_cb")[key+["pred_cb"]]
    except Exception as e:
        print(f"[WARN] CatBoost yüklenemedi: {e}")
        cb_df = feats_te[key].copy(); cb_df["pred_cb"] = 0.0

    # TA
    ta = score_timeaware_baseline(feats_te)
    ta = normalize_in_session(ta[key + ["pred_final"]].rename(columns={"pred_final":"pred_ta"}), "pred_ta")

    # combine (rank-average)
    df = ltr.merge(xgb_df, on=key, how="left") \
            .merge(cb_df,  on=key, how="left") \
            .merge(ta,     on=key, how="left")

    for c in ["pred_ltr","pred_xgb","pred_cb","pred_ta"]:
        if c not in df.columns: df[c] = 0.0
        df[c + "_r01"] = _rank01_in_session(df, c)

    df["pred_final"] = (
        w_ltr * df["pred_ltr_r01"] +
        w_xgb * df["pred_xgb_r01"] +
        w_cb  * df["pred_cb_r01"]  +
        w_ta  * df["pred_ta_r01"]
    ).astype("float32")

    # küçük “tekrar bonusları”
    missing_cols = [
        "clicked_before_item_sess","added_to_cart_before_item_sess","added_to_fav_before_item_sess",
    ]
    for c in missing_cols:
        if c not in feats_te.columns: feats_te[c] = 0.0
    if "seen_before" not in feats_te.columns: feats_te["seen_before"] = 0.0
    sel_cols = key + ["seen_before"] + missing_cols
    feats_te[sel_cols] = feats_te[sel_cols].fillna(0)
    df = df.merge(feats_te[sel_cols], on=key, how="left")

    df["repeat_bonus"] = (
        0.02 * df["seen_before"].fillna(0) +
        0.04 * (df["clicked_before_item_sess"].fillna(0) > 0).astype("float32") +
        0.08 * (df["added_to_cart_before_item_sess"].fillna(0) > 0).astype("float32") +
        0.05 * (df["added_to_fav_before_item_sess"].fillna(0) > 0).astype("float32")
    ).astype("float32")

    df["pred_final"] = (df["pred_final"] + df["repeat_bonus"]).astype("float32")
    df = normalize_in_session(df, "pred_final")

    idx = load_sample_submission_session_ids(os.path.join(DATA_DIR, "sample_submission.csv"))
    make_submission(df, out_path, session_index=idx)
    print(f"[TIMER] Ensemble infer done (w_ltr={w_ltr:.2f}, w_xgb={w_xgb:.2f}, w_cb={w_cb:.2f}, w_ta={w_ta:.2f})")

def run_offline_eval(alpha_ltr: float, w_ltr: float, w_xgb: float, w_cb: float, w_ta: float,
                     metric: str = "ndcg", k: int = 100):
    print("[TIMER] Offline eval ...]")
    train = load_train_sessions()
    feats = assemble_timeaware_features(train, windows=(7, 30, 60))
    feats = _sort_for_grouping(feats)
    tr, va = split_time_holdout(feats, holdout_days=7)
    va = va.copy()
    key = ["session_id","content_id_hashed"]

    # LTR
    try:
        m_click = load_lgb(os.path.join(MODELS_DIR,"lgb_click.txt"))
        m_order = load_lgb(os.path.join(MODELS_DIR,"lgb_order.txt"))
        va["pred_click"] = predict_rank_lgb(va, m_click)
        va["pred_order"] = predict_rank_lgb(va, m_order)
        va["pred_ltr"]   = (1.0 - alpha_ltr) * va["pred_click"] + alpha_ltr * va["pred_order"]
    except Exception as e:
        print(f"[WARN] LTR yok: {e}"); va["pred_ltr"] = 0.0

    # XGB
    try:
        xgb_feat_cols = pd.read_csv(os.path.join(MODELS_DIR,"xgb_rank_features.txt"), header=None).iloc[:,0].tolist()
        X = to_float32(ensure_feature_columns(va, xgb_feat_cols), xgb_feat_cols).values
        xm = XGBRanker(); xm.load_model(os.path.join(MODELS_DIR,"xgb_rank.json"))
        va["pred_xgb"] = xm.predict(X).astype("float32")
    except Exception as e:
        print(f"[WARN] XGB yok: {e}"); va["pred_xgb"] = 0.0

    # CB
    try:
        cb_feat_cols = pd.read_csv(os.path.join(MODELS_DIR,"cb_rank_features.txt"), header=None).iloc[:,0].tolist()
        Xcb = ensure_feature_columns(to_float32(va, cb_feat_cols), cb_feat_cols)
        grp = map_group_ids(va)
        cbm = CatBoostRanker(); cbm.load_model(os.path.join(MODELS_DIR,"cb_rank.cbm"))
        va["pred_cb"] = cbm.predict(Pool(Xcb, group_id=grp)).astype("float32")
    except Exception as e:
        print(f"[WARN] CB yok: {e}"); va["pred_cb"] = 0.0

    # TA
    ta = score_timeaware_baseline(va)
    va["pred_ta"] = ta["pred_final"].values

    for c in ["pred_ltr","pred_xgb","pred_cb","pred_ta"]:
        va[c + "_r01"] = _rank01_in_session(va, c)
    va["pred_final"] = (
        w_ltr * va["pred_ltr_r01"] +
        w_xgb * va["pred_xgb_r01"] +
        w_cb  * va["pred_cb_r01"]  +
        w_ta  * va["pred_ta_r01"]
    ).astype("float32")

    metric_l = metric.lower()
    if metric_l == "map":
        score = mapk_grouped(va, label_col="ordered", score_col="pred_final", k=k)
        print(f"[OFFLINE] MAP@{k}: {score:.6f}")
    elif metric_l == "auc":
        final, click_auc, order_auc, n_click, n_order = group_auc_weighted(va, score_col="pred_final")
        print(f"[OFFLINE] GroupAUC (weighted) = {final:.6f} | "
              f"click_auc={click_auc:.6f} (sessions={n_click}) | "
              f"order_auc={order_auc:.6f} (sessions={n_order})")
    else:
        score = ndcg_grouped(va, label_col="ordered", score_col="pred_final", k=k)
        print(f"[OFFLINE] NDCG@{k}: {score:.6f}")

    print("[TIMER] Offline eval done")

# ---------------------- CLI ----------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline_timeaware", action="store_true")
    ap.add_argument("--train_ltr", action="store_true")
    ap.add_argument("--infer_ltr", action="store_true")
    ap.add_argument("--infer_blend", action="store_true")

    ap.add_argument("--train_xgb", action="store_true", help="XGBoost Ranker eğit")
    ap.add_argument("--infer_xgb", action="store_true", help="XGBoost Ranker infer")
    ap.add_argument("--train_cat", action="store_true", help="CatBoost YetiRank eğit")
    ap.add_argument("--infer_cat", action="store_true", help="CatBoost YetiRank infer")

    ap.add_argument("--infer_ensemble", action="store_true", help="LTR+XGB+CB+TA ensemble infer")
    ap.add_argument("--w_ltr", type=float, default=0.50)
    ap.add_argument("--w_xgb", type=float, default=0.25)
    ap.add_argument("--w_cb",  type=float, default=0.25)
    ap.add_argument("--w_ta",  type=float, default=0.15)

    ap.add_argument("--offline_eval", action="store_true")
    ap.add_argument("--metric", type=str, default="ndcg", choices=["ndcg","map","auc"])
    ap.add_argument("--k", type=int, default=100)

    ap.add_argument("--alpha", type=float, default=0.80, help="LTR içi order ağırlığı")
    ap.add_argument("--beta",  type=float, default=0.30, help="Blend ağırlığı: LTR (1-beta) ⊕ TA (beta)")
    ap.add_argument("--out",   type=str, default="outputs/submission.csv")
    ap.add_argument("--use_rel", action="store_true", help="XGB/CB eğitiminde ordered yerine relevance label kullan")
    ap.add_argument("--check_sub", type=str, help="Kontrol edilecek submission dosyası yolu")

    args = ap.parse_args()
    if args.check_sub:
        validate_submission(args.check_sub, os.path.join(DATA_DIR, "sample_submission.csv"))
    elif args.baseline_timeaware:
        run_baseline_timeaware(args.out)
    elif args.train_ltr:
        run_train_ltr(alpha=args.alpha)
    elif args.infer_ltr:
        run_infer_ltr(args.out, alpha=args.alpha)
    elif args.infer_blend:
        run_infer_blend(args.out, alpha=args.alpha, beta=args.beta)
    elif args.train_xgb:
        run_train_xgb()
    elif args.infer_xgb:
        run_infer_xgb(args.out)
    elif args.train_cat:
        run_train_cat()
    elif args.infer_cat:
        run_infer_cat(args.out)
    elif args.infer_ensemble:
        run_infer_ensemble(
            out_path=args.out,
            alpha_ltr=args.alpha,
            w_ltr=args.w_ltr,
            w_xgb=args.w_xgb,
            w_cb=args.w_cb,
            w_ta=args.w_ta
        )
    elif args.offline_eval:
        run_offline_eval(
            alpha_ltr=args.alpha,
            w_ltr=args.w_ltr, w_xgb=args.w_xgb, w_cb=args.w_cb, w_ta=args.w_ta,
            metric=args.metric, k=args.k
        )
    else:
        print("Örnekler:\n"
              "  python one_run_v9.py --baseline_timeaware --out outputs/ta.csv\n"
              "  python one_run_v9.py --train_ltr --alpha 0.80\n"
              "  python one_run_v9.py --infer_ltr --alpha 0.80 --out outputs/sub_ltr.csv\n"
              "  python one_run_v9.py --train_xgb\n"
              "  python one_run_v9.py --infer_xgb --out outputs/sub_xgb.csv\n"
              "  python one_run_v9.py --train_cat\n"
              "  python one_run_v9.py --infer_cat --out outputs/sub_cat.csv\n"
              "  python one_run_v9.py --infer_blend --alpha 0.80 --beta 0.30 --out outputs/sub_blend.csv\n"
              "  python one_run_v9.py --infer_ensemble --alpha 0.85 --w_ltr 0.70 --w_xgb 0.15 --w_cb 0.15 --w_ta 0.00 --out outputs/sub_ens.csv")
