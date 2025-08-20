# -*- coding: utf-8 -*-
"""
one_run_v5.py — Tek dosyalık pipeline

Komutlar (örnekler en altta):
  --baseline_timeaware                        : TA özellikleri + TA baseline ile submission
  --train_ltr --alpha 0.80                    : TA özelliklerden LTR (click & order) eğit
  --infer_ltr  --alpha 0.80 --out out.csv     : Kaydedilen LTR modellerle submission
  --infer_blend --alpha 0.80 --beta 0.30      : LTR ⊕ TA blend submission
  --train_xgb / --infer_xgb                   : XGBoost Ranker eğitim / infer
  --train_cat / --ianfer_cat                   : CatBoost YetiRank eğitim / infer
  --infer_ensemble [w_ltr w_xgb w_cb w_ta]    : LTR + XGB + CatBoost + TA ensemble
"""

import os, time, argparse, numpy as np, pandas as pd
import duckdb
import lightgbm as lgb

# XGBoost & CatBoost opsiyonel — sistemde yoksa import hatası vermesin
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
DATA_DIR = "data"
MODELS_DIR = "models"
os.makedirs("outputs", exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

class timer:
    def __init__(self, msg): self.msg = msg
    def __enter__(self): self.t0=time.time(); print(f"[TIMER] {self.msg} ..."); return self
    def __exit__(self, *a): print(f"[TIMER] {self.msg} done in {time.time()-self.t0:.2f}s")

def set_seed(s=42):
    np.random.seed(s)

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

def make_submission(scored: pd.DataFrame, out_path: str,
                    session_index: pd.Series | None = None,
                    expected_sessions: int | None = None):
    """scored: ['session_id','content_id_hashed','pred_final']"""
    sub = (scored.sort_values(["session_id","pred_final"], ascending=[True, False])
                 .groupby("session_id")["content_id_hashed"]
                 .apply(lambda x: " ".join(x.astype(str))).reset_index())
    sub.columns = ["session_id","prediction"]
    sub["session_id"] = sub["session_id"].astype(str)

    if session_index is not None:
        sub = session_index.to_frame().merge(sub, on="session_id", how="left").fillna({"prediction": ""})
    if expected_sessions is not None:
        assert sub.shape[0] == expected_sessions, f"Submission rows {sub.shape[0]} != {expected_sessions}"

    sub.to_csv(out_path, index=False)
    print(f"[OK] Submission written -> {out_path} (rows={len(sub):,})")


# ---------------------- time-aware özellik inşası (DuckDB) ----------------------
def _mk_roll(win: int, alias_prefix: str, col: str, part_cols: str) -> str:
    return (f"SUM({col}) OVER (PARTITION BY {part_cols} ORDER BY d "
            f"RANGE BETWEEN INTERVAL {win} DAY PRECEDING AND CURRENT ROW) "
            f"AS {alias_prefix}_{win}d")
def _safe_get_column(df: pd.DataFrame, col_name: str, default_value: float = 0.0) -> pd.Series:
    """DataFrame'den güvenli sütun erişimi - yoksa default_value ile Series döner"""
    if col_name in df.columns:
        return df[col_name].fillna(default_value)
    else:
        return pd.Series(default_value, index=df.index, dtype='float64')


def assemble_timeaware_features(sessions: pd.DataFrame, windows=(3,7,14,30,60)) -> pd.DataFrame:
    print("[TA] DuckDB builder -> start")
    need = [c for c in [
        "ts_hour","search_term_normalized","content_id_hashed","session_id",
        "clicked","ordered","added_to_cart","added_to_fav","user_id_hashed"
    ] if c in sessions.columns]
    s = sessions[need].copy()
    s["ts_hour"] = pd.to_datetime(s["ts_hour"], utc=False)
    s["session_date"] = s["ts_hour"].dt.floor("D")

    # Basit fallback: eğer memory/DuckDB sorunları varsa sadece temel feature'ları döndür
    try:
        con = duckdb.connect()
        try:
            n = max(1, min(os.cpu_count() or 4, 16))
            con.execute(f"PRAGMA threads={n};")
        except Exception:
            pass
        con.register("sessions_df", s)

        # Burada gerçek DuckDB SQL'leri gelir...
        # Eğer herhangi bir hata olursa aşağıdaki except bloğuna düşer
        
        # Şimdilik basit bir test yapalım
        test_result = con.execute("SELECT COUNT(*) FROM sessions_df").fetchone()
        print(f"[TA] DuckDB test successful: {test_result[0]} rows")
        
        # Gerçek feature engineering burada olacak
        # Şimdilik sadece giriş verilerini döndürelim
        out = s.copy()
        
        # Temel trend feature'ları ekle
        for col in ["tc_ctr_7d", "tc_ctr_14d", "tc_ctr_30d", "order_rate_7d", "order_rate_14d", "order_rate_30d"]:
            if col not in out.columns:
                out[col] = 0.0
        
        # Yeni feature'lar için placeholder
        for col in ["discount_pct", "rating_avg", "user_term_ctr_30d", 
                   "user_content_fashion_order_rate_30d", "content_search_ctr_7d"]:
            if col not in out.columns:
                out[col] = 0.0
                
    except Exception as e:
        print(f"[TA] DuckDB error: {e}, falling back to basic features")
        out = s.copy()
        
        # Minimum feature set
        basic_features = ["tc_ctr_7d", "tc_ctr_14d", "tc_ctr_30d", "order_rate_7d", 
                         "order_rate_14d", "order_rate_30d", "discount_pct", "rating_avg",
                         "user_term_ctr_30d", "user_content_fashion_order_rate_30d", 
                         "content_search_ctr_7d"]
        
        for feat in basic_features:
            if feat not in out.columns:
                out[feat] = np.random.random(len(out)) * 0.1  # Küçük random değerler
    
    # In-session feature'ları ekle
    try:
        out = add_in_session_features(out) 
    except Exception as e:
        print(f"[TA] In-session features error: {e}")
        # Temel in-session feature'lar
        if "sess_step_idx" not in out.columns:
            out["sess_step_idx"] = out.groupby("session_id").cumcount().astype("int32")
        if "seen_before" not in out.columns:
            out["seen_before"] = 0

    # Trend sinyalleri
    out["trend_order_rate_7v30"] = _safe_get_column(out, "order_rate_7d") - _safe_get_column(out, "order_rate_30d")
    out["trend_tc_ctr_7v30"] = _safe_get_column(out, "tc_ctr_7d") - _safe_get_column(out, "tc_ctr_30d")

    out = reduce_memory_df(out)
    print("[TA] DuckDB builder -> done")
    return out



def add_in_session_features(df: pd.DataFrame) -> pd.DataFrame:
    # deterministik sıra
    df = df.sort_values(["session_id","ts_hour","content_id_hashed"]).reset_index(drop=True)

    # oturum adımı (0,1,2,...) ve item tekrar indeksi (aynı item oturumda kaçıncı kez)
    df["sess_step_idx"] = df.groupby("session_id").cumcount().astype("int32")
    g_item = df.groupby(["session_id","content_id_hashed"], sort=False)
    df["item_occ_idx"] = g_item.cumcount().astype("int16")
    df["seen_before"]  = (df["item_occ_idx"] > 0).astype("int8")

    # bu item oturumda en son ne zaman/kaç adım önce görüldü
    prev_step = g_item["sess_step_idx"].shift()
    df["steps_since_item_last_seen"] = (df["sess_step_idx"] - prev_step).fillna(1e6).clip(0, 1e6).astype("float32")
    prev_time = g_item["ts_hour"].shift()
    df["secs_since_item_last_seen"]  = (df["ts_hour"] - prev_time).dt.total_seconds().fillna(1e9).clip(0, 1e9).astype("float32")

    # bu item için geçmiş etkileşim sayıları (mevcut satır HARİÇ)
    for lab in ["clicked","added_to_cart","added_to_fav","ordered"]:
        if lab in df.columns:
            cum = g_item[lab].cumsum()
            df[f"{lab}_before_item_sess"] = (cum - df[lab]).astype("int16")

    # oturum süresi (sn) ve oturum başlangıcından bu yana adım
    sess_start = df.groupby("session_id")["ts_hour"].transform("min")
    df["secs_since_session_start"] = (df["ts_hour"] - sess_start).dt.total_seconds().astype("float32")
    
    # mevcut liste: ["order_rate_7d","tc_ctr_7d","discount_pct","rating_avg"]
    for base in ["order_rate_7d","tc_ctr_7d","discount_pct","rating_avg",
                 "log_discounted_price","trend_order_7v30","trend_tcctr_7v30"]:
        if base in df.columns:
            df[f"{base}_rank_sess"] = df.groupby("session_id")[base].rank(method="first", pct=True).astype("float32")
        # ucuzluk ve indirim rütbeleri (oturum içi)
        if "log_discounted_price" in df.columns:
            df["cheap_rank_sess"] = 1.0 - df.groupby("session_id")["log_discounted_price"]\
                .rank(method="first", pct=True)  # 1 = en ucuz

        if "discount_pct" in df.columns:
            df["discount_rank_sess"] = df.groupby("session_id")["discount_pct"]\
                .rank(method="first", ascending=False, pct=True)  # 1 = en yüksek indirim

    
    # oturum-içi göreli rütbeler (global öncülleri oturum içinde normalize et)
    for base in ["order_rate_7d","tc_ctr_7d","discount_pct","rating_avg"]:
        if base in df.columns:
            df[f"{base}_rank_sess"] = df.groupby("session_id")[base].rank(method="first", pct=True).astype("float32")
        # ucuzluk ve indirim rütbeleri (oturum içi)
        if "log_discounted_price" in df.columns:
            df["cheap_rank_sess"] = 1.0 - df.groupby("session_id")["log_discounted_price"]\
                .rank(method="first", pct=True)  # 1 = en ucuz

        if "discount_pct" in df.columns:
            df["discount_rank_sess"] = df.groupby("session_id")["discount_pct"]\
                .rank(method="first", ascending=False, pct=True)  # 1 = en yüksek indirim

    return df


# ---------------------- basit TA baseline ----------------------
def score_timeaware_baseline(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["pred_click"] = 0.60*out.get("tc_ctr_7d", pd.Series(0, index=out.index)) + 0.25*out.get("tc_ctr_30d", pd.Series(0, index=out.index)) + 0.15*out.get("tc_ctr_14d", pd.Series(0, index=out.index))
    out["pred_order"] = 0.65*out.get("order_rate_7d", pd.Series(0, index=out.index)) + 0.20*out.get("order_rate_30d", pd.Series(0, index=out.index)) + 0.15*out.get("order_rate_14d", pd.Series(0, index=out.index))
    
    # Güvenli sütun erişimi
    discount_pct = out.get("discount_pct", pd.Series(0.0, index=out.index))
    if isinstance(discount_pct, (int, float)):
        discount_pct = pd.Series(discount_pct, index=out.index)
    
    rating_avg = out.get("rating_avg", pd.Series(0.0, index=out.index))
    if isinstance(rating_avg, (int, float)):
        rating_avg = pd.Series(rating_avg, index=out.index)
    
    user_term_ctr = out.get("user_term_ctr_30d", pd.Series(0.0, index=out.index))
    if isinstance(user_term_ctr, (int, float)):
        user_term_ctr = pd.Series(user_term_ctr, index=out.index)
    
    # Yeni feature'lar için güvenli erişim
    user_content_fashion_order_rate = out.get("user_content_fashion_order_rate_30d", pd.Series(0.0, index=out.index))
    if isinstance(user_content_fashion_order_rate, (int, float)):
        user_content_fashion_order_rate = pd.Series(user_content_fashion_order_rate, index=out.index)
    
    content_search_ctr = out.get("content_search_ctr_7d", pd.Series(0.0, index=out.index))
    if isinstance(content_search_ctr, (int, float)):
        content_search_ctr = pd.Series(content_search_ctr, index=out.index)
    
    out["pred_order"] += 0.08*discount_pct.clip(lower=0).fillna(0)
    out["pred_order"] += 0.04*(rating_avg.fillna(0)/5.0)
    out["pred_click"] += 0.05*user_term_ctr.fillna(0)
    
    # Yeni feature'lar
    out["pred_order"] += 0.03*user_content_fashion_order_rate.fillna(0)
    out["pred_click"] += 0.02*content_search_ctr.fillna(0)
    
    out["pred_final"] = 0.85*out["pred_order"] + 0.15*out["pred_click"]
    return out

def normalize_in_session(df: pd.DataFrame, score_col="pred_final") -> pd.DataFrame:
    out = df.copy()
    grp = out.groupby("session_id")[score_col]
    out[score_col] = (out[score_col] - grp.transform("min")) / (grp.transform("max") - grp.transform("min") + 1e-8)
    return out


# ---------------------- yardımcılar: feature list / split ----------------------
# BUNU GÜNCELLE
LABEL_COLS = {"clicked", "ordered", "added_to_cart", "added_to_fav"}  # << eklendi
ID_COLS = {
    "session_id", "content_id_hashed", "ts_hour", "session_date",
    "search_term_normalized", "user_id_hashed"
}


def _feature_cols(df: pd.DataFrame) -> list:
    cols = []
    for c in df.columns:
        if c in LABEL_COLS or c in ID_COLS: continue
        if pd.api.types.is_numeric_dtype(df[c]): cols.append(c)
    return cols

def split_time_holdout(df: pd.DataFrame, holdout_days=7, fallback_q=0.8):
    ts = pd.to_datetime(df["ts_hour"])
    cutoff = ts.max().normalize() - pd.Timedelta(days=holdout_days-1)
    tr = df[ts < cutoff].copy(); va = df[ts >= cutoff].copy()
    if len(tr)==0 or len(va)==0:
        q = ts.quantile(fallback_q); tr=df[ts<q].copy(); va=df[ts>=q].copy()
        print(f"[SPLIT] Fallback quantile used at {q}")
    else:
        print(f"[SPLIT] cutoff={cutoff.date()}  train={len(tr):,} rows  valid={len(va):,} rows")
    return tr, va


# ---------------------- LTR (LightGBM LambdaRank) ----------------------
def _callbacks():
    return [lgb.early_stopping(stopping_rounds=600, verbose=True),
            lgb.log_evaluation(period=50)]

def _group_counts(df: pd.DataFrame) -> np.ndarray:
    return df.groupby("session_id").size().astype(int).values

def train_ltr_models(tr: pd.DataFrame, va: pd.DataFrame, feat_cols: list):
    # click
    dtr = lgb.Dataset(tr[feat_cols], label=tr["clicked"].astype(int), group=_group_counts(tr))
    dva = lgb.Dataset(va[feat_cols], label=va["clicked"].astype(int), group=_group_counts(va))
    params_click = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "eval_at": [5, 10, 20, 100],
        "boosting": "gbdt",
        "learning_rate": 0.045,
        "num_leaves": 127,
        "max_depth": -1,
        "min_data_in_leaf": 25,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.75,
        "bagging_freq": 1,
        "reg_alpha": 0.1,
        "reg_lambda": 1.2,
        "verbose": -1,
    }
    m_click = lgb.train(params_click, dtr, num_boost_round=10000,
                        valid_sets=[dtr, dva], valid_names=["train", "valid"],
                        callbacks=_callbacks())
    # order
    dtr = lgb.Dataset(tr[feat_cols], label=tr["ordered"].astype(int), group=_group_counts(tr))
    dva = lgb.Dataset(va[feat_cols], label=va["ordered"].astype(int), group=_group_counts(va))
    params_order = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "eval_at": [5, 10, 20, 100],
        "boosting": "gbdt",
        "learning_rate": 0.045,
        "num_leaves": 127,
        "max_depth": -1,
        "min_data_in_leaf": 25,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.75,
        "bagging_freq": 1,
        "reg_alpha": 0.1,
        "reg_lambda": 1.2,
        "verbose": -1,
    }
    m_order = lgb.train(params_order, dtr, num_boost_round=10000,
                        valid_sets=[dtr, dva], valid_names=["train", "valid"],
                        callbacks=_callbacks())
    return m_click, m_order

def predict_rank_lgb(df: pd.DataFrame, model: lgb.Booster) -> np.ndarray:
    feat_cols = list(model.feature_name())
    X = ensure_feature_columns(df, feat_cols)  # << eksikler 0.0 ile tamamlanır
    return model.predict(X, num_iteration=getattr(model, "best_iteration", None))


def save_lgb(model: lgb.Booster, path:str): model.save_model(path)
def load_lgb(path:str) -> lgb.Booster: return lgb.Booster(model_file=path)


# ---------------------- XGBoost Ranker ----------------------
def get_numeric_feature_cols(df: pd.DataFrame) -> list:
    cols = []
    for c in df.columns:
        if c in LABEL_COLS or c in ID_COLS: continue
        if pd.api.types.is_numeric_dtype(df[c]): cols.append(c)
    return cols

def to_float32(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = out[c].astype("float32")
        else:
            # modele beklenen ama DF'te olmayan feature'ları 0.0 ile oluştur
            out[c] = np.float32(0.0)
    return out


def ensure_feature_columns(df: pd.DataFrame, feat_cols: list) -> pd.DataFrame:
    """Modelin beklediği her feature sütunu DataFrame’de varsa kullan, yoksa 0.0 ile oluştur."""
    out = df.copy()
    for c in feat_cols:
        if c not in out.columns:
            out[c] = 0.0
    # fazladan kolonlar model tarafından okunmaz; sadece sırayı sabitliyoruz
    return out[feat_cols]


def build_relevance(df: pd.DataFrame) -> pd.Series:
    o = df.get("ordered", 0).fillna(0).astype(int)
    c = df.get("clicked", 0).fillna(0).astype(int)
    a = df.get("added_to_cart", 0).fillna(0).astype(int)
    f = df.get("added_to_fav", 0).fillna(0).astype(int)
    # Tümü pozitif TAM SAYI: order > cart > fav > click
    rel = (4*o + 3*a + 2*f + 1*c).astype("int32")
    return pd.Series(rel, index=df.index)



def make_group_sizes(df: pd.DataFrame) -> np.ndarray:
    return df.groupby("session_id").size().values.astype(int)

def map_group_ids(df: pd.DataFrame) -> np.ndarray:
    sid = df["session_id"].astype("category").cat.codes.values
    return sid.astype(np.int32)

def run_train_xgb():
    assert XGBRanker is not None, "xgboost kurulu değil: pip install xgboost"
    print("[TIMER] XGBRanker train ...")
    train = load_train_sessions()
    feats = assemble_timeaware_features(train, windows=(3,7,14,30,60))
    feat_cols = get_numeric_feature_cols(feats)

    tr, va = split_time_holdout(feats, holdout_days=7)

    y_tr = build_relevance(tr).values
    y_va = build_relevance(va).values
    X_tr = to_float32(tr, feat_cols)[feat_cols].values
    X_va = to_float32(va, feat_cols)[feat_cols].values

    group_tr = make_group_sizes(tr)
    group_va = make_group_sizes(va)

    ranker = XGBRanker(
        objective="rank:ndcg",
        eval_metric=["ndcg@5","ndcg@10","ndcg@100"],
        tree_method="hist",
        max_depth=12,
        n_estimators=6000,
        learning_rate=0.04,
        subsample=0.75,
        colsample_bytree=0.75,
        reg_alpha=0.1,
        reg_lambda=1.5,
        random_state=42,
    )
    ranker.fit(
        X_tr, y_tr,
        group=group_tr.tolist(),
        eval_set=[(X_va, y_va)],
        eval_group=[group_va.tolist()],
        verbose=True,
        early_stopping_rounds=600,
    )


    os.makedirs("models", exist_ok=True)
    ranker.save_model("models/xgb_rank.json")
    pd.Series(feat_cols).to_csv("models/xgb_rank_features.txt", index=False, header=False)
    print("[XGB] model saved -> models/xgb_rank.json & xgb_rank_features.txt")
    print("[TIMER] XGBRanker train done")

def run_infer_xgb(out_path: str):
    assert xgb is not None, "xgboost kurulu değil: pip install xgboost"
    print("[TIMER] XGBRanker infer ...")
    test = load_test_sessions()
    feats_te = assemble_timeaware_features(test, windows=(3,7,14,30,60))

    feat_cols = pd.read_csv("models/xgb_rank_features.txt", header=None).iloc[:,0].tolist()
    X_te = to_float32(ensure_feature_columns(feats_te, feat_cols), feat_cols).values

    model = XGBRanker()
    model.load_model("models/xgb_rank.json")
    preds = model.predict(X_te)

    out = feats_te[["session_id","content_id_hashed"]].copy()
    out["pred_final"] = preds.astype("float32")
    out = normalize_in_session(out, "pred_final")

    idx = load_sample_submission_session_ids(os.path.join(DATA_DIR, "sample_submission.csv"))
    make_submission(out, out_path, session_index=idx)
    print("[TIMER] XGBRanker infer done")


# ---------------------- CatBoost YetiRank ----------------------
def run_train_cat():
    assert CatBoostRanker is not None, "catboost kurulu değil: pip install catboost"
    print("[TIMER] CatBoost YetiRank train ...")
    train = load_train_sessions()
    feats = assemble_timeaware_features(train, windows=(3,7,14,30,60))
    feat_cols = get_numeric_feature_cols(feats)

    tr, va = split_time_holdout(feats, holdout_days=7)
    tr = _sort_for_grouping(tr)
    va = _sort_for_grouping(va)

    # feature matrisi
    X_tr = to_float32(ensure_feature_columns(tr, feat_cols), feat_cols)
    X_va = to_float32(ensure_feature_columns(va, feat_cols), feat_cols)

    # label & group (string session_id de verebilirsin)
    y_tr = build_relevance(tr).values
    y_va = build_relevance(va).values
    grp_tr = tr["session_id"].astype(str).values
    grp_va = va["session_id"].astype(str).values

    train_pool = Pool(X_tr, label=y_tr, group_id=grp_tr)
    valid_pool = Pool(X_va, label=y_va, group_id=grp_va)

    model = CatBoostRanker(
        loss_function="YetiRank",
        eval_metric="NDCG:top=10",
        iterations=8000,
        learning_rate=0.04,
        depth=10,
        l2_leaf_reg=2.0,
        random_strength=1.5,
        bootstrap_type="Bayesian",
        bagging_temperature=0.5,
        od_type="Iter",
        od_wait=800,
        verbose=50,
        random_seed=42,
    )

    model.fit(train_pool, eval_set=valid_pool, use_best_model=True)
    os.makedirs("models", exist_ok=True)
    model.save_model("models/cb_rank.cbm")
    pd.Series(feat_cols).to_csv("models/cb_rank_features.txt", index=False, header=False)
    print("[CatBoost] model saved -> models/cb_rank.cbm & cb_rank_features.txt")
    print("[TIMER] CatBoost YetiRank train done")


def make_submission(scored: pd.DataFrame, out_path: str,
                    session_index: pd.Series | None = None,
                    expected_sessions: int | None = None):
    key = ["session_id","content_id_hashed"]
    # sadece kandidat set + tekille
    scored = scored[key + ["pred_final"]].drop_duplicates(key)
    # her session kendi skoruna göre sırala (kırpma yok, padding yok)
    sub = (scored.sort_values(["session_id","pred_final"], ascending=[True, False])
                 .groupby("session_id")["content_id_hashed"]
                 .apply(lambda x: " ".join(x.astype(str))).reset_index())
    sub.columns = ["session_id","prediction"]
    sub["session_id"] = sub["session_id"].astype(str)

    if session_index is not None:
        sub = session_index.to_frame().merge(sub, on="session_id", how="left").fillna({"prediction": ""})
    if expected_sessions is not None:
        assert sub.shape[0] == expected_sessions

    sub.to_csv(out_path, index=False)
    print(f"[OK] Submission written -> {out_path} (rows={len(sub):,})")


def _sort_for_grouping(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(["session_id","ts_hour","content_id_hashed"]).reset_index(drop=True)

def run_infer_cat(out_path: str):
    assert CatBoostRanker is not None, "catboost kurulu değil: pip install catboost"
    print("[TIMER] CatBoost YetiRank infer ...")
    test = load_test_sessions()
    feats_te = assemble_timeaware_features(test, windows=(3,7,14,30,60))
    df = _sort_for_grouping(feats_te)

    feat_cols = pd.read_csv("models/cb_rank_features.txt", header=None).iloc[:,0].tolist()
    X_te = to_float32(ensure_feature_columns(df, feat_cols), feat_cols)
    grp_te = df["session_id"].astype(str).values
    pool_te = Pool(X_te, group_id=grp_te)

    model = CatBoostRanker()
    model.load_model("models/cb_rank.cbm")
    preds = model.predict(pool_te)

    out = df[["session_id","content_id_hashed"]].copy()
    out["pred_final"] = preds.astype("float32")
    out = normalize_in_session(out, "pred_final")

    idx = load_sample_submission_session_ids(os.path.join(DATA_DIR, "sample_submission.csv"))
    make_submission(out, out_path, session_index=idx)
    print("[TIMER] CatBoost YetiRank infer done")

# --- METRICS: MAP@K ve NDCG@K (tam yarışma formatı) ---
def _dcg_at_k(rels, k=100):
    import numpy as np
    rels = np.asfarray(rels)[:k]
    if rels.size == 0: return 0.0
    discounts = 1.0 / np.log2(np.arange(2, rels.size + 2))
    return float(np.sum((2.0 ** rels - 1.0) * discounts))

def ndcg_grouped(df: pd.DataFrame, label_col="ordered", score_col="pred_final", k=100) -> float:
    vals = []
    for _, g in df.groupby("session_id", sort=False):
        g = g.sort_values(score_col, ascending=False)
        rels = g[label_col].values
        if (rels > 0).sum() == 0:  # hiç pozitif yoksa oturumu skordan hariç tut
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
        if rels.sum() == 0:  # hiç pozitif yoksa oturumu skordan hariç tut
            continue
        hits = 0; ap = 0.0; n = min(k, len(rels))
        for i in range(n):
            if rels[i]:
                hits += 1
                ap += hits / (i + 1)
        ap /= min(rels.sum(), k)
        vals.append(ap)
    return float(np.mean(vals)) if vals else 0.0


# --- RANK-AVERAGE BLEND: skoru değil oturum içi sırayı karıştır ---
def _rank01_in_session(df: pd.DataFrame, col: str) -> pd.Series:
    r = df.groupby("session_id")[col].rank(method="first", ascending=False)
    n = df.groupby("session_id")[col].transform("count")
    return (n - r) / (n - 1 + 1e-9)  # 1=best, 0=worst



# ---------------------- runner'lar ----------------------
def run_baseline_timeaware(out_path: str):
    set_seed(42)
    with timer("baseline_timeaware"):
        train = load_train_sessions()
        feats_tr = assemble_timeaware_features(train, windows=(3,7,14,30,60))
        scored_tr = score_timeaware_baseline(feats_tr)
        tr, va = split_time_holdout(scored_tr, holdout_days=7)
        va = normalize_in_session(va, "pred_final")
        print("[INFO] TA valid prepared.")

        test = load_test_sessions()
        feats_te = assemble_timeaware_features(test, windows=(3,7,14,30,60))
        scored_te = score_timeaware_baseline(feats_te)
        scored_te = normalize_in_session(scored_te, "pred_final")

        idx = load_sample_submission_session_ids()
        make_submission(scored_te[["session_id","content_id_hashed","pred_final"]],
                        out_path, session_index=idx)

def run_train_ltr(alpha: float):
    set_seed(42)
    with timer("LTR train"):
        train = load_train_sessions()
        feats = assemble_timeaware_features(train, windows=(3,7,14,30,60))
        feat_cols = _feature_cols(feats)
        tr, va = split_time_holdout(feats, holdout_days=7)
        m_click, m_order = train_ltr_models(tr, va, feat_cols)

        # hızlı alpha taraması (log amaçlı)
        for a in [0.70,0.74,0.78,0.80,0.82,0.85]:
            tmp = va.copy()
            tmp["pred_click"] = predict_rank_lgb(tmp, m_click)
            tmp["pred_order"] = predict_rank_lgb(tmp, m_order)
            tmp["pred_final"] = (1-a)*tmp["pred_click"] + a*tmp["pred_order"]
            tmp = normalize_in_session(tmp, "pred_final")
            proxy = tmp.groupby("session_id")["pred_final"].mean().mean()
            print(f"[ALPHA] a={a:.2f} (valid mean score proxy={proxy:.6f})")

        save_lgb(m_click, os.path.join(MODELS_DIR,"lgb_click.txt"))
        save_lgb(m_order, os.path.join(MODELS_DIR,"lgb_order.txt"))
        print("[LTR] models saved.")


def run_infer_ltr(out_path: str, alpha: float):
    set_seed(42)
    with timer("LTR infer"):
        test = load_test_sessions()
        feats_te = assemble_timeaware_features(test, windows=(3,7,14,30,60))

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
        feats = assemble_timeaware_features(test, windows=(3,7,14,30,60))

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

        idx = load_sample_submission_session_ids()
        make_submission(df[["session_id","content_id_hashed","pred_final"]],
                        out_path, session_index=idx)

def run_infer_ensemble(out_path: str,
                       alpha_ltr: float = 0.85,
                       w_ltr: float = 0.55,
                       w_xgb: float = 0.30,
                       w_cb:  float = 0.20,
                       w_ta:  float = 0.10):
    print("[TIMER] Ensemble infer ...")
    # ağırlıkları normalize et
    ws = np.array([w_ltr, w_xgb, w_cb, w_ta], dtype="float64")
    if ws.sum() <= 0:
        ws = np.array([1.0, 0.0, 0.0, 0.0])
    ws = ws / ws.sum()
    w_ltr, w_xgb, w_cb, w_ta = ws.tolist()

    test = load_test_sessions()
    feats_te = assemble_timeaware_features(test, windows=(3,7,14,30,60))
    feats_te = feats_te.sort_values(["session_id","ts_hour","content_id_hashed"]).reset_index(drop=True)
    key = ["session_id","content_id_hashed"]

    # LTR skorları
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

    # XGB skorları
    try:
        xgb_feat_cols = pd.read_csv("models/xgb_rank_features.txt", header=None).iloc[:,0].tolist()
        xgb_model = XGBRanker(); xgb_model.load_model("models/xgb_rank.json")

        fe_mat = to_float32(ensure_feature_columns(feats_te, xgb_feat_cols), xgb_feat_cols)
        preds_xgb = xgb_model.predict(fe_mat.values)

        xgb_df = feats_te[key].copy()
        xgb_df["pred_xgb"] = preds_xgb.astype("float32")
        xgb_df = normalize_in_session(xgb_df, "pred_xgb")[key+["pred_xgb"]]

    except Exception as e:
        print(f"[WARN] XGB yüklenemedi: {e}")
        xgb_df = feats_te[key].copy(); xgb_df["pred_xgb"] = 0.0

    # CatBoost skorları
    try:
        cb_feat_cols = pd.read_csv("models/cb_rank_features.txt", header=None).iloc[:,0].tolist()
        X_cb = ensure_feature_columns(to_float32(feats_te, cb_feat_cols), cb_feat_cols)  # eksik feature güvence
        cb_model = CatBoostRanker(); cb_model.load_model("models/cb_rank.cbm")

        X_cb = ensure_feature_columns(to_float32(feats_te, cb_feat_cols), cb_feat_cols)  # eksik feature güvence
        grp_te = map_group_ids(feats_te)
        pool_te = Pool(X_cb, group_id=grp_te)
        preds_cb = cb_model.predict(pool_te)

        cb_df = feats_te[key].copy()
        cb_df["pred_cb"] = preds_cb.astype("float32")
        cb_df = normalize_in_session(cb_df, "pred_cb")[key+["pred_cb"]]

    except Exception as e:
        print(f"[WARN] CatBoost yüklenemedi: {e}")
        cb_df = feats_te[key].copy(); cb_df["pred_cb"] = 0.0

    # TA baseline
    ta = score_timeaware_baseline(feats_te)
    ta = normalize_in_session(ta[key + ["pred_final"]].rename(columns={"pred_final":"pred_ta"}), "pred_ta")

    # Birleştir + ağırlıklandır
    df = ltr.merge(xgb_df, on=key, how="left") \
            .merge(cb_df,  on=key, how="left") \
            .merge(ta,     on=key, how="left")

    # rank-average özelliklerine çevir
    for c in ["pred_ltr","pred_xgb","pred_cb","pred_ta"]:
        if c not in df.columns: df[c] = 0.0
        df[c + "_r01"] = _rank01_in_session(df, c)

    # ağırlıklı rank-average
    df["pred_final"] = (
        w_ltr * df["pred_ltr_r01"] +
        w_xgb * df["pred_xgb_r01"] +
        w_cb  * df["pred_cb_r01"]  +
        w_ta  * df["pred_ta_r01"]
    ).astype("float32")

    # oturum içi sinyalleri getir ve küçük bonus ver
    # Eksik olabilecek in-session kolonlarını güvenle oluştur
    missing_cols = [
        "clicked_before_item_sess",
        "added_to_cart_before_item_sess",
        "added_to_fav_before_item_sess",
    ]
    for c in missing_cols:
        if c not in feats_te.columns:
            feats_te[c] = 0.0
    if "seen_before" not in feats_te.columns:
        feats_te["seen_before"] = 0.0
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

    df = normalize_in_session(df, "pred_final")  # hafif min-max, opsiyonel


    df = normalize_in_session(df, "pred_final")
    idx = load_sample_submission_session_ids(os.path.join(DATA_DIR, "sample_submission.csv"))
    make_submission(df, out_path, session_index=idx)
    print(f"[TIMER] Ensemble infer done (w_ltr={w_ltr:.2f}, w_xgb={w_xgb:.2f}, w_cb={w_cb:.2f}, w_ta={w_ta:.2f})")


def run_offline_eval(alpha_ltr: float, w_ltr: float, w_xgb: float, w_cb: float, w_ta: float,
                     metric: str = "ndcg", k: int = 100):
    print("[TIMER] Offline eval ...")
    train = load_train_sessions()
    feats = assemble_timeaware_features(train, windows=(3,7,14,30,60))
    feats = _sort_for_grouping(feats)
    key = ["session_id","content_id_hashed"]

    # valid holdout ayır
    tr, va = split_time_holdout(feats, holdout_days=7)
    va = va.copy()

    # LTR
    try:
        m_click = load_lgb(os.path.join(MODELS_DIR,"lgb_click.txt"))
        m_order = load_lgb(os.path.join(MODELS_DIR,"lgb_order.txt"))
        va["pred_click"] = predict_rank_lgb(va, m_click)
        va["pred_order"] = predict_rank_lgb(va, m_order)
        va["pred_ltr"]   = (1.0 - alpha_ltr) * va["pred_click"] + alpha_ltr * va["pred_order"]
    except Exception as e:
        print(f"[WARN] LTR yok/yüklenemedi: {e}"); va["pred_ltr"] = 0.0

    # XGB
    try:
        xgb_feat_cols = pd.read_csv("models/xgb_rank_features.txt", header=None).iloc[:,0].tolist()
        X = to_float32(ensure_feature_columns(va, xgb_feat_cols), xgb_feat_cols).values
        xm = XGBRanker(); xm.load_model("models/xgb_rank.json")
        va["pred_xgb"] = xm.predict(X).astype("float32")
    except Exception as e:
        print(f"[WARN] XGB yok/yüklenemedi: {e}"); va["pred_xgb"] = 0.0

    # CatBoost
    try:
        cb_feat_cols = pd.read_csv("models/cb_rank_features.txt", header=None).iloc[:,0].tolist()
        Xcb = ensure_feature_columns(to_float32(va, cb_feat_cols), cb_feat_cols)
        grp = map_group_ids(va)
        cbm = CatBoostRanker(); cbm.load_model("models/cb_rank.cbm")
        va["pred_cb"] = cbm.predict(Pool(Xcb, group_id=grp)).astype("float32")
    except Exception as e:
        print(f"[WARN] CatBoost yok/yüklenemedi: {e}"); va["pred_cb"] = 0.0

    # TA
    ta = score_timeaware_baseline(va)
    va["pred_ta"] = ta["pred_final"].values

    # rank-average
    for c in ["pred_ltr","pred_xgb","pred_cb","pred_ta"]:
        va[c + "_r01"] = _rank01_in_session(va, c)
    va["pred_final"] = (
        w_ltr * va["pred_ltr_r01"] +
        w_xgb * va["pred_xgb_r01"] +
        w_cb  * va["pred_cb_r01"]  +
        w_ta  * va["pred_ta_r01"]
    ).astype("float32")

    # metrik: label olarak ORDERED kullan (yarışma LB ile hizalanır)
    if metric.lower() == "map":
        score = mapk_grouped(va, label_col="ordered", score_col="pred_final", k=k)
        print(f"[OFFLINE] MAP@{k}: {score:.6f}")
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

    # Ranker eğitim/çıkarsama
    ap.add_argument("--train_xgb", action="store_true", help="XGBoost Ranker eğit")
    ap.add_argument("--infer_xgb", action="store_true", help="XGBoost Ranker infer")
    ap.add_argument("--train_cat", action="store_true", help="CatBoost YetiRank eğit")
    ap.add_argument("--infer_cat", action="store_true", help="CatBoost YetiRank infer")

    # Ensemble
    ap.add_argument("--infer_ensemble", action="store_true", help="LTR+XGB+CB+TA ensemble infer")
    ap.add_argument("--w_ltr", type=float, default=0.55)
    ap.add_argument("--w_xgb", type=float, default=0.30)
    ap.add_argument("--w_cb",  type=float, default=0.20)
    ap.add_argument("--w_ta",  type=float, default=0.10)

    ap.add_argument("--offline_eval", action="store_true")
    ap.add_argument("--metric", type=str, default="ndcg", choices=["ndcg","map"])
    ap.add_argument("--k", type=int, default=100)


    ap.add_argument("--alpha", type=float, default=0.85, help="LTR içi order ağırlığı")
    ap.add_argument("--beta",  type=float, default=0.30, help="Blend ağırlığı: LTR (1-beta) ⊕ TA (beta)")
    ap.add_argument("--out",   type=str, default="outputs/submission.csv")
    args = ap.parse_args()

    if args.baseline_timeaware:
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
              "  python one_run.py --baseline_timeaware --out outputs/ta.csv\n"
              "  python one_run.py --train_ltr --alpha 0.80\n"
              "  python one_run.py --infer_ltr --alpha 0.80 --out outputs/sub_ltr.csv\n"
              "  python one_run.py --train_xgb\n"
              "  python one_run.py --infer_xgb --out outputs/sub_xgb.csv\n"
              "  python one_run.py --train_cat\n"
              "  python one_run.py --infer_cat --out outputs/sub_cat.csv\n"
              "  python one_run.py --infer_blend --alpha 0.80 --beta 0.30 --out outputs/sub_blend.csv\n"
              "  python one_run.py --infer_ensemble --alpha 0.80 --w_ltr 0.55 --w_xgb 0.20 --w_cb 0.20 --w_ta 0.05 --out outputs/sub_ens.csv")
