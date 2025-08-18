# -*- coding: utf-8 -*-
"""
train_rank_ltr.py
LambdaMART (LightGBM lambdarank) eğitimi ve çıkarımı:
- Binary: label={clicked|ordered}
- Graded: label=graded_label (0,1,2,...)
- Grup = session_id
Erken durdurma: CALLBACK ile (sürüm-uyumlu)
"""

from __future__ import annotations
from typing import List, Tuple
import os
import numpy as np
import pandas as pd

def _ensure_dir(p: str):
    os.makedirs(os.path.dirname(p), exist_ok=True)

def _lgb():
    import lightgbm as lgb
    return lgb

# --------- Feature seçimi ---------
ID_COLS    = {"session_id", "content_id_hashed"}
LABEL_COLS = {"clicked", "ordered", "graded_label"}

# --- ek: güvenli feature seçici ---
def get_feature_cols(df: pd.DataFrame) -> list[str]:
    drop = {
        "session_id","content_id_hashed","user_id_hashed",
        "ts_hour","session_date","search_term_normalized",
        "clicked","ordered","added_to_cart","added_to_fav",
        "level1_category_name","cv_tags"
    }
    cols = [c for c in df.columns
            if c not in drop and pd.api.types.is_numeric_dtype(df[c])]
    return cols

def _make_group(df: pd.DataFrame) -> np.ndarray:
    return df.groupby("session_id", sort=False).size().to_numpy()

# --------- Ortak eğitim yardımcıları ---------
def _common_params(seed: int = 42) -> dict:
    # hafif sıkılaştırılmış defaultlar
    return {
        "objective": "lambdarank",
        "metric": ["ndcg"],
        "eval_at": [10, 20, 50],
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_data_in_leaf": 50,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.75,
        "bagging_freq": 1,
        "lambda_l2": 4.0,
        "max_depth": -1,
        "verbosity": -1,
        "deterministic": True,
        "force_row_wise": True,
        "seed": seed,
    }

def _callbacks():
    lgb = _lgb()
    # Sürümden bağımsız early stopping & log
    return [
        lgb.early_stopping(stopping_rounds=200, first_metric_only=False),
        lgb.log_evaluation(period=100),
    ]

# --------- Binary LambdaMART ---------
def train_lambdarank(tr: pd.DataFrame, va: pd.DataFrame, label_col: str, feat_cols: list[str]):
    import lightgbm as lgb

    # grup (session bazlı)
    g_tr = tr.groupby("session_id").size().values
    g_va = va.groupby("session_id").size().values

    # click modelinde siparişe ek ağırlık
    w_tr = w_va = None
    if label_col == "clicked" and "ordered" in tr.columns:
        w_tr = (1.0 + 2.0 * tr["ordered"].astype(float)).values
        w_va = (1.0 + 2.0 * va["ordered"].astype(float)).values

    dtr = lgb.Dataset(tr[feat_cols], label=tr[label_col], group=g_tr, weight=w_tr, free_raw_data=False)
    dva = lgb.Dataset(va[feat_cols], label=va[label_col], group=g_va, weight=w_va, reference=dtr, free_raw_data=False)

    params = dict(
        objective="lambdarank",
        metric="ndcg",
        ndcg_eval_at=[10,20,50],
        learning_rate=0.05,
        num_leaves=127,
        min_data_in_leaf=300,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=1,
        lambda_l2=2.0,
        verbose=-1,
    )

    # LightGBM 4.x: early_stopping_rounds paramı yerine callback kullanılır
    def _callbacks():
        import lightgbm as lgb
        return [
            lgb.early_stopping(stopping_rounds=200, verbose=True),
            lgb.log_evaluation(period=100),
        ]

    model = lgb.train(
        params=params,
        train_set=dtr,
        valid_sets=[dtr, dva],
        valid_names=["train","valid"],
        num_boost_round=2000,
        callbacks=_callbacks(),   # <--- kritik
    )
    return model
# --------- Graded LambdaMART ---------
def train_lambdarank_graded(tr_df: pd.DataFrame, va_df: pd.DataFrame,
                            feat_cols: list[str]):
    lgb = _lgb()
    # graded labels: 0..K (neg/tık/sipariş vb. mapping’i sen hazırlıyorsun)
    y_tr = tr_df["graded_label"].astype(int).to_numpy()
    y_va = va_df["graded_label"].astype(int).to_numpy()
    g_tr = _make_group(tr_df)
    g_va = _make_group(va_df)

    dtr = lgb.Dataset(tr_df[feat_cols], label=y_tr, group=g_tr, free_raw_data=False)
    dva = lgb.Dataset(va_df[feat_cols], label=y_va, group=g_va, reference=dtr, free_raw_data=False)

    params = _common_params(seed=2025)
    model = lgb.train(
        params,
        dtr,
        num_boost_round=3000,
        valid_sets=[dtr, dva],
        valid_names=["train", "valid"],
        callbacks=_callbacks(),
    )
    return model

# --------- Tahmin / Kaydet / Yükle ---------
def predict_on_df(df: pd.DataFrame, model, feat_cols: list[str] | None = None) -> np.ndarray:
    """Modelin gördüğü feature isim sırasına göre tahmin yap (shape-check off)."""
    # Modelin gördüğü isimleri kullanmak shape uyuşmazlığını önler
    model_feat_names = list(model.feature_name())
    cols = [c for c in model_feat_names if c in df.columns]
    X = df[cols].astype(np.float32)
    return model.predict(
        X,
        num_iteration=getattr(model, "best_iteration_", None),
        predict_disable_shape_check=True
    )

def save_lgb_model(model, path: str):
    _ensure_dir(path)
    model.save_model(path)

def load_lgb_model(path: str):
    lgb = _lgb()
    return lgb.Booster(model_file=path)

# ---- graded alias'lar (aynı I/O) ----
def save_lgb_model_graded(model, path: str):
    save_lgb_model(model, path)

def load_lgb_model_graded(path: str):
    return load_lgb_model(path)
