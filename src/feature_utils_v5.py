# src/feature_utils_v5.py
import pandas as pd
import numpy as np

def add_item_ctr_cvr(df: pd.DataFrame, logs_df: pd.DataFrame) -> pd.DataFrame:
    """
    df: train/test tablosu (item_id kolonu olmalı)
    logs_df: SADECE TRAIN'den gelen geçmiş (item_id, clicked, ordered) -> leakage yok
    Ürün bazlı CTR/CVR ekler.
    """
    need = {"item_id","clicked","ordered"}
    missing = need - set(logs_df.columns)
    if missing:
        raise ValueError(f"logs_df missing columns: {missing}")

    stats = logs_df.groupby("item_id", observed=True).agg(
        views=("item_id", "count"),
        clicks=("clicked", "sum"),
        orders=("ordered", "sum")
    ).reset_index()

    stats["ctr"] = stats["clicks"] / (stats["views"] + 1e-6)
    stats["cvr"] = stats["orders"] / (stats["clicks"] + 1e-6)

    out = df.merge(stats[["item_id","ctr","cvr"]], on="item_id", how="left")
    out[["ctr","cvr"]] = out[["ctr","cvr"]].fillna(0.0)
    return out


def add_session_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    df: (session_id, item_id) içermeli. Train'de clicked/ordered varsa ek sayımlar üretir.
    Testte bu iki kolon yoksa 0'lanır.
    """
    out = df.copy()
    if "clicked" not in out.columns:  out["clicked"] = 0
    if "ordered" not in out.columns:  out["ordered"] = 0

    sess_stats = out.groupby("session_id", observed=True).agg(
        session_items=("item_id","count"),
        session_clicks=("clicked","sum"),
        session_orders=("ordered","sum"),
    ).reset_index()
    out = out.merge(sess_stats, on="session_id", how="left")

    # pozisyon ve göreli pozisyon (mevcut sırasına göre; ileride gerçek gösterim sırası varsa onu kullan)
    out["pos_in_session"] = out.groupby("session_id").cumcount()
    out["rel_pos_in_session"] = out["pos_in_session"] / out["session_items"].clip(lower=1)

    return out


def add_term_item_affinity(df: pd.DataFrame, logs_df: pd.DataFrame) -> pd.DataFrame:
    """
    df: (term_id, item_id)
    logs_df: SADECE TRAIN geçmişi (term_id, item_id, clicked, ordered)
    Term×Item eşleşmeleri için CTR/CVR ekler.
    """
    need = {"term_id","item_id","clicked","ordered"}
    missing = need - set(logs_df.columns)
    if missing:
        # bazen term_id yoksa (örn. normalization), bu adımı atlayalım
        return df.assign(term_item_ctr=0.0, term_item_cvr=0.0)

    aff = logs_df.groupby(["term_id","item_id"], observed=True).agg(
        ti_views=("item_id", "count"),
        ti_clicks=("clicked","sum"),
        ti_orders=("ordered","sum"),
    ).reset_index()

    aff["term_item_ctr"] = aff["ti_clicks"] / (aff["ti_views"] + 1e-6)
    aff["term_item_cvr"] = aff["ti_orders"] / (aff["ti_clicks"] + 1e-6)

    out = df.merge(
        aff[["term_id","item_id","term_item_ctr","term_item_cvr"]],
        on=["term_id","item_id"], how="left"
    )
    out[["term_item_ctr","term_item_cvr"]] = out[["term_item_ctr","term_item_cvr"]].fillna(0.0)
    return out
