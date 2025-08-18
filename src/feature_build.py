# -*- coding: utf-8 -*-
"""
feature_build.py
v0 Baseline özellikleri (ZAMAN FİLTRESİ YOK - hızlıca skor almak için):
- term ⨯ content CTR (content/top_terms_log)
- content sitewide oranları (click/order rate)
- basit metin eşleşmesi: search_term_normalized ile cv_tags overlap (çok kaba)
Sonraki sürümde: time-aware (<= ts_hour), window'lu agregasyon, kullanıcı geçmişi vb.
"""

import numpy as np
import pandas as pd
from .dataio import (
    load_content_top_terms, load_content_sitewide, load_content_meta
)
from .utils import reduce_memory_df

def build_global_term_content_ctr(alpha: float = 1.0, beta: float = 3.0) -> pd.DataFrame:
    """
    term⨯content bazında global CTR (beta smoothing).
    score = (click + α) / (impression + α + β)
    """
    ctt = load_content_top_terms()
    grp = ctt.groupby(["content_id_hashed", "search_term_normalized"], observed=True).agg(
        click=("total_search_click", "sum"),
        imp=("total_search_impression", "sum")
    ).reset_index()
    grp["tc_ctr"] = (grp["click"] + alpha) / (grp["imp"] + alpha + beta)
    grp = grp[["content_id_hashed","search_term_normalized","tc_ctr"]]
    return reduce_memory_df(grp)

def build_global_content_rates(alpha: float = 1.0, beta: float = 3.0) -> pd.DataFrame:
    """
    content bazında global sitewide oranları.
    order_rate, click_rate (beta smoothing).
    """
    sw = load_content_sitewide()
    agg = sw.groupby("content_id_hashed", observed=True).agg(
        order_sum=("total_order","sum"),
        click_sum=("total_click","sum"),
        cart_sum=("total_cart","sum"),
        fav_sum=("total_fav","sum"),
    ).reset_index()
    agg["order_rate"] = (agg["order_sum"] + alpha) / (agg["order_sum"] + agg["click_sum"] + alpha + beta)
    agg["click_rate"] = (agg["click_sum"] + alpha) / (agg["click_sum"] + alpha + beta)
    return reduce_memory_df(agg[["content_id_hashed","order_rate","click_rate"]])

def very_crude_term_tag_overlap() -> pd.DataFrame:
    """
    cv_tags içinde aranan terim geçiyor mu? (kaba bayrak)
    Not: cv_tags serbest metin; contains ile kaba eşleşme kullanıyoruz.
    """
    meta = load_content_meta()[["content_id_hashed","cv_tags"]].copy()
    # küçük harfe indir, NaN -> ""
    meta["cv_tags"] = meta["cv_tags"].fillna("").str.lower()
    return meta

def assemble_baseline_features(sessions: pd.DataFrame) -> pd.DataFrame:
    """
    Train/Test oturum tablolarına baseline özellikleri merge eder.
    Dönüş: sessions + ['tc_ctr','order_rate','click_rate','tag_hit'] + sentinel NaN'lar 0 ile doldurulur.
    """
    term_ctr = build_global_term_content_ctr()
    rates = build_global_content_rates()
    tags = very_crude_term_tag_overlap()

    df = sessions.merge(term_ctr, on=["content_id_hashed","search_term_normalized"], how="left")
    df = df.merge(rates, on="content_id_hashed", how="left")
    df = df.merge(tags, on="content_id_hashed", how="left")

    # aranan terim cv_tags içinde geçiyor mu (kaba)
    st = df["search_term_normalized"].astype(str).str.lower()
    cv = df["cv_tags"].fillna("")
    df["tag_hit"] = (st.str.len() > 0) & cv.str.contains(st, regex=False)
    df["tag_hit"] = df["tag_hit"].astype("int8")

    # NaN doldurma (agregasyon olmayanlar)
    for col in ["tc_ctr","order_rate","click_rate"]:
        if col in df.columns:
            df[col] = df[col].fillna(0.0).astype("float32")
    return df
