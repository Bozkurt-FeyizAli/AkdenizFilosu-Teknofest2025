# -*- coding: utf-8 -*-
"""
feature_build_timeaware.py (VEKTÖRİZE, HIZLI)
Time-aware özellik üretimi:
- Tüm agregasyonlar session.ts_hour tarihinden ÖNCEKİ verilerle yapılır (leakage yok)
- 7g / 30g / 90g pencerelerinde content & term⨯content rolling toplamlar
- merge_asof ile oturumlara en yakın (<= ts) günlük agregasyonlar bağlanır
"""

from __future__ import annotations
import os
from typing import Iterable, Tuple
import numpy as np
import pandas as pd

from .dataio import (
    load_content_top_terms, load_content_sitewide, load_content_meta
)
from .utils import reduce_memory_df

# ------------------------------------------------------------
# Yardımcılar
# ------------------------------------------------------------
def _ensure_dir(p: str):
    os.makedirs(os.path.dirname(p), exist_ok=True)

def _windows_to_tag(windows: Iterable[int]) -> str:
    return "_".join(str(int(w)) for w in windows)

# ------------------------------------------------------------
# Rolling hazırlıkları (VEKTÖRİZE)
# ------------------------------------------------------------
def build_sitewide_rolling(windows: Tuple[int, ...] = (7, 30, 90),
                           cache: bool = True) -> pd.DataFrame:
    """
    content/sitewide_log.parquet -> content_id_hashed, date bazında rolling toplamlar.
    Dönen kolonlar:
      click_{win}d, order_{win}d  (win ∈ windows)
    """
    tag = _windows_to_tag(windows)
    cache_path = f"features/sitewide_roll_{tag}.parquet"
    if cache and os.path.exists(cache_path):
        return pd.read_parquet(cache_path)

    sw = load_content_sitewide()[["content_id_hashed", "date", "total_click", "total_order"]].copy()
    sw = sw.sort_values(["content_id_hashed", "date"])
    # Her window için rolling toplamları hesapla
    out = sw[["content_id_hashed", "date"]].drop_duplicates().copy()
    for win in windows:
        rolled = (
            sw.groupby("content_id_hashed", group_keys=False)
              .apply(lambda g: g.rolling(f"{win}D", on="date")[["total_click", "total_order"]].sum())
              .reset_index(drop=True)
        )
        rolled = pd.concat([sw[["content_id_hashed", "date"]].reset_index(drop=True), rolled], axis=1)
        rolled = rolled.rename(columns={
            "total_click": f"click_{win}d",
            "total_order": f"order_{win}d"
        })
        out = out.merge(
            rolled[["content_id_hashed", "date", f"click_{win}d", f"order_{win}d"]],
            on=["content_id_hashed", "date"],
            how="left"
        )

    out = reduce_memory_df(out.fillna(0.0))
    if cache:
        _ensure_dir(cache_path)
        out.to_parquet(cache_path, index=False)
    return out


def build_topterms_rolling(windows: Tuple[int, ...] = (7, 30, 90),
                           cache: bool = True) -> pd.DataFrame:
    """
    content/top_terms_log.parquet -> (content_id_hashed, search_term_normalized, date) bazında rolling toplamlar.
    Dönen kolonlar:
      imp_{win}d, clk_{win}d  (win ∈ windows)
    """
    tag = _windows_to_tag(windows)
    cache_path = f"features/topterms_roll_{tag}.parquet"
    if cache and os.path.exists(cache_path):
        return pd.read_parquet(cache_path)

    ctt = load_content_top_terms()[[
        "content_id_hashed", "search_term_normalized", "date",
        "total_search_impression", "total_search_click"
    ]].copy()
    ctt = ctt.sort_values(["content_id_hashed", "search_term_normalized", "date"])

    def _add_rolls(g: pd.DataFrame) -> pd.DataFrame:
        d = g.copy()
        for win in windows:
            r_imp = d.rolling(f"{win}D", on="date")["total_search_impression"].sum()
            r_clk = d.rolling(f"{win}D", on="date")["total_search_click"].sum()
            d[f"imp_{win}d"] = r_imp.values.astype("float32")
            d[f"clk_{win}d"] = r_clk.values.astype("float32")
        return d

    rolled = (
        ctt.groupby(["content_id_hashed", "search_term_normalized"], group_keys=False)
           .apply(_add_rolls)
           .reset_index(drop=True)
    )
    keep_cols = ["content_id_hashed", "search_term_normalized", "date"] + \
                [f"imp_{w}d" for w in windows] + [f"clk_{w}d" for w in windows]
    rolled = reduce_memory_df(rolled[keep_cols].fillna(0.0))

    if cache:
        _ensure_dir(cache_path)
        rolled.to_parquet(cache_path, index=False)
    return rolled


# ------------------------------------------------------------
# Ana assembler (merge_asof ile oturumlara bağla)
# ------------------------------------------------------------
def assemble_timeaware_features(sessions: pd.DataFrame,
                                windows: Tuple[int, ...] = (7, 30, 90)) -> pd.DataFrame:
    """
    Her satır (session-item) için time-aware özellik seti döner:
      - tc_ctr_{7,30,90}d
      - click_rate_{7,30,90}d
      - order_rate_{7,30,90}d
      - days_since_creation
    """
    # 1) Yardımcı tarih kolonları
    df = sessions.copy()
    df["date"] = pd.to_datetime(df["ts_hour"], utc=False).dt.floor("D")

    # 2) Rolling tabloları hazırla/oku (vektörize)
    sw_roll = build_sitewide_rolling(windows=windows, cache=True)
    tt_roll = build_topterms_rolling(windows=windows, cache=True)

    # 3) Content sitewide rolling'i oturumlara bağla
    #    (content_id_hashed eşleşmesi ve date <= session_date olacak şekilde)
    df = pd.merge_asof(
        df.sort_values("date"),
        sw_roll.sort_values("date"),
        by="content_id_hashed",
        left_on="date", right_on="date",
        direction="backward",
        allow_exact_matches=True
    )

    # 4) Term⨯Content rolling'i oturumlara bağla
    df = pd.merge_asof(
        df.sort_values("date"),
        tt_roll.sort_values("date"),
        by=["content_id_hashed", "search_term_normalized"],
        left_on="date", right_on="date",
        direction="backward",
        allow_exact_matches=True
    )

    # 5) Rate/CTR türeteçleri (smoothing ile)
    for win in windows:
        # sitewide click/order
        clk = df.get(f"click_{win}d", 0.0).astype("float32")
        ord_ = df.get(f"order_{win}d", 0.0).astype("float32")
        # CTR: term⨯content
        imp = df.get(f"imp_{win}d", 0.0).astype("float32")
        tclk = df.get(f"clk_{win}d", 0.0).astype("float32")

        # Smoothing aynı mantık: (num + 1) / (den + 1 + 3)
        df[f"click_rate_{win}d"] = (clk + 1.0) / (clk + 1.0 + 3.0)
        df[f"order_rate_{win}d"] = (ord_ + 1.0) / (clk + 1.0 + 3.0)  # siparişi "tıklamaya göre" normalize etmek
        df[f"tc_ctr_{win}d"] = (tclk + 1.0) / (imp + 1.0 + 3.0)

    # 6) Meta: days_since_creation
    meta = load_content_meta()[["content_id_hashed", "content_creation_date"]]
    df = df.merge(meta, on="content_id_hashed", how="left")
    # creation_date NaT ise -1
    cd = pd.to_datetime(df["content_creation_date"], errors="coerce")
    ds = (pd.to_datetime(df["ts_hour"], utc=False) - cd).dt.days
    df["days_since_creation"] = ds.fillna(-1).astype("int32")
    df = df.drop(columns=["content_creation_date"], errors="ignore")

    # 7) NaN -> 0 (oranlar/ctr'ler için)
    fill_cols = []
    for win in windows:
        fill_cols += [f"click_rate_{win}d", f"order_rate_{win}d", f"tc_ctr_{win}d"]
    df[fill_cols] = df[fill_cols].fillna(0.0).astype("float32")

    return reduce_memory_df(df)
