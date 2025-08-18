# -*- coding: utf-8 -*-
"""
train_rank.py
- v0 baseline scorer
- v1 time-aware baseline scorer (7d/30d/90d pencereleri blend)
"""

import numpy as np
import pandas as pd

def score_baseline(df: pd.DataFrame,
                   w_order: float = 0.6,
                   w_click: float = 0.3,
                   w_tag: float = 0.1):
    df = df.copy()
    df["pred_click"] = 0.7*df["tc_ctr"] + 0.3*df["click_rate"] + 0.05*df.get("tag_hit", 0)
    df["pred_order"] = 0.7*df["order_rate"] + 0.3*df["tc_ctr"] + 0.02*df.get("tag_hit", 0)
    df["pred_final"] = w_order*df["pred_order"] + w_click*df["pred_click"] + w_tag*df.get("tag_hit", 0)
    return df

# --- YENİ: time-aware sürüm ---
def score_timeaware_baseline(df: pd.DataFrame,
                             alpha_tc=(0.5, 0.3, 0.2),
                             alpha_cr=(0.5, 0.3, 0.2),
                             alpha_or=(0.6, 0.25, 0.15),
                             w_order=0.75, w_click=0.25):
    """
    7d/30d/90d pencerelerini karıştırır:
    tc_ctr_blend   = 0.5*7d + 0.3*30d + 0.2*90d
    click_rate_blend / order_rate_blend benzer.
    """
    df = df.copy()

    # güvenli get helper
    g = lambda c: df[c] if c in df.columns else 0.0

    tc = (alpha_tc[0]*g("tc_ctr_7d") + alpha_tc[1]*g("tc_ctr_30d") + alpha_tc[2]*g("tc_ctr_90d")).astype("float32")
    cr = (alpha_cr[0]*g("click_rate_7d") + alpha_cr[1]*g("click_rate_30d") + alpha_cr[2]*g("click_rate_90d")).astype("float32")
    orr = (alpha_or[0]*g("order_rate_7d") + alpha_or[1]*g("order_rate_30d") + alpha_or[2]*g("order_rate_90d")).astype("float32")

    # recency bonus (yeni ürünlere küçük +): ~0.02 * exp(-days/90)
    days = g("days_since_creation")
    recency_bonus = np.where((days >= 0) & (days < 3650), np.exp(-days/90.0)*0.02, 0.0).astype("float32")

    # tahminler
    df["pred_click"] = (0.7*tc + 0.3*cr + recency_bonus).astype("float32")
    df["pred_order"] = (0.7*orr + 0.3*tc + 0.5*recency_bonus).astype("float32")

    # tek sıralama skoru (order ağır)
    df["pred_final"] = (w_order*df["pred_order"] + w_click*df["pred_click"]).astype("float32")
    return df
