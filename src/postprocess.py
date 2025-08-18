# -*- coding: utf-8 -*-
"""
postprocess.py
Basit yeniden-sıralama: oturum içi min-max normalizasyon ve hafif fiyat cezası/bonusu (meta yoksa pas).
Not: V0'da fiyat yok; sadece normalizasyon yapıyoruz.
"""

import pandas as pd

def normalize_in_session(df: pd.DataFrame, score_col: str = "pred_final") -> pd.DataFrame:
    def _scale(g):
        x = g[score_col].values
        if x.max() == x.min():
            g["score_norm"] = 0.5  # hepsi aynıysa
        else:
            g["score_norm"] = (x - x.min()) / (x.max() - x.min())
        return g
    df = df.groupby("session_id", group_keys=False).apply(_scale)
    return df
