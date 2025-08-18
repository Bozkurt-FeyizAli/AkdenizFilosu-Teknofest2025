# -*- coding: utf-8 -*-
"""
metric.py
Trendyol değerlendirmesine yakın offline metrik:
- Oturum içi AUC (click / order ayrı)
- Pozitif aksiyon içeren oturumlarda AUC hesaplanır, sonra ortalama alınır.
- Nihai skor: order'a daha yüksek ağırlık (varsayılan 0.75).
"""

from typing import Tuple, Dict
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

def _mean_session_auc(df: pd.DataFrame, label_col: str, pred_col: str) -> float:
    """
    Her session_id için (eğer o label'da en az bir pozitif ve bir negatif varsa) AUC hesaplar,
    oturumlar üzerinde basit ortalama döner.
    """
    aucs = []
    for sid, g in df.groupby("session_id"):
        y = g[label_col].values
        # en az bir 1 ve bir 0 olmalı
        if (y.sum() > 0) and (y.sum() < len(y)):
            try:
                aucs.append(roc_auc_score(y, g[pred_col].values))
            except ValueError:
                # tekil değer veya başka sorunlarda oturumu atla
                continue
    return float(np.mean(aucs)) if aucs else np.nan

def evaluate_sessions(df_pred: pd.DataFrame,
                      click_score_col: str = "pred_click",
                      order_score_col: str = "pred_order",
                      w_order: float = 0.75) -> Dict[str, float]:
    """
    df_pred: train_sessions + tahmin skorları (kolonlar: clicked, ordered, session_id, ...).
    Dönen: {'auc_click':..., 'auc_order':..., 'final':...}
    """
    auc_c = _mean_session_auc(df_pred, "clicked", click_score_col)
    auc_o = _mean_session_auc(df_pred, "ordered", order_score_col)
    final = w_order * auc_o + (1.0 - w_order) * auc_c
    return {"auc_click": auc_c, "auc_order": auc_o, "final": final}
