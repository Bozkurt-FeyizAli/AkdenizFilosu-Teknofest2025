# -*- coding: utf-8 -*-
"""
evaluator.py
Kaggle submission formatındaki bir dosyayı (session_id,prediction) alır,
train_sessions.parquet (etiketli) ile eşleştirip yarışmanın tarifine uygun
oturum içi AUC(click/order ayrı) + ağırlıklı final metriğini hesaplar.

Notlar:
- Submission'daki prediction: her session_id için content_id_hashed'lerin
  boşlukla ayrılmış sıralı listesi.
- AUC hesaplamak için her item'a bir skor gerekir. Burada sırayı skora çeviririz:
  en üstteki en yüksek skor olacak şekilde score = (N + 1 - rank).
- Yalnızca ilgili session'ın gerçek item'larıyla (inner-join) değerlendiririz.
- Oturum içinde (label'da) en az bir pozitif ve bir negatif varsa AUC hesaplanır.
"""

from typing import Dict
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

def _explode_submission(sub_df: pd.DataFrame) -> pd.DataFrame:
    """
    submission (session_id,prediction) -> satır satır (session_id, content_id_hashed, rank, score)
    """
    # prediction alanını list'e çevir
    sub_df = sub_df.copy()
    sub_df["prediction_list"] = sub_df["prediction"].str.strip().str.split()
    sub_df = sub_df.dropna(subset=["prediction_list"])

    # patlat
    ex = sub_df.explode("prediction_list").rename(columns={"prediction_list":"content_id_hashed"})
    # 1-based rank: üstteki 1 (en iyi), alttaki N
    ex["rank"] = ex.groupby("session_id").cumcount() + 1
    # Skor: büyük daha iyi; AUC için yeterli
    # max rank'i bulup ters sıralı skor yapalım
    ex["max_rank"] = ex.groupby("session_id")["rank"].transform("max")
    ex["score"] = (ex["max_rank"] + 1 - ex["rank"]).astype("float32")
    return ex[["session_id","content_id_hashed","rank","score"]]

def _mean_session_auc(df: pd.DataFrame, label_col: str, score_col: str) -> float:
    aucs = []
    for sid, g in df.groupby("session_id"):
        y = g[label_col].values
        # hem pozitif hem negatif olmalı
        if (y.sum() > 0) and (y.sum() < len(y)):
            try:
                aucs.append(roc_auc_score(y, g[score_col].values))
            except ValueError:
                continue
    return float(np.mean(aucs)) if aucs else np.nan

def evaluate_submission(pred_csv: str, ground_parquet: str, w_order: float = 0.75) -> Dict[str, float]:
    """
    pred_csv: Kaggle submission dosyası (session_id,prediction)
    ground_parquet: train_sessions.parquet (veya aynı şemalı etiketli set)
    w_order: final ağırlık (order daha önemli)
    """
    # 1) submission oku ve patlat
    sub = pd.read_csv(pred_csv)
    if not {"session_id","prediction"}.issubset(sub.columns):
        raise ValueError("Submission CSV 'session_id' ve 'prediction' kolonlarını içermeli.")
    ex = _explode_submission(sub)

    # 2) ground (etiketli oturum ürünleri) oku
    cols = ["session_id","content_id_hashed","clicked","ordered"]
    g = pd.read_parquet(ground_parquet, columns=cols)

    # 3) inner-join: sadece gerçek oturumdaki ürünleri değerlendir
    j = ex.merge(g, on=["session_id","content_id_hashed"], how="inner")

    # 4) oturum içi AUC'ler
    auc_click = _mean_session_auc(j, "clicked", "score")
    auc_order = _mean_session_auc(j, "ordered", "score")
    final = w_order * auc_order + (1.0 - w_order) * auc_click

    # Basit kapsama istatistikleri
    covered_sessions = j["session_id"].nunique()
    total_sessions = g["session_id"].nunique()
    coverage = covered_sessions / max(1, total_sessions)

    return {
        "auc_click": auc_click,
        "auc_order": auc_order,
        "final": final,
        "sessions_covered": int(covered_sessions),
        "sessions_total": int(total_sessions),
        "coverage": float(coverage)
    }
