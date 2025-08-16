# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from typing import Tuple

# metric_wrappers_v6.py
# -*- coding: utf-8 -*-
import numpy as np
from sklearn.metrics import roc_auc_score, ndcg_score

def _auc_group(y: np.ndarray, s: np.ndarray):
    pos = (y == 1)
    n1 = int(pos.sum()); n0 = int(len(y) - n1)
    if n1 == 0 or n1 == len(y):  # sadece tek sınıf → AUC tanımsız, atla
        return None
    order = np.argsort(s)
    ranks = np.empty_like(order); ranks[order] = np.arange(len(s))
    u = ranks[pos].sum() - n1 * (n1 - 1) / 2.0
    return float(u / (n1 * n0))

def auc_grouped(df: pd.DataFrame, label_col: str, score_col: str, group_col: str) -> float:
    aucs = []
    for _, g in df.groupby(group_col, observed=True):
        a = _auc_group(g[label_col].values, g[score_col].values)
        if a is not None:
            aucs.append(a)
    return float(np.mean(aucs)) if len(aucs) else float("nan")

def trendyol_final(
    df: pd.DataFrame,
    w_click: float = 0.3,
    w_order: float = 0.7,
    group_col: str = "session_id",
    click_label: str = "clicked",
    order_label: str = "ordered",
    click_score: str = "p_click",
    order_score: str = "p_order",
) -> Tuple[float, float, float]:
    """
    Dönüş: (final, click_auc_mean, order_auc_mean)
    """
    ac = auc_grouped(df, click_label, click_score, group_col)
    ao = auc_grouped(df, order_label, order_score, group_col)
    return w_click * ac + w_order * ao, ac, ao


# =============== Ranking Metrics ===================
def auc_score(y_true, y_pred):
    try:
        return roc_auc_score(y_true, y_pred)
    except:
        return 0.5

def ndcg_at_k(y_true, y_pred, k=10, group=None):
    """
    NDCG@K, optionally grouped (e.g. by query/session).
    """
    if group is None:
        return ndcg_score([y_true], [y_pred], k=k)
    scores = []
    for g in np.unique(group):
        mask = (group == g)
        if mask.sum() > 1:
            scores.append(ndcg_score([y_true[mask]], [y_pred[mask]], k=k))
    return np.mean(scores) if scores else 0.0

# =============== Wrapper for LightGBM ==============
def lgbm_auc_metric(y_pred, dataset):
    y_true = dataset.get_label()
    return "auc", auc_score(y_true, y_pred), True

def lgbm_ndcg10_metric(y_pred, dataset):
    y_true = dataset.get_label()
    group = dataset.get_group() if dataset.get_group() is not None else None
    return "ndcg@10", ndcg_at_k(y_true, y_pred, k=10, group=group), True
