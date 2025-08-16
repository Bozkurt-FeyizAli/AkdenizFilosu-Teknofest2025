import numpy as np
import pandas as pd
# Kaggle’ın verdiği dosyayı runtime’da import edeceğiz
# from trendyol_metric_group_auc import auc_grouped

def auc_grouped_fast(df, label_col, score_col, group_col):
    """
    Her group (session) için AUC; sadece pozitif içeren gruplar dahil.
    """
    aucs = []
    for _, g in df.groupby(group_col, observed=True):
        y = g[label_col].values
        s = g[score_col].values
        if y.sum() == 0 or y.sum() == len(y):
            continue
        # rank-based AUC (Mann–Whitney U)
        order = np.argsort(s)
        ranks = np.empty_like(order)
        ranks[order] = np.arange(len(s))
        pos = y == 1
        n1 = pos.sum(); n0 = (~pos).sum()
        u = ranks[pos].sum() - n1*(n1-1)/2.0
        aucs.append(u / (n1*n0))
    if not aucs:
        return np.nan
    return float(np.mean(aucs))

def trendyol_final_score(df, click_label="clicked", order_label="ordered",
                         click_score="score_click", order_score="score_order",
                         group_col="session_id", w_click=0.3, w_order=0.7):
    auc_click = auc_grouped_fast(df, click_label, click_score, group_col)
    auc_order = auc_grouped_fast(df, order_label, order_score, group_col)
    final = w_click * auc_click + w_order * auc_order
    return final, auc_click, auc_order

def auc_grouped_from_single_score(df, label_col, score_col, group_col):
    """
    Tek bir skorla (s) oturum başına AUC: pozitiflerin negatiflerden önde olma olasılığı.
    Pozitiflerin kendi arası sırası önemli değildir.
    """
    return auc_grouped_fast(df[[group_col, label_col, score_col]].copy(),
                            label_col=label_col, score_col=score_col, group_col=group_col)

def final_metric_from_single_score(df, score_col="s",
                                   group_col="session_id",
                                   click_label="clicked", order_label="ordered",
                                   w_click=0.3, w_order=0.7):
    """
    Tek skor s ile iki AUC (click/order) hesaplar ve yarışmanın ağırlıklarıyla toplar.
    """
    auc_c = auc_grouped_from_single_score(df, click_label, score_col, group_col)
    auc_o = auc_grouped_from_single_score(df, order_label, score_col, group_col)
    return w_click * auc_c + w_order * auc_o, auc_c, auc_o
