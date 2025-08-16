import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

def kfold_target_encode(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    group_cols,
    target_col: str,
    n_splits: int = 5,
    smoothing: float = 20.0,
    global_prior: float | None = None,
    seed: int = 42,
    out_col: str | None = None,
):
    """
    Sızıntısız KFold target encoding.
    - group_cols: ['search_term_normalized'] veya ['search_term_normalized','leaf_category_name'] gibi
    - target_col: 'clicked' ya da 'ordered'
    - smoothing: grubu gözlem sayısına göre global prior'a doğru yumuşatma
    """
    if out_col is None:
        out_col = f"te_{'_'.join(group_cols)}__{target_col}"

    df_train = df_train.copy()
    df_test = df_test.copy()

    if global_prior is None:
        global_prior = df_train[target_col].mean()

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros(len(df_train), dtype=float)

    for tr_idx, va_idx in kf.split(df_train):
        tr = df_train.iloc[tr_idx]
        va = df_train.iloc[va_idx]
        agg = tr.groupby(group_cols)[target_col].agg(["mean", "count"]).reset_index()
        # smoothing
        agg[out_col] = (agg["count"] * agg["mean"] + smoothing * global_prior) / (agg["count"] + smoothing)
        agg = agg[group_cols + [out_col]]
        va = va.merge(agg, on=group_cols, how="left")
        oof[va_idx] = va[out_col].fillna(global_prior).values

    df_train[out_col] = oof

    # test mapping train’den
    agg_full = df_train.groupby(group_cols)[target_col].agg(["mean", "count"]).reset_index()
    agg_full[out_col] = (agg_full["count"] * agg_full["mean"] + smoothing * global_prior) / (agg_full["count"] + smoothing)
    df_test = df_test.merge(agg_full[group_cols + [out_col]], on=group_cols, how="left")
    df_test[out_col] = df_test[out_col].fillna(global_prior)
    return df_train, df_test, out_col
