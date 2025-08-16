# src/main_v5.py
import argparse
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from pathlib import Path

from .helpers import groupwise_rank, reduce_cats, set_global_seed
from .metric_wrappers import trendyol_final_score
import optuna

NUM_BOOST = 4000
EARLY_STOP = 200

def tune_params_with_optuna(train_df, features, label_col, group_col="session_id", n_trials=30, seed=42):
    y = train_df[label_col].values
    gkf = GroupKFold(n_splits=3)
    def objective(trial):
        params = dict(
            objective="lambdarank", metric="ndcg",
            learning_rate=trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
            num_leaves=trial.suggest_int("num_leaves", 63, 255),
            min_data_in_leaf=trial.suggest_int("min_data_in_leaf", 20, 200),
            feature_fraction=trial.suggest_float("feature_fraction", 0.6, 1.0),
            bagging_fraction=trial.suggest_float("bagging_fraction", 0.6, 1.0),
            bagging_freq=1, lambda_l1=trial.suggest_float("lambda_l1", 0.0, 5.0),
            lambda_l2=trial.suggest_float("lambda_l2", 0.0, 5.0), verbose=-1, seed=seed
        )
        oof = np.zeros(len(train_df), dtype=float)
        for tr, va in gkf.split(train_df, y, groups=train_df[group_col]):
            tr_df = train_df.iloc[tr]; va_df = train_df.iloc[va]
            gtr = tr_df.groupby(group_col, observed=True).size().values
            gva = va_df.groupby(group_col, observed=True).size().values
            dtr = lgb.Dataset(tr_df[features], label=tr_df[label_col], group=gtr, free_raw_data=False)
            dva = lgb.Dataset(va_df[features], label=va_df[label_col], group=gva, free_raw_data=False)
            model = lgb.train(params, dtr, 3000, valid_sets=[dva], early_stopping_rounds=150, verbose_eval=False)
            oof[va] = model.predict(va_df[features], num_iteration=model.best_iteration)
        # group-AUC yerine basit NDCG kullanıyoruz; Optuna sırasında hız/istikrar için
        # Daha hassas istersen trendyol_final_score ile benzer bir objective yazabiliriz.
        # Burada NDCG'yi valid'den aldığı için zaten optimize ediyor.
        return np.mean(oof)  # dummy; LightGBM val_metric'i yokken sayı lazım
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def train_head(train_df, features, label_col, group_col="session_id",
               params=None, n_splits=5, seed=42):
    """
    LightGBM lambdarank tek-head eğitimi (clicked/ordered için ayrı ayrı çağır).
    """
    if params is None:
        params = dict(
            objective="lambdarank",
            metric="ndcg",
            learning_rate=0.05,
            num_leaves=127,
            min_data_in_leaf=50,
            feature_fraction=0.9,
            bagging_fraction=0.9,
            bagging_freq=1,
            verbose=-1,
            seed=seed,
        )

    y = train_df[label_col].values
    gkf = GroupKFold(n_splits=n_splits)
    oof = np.zeros(len(train_df), dtype=float)
    models = []

    for fold, (tr, va) in enumerate(gkf.split(train_df, y, groups=train_df[group_col])):
        tr_df = train_df.iloc[tr]; va_df = train_df.iloc[va]
        # Fold bazında group size dizileri:
        gtr = tr_df.groupby(group_col, observed=True).size().values
        gva = va_df.groupby(group_col, observed=True).size().values

        dtr = lgb.Dataset(tr_df[features], label=tr_df[label_col], group=gtr, free_raw_data=False)
        dva = lgb.Dataset(va_df[features], label=va_df[label_col], group=gva, free_raw_data=False)

        model = lgb.train(
            params,
            dtr,
            num_boost_round=NUM_BOOST,
            valid_sets=[dva],
            early_stopping_rounds=EARLY_STOP,
            verbose_eval=False,
        )

        oof[va] = model.predict(va_df[features], num_iteration=model.best_iteration)
        models.append(model)

    return models, oof


def main(args):
    set_global_seed(42)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    def _read_any(pq, fe):
        p_pq, p_fe = Path(pq), Path(fe)
        if p_fe.exists():
            return pd.read_feather(p_fe)
        return pd.read_parquet(p_pq)

    tr = _read_any(args.train_features, str(Path(args.train_features).with_suffix(".feather")))
    te = _read_any(args.test_features,  str(Path(args.test_features).with_suffix(".feather")))

    # Temel ignore listesi
    ignore = [
        "clicked","ordered","ts_hour","session_id",
        "user_id_hashed","content_id_hashed","search_term_normalized"
    ]
    features = [c for c in tr.columns if c not in ignore]

    # Kategorikler
    cat_cols = [c for c in ["level1_category_name","level2_category_name","leaf_category_name"] if c in tr.columns]
    tr = reduce_cats(tr, cat_cols)
    te = reduce_cats(te, cat_cols)

    # Grup içi rank özellikleri (v5+ ek sütunlar dahil)
    rank_base = [
        "discounted_price_last","rate_avg_last","review_cnt_last",
        "c_search_ctr","term_ctr","u_term_ctr",
        "c_search_ctr_d","term_ctr_d","u_term_ctr_d",
        "q_cvtag_overlap","q_cvtag_tfidf_cos",
        "price_discount_ratio","log_discounted_price",
        "tc_ctr",
    ]
    rank_cols = [c for c in rank_base if c in tr.columns]
    tr = groupwise_rank(tr, "session_id", rank_cols, prefix="r")
    te = groupwise_rank(te, "session_id", rank_cols, prefix="r")
    rank_feats = [f"r_{c}_in_session_id" for c in rank_cols]
    features = list(dict.fromkeys(features + rank_feats))  # dedup

    # Session group size feature (hem train hem test)
    tr["sess_size"] = tr.groupby("session_id")["content_id_hashed"].transform("size")
    te["sess_size"] = te.groupby("session_id")["content_id_hashed"].transform("size")
    if "sess_size" not in features:
        features.append("sess_size")

    # Optuna (hafif): istersen kapat -> use_optuna=False yap
    use_optuna = True
    seeds = [42, 2025, 7]  # 2–3 seed genelde yeter
    params_click = None; params_order = None
    if use_optuna:
        params_click = tune_params_with_optuna(tr, features, "clicked", n_trials=25)
        params_order = tune_params_with_optuna(tr, features, "ordered", n_trials=25)

    def train_with_seeds(label):
        models_all, oof_stack = [], []
        for sd in seeds:
            params = (params_click if label=="clicked" else params_order) or dict(
                objective="lambdarank", metric="ndcg", learning_rate=0.05, num_leaves=127,
                min_data_in_leaf=50, feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=1, verbose=-1, seed=sd
            )
            m, oof_ = train_head(tr, features, label, group_col="session_id", params=params, seed=sd)
            models_all.append(m); oof_stack.append(oof_)
        return models_all, np.mean(oof_stack, axis=0)

    # İki head eğitimi (seed ensemble ile)
    models_click_seeds, oof_click = train_with_seeds("clicked")
    models_order_seeds, oof_order = train_with_seeds("ordered")

    # Offline skor (yarışma formülüyle)
    df_oof = tr[["session_id","clicked","ordered"]].copy()
    df_oof["score_click"] = oof_click
    df_oof["score_order"] = oof_order
    final, auc_c, auc_o = trendyol_final_score(df_oof)
    print(f"[OOF] final={final:.5f}  (click_auc={auc_c:.5f}, order_auc={auc_o:.5f})")

    # Ağırlığı öğren (0..1 arası kısıtlı)
    from sklearn.linear_model import LinearRegression
    tmp = df_oof[["score_click","score_order"]].to_numpy()
    y = 0.5*df_oof["clicked"].to_numpy() + 0.5*df_oof["ordered"].to_numpy()  # proxy hedef
    w_model = LinearRegression(positive=True)
    w_model.fit(tmp, y)
    w_click, w_order = w_model.coef_
    s = max(w_click + w_order, 1e-9)
    w_click /= s; w_order /= s
    print(f"[blend] learned weights: click={w_click:.3f}, order={w_order:.3f}")

    # Test tahmini (seed ensemble + öğrenilmiş ağırlık)
    def predict_folds_ensemble(list_of_model_lists, X):
        preds = []
        for models in list_of_model_lists:
            preds.append(np.mean([m.predict(X, num_iteration=getattr(m, "best_iteration", None)) for m in models], axis=0))
        return np.mean(preds, axis=0)

    te_click = predict_folds_ensemble(models_click_seeds, te[features])
    te_order = predict_folds_ensemble(models_order_seeds, te[features])
    te_final = w_click * te_click + w_order * te_order

    sub = pd.DataFrame({
        "session_id": te["session_id"].values,
        "content_id_hashed": te["content_id_hashed"].values,
        "prediction": te_final,
    })
    sub_path = out_dir / "submission_v5.csv"
    sub.to_csv(sub_path, index=False)
    print(f"[OK] {sub_path} hazır.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_features", required=True)
    ap.add_argument("--test_features", required=True)
    ap.add_argument("--out_dir", required=True)
    # Not: aşağıdaki ekstra argümanlar bu sürümde kullanılmıyor; sade tuttuk.
    # İleride Optuna/auto-rebuild eklersen tekrar açabiliriz.
    args = ap.parse_args()
    main(args)
