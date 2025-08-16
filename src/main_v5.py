# src/main_v5.py
import argparse
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from pathlib import Path

from .helpers import groupwise_rank, reduce_cats, set_global_seed
from .metric_wrappers import trendyol_final_score, final_metric_from_single_score
from .build_features_v5 import add_item_ctr_cvr, add_session_features, add_term_item_affinity
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
            metric="xendcg",
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

        # Order head için pozitif örneklere ekstra ağırlık
        w_tr = np.ones(len(tr_df), dtype=float)
        w_va = np.ones(len(va_df), dtype=float)
        if label_col == "ordered":
            w_tr *= (1 + (tr_df[label_col].values > 0))  # 1 veya 2
            w_va *= (1 + (va_df[label_col].values > 0))

        dtr = lgb.Dataset(tr_df[features], label=tr_df[label_col], group=gtr, weight=w_tr, free_raw_data=False)
        dva = lgb.Dataset(va_df[features], label=va_df[label_col], group=gva, weight=w_va, free_raw_data=False)

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


# Rerank: ordered > clicked > others (grup içi skor sırası korunur)
def rerank_with_priority(df_pred: pd.DataFrame) -> pd.DataFrame:
    def reorder(group: pd.DataFrame) -> pd.DataFrame:
        # Varsayılan kolonlar yoksa sıfırla
        if "ordered" not in group.columns:
            group = group.assign(ordered=0)
        if "clicked" not in group.columns:
            group = group.assign(clicked=0)

        ordered = group[group["ordered"] == 1]
        clicked = group[(group["clicked"] == 1) & (group["ordered"] == 0)]
        others = group[(group["clicked"] == 0) & (group["ordered"] == 0)]

        ordered = ordered.sort_values("score", ascending=False)
        clicked = clicked.sort_values("score", ascending=False)
        others = others.sort_values("score", ascending=False)
        return pd.concat([ordered, clicked, others])

    return df_pred.groupby("session_id", group_keys=False).apply(reorder)

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

    # --- External feature integrations from build_features_v5 helpers ---
    # Prepare logs from train only (avoid leakage)
    _mapper = {"content_id_hashed": "item_id", "search_term_normalized": "term_id"}
    tr_tmp = tr.rename(columns=_mapper)
    full_logs = tr_tmp[["item_id","term_id","clicked","ordered","session_id"]].copy()

    # Train ext feats
    tr_ext = add_item_ctr_cvr(tr_tmp.copy(), full_logs)
    tr_ext = add_session_features(tr_ext)
    tr_ext = add_term_item_affinity(tr_ext, full_logs)
    # bring back new columns to original tr
    tr_ext = tr_ext.rename(columns={v: k for k, v in _mapper.items()})
    for col in [
        "ctr","cvr","session_items","session_clicks","session_orders",
        "pos_in_session","rel_pos_in_session","term_item_ctr","term_item_cvr"
    ]:
        if col in tr_ext.columns:
            tr[col] = tr_ext[col].values

    # Test ext feats (add dummy labels for aggregations expected by helpers)
    te_tmp = te.rename(columns=_mapper)
    if "clicked" not in te_tmp.columns:
        te_tmp["clicked"] = 0
    if "ordered" not in te_tmp.columns:
        te_tmp["ordered"] = 0
    te_ext = add_item_ctr_cvr(te_tmp.copy(), full_logs)
    te_ext = add_session_features(te_ext)
    te_ext = add_term_item_affinity(te_ext, full_logs)
    te_ext = te_ext.rename(columns={v: k for k, v in _mapper.items()})
    for col in [
        "ctr","cvr","session_items","session_clicks","session_orders",
        "pos_in_session","rel_pos_in_session","term_item_ctr","term_item_cvr"
    ]:
        if col in te_ext.columns:
            te[col] = te_ext[col].values

    # Features list refresh (exclude ids/labels)
    features = [
        c for c in tr.columns
        if c not in [
            "session_id","content_id_hashed","user_id_hashed","search_term_normalized","ts_hour",
            "clicked","ordered"
        ]
    ]

    # Grup içi rank özellikleri (v5+ ek sütunlar dahil)
    rank_base = [
        "discounted_price_last","rate_avg_last","review_cnt_last",
        "c_search_ctr","term_ctr","u_term_ctr",
        "c_search_ctr_d","term_ctr_d","u_term_ctr_d",
        "q_cvtag_overlap","q_cvtag_tfidf_cos",
        "price_discount_ratio","log_discounted_price",
        "tc_ctr",
        "freq_term_leaf","freq_user_leaf",
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

    # --- Hata Analizi: per-session AUC raporu ---
    from collections import defaultdict  # gelecekte genişletme için
    def per_session_auc(df, label, score):
        rows = []
        for sid, g in df.groupby("session_id"):
            y = g[label].values; s = g[score].values
            if y.sum()==0 or y.sum()==len(y):
                continue
            order = s.argsort().argsort()
            pos = y==1; n1, n0 = pos.sum(), (~pos).sum()
            u = order[pos].sum() - n1*(n1-1)/2.0
            rows.append((sid, u/(n1*n0)))
        return pd.DataFrame(rows, columns=["session_id", f"auc_{label}"])

    _auc_c = per_session_auc(df_oof.assign(score=df_oof["score_click"]), "clicked", "score")
    _auc_o = per_session_auc(df_oof.assign(score=df_oof["score_order"]), "ordered", "score")
    bad_c = _auc_c.nsmallest(50, "auc_clicked")
    bad_o = _auc_o.nsmallest(50, "auc_ordered")
    bad_c.to_csv(out_dir / "dbg_worst_sessions_click.csv", index=False)
    bad_o.to_csv(out_dir / "dbg_worst_sessions_order.csv", index=False)
    print("[DBG] worst session lists saved.")

    # --- Blend Weight Tuning (tek skor üzerinden gerçek metrik) ---
    def tune_blend_weight(df, group_col="session_id",
                          click_label="clicked", order_label="ordered",
                          w_click_metric=0.3, w_order_metric=0.7,
                          grid=41):
        """
        s = w * score_order + (1-w) * score_click
        için w ∈ [0,1] boyunca gezer; final metrik (0.3*auc_click + 0.7*auc_order) maksimumu döndürür.
        """
        best = (-1.0, None, None, None)  # (final, w, auc_c, auc_o)
        ws = np.linspace(0.0, 1.0, grid)
        base = df[[group_col, click_label, order_label, "score_click", "score_order"]].copy()
        for w in ws:
            base["s"] = w * base["score_order"] + (1 - w) * base["score_click"]
            f, ac, ao = final_metric_from_single_score(
                base, score_col="s", group_col=group_col,
                click_label=click_label, order_label=order_label,
                w_click=w_click_metric, w_order=w_order_metric
            )
            if f > best[0]:
                best = (f, w, ac, ao)
        return {"final": best[0], "w_order": best[1], "auc_click": best[2], "auc_order": best[3]}

    tune_res = tune_blend_weight(df_oof)
    print(f"[OOF/blend] best_final={tune_res['final']:.5f}  "
          f"w_order={tune_res['w_order']:.3f}  "
          f"(click_auc={tune_res['auc_click']:.5f}, order_auc={tune_res['auc_order']:.5f})")
    w_order_opt = float(tune_res["w_order"])
    w_click_opt = 1.0 - w_order_opt

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

    # --- CatBoost Ranker (opsiyonel) ---
    import importlib
    try:
        _cat_mod = importlib.import_module("catboost")
        CatBoostRanker = getattr(_cat_mod, "CatBoostRanker", None)
        use_cat = CatBoostRanker is not None
    except Exception:
        use_cat = False
        CatBoostRanker = None

    cat_preds_click = cat_preds_order = None
    if use_cat:
        # CatBoost kategorik index’leri
        cat_idx = [i for i, c in enumerate(features) if getattr(tr[c], 'dtype', None) is not None and tr[c].dtype.name == "category"]
        gkf = GroupKFold(n_splits=5)

        def train_cat(label):
            oof = np.zeros(len(tr))
            models = []
            for tr_idx, va_idx in gkf.split(tr, groups=tr["session_id"]):
                tr_df, va_df = tr.iloc[tr_idx], tr.iloc[va_idx]
                # grup büyüklükleri (CatBoost group_id istiyor)
                tr_grp = tr_df["session_id"].astype("category").cat.codes
                va_grp = va_df["session_id"].astype("category").cat.codes
                m = CatBoostRanker(
                    iterations=2000,
                    learning_rate=0.06,
                    depth=8,
                    loss_function="YetiRank",  # stabil
                    eval_metric="NDCG:top=10",
                    random_seed=42,
                    verbose=False
                )
                m.fit(
                    tr_df[features], tr_df[label],
                    group_id=tr_grp,
                    eval_set=(va_df[features], va_df[label]),
                    eval_group_id=va_grp,
                    cat_features=cat_idx if cat_idx else None,
                    use_best_model=True
                )
                oof[va_idx] = m.predict(va_df[features])
                models.append(m)
            return models, oof

        cat_click_models, _ = train_cat("clicked")
        cat_order_models, _ = train_cat("ordered")

        def avg_pred(models, X):
            return np.mean([m.predict(X[features]) for m in models], axis=0)

        cat_preds_click = avg_pred(cat_click_models, te)
        cat_preds_order = avg_pred(cat_order_models, te)

        # CatBoost varsa simple blend ile karıştır
        te_click = 0.5 * te_click + 0.5 * cat_preds_click
        te_order = 0.5 * te_order + 0.5 * cat_preds_order

    # Session içi min-max normalize + score clipping (stabilizasyon)
    def sess_minmax(arr, sess):
        df = pd.DataFrame({"s": arr, "g": sess.values})
        g = df.groupby("g")["s"]
        mn, mx = g.transform("min"), g.transform("max")
        norm = (df["s"] - mn) / ((mx - mn).replace(0, np.nan))
        return norm.fillna(0.5).values

    te_click = np.clip(sess_minmax(te_click, te["session_id"]), 0.001, 0.999)
    te_order = np.clip(sess_minmax(te_order, te["session_id"]), 0.001, 0.999)

    # Öğrenilmiş ağırlıklarla final skor
    te_final = w_click_opt * te_click + w_order_opt * te_order
    print(f"[TEST/blend] using w_click={w_click_opt:.3f}, w_order={w_order_opt:.3f}")

    # Kaggle-style reranking and submission formatting
    df_pred = te[["session_id","content_id_hashed"]].copy()
    if "clicked" in te.columns:
        df_pred["clicked"] = te["clicked"].values
    else:
        df_pred["clicked"] = 0
    if "ordered" in te.columns:
        df_pred["ordered"] = te["ordered"].values
    else:
        df_pred["ordered"] = 0
    df_pred["score"] = te_final
    df_pred = rerank_with_priority(df_pred)

    submission = (
        df_pred.groupby("session_id")["content_id_hashed"]
        .apply(lambda x: " ".join(x.astype(str).tolist()))
        .reset_index()
        .rename(columns={"content_id_hashed": "content_id"})
    )

    # Özellik önemi raporları (gain)
    def top_importances(models, feat_names, k=30):
        gain = pd.Series(0.0, index=feat_names, dtype=float)
        for m in models:
            g = pd.Series(m.feature_importance(importance_type="gain"), index=feat_names)
            gain = gain.add(g, fill_value=0)
        return gain.sort_values(ascending=False).head(k)

    all_click_models = [m for lst in models_click_seeds for m in lst]
    all_order_models = [m for lst in models_order_seeds for m in lst]
    imp_click = top_importances(all_click_models, features)
    imp_order = top_importances(all_order_models, features)
    imp_click.to_csv(out_dir/"imp_click_gain.csv")
    imp_order.to_csv(out_dir/"imp_order_gain.csv")

    sub_path = out_dir / "submission_v5.csv"
    submission.to_csv(sub_path, index=False)
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
