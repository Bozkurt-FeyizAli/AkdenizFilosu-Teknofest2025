# -*- coding: utf-8 -*-
import argparse, numpy as np, pandas as pd, lightgbm as lgb, xgboost as xgb
from pathlib import Path

from helpers_v6 import (
    reduce_cats, reduce_mem_usage, safe_div,
    tfidf_cosine_sparse, aggregate_with_decay, covisitation_pairs,
    add_price_review_features,            # ← YENİ
    add_user_term_category_features,      # ← YENİ
    add_time_patterns,                   # ← YENİ
    reduce_cats, reduce_mem_usage,
    safe_div,
    tfidf_cosine_sparse,           # TF-IDF(q, cv_tags) cosine
    aggregate_with_decay,          # λ-decay agregat
    covisitation_pairs, 
)


from metric_wrappers_v6 import trendyol_final


# -------------------------- time-aware folds --------------------------
def _time_group_folds(df, n_splits=5, sess_col="session_id", ts_col="ts_hour"):
    ss = df.groupby(sess_col)[ts_col].min().sort_values().index.values
    chunks = np.array_split(ss, n_splits)
    folds = []
    for i in range(n_splits):
        val_sess = set(chunks[i])
        tr_idx = df.index[~df[sess_col].isin(val_sess)].values
        va_idx = df.index[df[sess_col].isin(val_sess)].values
        folds.append((tr_idx, va_idx))
    return folds


# -------------------------- Feature Augmenters (v6.1) --------------------------
def add_tfidf_similarity(df, data_dir):
    """ search_term_normalized ↔ cv_tags TF-IDF cosine """
    meta = pd.read_parquet(Path(data_dir) / "content/metadata.parquet",
                           columns=["content_id_hashed","cv_tags"])
    tmp = df.merge(meta, on="content_id_hashed", how="left")
    sim = tfidf_cosine_sparse(
        tmp["search_term_normalized"].astype(str),
        tmp["cv_tags"].astype(str),
        max_features=50_000
    )
    df["tfidf_q_cv"] = sim.astype("float32")
    return df

def add_decay_aggregates(df, data_dir):
    """ λ-decay ile içerik / terim / kullanıcı agregatları """
    # content/sitewide_log: decay total_click, total_order → join by content_id
    c_site = pd.read_parquet(Path(data_dir)/"content/sitewide_log.parquet",
                             columns=["date","content_id_hashed","total_click","total_order"])
    c_site["date"] = pd.to_datetime(c_site["date"])
    agg_click = aggregate_with_decay(c_site.rename(columns={"date":"ts"}),
                                     ["content_id_hashed"], "total_click", "ts",
                                     half_life=7*24*3600)
    agg_order = aggregate_with_decay(c_site.rename(columns={"date":"ts"}),
                                     ["content_id_hashed"], "total_order", "ts",
                                     half_life=14*24*3600)
    agg_c = agg_click.merge(agg_order, on="content_id_hashed", how="outer").fillna(0)
    df = df.merge(agg_c, on="content_id_hashed", how="left")

    # term/search_log: decay impression/click → CTR
    tlog = pd.read_parquet(Path(data_dir)/"term/search_log.parquet",
                           columns=["ts_hour","search_term_normalized",
                                    "total_search_impression","total_search_click"])
    tlog["ts_hour"] = pd.to_datetime(tlog["ts_hour"])
    dimp = aggregate_with_decay(tlog.rename(columns={"ts_hour":"ts"}),
                                ["search_term_normalized"], "total_search_impression", "ts",
                                half_life=10*24*3600)
    dclk = aggregate_with_decay(tlog.rename(columns={"ts_hour":"ts"}),
                                ["search_term_normalized"], "total_search_click", "ts",
                                half_life=10*24*3600)
    dt = dimp.merge(dclk, on="search_term_normalized", how="outer").fillna(0)
    df["term_ctr_decay"] = 0.0
    df = df.merge(
        dt.assign(term_ctr_decay=safe_div(dt["total_search_click_decay"],
                                          dt["total_search_impression_decay"]))[
            ["search_term_normalized","term_ctr_decay"]],
        on="search_term_normalized", how="left"
    )

    # user/sitewide_log: decay order/click as user recency preference
    u_site = pd.read_parquet(Path(data_dir)/"user/sitewide_log.parquet",
                             columns=["ts_hour","user_id_hashed","total_click","total_order"])
    u_site["ts_hour"] = pd.to_datetime(u_site["ts_hour"])
    uo = aggregate_with_decay(u_site.rename(columns={"ts_hour":"ts"}),
                              ["user_id_hashed"], "total_order", "ts",
                              half_life=21*24*3600)
    uc = aggregate_with_decay(u_site.rename(columns={"ts_hour":"ts"}),
                              ["user_id_hashed"], "total_click", "ts",
                              half_life=7*24*3600)
    du = uo.merge(uc, on="user_id_hashed", how="outer").fillna(0)
    df = df.merge(du, on="user_id_hashed", how="left")
    return df

def add_cross_counts(df_train_sessions, tr, te):
    """ User×Category ve User×Term sayımları (train geçmişinden) """
    tdf = df_train_sessions[["user_id_hashed","search_term_normalized",
                             "content_id_hashed","ts_hour"]].copy()
    # leaf_category_name’ı tr’den çek
    catmap = tr[["content_id_hashed","leaf_category_name"]].drop_duplicates()
    tdf = tdf.merge(catmap, on="content_id_hashed", how="left")

    uc = tdf.groupby(["user_id_hashed","leaf_category_name"]).size().reset_index(name="uxcat_cnt")
    ut = tdf.groupby(["user_id_hashed","search_term_normalized"]).size().reset_index(name="uxterm_cnt")

    tr = tr.merge(uc, on=["user_id_hashed","leaf_category_name"], how="left") \
           .merge(ut, on=["user_id_hashed","search_term_normalized"], how="left")
    te = te.merge(uc, on=["user_id_hashed","leaf_category_name"], how="left") \
           .merge(ut, on=["user_id_hashed","search_term_normalized"], how="left")
    tr[["uxcat_cnt","uxterm_cnt"]] = tr[["uxcat_cnt","uxterm_cnt"]].fillna(0).astype("float32")
    te[["uxcat_cnt","uxterm_cnt"]] = te[["uxcat_cnt","uxterm_cnt"]].fillna(0).astype("float32")
    return tr, te

def add_covis_score(tr, te, data_dir):
    """ Basit co-visitation skoru: item-item pair sayımlarından normalize skor """
    ufl = pd.read_parquet(Path(data_dir)/"user/fashion_sitewide_log.parquet",
                          columns=["ts_hour","user_id_hashed","content_id_hashed","total_click"])
    ufl = ufl.dropna(subset=["user_id_hashed","content_id_hashed"])
    pairs = covisitation_pairs(ufl.rename(columns={"ts_hour":"ts"}),
                               user_col="user_id_hashed",
                               item_col="content_id_hashed",
                               time_col="ts", max_window=30)
    if pairs.empty:
        tr["item_covis_score"] = 0.0; te["item_covis_score"] = 0.0
        return tr, te
    a_sum = pairs.groupby("content_id_hashed_A")["cnt"].transform("sum")
    pairs["covis_score"] = pairs["cnt"] / (a_sum + 1e-9)
    item_score = pairs.groupby("content_id_hashed_B")["covis_score"].mean().reset_index()
    item_score = item_score.rename(columns={"content_id_hashed_B":"content_id_hashed",
                                            "covis_score":"item_covis_score"})
    tr = tr.merge(item_score, on="content_id_hashed", how="left")
    te = te.merge(item_score, on="content_id_hashed", how="left")
    tr["item_covis_score"] = tr["item_covis_score"].fillna(0).astype("float32")
    te["item_covis_score"] = te["item_covis_score"].fillna(0).astype("float32")
    return tr, te


# -------------------------- Trainers & Ensembling --------------------------
NUM_BOOST=4000; ES=200

def _lgb_train(dtr, dva, params, rounds=NUM_BOOST, es=ES):
    return lgb.train(
        params, dtr, num_boost_round=rounds, valid_sets=[dva],
        callbacks=[lgb.early_stopping(es), lgb.log_evaluation(period=0)]
    )

def train_ranker_lgb(df, feats, label):
    params = dict(
        objective="lambdarank", metric="ndcg", learning_rate=0.05,
        num_leaves=127, min_data_in_leaf=60, feature_fraction=0.9,
        bagging_fraction=0.9, bagging_freq=1, verbose=-1, seed=42,
        eval_at=[10,20,50]
    )
    oof = np.zeros(len(df)); models=[]
    for tr_idx, va_idx in _time_group_folds(df, n_splits=5):
        tr, va = df.iloc[tr_idx], df.iloc[va_idx]
        gtr = tr.groupby("session_id", observed=True).size().values
        gva = va.groupby("session_id", observed=True).size().values
        dtr = lgb.Dataset(tr[feats], label=tr[label], group=gtr, free_raw_data=False)
        dva = lgb.Dataset(va[feats], label=va[label], group=gva, free_raw_data=False)
        m = _lgb_train(dtr, dva, params)
        oof[va_idx] = m.predict(va[feats], num_iteration=m.best_iteration)
        models.append(m)
    return models, oof

def train_ranker_xgb(df, feats, label):
    oof = np.zeros(len(df)); models=[]
    for tr_idx, va_idx in _time_group_folds(df, n_splits=5):
        tr, va = df.iloc[tr_idx], df.iloc[va_idx]
        gtr = tr.groupby("session_id", observed=True).size().tolist()
        gva = va.groupby("session_id", observed=True).size().tolist()
        m = xgb.XGBRanker(
            objective="rank:ndcg", eval_metric="ndcg", learning_rate=0.06,
            max_depth=8, subsample=0.9, colsample_bytree=0.9, min_child_weight=30,
            n_estimators=2000, tree_method="hist", random_state=42
        )
        m.fit(tr[feats], tr[label], group=gtr,
              eval_set=[(va[feats], va[label])], eval_group=[gva], verbose=False)
        oof[va_idx] = m.predict(va[feats])
        models.append(m)
    return models, oof

def blend(oof_click_lgb, oof_order_lgb, oof_click_xgb, oof_order_xgb, df_labels):
    best = (-1, 0, 0)  # score, w_lgb_click, w_lgb_order
    base = df_labels[["session_id","clicked","ordered"]].copy()
    for w1 in np.linspace(0,1,11):
        for w2 in np.linspace(0,1,11):
            tmp = base.copy()
            tmp["p_click"] = w1*oof_click_lgb + (1-w1)*oof_click_xgb
            tmp["p_order"] = w2*oof_order_lgb + (1-w2)*oof_order_xgb
            f,_,_ = trendyol_final(tmp)
            if f > best[0]:
                best = (f, w1, w2)
    return best

def predict_ensemble(models_lgb_c, models_lgb_o, models_xgb_c, models_xgb_o, test, feats, w1, w2):
    def _avg(models, X):
        preds = []
        for m in models:
            try:
                preds.append(m.predict(X, num_iteration=m.best_iteration))
            except TypeError:
                preds.append(m.predict(X))
        return np.mean(preds, axis=0)
    pc = w1*_avg(models_lgb_c, test[feats]) + (1-w1)*_avg(models_xgb_c, test[feats])
    po = w2*_avg(models_lgb_o, test[feats]) + (1-w2)*_avg(models_xgb_o, test[feats])
    return 0.3*pc + 0.7*po  # yarışmanın 0.3/0.7 finali


# -------------------------- Main --------------------------
def main(args):
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    tr = pd.read_parquet(args.train_features)
    te = pd.read_parquet(args.test_features)

    # === v6.1 feature augment ===
    tr = add_tfidf_similarity(tr, args.data_dir)
    te = add_tfidf_similarity(te, args.data_dir)

    tr = add_decay_aggregates(tr, args.data_dir)
    te = add_decay_aggregates(te, args.data_dir)

    train_sessions = pd.read_parquet(Path(args.data_dir)/"train_sessions.parquet",
                                     columns=["user_id_hashed","content_id_hashed","search_term_normalized","ts_hour"])
    tr, te = add_cross_counts(train_sessions, tr, te)

    tr, te = add_covis_score(tr, te, args.data_dir)

    # 5) Price/Review/Discount özellikleri
    tr = add_price_review_features(tr, args.data_dir)
    te = add_price_review_features(te, args.data_dir)

    # 6) User×Term & User×Category tercihleri
    tr, te = add_user_term_category_features(tr, te, args.data_dir)

    # 7) Zaman kalıpları (hour/dow tabanlı tercih sinyalleri)
    tr = add_time_patterns(tr)
    te = add_time_patterns(te)

    # model input
    ignore = ["clicked","ordered","ts_hour","session_id","user_id_hashed","content_id_hashed","search_term_normalized"]
    feats = [c for c in tr.columns if c not in ignore]
    cat_cols = [c for c in ["level1_category_name","level2_category_name","leaf_category_name"] if c in tr.columns]
    tr = reduce_cats(tr, cat_cols); te = reduce_cats(te, cat_cols)
    tr = reduce_mem_usage(tr); te = reduce_mem_usage(te)

    # Ranker eğitimleri (click/order ayrı)
    lgb_c, oof_c_lgb = train_ranker_lgb(tr, feats, "clicked")
    lgb_o, oof_o_lgb = train_ranker_lgb(tr, feats, "ordered")
    xgb_c, oof_c_xgb = train_ranker_xgb(tr, feats, "clicked")
    xgb_o, oof_o_xgb = train_ranker_xgb(tr, feats, "ordered")

    # OOF blend ağırlık araması
    best_f, w_c, w_o = blend(oof_c_lgb, oof_o_lgb, oof_c_xgb, oof_o_xgb, tr)
    print(f"[OOF v6.1] final={best_f:.6f}  (best w_click={w_c:.2f}, w_order={w_o:.2f})")

    # Test tahmini ve submission
    te_final = predict_ensemble(lgb_c, lgb_o, xgb_c, xgb_o, te, feats, w_c, w_o)
    sub = pd.DataFrame({
        "session_id": te["session_id"].values,
        "content_id_hashed": te["content_id_hashed"].values,
        "prediction": te_final.astype("float32")
    })
    sub.to_csv(out/"submission_v6.csv", index=False)
    print("[OK] outputs/submission_v6.csv yazıldı.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_features", required=True)
    ap.add_argument("--test_features", required=True)
    ap.add_argument("--data_dir", required=True, help="Parquet kök klasörü (data/)")
    ap.add_argument("--out_dir", required=True)
    main(ap.parse_args())
