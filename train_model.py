# train_model.py
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import roc_auc_score
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
pd.set_option("display.max_columns", 200)
warnings.filterwarnings("ignore", category=FutureWarning)  # 👈 eklendi
warnings.filterwarnings("ignore", category=UserWarning)

# ===== helpers =====
def safe_div(a, b):
    a = a.astype(float); b = b.astype(float)
    out = np.zeros_like(a, dtype=float)
    mask = b > 0
    out[mask] = a[mask] / b[mask]
    return out

def season_from_month(m):
    if m in [12,1,2]: return 0
    if m in [3,4,5]:  return 1
    if m in [6,7,8]:  return 2
    return 3

def add_time_feats(df, col="ts_hour"):
    dt = pd.to_datetime(df[col])
    df["hour"] = dt.dt.hour
    df["dow"] = dt.dt.dayofweek
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["month"] = dt.dt.month
    df["season"] = df["month"].apply(season_from_month)
    return df

def latest_by_key_fast(df, key, timecol):
    idx = df.groupby(key)[timecol].idxmax()
    return df.loc[idx].reset_index(drop=True)

def group_sizes_by_session(df):
    # 👇 observed=True eklendi (FutureWarning gider)
    return df.groupby("session_id", observed=True).size().values


def add_session_price_position(df):
    tmp = df[["session_id", "discounted_price"]].copy()
    # 👇 observed=True ekledik (sadece mevcut grupları dikkate alır; uyarı yok)
    tmp["price_rank"] = tmp.groupby("session_id", observed=True)["discounted_price"].rank(method="average")
    tmp["price_mean"] = tmp.groupby("session_id", observed=True)["discounted_price"].transform("mean")
    tmp["price_std"]  = tmp.groupby("session_id", observed=True)["discounted_price"].transform("std").replace(0, np.nan)
    tmp["price_z"]    = (tmp["discounted_price"] - tmp["price_mean"]) / tmp["price_std"]
    return df.join(tmp[["price_rank", "price_z"]])


def robust_pref_table(df, group_keys, price_col="discounted_price"):
    if df.empty:
        return pd.DataFrame(columns=group_keys + ["price_median","q25","q75","band_width"])
    gb = df.groupby(group_keys)[price_col]
    out = gb.agg(
        price_median="median",
        q25=lambda s: s.quantile(0.25),
        q75=lambda s: s.quantile(0.75),
    ).reset_index()
    out["band_width"] = (out["q75"] - out["q25"]).replace(0, np.nan)
    return out

def merge_pref_with_fallback(base_df, pref_tables, keys_list):
    X = base_df.copy()
    for c in ["price_median","q25","q75","band_width"]:
        if c not in X.columns: X[c] = np.nan
    filled = pd.Series(False, index=X.index)
    for pref, keys in zip(pref_tables, keys_list):
        if pref is None or pref.empty: continue
        need = ~filled
        if need.any():
            cols_to_merge = keys + ["price_median","q25","q75","band_width"]
            tmp = X.loc[need, keys].merge(pref[cols_to_merge], on=keys, how="left")
            idx = X.index[need]
            for c in ["price_median","q25","q75","band_width"]:
                vals = tmp[c].values
                write_mask = ~pd.isna(vals)
                X.loc[idx[write_mask], c] = vals[write_mask]
            filled = ~X["price_median"].isna()

    bw = X["band_width"]
    if bw.isna().all(): bw = pd.Series(0.1, index=X.index)
    bw = bw.fillna(bw.median() if not np.isnan(bw.median()) else 0.1)
    bw = bw.replace(0, bw.median() if not np.isnan(bw.median()) else 0.1)

    X["price_delta"] = X["discounted_price"] - X["price_median"]
    X.loc[X["price_median"].isna(), "price_delta"] = np.nan
    X["abs_price_delta"] = X["price_delta"].abs()
    X["z_price_delta"] = X["price_delta"] / (bw + 1e-6)
    X["in_band"] = ((X["discounted_price"] >= X["q25"]) & (X["discounted_price"] <= X["q75"])).astype(float)
    X["price_gauss_affinity"] = np.exp(-((X["abs_price_delta"]) / (bw + 1e-6)) ** 2)

    for col in ["price_delta","abs_price_delta","z_price_delta","in_band","price_gauss_affinity"]:
        X[col] = X[col].fillna(0.0)
    return X

def chunked_merge(left_df, right_df, on_cols, how="left", chunk_size=500_000):
    """Right küçük/orta, left büyükken RAM dostu merge."""
    parts = []
    for start in range(0, len(left_df), chunk_size):
        chunk = left_df.iloc[start:start+chunk_size]
        parts.append(chunk.merge(right_df, on=on_cols, how=how))
    return pd.concat(parts, ignore_index=True)

# ===== main =====
def main(args):
    DATA = Path(args.data_dir)

    print("[1/9] train/test parquet okunuyor...")
    train = pd.read_parquet(
        DATA / "train_sessions.parquet",
        columns=["ts_hour","search_term_normalized","clicked","ordered","user_id_hashed","content_id_hashed","session_id"],
    )
    test = pd.read_parquet(
        DATA / "test_sessions.parquet",
        columns=["ts_hour","search_term_normalized","user_id_hashed","content_id_hashed","session_id"],
    )
    for df in [train, test]:
        for c in ["user_id_hashed","content_id_hashed","search_term_normalized","session_id"]:
            df[c] = df[c].astype("category")
    train = add_time_feats(train, "ts_hour")
    test  = add_time_feats(test, "ts_hour")

    print("[2/9] content tabloları okunuyor...")
    meta = pd.read_parquet(
        DATA / "content" / "metadata.parquet",
        columns=["content_id_hashed","level1_category_name","level2_category_name","leaf_category_name",
                 "attribute_type_count","total_attribute_option_count","merchant_count","filterable_label_count",
                 "content_creation_date"],
    )
    price = pd.read_parquet(
        DATA / "content" / "price_rate_review_data.parquet",
        columns=["content_id_hashed","update_date","original_price","selling_price","discounted_price",
                 "content_review_count","content_review_wth_media_count","content_rate_count","content_rate_avg"],
    )
    c_search_log = pd.read_parquet(
        DATA / "content" / "search_log.parquet",
        columns=["content_id_hashed","date","total_search_impression","total_search_click"],
    )
    c_sitewide_log = pd.read_parquet(
        DATA / "content" / "sitewide_log.parquet",
        columns=["content_id_hashed","date","total_click","total_cart","total_fav","total_order"],
    )
    print("[3/9] top_terms okunuyor ve tekilleştiriliyor...")
    top_terms = pd.read_parquet(
        DATA / "content" / "top_terms_log.parquet",
        columns=["content_id_hashed","search_term_normalized","total_search_impression","total_search_click"],
    )
    for c in ["content_id_hashed","search_term_normalized"]:
        top_terms[c] = top_terms[c].astype("category")
    top_terms_small = (
        top_terms.groupby(["content_id_hashed","search_term_normalized"], as_index=False, observed=True)
                 .agg(tt_search_impr=("total_search_impression","sum"),
                      tt_search_click=("total_search_click","sum"))
    )
    # düşük gösterimlileri kırp (opsiyonel ama RAM dostu)
    if len(top_terms_small) > 1_000_000:
        q = top_terms_small["tt_search_impr"].quantile(0.05)
        top_terms_small = top_terms_small[top_terms_small["tt_search_impr"] >= q]
    top_terms_small["term_ctr"] = safe_div(
        top_terms_small["tt_search_click"].astype("float32"),
        top_terms_small["tt_search_impr"].astype("float32")
    )
    top_terms_small["tt_search_impr"] = top_terms_small["tt_search_impr"].astype("float32")
    top_terms_small["tt_search_click"] = top_terms_small["tt_search_click"].astype("float32")
    # (⚠️ ÖNEMLİ) — Burada asla yeniden top_terms_small = top_terms.copy() YAPMIYORUZ.

    print("[4/9] content özetleri hazırlanıyor...")
    content_price = latest_by_key_fast(price, "content_id_hashed", "update_date")[
        ["content_id_hashed","original_price","selling_price","discounted_price",
         "content_review_count","content_review_wth_media_count","content_rate_count","content_rate_avg"]
    ].copy()
    content_search = latest_by_key_fast(c_search_log, "content_id_hashed", "date")[
        ["content_id_hashed","total_search_impression","total_search_click"]
    ].copy()
    content_search["content_search_ctr"] = safe_div(
        content_search["total_search_click"], content_search["total_search_impression"]
    )
    content_site = latest_by_key_fast(c_sitewide_log, "content_id_hashed", "date")[
        ["content_id_hashed","total_click","total_cart","total_fav","total_order"]
    ].copy()
    content_site["content_click_to_order"] = safe_div(content_site["total_order"], content_site["total_click"])
    content_site["content_cart_to_order"]  = safe_div(content_site["total_order"], content_site["total_cart"])

    print("[5/9] user tabloları okunuyor...")
    user_meta = pd.read_parquet(
        DATA / "user" / "metadata.parquet",
        columns=["user_id_hashed","user_gender","user_birth_year","user_tenure_in_days"],
    )
    u_site = pd.read_parquet(
        DATA / "user" / "sitewide_log.parquet",
        columns=["user_id_hashed","ts_hour","total_click","total_cart","total_fav","total_order"],
    )
    user_basic = user_meta.copy()
    user_last  = latest_by_key_fast(u_site, "user_id_hashed", "ts_hour")[
        ["user_id_hashed","total_click","total_cart","total_fav","total_order"]
    ].copy()
    user_last["user_click_to_order"] = safe_div(user_last["total_order"], user_last["total_click"])
    user_last["user_cart_to_order"]  = safe_div(user_last["total_order"], user_last["total_cart"])

    print("[6/9] büyük merge (chunked) başlıyor...")
    # Büyük merge: önce content/user taraflarını ekleyelim
    def build_matrix_base(df):
        X = (df.merge(meta, on="content_id_hashed", how="left")
               .merge(content_price, on="content_id_hashed", how="left")
               .merge(content_search, on="content_id_hashed", how="left")
               .merge(content_site,   on="content_id_hashed", how="left")
               .merge(user_basic, on="user_id_hashed", how="left")
               .merge(user_last,  on="user_id_hashed", how="left"))
        return X

    trainX = build_matrix_base(train)
    testX  = build_matrix_base(test)

    # Son olarak top_terms_small ile (content_id, term) bazında merge — CHUNKED
    join_cols = ["content_id_hashed","search_term_normalized"]
    for df in [trainX, testX, top_terms_small]:
        for c in join_cols:
            if c in df.columns:
                df[c] = df[c].astype("category")

    print("   - trainX ⨝ top_terms_small (chunk)...")
    trainX = chunked_merge(trainX, top_terms_small, on_cols=join_cols, how="left", chunk_size=400_000)
    print("   - testX  ⨝ top_terms_small (chunk)...")
    testX  = chunked_merge(testX,  top_terms_small, on_cols=join_cols, how="left", chunk_size=400_000)

    print("[7/9] feature mühendisliği...")
    for df in [trainX, testX]:
        ccd = pd.to_datetime(df["content_creation_date"], errors="coerce")
        ts  = pd.to_datetime(df["ts_hour"], errors="coerce")
        df["content_age_days"] = (ts - ccd).dt.days
        df.loc[df["content_age_days"].isna(), "content_age_days"] = df["content_age_days"].median()

    trainX = add_session_price_position(trainX)
    testX  = add_session_price_position(testX)

    cat_cols = ["level1_category_name","level2_category_name","leaf_category_name","user_gender"]
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    trainX[cat_cols] = enc.fit_transform(trainX[cat_cols].astype(str))
    testX[cat_cols]  = enc.transform(testX[cat_cols].astype(str))

    for df in [trainX, testX]:
        df["age"] = pd.to_datetime(df["ts_hour"]).dt.year - df["user_birth_year"].fillna(0)
        df.loc[(df["user_birth_year"]<=0) | (df["user_birth_year"].isna()), "age"] = -1
        df["discount_rate"] = np.where(
            (df["original_price"]>0) & (df["discounted_price"]>0),
            1 - (df["discounted_price"]/df["original_price"]), 0
        )

    num_cols = [
        "hour","dow","is_weekend","month","season",
        "attribute_type_count","total_attribute_option_count","merchant_count","filterable_label_count",
        "original_price","selling_price","discounted_price",
        "content_review_count","content_review_wth_media_count","content_rate_count","content_rate_avg",
        "total_search_impression","total_search_click","content_search_ctr",
        "total_click","total_cart","total_fav","total_order",
        "content_click_to_order","content_cart_to_order",
        "user_birth_year","user_tenure_in_days",
        "user_click_to_order","user_cart_to_order",
        "tt_search_impr","tt_search_click","term_ctr",
        "content_age_days","price_rank","price_z","age","discount_rate",
    ]
    for col in num_cols:
        if col not in trainX.columns: trainX[col] = np.nan
        if col not in testX.columns:  testX[col]  = np.nan
    for df in [trainX, testX]:
        df[num_cols] = df[num_cols].astype("float32").fillna(-1)

    print("[8/9] sızıntısız fiyat-tercih fallback hesaplanıyor...")
    cutoff = trainX["ts_hour"].quantile(0.8)
    hist = trainX[trainX["ts_hour"] < cutoff].copy()
    valid_part = trainX[trainX["ts_hour"] >= cutoff].copy()

    pref_base = hist[hist["ordered"]==1].copy()
    if pref_base.empty: pref_base = hist[hist["clicked"]==1].copy()

    pref_usr_season_cat = robust_pref_table(pref_base, ["user_id_hashed","season","leaf_category_name"])
    pref_usr_season     = robust_pref_table(pref_base, ["user_id_hashed","season"])
    pref_term_season    = robust_pref_table(pref_base, ["search_term_normalized","season"])
    pref_global_season  = robust_pref_table(pref_base, ["season"])

    trainX_valid_feats = merge_pref_with_fallback(
        valid_part,
        pref_tables=[pref_usr_season_cat, pref_usr_season, pref_term_season, pref_global_season],
        keys_list=[["user_id_hashed","season","leaf_category_name"],
                   ["user_id_hashed","season"],
                   ["search_term_normalized","season"],
                   ["season"]]
    )
    trainX_train_feats = merge_pref_with_fallback(
        hist,
        pref_tables=[pref_usr_season_cat, pref_usr_season, pref_term_season, pref_global_season],
        keys_list=[["user_id_hashed","season","leaf_category_name"],
                   ["user_id_hashed","season"],
                   ["search_term_normalized","season"],
                   ["season"]]
    )
    trainX_enh = pd.concat([trainX_train_feats, trainX_valid_feats], axis=0).sort_index()

    pref_base_all = trainX[trainX["ordered"]==1].copy()
    if pref_base_all.empty: pref_base_all = trainX[trainX["clicked"]==1].copy()
    pref_usr_season_cat_all = robust_pref_table(pref_base_all, ["user_id_hashed","season","leaf_category_name"])
    pref_usr_season_all     = robust_pref_table(pref_base_all, ["user_id_hashed","season"])
    pref_term_season_all    = robust_pref_table(pref_base_all, ["search_term_normalized","season"])
    pref_global_season_all  = robust_pref_table(pref_base_all, ["season"])

    testX_enh = merge_pref_with_fallback(
        testX,
        pref_tables=[pref_usr_season_cat_all, pref_usr_season_all, pref_term_season_all, pref_global_season_all],
        keys_list=[["user_id_hashed","season","leaf_category_name"],
                   ["user_id_hashed","season"],
                   ["search_term_normalized","season"],
                   ["season"]]
    )

    extra_price_feats = ["price_delta","abs_price_delta","z_price_delta","in_band","price_gauss_affinity"]
    feat_cols = ["level1_category_name","level2_category_name","leaf_category_name","user_gender"] + num_cols + extra_price_feats
    for col in feat_cols:
        if col not in trainX_enh.columns: trainX_enh[col] = -1
        if col not in testX_enh.columns:  testX_enh[col]  = -1

    print("[9/9] ranker eğitimi...")
    trainX_enh["relevance"] = 3*trainX_enh.get("ordered",0).astype(int) + 1*trainX_enh.get("clicked",0).astype(int)
    tr = trainX_enh[trainX_enh["ts_hour"] < cutoff].copy()
    va = trainX_enh[trainX_enh["ts_hour"] >= cutoff].copy()
    X_tr, y_tr = tr[feat_cols], tr["relevance"]
    X_va, y_va = va[feat_cols], va["relevance"]
    g_tr = group_sizes_by_session(tr)
    g_va = group_sizes_by_session(va)

    ranker = lgb.LGBMRanker(
        objective="lambdarank", metric="ndcg",
        n_estimators=2000, learning_rate=0.05,
        num_leaves=96, subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=-1, max_bin=255,
    )
    ranker.fit(
        X_tr, y_tr, group=g_tr,
        eval_set=[(X_va, y_va)], eval_group=[g_va],
        eval_at=[5,10,20],
        callbacks=[lgb.early_stopping(150), lgb.log_evaluation(100)],
    )

    def session_auc(df_with_targets, scores, target_col):
    #""" Yarışma tanımına uygun: sadece pozitif içeren oturumlarda AUC, sonra ortalama """
        tmp = df_with_targets.copy()
        tmp["__score__"] = scores
        aucs = []
        for sid, g in tmp.groupby("session_id", observed=True):
            y = g[target_col].astype(int).values
            s = g["__score__"].values
            # en az bir pozitif ve bir negatif olmalı
            if (y.sum() > 0) and ((y==0).sum() > 0):
                try:
                    aucs.append(roc_auc_score(y, s))
                except Exception:
                    pass
        return float(np.mean(aucs)) if len(aucs) else float("nan")

    # --- VALIDASYON SKORLARI (ranker skoru ile) ---
    va_scores = ranker.predict(va[feat_cols], raw_score=False)

    click_session_auc  = session_auc(va[["session_id","clicked"]].assign(clicked=va.get("clicked",0)),
                                     va_scores, "clicked")
    order_session_auc  = session_auc(va[["session_id","ordered"]].assign(ordered=va.get("ordered",0)),
                                     va_scores, "ordered")

    W_ORDER, W_CLICK = 0.7, 0.3   # resmi ağırlık farklıysa burayı değiştir
    final_local_score = W_ORDER * order_session_auc + W_CLICK * click_session_auc

    print(f"[LOCAL] Click session-AUC : {click_session_auc:.6f}")
    print(f"[LOCAL] Order session-AUC : {order_session_auc:.6f}")
    print(f"[LOCAL] Final weighted    : {final_local_score:.6f}")

    print("Test skorlama ve submission yazılıyor...")
    test_scores = ranker.predict(testX_enh[feat_cols], raw_score=False)
    out = testX_enh[["session_id","content_id_hashed"]].copy()
    out["score"] = test_scores
    submission = (
        out.sort_values(["session_id","score"], ascending=[True,False])
           .groupby("session_id")["content_id_hashed"]
           .apply(lambda x: " ".join(x.tolist()))
           .reset_index()
           .rename(columns={"content_id_hashed":"prediction"})
    )
    submission.to_csv(args.out, index=False)
    print(f"Saved submission to: {args.out}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--out", type=str, default="submission.csv")
    args = parser.parse_args()
    main(args)
