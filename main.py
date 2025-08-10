# main.py
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import roc_auc_score
import warnings

# Uyarıları bastır
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option("display.max_columns", 200)

# Kendi modüllerimizi import edelim
from features.content_features import generate_content_features
from features.user_features import generate_user_features
from features.interaction_features import (
    generate_top_terms_features,
    add_session_context_features,
    add_price_preference_features,
)
from utils.helpers import add_time_feats, group_sizes_by_session, chunked_merge

def main(args):
    DATA = Path(args.data_dir)

    # =================================================================
    # 1. Veri Yükleme ve Temel Hazırlık
    # =================================================================
    print("[1/7] Ana train/test verileri okunuyor...")
    train = pd.read_parquet(DATA / "train_sessions.parquet", columns=["ts_hour", "search_term_normalized", "clicked", "ordered", "user_id_hashed", "content_id_hashed", "session_id"])
    test = pd.read_parquet(DATA / "test_sessions.parquet", columns=["ts_hour", "search_term_normalized", "user_id_hashed", "content_id_hashed", "session_id"])

    for df in [train, test]:
        for c in ["user_id_hashed", "content_id_hashed", "search_term_normalized", "session_id"]:
            df[c] = df[c].astype("category")
    
    train = add_time_feats(train, "ts_hour")
    test = add_time_feats(test, "ts_hour")
    
    # =================================================================
    # 2. Özellik Modüllerini Çalıştır
    # =================================================================
    print("[2/7] Özellik modülleri çalıştırılıyor...")
    content_df = generate_content_features(DATA) # Yunus'un modülü
    user_df = generate_user_features(DATA)       # Özkan'ın modülü
    top_terms_df = generate_top_terms_features(DATA) # Muhammed'in modülü (kısım 1)

    # =================================================================
    # 3. Ana Tabloları Birleştir
    # =================================================================
    print("[3/7] Ana tablolar birleştiriliyor...")
    def build_matrix(df, content_df, user_df, top_terms_df):
        print(f"    - {('Train' if 'clicked' in df.columns else 'Test')} matrisi oluşturuluyor...")
        X = (df.merge(content_df, on="content_id_hashed", how="left")
               .merge(user_df, on="user_id_hashed", how="left"))
        
        join_cols = ["content_id_hashed", "search_term_normalized"]
        X = chunked_merge(X, top_terms_df, on_cols=join_cols, how="left", chunk_size=400_000)
        return X

    trainX = build_matrix(
    df=train,
    content_df=content_df,
    user_df=user_df,
    top_terms_df=top_terms_df
    )

    testX = build_matrix(
        df=test,
        content_df=content_df,
        user_df=user_df,
        top_terms_df=top_terms_df
    )
    del content_df, user_df, top_terms_df # RAM temizliği
    
    # =================================================================
    # 4. Etkileşim ve Gelişmiş Özellikleri Ekle
    # =================================================================
    print("[4/7] Etkileşim ve bağlam özellikleri ekleniyor (Muhammed)...")
    trainX = add_session_context_features(trainX)
    testX = add_session_context_features(testX)

    cutoff = trainX["ts_hour"].quantile(0.8) # Validasyon için zaman sınırı
    trainX_enh, testX_enh = add_price_preference_features(trainX, testX, cutoff)

    # =================================================================
    # 5. Son Özellik Mühendisliği Adımları
    # =================================================================
    print("[5/7] Son özellik mühendisliği adımları...")
    for df in [trainX_enh, testX_enh]:
        ccd = pd.to_datetime(df["content_creation_date"], errors="coerce")
        ts = pd.to_datetime(df["ts_hour"], errors="coerce")
        df["content_age_days"] = (ts - ccd).dt.days
        df.loc[df["content_age_days"].isna(), "content_age_days"] = df["content_age_days"].median()
        
        df["age"] = pd.to_datetime(df["ts_hour"]).dt.year - df["user_birth_year"].fillna(0)
        df.loc[(df["user_birth_year"] <= 0) | (df["user_birth_year"].isna()), "age"] = -1
        
        df["discount_rate"] = np.where(
            (df["original_price"] > 0) & (df["discounted_price"] > 0),
            1 - (df["discounted_price"] / df["original_price"]), 0
        )

    cat_cols = ["level1_category_name", "level2_category_name", "leaf_category_name", "user_gender"]
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    trainX_enh[cat_cols] = enc.fit_transform(trainX_enh[cat_cols].astype(str))
    testX_enh[cat_cols] = enc.transform(testX_enh[cat_cols].astype(str))

    num_cols = ["hour", "dow", "is_weekend", "month", "season", "attribute_type_count", "total_attribute_option_count", "merchant_count", "filterable_label_count", "original_price", "selling_price", "discounted_price", "content_review_count", "content_review_wth_media_count", "content_rate_count", "content_rate_avg", "total_search_impression", "total_search_click", "content_search_ctr", "total_click", "total_cart", "total_fav", "total_order", "content_click_to_order", "content_cart_to_order", "user_birth_year", "user_tenure_in_days", "user_click_to_order", "user_cart_to_order", "tt_search_impr", "tt_search_click", "term_ctr", "content_age_days", "price_rank", "price_z", "age", "discount_rate"]
    extra_price_feats = ["price_delta", "abs_price_delta", "z_price_delta", "in_band", "price_gauss_affinity"]
    feat_cols = cat_cols + num_cols + extra_price_feats
    
    for df in [trainX_enh, testX_enh]:
        for col in feat_cols:
            if col not in df.columns: df[col] = -1
        df[feat_cols] = df[feat_cols].astype("float32").fillna(-1)

    # =================================================================
    # 6. Model Eğitimi ve Validasyon
    # =================================================================
    print("[6/7] Ranker modeli eğitiliyor...")
    trainX_enh["relevance"] = 3 * trainX_enh.get("ordered", 0).astype(int) + 1 * trainX_enh.get("clicked", 0).astype(int)
    
    tr = trainX_enh[trainX_enh["ts_hour"] < cutoff].copy()
    va = trainX_enh[trainX_enh["ts_hour"] >= cutoff].copy()
    X_tr, y_tr = tr[feat_cols], tr["relevance"]
    X_va, y_va = va[feat_cols], va["relevance"]
    g_tr = group_sizes_by_session(tr)
    g_va = group_sizes_by_session(va)

    ranker = lgb.LGBMRanker(
        objective="lambdarank", metric="ndcg", n_estimators=2000, learning_rate=0.05,
        num_leaves=96, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, max_bin=255,
    )
    ranker.fit(
        X_tr, y_tr, group=g_tr,
        eval_set=[(X_va, y_va)], eval_group=[g_va], eval_at=[10, 20],
        callbacks=[lgb.early_stopping(150, verbose=False), lgb.log_evaluation(100)],
    )
    
    def session_auc(df_with_targets, scores, target_col):
        tmp = df_with_targets.copy()
        tmp["__score__"] = scores
        aucs = [roc_auc_score(g[target_col], g["__score__"]) for _, g in tmp.groupby("session_id", observed=True) if g[target_col].nunique() > 1]
        return float(np.mean(aucs)) if aucs else float("nan")

    va_scores = ranker.predict(va[feat_cols])
    click_auc = session_auc(va[["session_id", "clicked"]], va_scores, "clicked")
    order_auc = session_auc(va[["session_id", "ordered"]], va_scores, "ordered")
    final_score = 0.7 * order_auc + 0.3 * click_auc
    print(f"\n[VALIDATION] Click AUC: {click_auc:.6f} | Order AUC: {order_auc:.6f} | Final Score: {final_score:.6f}\n")

    # =================================================================
    # 7. Tahmin ve Gönderim Dosyası
    # =================================================================
    print("[7/7] Test seti skorlanıyor ve gönderim dosyası oluşturuluyor...")
    test_scores = ranker.predict(testX_enh[feat_cols])
    out = testX_enh[["session_id", "content_id_hashed"]].copy()
    out["score"] = test_scores
    
    submission = (
        out.sort_values(["session_id", "score"], ascending=[True, False])
           .groupby("session_id")["content_id_hashed"]
           .apply(lambda x: " ".join(x.tolist()))
           .reset_index()
           .rename(columns={"content_id_hashed": "prediction"})
    )
    submission.to_csv(args.out, index=False)
    print(f"Başarıyla tamamlandı. Gönderim dosyası kaydedildi: {args.out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TEKNOFEST E-Ticaret Hackathonu için Modüler Eğitim Pipeline'ı")
    parser.add_argument("--data_dir", type=str, default="data", help="Veri setlerinin bulunduğu klasör.")
    parser.add_argument("--out", type=str, default="submission.csv", help="Oluşturulacak gönderim dosyasının adı.")
    args = parser.parse_args()
    main(args)