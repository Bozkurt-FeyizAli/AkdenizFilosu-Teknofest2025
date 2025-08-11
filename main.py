# main.py
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import roc_auc_score
import warnings
from utils.helpers import group_sizes_by_session, session_auc # session_auc'u helper'a taşıyabiliriz.
from build_features_polars import generate_features # Polars scriptimizi import ediyoruz

# Uyarıları bastır
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option("display.max_columns", 200)

def main(args):
    DATA = Path(args.data_dir)
    
    # =================================================================
    # 1. POLARS ile Özellikleri Üret (Eğer dosyalar yoksa)
    # =================================================================
    train_feature_path = DATA / "train_features.parquet"
    test_feature_path = DATA / "test_features.parquet"

    if not train_feature_path.exists() or not test_feature_path.exists() or args.force_rebuild:
        print("Özellik dosyaları bulunamadı veya yeniden oluşturma zorlandı.")
        print("Polars ile özellik üretimi başlatılıyor...")
        generate_features(DATA, is_train=True)
        generate_features(DATA, is_train=False)
    else:
        print("Mevcut özellik dosyaları kullanılacak.")

    # =================================================================
    # 2. Önceden İşlenmiş Verileri PANDAS ile Oku
    # =================================================================
    print("[1/4] Önceden işlenmiş train/test verileri okunuyor...")
    trainX = pd.read_parquet(train_feature_path)
    testX = pd.read_parquet(test_feature_path)

    # =================================================================
    # 3. Model için Son Hazırlıklar
    # =================================================================
    print("[2/4] Model için son hazırlıklar yapılıyor...")
    
    # Kategorik ve numerik sütunları belirle
    cat_cols = ["level1_category_name", "level2_category_name", "leaf_category_name", "user_gender"]
    
    # Polars'ta ürettiğimiz tüm sütunlardan gereksiz olanları çıkarıp kalanını numerik kabul edelim
    ignore_cols = ["ts_hour", "search_term_normalized", "clicked", "ordered", "added_to_cart", "added_to_fav", 
                   "user_id_hashed", "content_id_hashed", "session_id", "content_creation_date", "cv_tags", 
                   "update_date", "date"] + cat_cols
    
    num_cols = [col for col in trainX.columns if col not in ignore_cols]
    
    # Kategorik Sütunları Kodla
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    trainX[cat_cols] = enc.fit_transform(trainX[cat_cols].astype(str))
    testX[cat_cols] = enc.transform(testX[cat_cols].astype(str))

    feat_cols = cat_cols + num_cols
    
    # Veri tiplerini ve eksik değerleri kontrol et
    for df in [trainX, testX]:
        for col in feat_cols:
            if col not in df.columns: df[col] = -1.0
        df[feat_cols] = df[feat_cols].fillna(-1.0).astype("float32")

    # =================================================================
    # 4. Model Eğitimi ve Tahmin
    # =================================================================
    print("[3/4] Ranker modeli eğitiliyor...")
    
    # Zaman bazlı validasyon seti oluştur
    cutoff = trainX["ts_hour"].quantile(0.8)
    trainX["relevance"] = 3 * trainX.get("ordered", 0).astype(int) + 1 * trainX.get("clicked", 0).astype(int)
    
    tr = trainX[trainX["ts_hour"] < cutoff].copy()
    va = trainX[trainX["ts_hour"] >= cutoff].copy()
    
    X_tr, y_tr = tr[feat_cols], tr["relevance"]
    X_va, y_va = va[feat_cols], va["relevance"]
    g_tr = group_sizes_by_session(tr)
    g_va = group_sizes_by_session(va)

    ranker = lgb.LGBMRanker(
        objective="lambdarank", metric="ndcg", n_estimators=2000, learning_rate=0.05,
        num_leaves=96, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1,
    )
    ranker.fit(
        X_tr, y_tr, group=g_tr,
        eval_set=[(X_va, y_va)], eval_group=[g_va], eval_at=[20],
        callbacks=[lgb.early_stopping(150, verbose=False), lgb.log_evaluation(100)],
    )
    
    va_scores = ranker.predict(va[feat_cols])
    # Not: session_auc'u utils/helpers.py içine taşıdığınızdan emin olun
    click_auc = session_auc(va[["session_id", "clicked"]], va_scores, "clicked")
    order_auc = session_auc(va[["session_id", "ordered"]], va_scores, "ordered")
    final_score = 0.7 * order_auc + 0.3 * click_auc
    print(f"\n[VALIDATION] Click AUC: {click_auc:.6f} | Order AUC: {order_auc:.6f} | Final Score: {final_score:.6f}\n")

    print("[4/4] Test seti skorlanıyor ve gönderim dosyası oluşturuluyor...")
    test_scores = ranker.predict(testX[feat_cols])
    out = testX[["session_id", "content_id_hashed"]].copy()
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
    parser = argparse.ArgumentParser(description="Polars ve LightGBM ile Hızlandırılmış Eğitim Pipeline'ı")
    parser.add_argument("--data_dir", type=str, default="data", help="Veri setlerinin bulunduğu klasör.")
    parser.add_argument("--out", type=str, default="submission_polars.csv", help="Oluşturulacak gönderim dosyasının adı.")
    parser.add_argument("--force_rebuild", action="store_true", help="Mevcut olsalar bile özellik dosyalarını yeniden oluşturur.")
    args = parser.parse_args()
    main(args)