import argparse
from pathlib import Path
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import OrdinalEncoder
import warnings

# Eski script'lerimizden gerekli fonksiyonları import ediyoruz
from utils.helpers import session_auc 
from build_features_polars import generate_features 

# Uyarıları bastır
warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 200)

# =================================================================
# YENİ ÖZELLİK MÜHENDİSLİĞİ (Madde 3)
# =================================================================
def jaccard_similarity(list1, list2):
    """İki liste arasındaki Jaccard Benzerliğini hesaplar."""
    s1 = set(list1)
    s2 = set(list2)
    intersection = len(s1.intersection(s2))
    union = len(s1.union(s2))
    return intersection / union if union != 0 else 0.0

def add_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pandas üzerinde Jaccard ve Oturum İçi Rekabet özelliklerini ekler.
    """
    print("   - Gelişmiş özellikler ekleniyor (Jaccard, Oturum Rekabeti)...")
    
    # 1. Jaccard Benzerliği (Arama Terimi vs CV Etiketleri)
    # Boş değerleri ve None type'ları temizleyelim
    df['search_term_normalized'] = df['search_term_normalized'].fillna('').astype(str)
    df['cv_tags'] = df['cv_tags'].fillna('').astype(str)
    
    # Kelimelere ayır
    search_terms_split = df['search_term_normalized'].str.split()
    cv_tags_split = df['cv_tags'].str.split()
    
    # Jaccard benzerliğini hesapla
    df['term_tag_jaccard_similarity'] = [
        jaccard_similarity(term, tag) 
        for term, tag in zip(search_terms_split, cv_tags_split)
    ]

    # 2. Oturum İçi Rekabet Özellikleri
    # Fiyatın oturum ortalamasına göre durumu
    df['price_vs_session_avg'] = df['discounted_price'] / df.groupby('session_id')['discounted_price'].transform('mean')
    
    # Fiyatın oturum içindeki sıralaması (% olarak)
    df['price_rank_in_session'] = df.groupby('session_id')['discounted_price'].rank(pct=True)
    
    # Oturumdaki ürün sayısı
    df['session_item_count'] = df.groupby('session_id')['content_id_hashed'].transform('count')

    return df

def main(args):
    DATA = Path(args.data_dir)
    
    # ADIM 1: TEMEL ÖZELLİK ÜRETİMİ (Polars ile, Değişiklik Yok)
    train_feature_path = DATA / "train_features.parquet"
    test_feature_path = DATA / "test_features.parquet"
    if not train_feature_path.exists() or not test_feature_path.exists() or args.force_rebuild:
        generate_features(DATA, is_train=True)
        generate_features(DATA, is_train=False)
    
    # ADIM 2: VERİ OKUMA VE YENİ ÖZELLİKLERİ EKLEME
    print("\n[1/4] Veriler okunuyor ve gelişmiş özellikler ekleniyor...")
    trainX = pd.read_parquet(train_feature_path)
    testX = pd.read_parquet(test_feature_path)
    
    trainX = add_advanced_features(trainX)
    testX = add_advanced_features(testX)

    # ADIM 3: MODEL İÇİN SON HAZIRLIKLAR
    print("\n[2/4] Model için son hazırlıklar yapılıyor...")
    cat_cols = ["level1_category_name", "level2_category_name", "leaf_category_name", "user_gender"]
    ignore_cols = ["ts_hour", "search_term_normalized", "clicked", "ordered", "added_to_cart", "added_to_fav", 
               "user_id_hashed", "content_id_hashed", "session_id", "content_creation_date", 
               "cv_tags", "update_date", "date"] + cat_cols
    num_cols = [col for col in trainX.columns if col not in ignore_cols]
    feat_cols = cat_cols + num_cols
    
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    trainX[cat_cols] = enc.fit_transform(trainX[cat_cols].astype(str))
    testX[cat_cols] = enc.transform(testX[cat_cols].astype(str))
    
    for df in [trainX, testX]:
        df[feat_cols] = df[feat_cols].fillna(-1.0).astype("float32")
    
    cutoff = trainX["ts_hour"].quantile(0.8)
    tr = trainX[trainX["ts_hour"] < cutoff].copy()
    va = trainX[trainX["ts_hour"] >= cutoff].copy()
    
    X_tr, X_va = tr[feat_cols], va[feat_cols]

    # =================================================================
    # ADIM 4: YENİ MİMARİ - İKİ AYRI MODELİ EĞİTME (Madde 1)
    # =================================================================
    
    # --- Model 1: Sipariş (Order) Modeli ---
    print("\n[3/4] Model 1: Sipariş (Order) modeli eğitiliyor...")
    y_tr_order, y_va_order = tr['ordered'], va['ordered']
    
    model_order = lgb.LGBMClassifier(objective='binary', metric='auc', n_estimators=1000, learning_rate=0.05, num_leaves=31, random_state=42, n_jobs=-1)
    model_order.fit(X_tr, y_tr_order, eval_set=[(X_va, y_va_order)], callbacks=[lgb.early_stopping(50, verbose=False)])

    # --- Model 2: Tıklama (Click) Modeli ---
    print("   - Model 2: Tıklama (Click) modeli eğitiliyor...")
    y_tr_click, y_va_click = tr['clicked'], va['clicked']

    model_click = lgb.LGBMClassifier(objective='binary', metric='auc', n_estimators=1000, learning_rate=0.05, num_leaves=31, random_state=42, n_jobs=-1)
    model_click.fit(X_tr, y_tr_click, eval_set=[(X_va, y_va_click)], callbacks=[lgb.early_stopping(50, verbose=False)])
    
    # =================================================================
    # ADIM 5: SKORLARI BİRLEŞTİRME VE GÖNDERİM
    # =================================================================
    print("\n[4/4] Tahminler yapılıyor, skorlar birleştiriliyor ve gönderim dosyası oluşturuluyor...")
    
    # Validasyon seti için skorları birleştir
    va_scores_order = model_order.predict_proba(X_va)[:, 1]
    va_scores_click = model_click.predict_proba(X_va)[:, 1]
    va_scores_blended = (0.7 * va_scores_order) + (0.3 * va_scores_click)
    
    # Validasyon skorunu hesapla
    final_order_auc = session_auc(va, va_scores_blended, "ordered")
    final_click_auc = session_auc(va, va_scores_blended, "clicked")
    final_score = 0.7 * final_order_auc + 0.3 * final_click_auc
    print(f"\n[VALIDATION] Final Blended Score: {final_score:.6f}\n")
    
    # Test seti için skorları birleştir
    test_scores_order = model_order.predict_proba(testX[feat_cols])[:, 1]
    test_scores_click = model_click.predict_proba(testX[feat_cols])[:, 1]
    test_scores_blended = (0.7 * test_scores_order) + (0.3 * test_scores_click)
    
    # Gönderim dosyasını oluştur
    out = testX[["session_id", "content_id_hashed"]].copy()
    out["score"] = test_scores_blended
    submission = (
        out.sort_values(["session_id", "score"], ascending=[True, False])
           .groupby("session_id")["content_id_hashed"].apply(lambda x: " ".join(x.tolist()))
           .reset_index().rename(columns={"content_id_hashed": "prediction"})
    )
    submission.to_csv(args.out, index=False)
    print(f"Başarıyla tamamlandı. Gönderim dosyası kaydedildi: {args.out}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Strateji Yenileme: İki Uzman Modelli Pipeline")
    parser.add_argument("--data_dir", type=str, default="data", help="Veri klasörü.")
    parser.add_argument("--out", type=str, default="submission_strategy_refresh.csv", help="Gönderim dosyası adı.")
    parser.add_argument("--force_rebuild", action="store_true", help="Temel özellikleri yeniden oluşturur.")
    args = parser.parse_args()
    main(args)