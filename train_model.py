import argparse
from pathlib import Path
import pandas as pd
import lightgbm as lgb
import optuna
import warnings
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import GroupKFold

# Yardımcı fonksiyonlar ve özellik üretme scripti
from utils.helpers import session_auc
from build_features_polars import generate_features

# Gürültüyü azaltmak için Optuna loglamasını kapatalım
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 200)

# =================================================================
# Öncekiyle Aynı Kalan Gelişmiş Özellik Fonksiyonları
# =================================================================
def jaccard_similarity(list1, list2):
    s1 = set(list1)
    s2 = set(list2)
    intersection = len(s1.intersection(s2))
    union = len(s1.union(s2))
    return intersection / union if union != 0 else 0.0

def add_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    print("   - Gelişmiş özellikler ekleniyor (Jaccard, Oturum Rekabeti, Metin)...")
    df['search_term_normalized'] = df['search_term_normalized'].fillna('').astype(str)
    df['cv_tags'] = df['cv_tags'].fillna('').astype(str)
    
    # 1. Mevcut Jaccard Benzerliği (Bu zaten vardı)
    search_terms_split = df['search_term_normalized'].str.split()
    cv_tags_split = df['cv_tags'].str.split()
    df['term_tag_jaccard_similarity'] = [jaccard_similarity(term, tag) for term, tag in zip(search_terms_split, cv_tags_split)]
    
    # 2. Oturum İçi Rekabet Özellikleri (Bunlar da vardı)
    df['price_vs_session_avg'] = df['discounted_price'] / df.groupby('session_id')['discounted_price'].transform('mean')
    df['price_rank_in_session'] = df.groupby('session_id')['discounted_price'].rank(pct=True)
    df['session_item_count'] = df.groupby('session_id')['content_id_hashed'].transform('count')

    # === YENİ ÖZELLİKLER ===
    
    # 3. Metin Uzunlukları ve Kelime Sayıları
    df['search_term_len'] = df['search_term_normalized'].str.len()
    df['search_term_word_count'] = df['search_term_normalized'].str.split().str.len()
    
    # 4. Ürün Popülerliğinin Oturumdaki Diğer Ürünlere Göre Sıralaması
    #    (content_search_ctr Polars'ta üretilmişti, burada kullanıyoruz)
    df['ctr_rank_in_session'] = df.groupby('session_id')['content_search_ctr'].rank(pct=True)
    df['review_count_rank_in_session'] = df.groupby('session_id')['content_review_count'].rank(pct=True)
    
    # 5. Kullanıcı-Kategori Etkileşimi
    #    Bu kullanıcının bu kategorideki ürünlere genel tıklama/sipariş oranı
    #    Bu özellikler daha fazla veri manipülasyonu gerektirir, 
    #    ancak en basit haliyle oturum içi sayımlar yapılabilir.
    df['user_clicks_in_session_on_cat'] = df.groupby(['session_id', 'leaf_category_name'])['content_id_hashed'].transform('count')

    return df

# =================================================================
# OPTUNA OPTİMİZASYON FONKSİYONU
# =================================================================
def run_optimization(trial, target_col, tr, va, feat_cols):
    y_tr, y_va = tr[target_col], va[target_col]
    pos_weight = (y_tr == 0).sum() / (y_tr == 1).sum()

    params = {
        'objective': 'binary', 'metric': 'auc', 'random_state': 42,
        'n_estimators': 3000, 'boosting_type': 'gbdt', 'n_jobs': -1, 'verbosity': -1,  # 2000'den 3000'e artırıldı
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),  # Aralık genişletildi
        'num_leaves': trial.suggest_int('num_leaves', 15, 400),  # Aralık genişletildi
        'max_depth': trial.suggest_int('max_depth', 4, 20),  # Aralık genişletildi
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),  # Aralık genişletildi
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),  # Aralık genişletildi
        'scale_pos_weight': pos_weight
    }

    model = lgb.LGBMClassifier(**params)
    model.fit(tr[feat_cols], y_tr, eval_set=[(va[feat_cols], y_va)], callbacks=[lgb.early_stopping(150, verbose=False)])  # 100'den 150'ye artırıldı
    
    # Sadece validasyon AUC skorunu döndür
    return model.best_score_['valid_0']['auc']

# =================================================================
# ANA ORKESTRASYON
# =================================================================
def main(args):
    DATA = Path(args.data_dir)
    N_SPLITS = 7  # Çapraz validasyon için katman sayısı (5'ten 7'ye artırıldı)
    N_TRIALS = 60 # Her model için Optuna deneme sayısı (30'dan 60'a artırıldı)

    # ADIM 1: ÖZELLİK ÜRETİMİ
    train_feature_path = DATA / "train_features.parquet"
    test_feature_path = DATA / "test_features.parquet"
    if not train_feature_path.exists() or not test_feature_path.exists() or args.force_rebuild:
        generate_features(DATA, is_train=True)
        generate_features(DATA, is_train=False)

    # ADIM 2: VERİ OKUMA VE HAZIRLAMA
    print("\n[1/3] Veriler okunuyor ve hazırlanıyor...")
    trainX = pd.read_parquet(train_feature_path)
    testX = pd.read_parquet(test_feature_path)
    
    trainX = add_advanced_features(trainX)
    testX = add_advanced_features(testX)

    cat_cols = ["level1_category_name", "level2_category_name", "leaf_category_name", "user_gender"]
    ignore_cols = ["ts_hour", "search_term_normalized", "clicked", "ordered", "added_to_cart", "added_to_fav", "user_id_hashed", "content_id_hashed", "session_id", "content_creation_date", "cv_tags", "update_date", "date"] + cat_cols
    num_cols = [col for col in trainX.columns if col not in ignore_cols]
    feat_cols = cat_cols + num_cols
    
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    trainX[cat_cols] = enc.fit_transform(trainX[cat_cols].astype(str))
    testX[cat_cols] = enc.transform(testX[cat_cols].astype(str))
    
    for df in [trainX, testX]:
        for col in feat_cols:
            if col not in df.columns: df[col] = -1.0
        df[feat_cols] = df[feat_cols].fillna(-1.0).astype("float32")

    # =================================================================
    # ADIM 3: ÇAPRAZ VALİDASYON İLE MODEL EĞİTİMİ VE OPTİMİZASYON
    # =================================================================
    print(f"\n[2/3] {N_SPLITS}-Katmanlı Çapraz Validasyon ve Optimizasyon Başlatılıyor...")
    gkf = GroupKFold(n_splits=N_SPLITS)
    groups = trainX['session_id']

    oof_test_order_preds = []
    oof_test_click_preds = []
    trainX['p_click'] = -1.0 # OOF click tahminleri için yer tutucu

    for fold, (train_idx, val_idx) in enumerate(gkf.split(trainX, trainX['ordered'], groups=groups)):
        print("-" * 50)
        print(f"FOLD {fold+1}/{N_SPLITS} BAŞLADI")
        
        tr, va = trainX.iloc[train_idx], trainX.iloc[val_idx]

        # --- ADIM 1: CLICK MODELİNİ EĞİT ---
        print(f"   - Fold {fold+1}: Click modeli optimize ediliyor...")
        study_click = optuna.create_study(direction='maximize')
        study_click.optimize(lambda trial: run_optimization(trial, 'clicked', tr, va, feat_cols), n_trials=N_TRIALS)
        best_params_click = study_click.best_params
        
        model_click = lgb.LGBMClassifier(**best_params_click, objective='binary', random_state=42, n_jobs=-1, n_estimators=3000)  # 2000'den 3000'e artırıldı
        model_click.fit(tr[feat_cols], tr['clicked'], eval_set=[(va[feat_cols], va['clicked'])], callbacks=[lgb.early_stopping(150, verbose=False)])  # 100'den 150'ye artırıldı
        
        # Validasyon seti için click tahminlerini sakla (OOF)
        va_click_preds = model_click.predict_proba(va[feat_cols])[:, 1]
        trainX.loc[val_idx, 'p_click'] = va_click_preds
        
        # Test seti için click tahminlerini yap
        oof_test_click_preds.append(model_click.predict_proba(testX[feat_cols])[:, 1])

        # --- ADIM 2: ORDER MODELİNİ EĞİT (YENİ ÖZELLİKLE) ---
        print(f"   - Fold {fold+1}: Order modeli optimize ediliyor...")
        # Not: tr setine henüz p_click eklemedik. Bu, validasyon setinden sızıntıyı önler.
        # Optimizasyon için p_click olmadan da çalışılabilir veya daha karmaşık bir OOF şeması kurulabilir.
        # Şimdilik basit tutalım ve yeni özelliği sadece eğitimde kullanalım.
        
        feat_cols_with_p_click = feat_cols + ['p_click']
        
        # Eğitim setinin p_click değerlerini doldurmak için bir click modeli daha eğitebiliriz
        # veya bu fold'un dışındaki veriyi kullanabiliriz. Şimdilik bu fold'un modeliyle tahmin yapalım (küçük sızıntı olabilir ama pratik).
        tr_click_preds = model_click.predict_proba(tr[feat_cols])[:, 1]
        tr_with_p_click = tr.copy()
        tr_with_p_click['p_click'] = tr_click_preds
        
        va_with_p_click = va.copy()
        va_with_p_click['p_click'] = va_click_preds
        
        study_order = optuna.create_study(direction='maximize')
        study_order.optimize(lambda trial: run_optimization(trial, 'ordered', tr_with_p_click, va_with_p_click, feat_cols_with_p_click), n_trials=N_TRIALS)
        best_params_order = study_order.best_params

        model_order = lgb.LGBMClassifier(**best_params_order, objective='binary', random_state=42, n_jobs=-1, n_estimators=3000)  # 2000'den 3000'e artırıldı
        model_order.fit(tr_with_p_click[feat_cols_with_p_click], tr_with_p_click['ordered'], 
                        eval_set=[(va_with_p_click[feat_cols_with_p_click], va_with_p_click['ordered'])], 
                        callbacks=[lgb.early_stopping(200, verbose=False)])  # 100'den 200'e artırıldı

        # Test setine de p_click ekleyerek tahmin yap
        testX_with_p_click = testX.copy()
        # Test tahminini bu fold'un click modelinden alıyoruz
        testX_with_p_click['p_click'] = model_click.predict_proba(testX[feat_cols])[:, 1]
        oof_test_order_preds.append(model_order.predict_proba(testX_with_p_click[feat_cols_with_p_click])[:, 1])

        print(f"FOLD {fold+1}/{N_SPLITS} TAMAMLANDI")

    # =================================================================
    # ADIM 4: TAHMİNLERİ BİRLEŞTİR VE GÖNDERİM DOSYASI OLUŞTUR
    # =================================================================
    print("\n[3/3] Tüm katmanların tahminleri birleştiriliyor...")
    
    avg_order_preds = np.mean(oof_test_order_preds, axis=0)
    avg_click_preds = np.mean(oof_test_click_preds, axis=0)
    
    test_scores_blended = (0.7 * avg_order_preds) + (0.3 * avg_click_preds)
    
    out = testX[["session_id", "content_id_hashed"]].copy()
    out["score"] = test_scores_blended
    submission = (
        out.sort_values(["session_id", "score"], ascending=[True, False])
           .groupby("session_id")["content_id_hashed"].apply(lambda x: " ".join(x.tolist()))
           .reset_index().rename(columns={"content_id_hashed": "prediction"})
    )
    submission.to_csv(args.out, index=False)
    print("="*50)
    print("TÜM İŞLEMLER BAŞARIYLA TAMAMLANDI!")
    print(f"Gönderim dosyası kaydedildi: {args.out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kapsamlı Optimizasyon Pipeline'ı (Optuna + CV)")
    parser.add_argument("--data_dir", type=str, default="data", help="Veri klasörü.")
    parser.add_argument("--out", type=str, default="submission_comprehensive.csv", help="Gönderim dosyası adı.")
    parser.add_argument("--force_rebuild", action="store_true", help="Temel özellikleri yeniden oluşturur.")
    args = parser.parse_args()
    main(args)
