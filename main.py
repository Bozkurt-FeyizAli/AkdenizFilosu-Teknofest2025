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
    print("   - Gelişmiş özellikler ekleniyor (Jaccard, Oturum Rekabeti)...")
    df['search_term_normalized'] = df['search_term_normalized'].fillna('').astype(str)
    df['cv_tags'] = df['cv_tags'].fillna('').astype(str)
    search_terms_split = df['search_term_normalized'].str.split()
    cv_tags_split = df['cv_tags'].str.split()
    df['term_tag_jaccard_similarity'] = [jaccard_similarity(term, tag) for term, tag in zip(search_terms_split, cv_tags_split)]
    df['price_vs_session_avg'] = df['discounted_price'] / df.groupby('session_id')['discounted_price'].transform('mean')
    df['price_rank_in_session'] = df.groupby('session_id')['discounted_price'].rank(pct=True)
    df['session_item_count'] = df.groupby('session_id')['content_id_hashed'].transform('count')
    return df

# =================================================================
# OPTUNA OPTİMİZASYON FONKSİYONU
# =================================================================
def run_optimization(trial, target_col, tr, va, feat_cols):
    y_tr, y_va = tr[target_col], va[target_col]
    pos_weight = (y_tr == 0).sum() / (y_tr == 1).sum()

    params = {
        'objective': 'binary', 'metric': 'auc', 'random_state': 42,
        'n_estimators': 2000, 'boosting_type': 'gbdt', 'n_jobs': -1, 'verbosity': -1,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.08, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 300),
        'max_depth': trial.suggest_int('max_depth', 5, 15),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'scale_pos_weight': pos_weight
    }

    model = lgb.LGBMClassifier(**params)
    model.fit(tr[feat_cols], y_tr, eval_set=[(va[feat_cols], y_va)], callbacks=[lgb.early_stopping(100, verbose=False)])
    
    # Sadece validasyon AUC skorunu döndür
    return model.best_score_['valid_0']['auc']

# =================================================================
# ANA ORKESTRASYON
# =================================================================
def main(args):
    DATA = Path(args.data_dir)
    N_SPLITS = 5  # Çapraz validasyon için katman sayısı
    N_TRIALS = 45 # Her model için Optuna deneme sayısı

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
    
    # Oturumların bütünlüğünü korumak için GroupKFold kullanalım
    gkf = GroupKFold(n_splits=N_SPLITS)
    groups = trainX['session_id']
    
    oof_test_order_preds = []
    oof_test_click_preds = []
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(trainX, trainX['ordered'], groups=groups)):
        print("-" * 50)
        print(f"FOLD {fold+1}/{N_SPLITS} BAŞLADI")
        
        tr = trainX.iloc[train_idx]
        va = trainX.iloc[val_idx]
        
        # --- ORDER MODELİ İÇİN OPTİMİZASYON ---
        print(f"   - Fold {fold+1}: Order modeli için Optuna çalışıyor ({N_TRIALS} deneme)...")
        study_order = optuna.create_study(direction='maximize')
        study_order.optimize(lambda trial: run_optimization(trial, 'ordered', tr, va, feat_cols), n_trials=N_TRIALS)
        best_params_order = study_order.best_params
        print(f"   - Fold {fold+1}: Order modeli en iyi AUC: {study_order.best_value:.6f}")
        
        # --- CLICK MODELİ İÇİN OPTİMİZASYON ---
        print(f"   - Fold {fold+1}: Click modeli için Optuna çalışıyor ({N_TRIALS} deneme)...")
        study_click = optuna.create_study(direction='maximize')
        study_click.optimize(lambda trial: run_optimization(trial, 'clicked', tr, va, feat_cols), n_trials=N_TRIALS)
        best_params_click = study_click.best_params
        print(f"   - Fold {fold+1}: Click modeli en iyi AUC: {study_click.best_value:.6f}")

        # --- En iyi parametrelerle modelleri eğit ve test için tahmin yap ---
        print(f"   - Fold {fold+1}: Final modeller eğitiliyor ve test tahmini yapılıyor...")
        model_order = lgb.LGBMClassifier(**best_params_order, objective='binary', random_state=42, n_jobs=-1, n_estimators=2000)
        model_order.fit(tr[feat_cols], tr['ordered'], eval_set=[(va[feat_cols], va['ordered'])], callbacks=[lgb.early_stopping(100, verbose=False)])
        oof_test_order_preds.append(model_order.predict_proba(testX[feat_cols])[:, 1])

        model_click = lgb.LGBMClassifier(**best_params_click, objective='binary', random_state=42, n_jobs=-1, n_estimators=2000)
        model_click.fit(tr[feat_cols], tr['clicked'], eval_set=[(va[feat_cols], va['clicked'])], callbacks=[lgb.early_stopping(100, verbose=False)])
        oof_test_click_preds.append(model_click.predict_proba(testX[feat_cols])[:, 1])
        
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
