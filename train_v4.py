# train_v4.py

import argparse
from pathlib import Path
import pandas as pd
import lightgbm as lgb
import optuna
import warnings
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import GroupKFold

# v4 seviyesindeki Polars özellik üretme betiğini import ediyoruz
from build_features_polars_v4 import generate_features

# Gürültüyü azaltmak için Optuna loglamasını kapatalım
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 200)

# =================================================================
# Gelişmiş Özellik Fonksiyonu (Basitleştirilmiş)
# =================================================================
def jaccard_similarity(list1, list2):
    s1 = set(list1); s2 = set(list2)
    intersection = len(s1.intersection(s2)); union = len(s1.union(s2))
    return intersection / union if union != 0 else 0.0

def add_post_features(df: pd.DataFrame) -> pd.DataFrame:
    print("   - Pandas-tabanlı son özellikler ekleniyor (Jaccard)...")
    df['search_term_normalized'] = df['search_term_normalized'].fillna('').astype(str)
    df['cv_tags'] = df['cv_tags'].fillna('').astype(str)
    
    # Paralel Jaccard hesaplaması için daha hızlı bir yöntem (büyük veri setleri için)
    jaccard_scores = df[['search_term_normalized', 'cv_tags']].apply(
        lambda row: jaccard_similarity(row['search_term_normalized'].split(), row['cv_tags'].split()), axis=1
    )
    df['term_tag_jaccard_similarity'] = jaccard_scores
    
    return df

# =================================================================
# OPTUNA OPTİMİZASYON FONKSİYONU (LGBMRanker için)
# =================================================================
def run_ranker_optimization(trial, tr, va, feat_cols, group_tr, group_va):
    """
    LGBMRanker için Optuna hiperparametre optimizasyonu yapar.
    """
    params = {
        'objective': 'lambdarank',
        'metric': 'ndcg',  # Sıralama için standart metrik
        'random_state': 42,
        'n_estimators': 2000,
        'boosting_type': 'gbdt',
        'n_jobs': -1,
        'verbosity': -1,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 400),
        'max_depth': trial.suggest_int('max_depth', 4, 12),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'lambdarank_truncation_level': trial.suggest_int('lambdarank_truncation_level', 5, 20),
    }

    model = lgb.LGBMRanker(**params)
    model.fit(
        tr[feat_cols], tr['target'], group=group_tr,
        eval_set=[(va[feat_cols], va['target'])], eval_group=[group_va],
        callbacks=[lgb.early_stopping(100, verbose=False)]
    )
    
    # En iyi nDCG skorunu döndür
    return model.best_score_['valid_0']['ndcg@1'] # @1 veya @5 gibi bir metrik seçilebilir

# =================================================================
# ANA ORKESTRASYON
# =================================================================
def main(args):
    DATA = Path(args.data_dir)
    N_SPLITS = 5
    N_TRIALS = args.n_trials

    # ADIM 1: ÖZELLİK ÜRETİMİ (v4)
    train_feature_path = DATA / "train_features_v4.parquet"
    test_feature_path = DATA / "test_features_v4.parquet"
    if not train_feature_path.exists() or not test_feature_path.exists() or args.force_rebuild:
        print("v4 özellikleri bulunamadı veya yeniden oluşturma zorlandı. Oluşturuluyor...")
        generate_features(DATA, is_train=True)
        generate_features(DATA, is_train=False)

    # ADIM 2: VERİ OKUMA VE HAZIRLAMA
    print("\n[1/4] Veriler okunuyor ve hazırlanıyor...")
    trainX = pd.read_parquet(train_feature_path)
    testX = pd.read_parquet(test_feature_path)
    
    #trainX = add_post_features(trainX)
    #testX = add_post_features(testX)

    # (v4 DEĞİŞİKLİĞİ) Birleşik hedef değişkeni oluşturma
    trainX['target'] = trainX['clicked'] + trainX['ordered']

    cat_cols = ["level1_category_name", "leaf_category_name", "user_gender"]
    ignore_cols = ["ts_hour", "search_term_normalized", "clicked", "ordered", "added_to_cart", "added_to_fav",
                   "user_id_hashed", "content_id_hashed", "session_id", "content_creation_date", "cv_tags",
                   "target", "level2_category_name"] # L2 genellikle L1 ve Leaf arasında gereksizdir
    feat_cols = [col for col in trainX.columns if col not in ignore_cols and "_id_" not in col]
    
    for df in [trainX, testX]: df[feat_cols] = df[feat_cols].fillna(-1.0).astype("float32")
        
    print(f"Kullanılacak özellik sayısı: {len(feat_cols)}")

    # =================================================================
    # ADIM 3: ÇAPRAZ VALİDASYON İLE LGBMRanker EĞİTİMİ
    # =================================================================
    print(f"\n[2/4] LGBMRanker için {N_SPLITS}-Katmanlı Çapraz Validasyon Başlatılıyor...")
    
    gkf = GroupKFold(n_splits=N_SPLITS)
    groups = trainX['session_id']
    oof_test_preds = []
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(trainX, trainX['target'], groups=groups)):
        print("-" * 50); print(f"FOLD {fold+1}/{N_SPLITS} BAŞLADI")
        
        tr, va = trainX.iloc[train_idx], trainX.iloc[val_idx]
        group_tr = tr.groupby('session_id', sort=False).size().to_numpy()
        group_va = va.groupby('session_id', sort=False).size().to_numpy()
        
        # --- TEK BİR SIRALAMA MODELİ İÇİN OPTİMİZASYON ---
        print(f"   - Fold {fold+1}: LGBMRanker için Optuna çalışıyor ({N_TRIALS} deneme)...")
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: run_ranker_optimization(trial, tr, va, feat_cols, group_tr, group_va), n_trials=N_TRIALS)
        
        best_params = study.best_params
        print(f"   - Fold {fold+1}: En iyi nDCG: {study.best_value:.6f}")
        
        # --- En iyi parametrelerle final modeli eğit ve test için tahmin yap ---
        print(f"   - Fold {fold+1}: Final model eğitiliyor ve test tahmini yapılıyor...")
        model = lgb.LGBMRanker(objective='lambdarank', metric='ndcg', random_state=42, n_jobs=-1, n_estimators=4000, **best_params)
        model.fit(tr[feat_cols], tr['target'], group=group_tr,
                  eval_set=[(va[feat_cols], va['target'])], eval_group=[group_va],
                  callbacks=[lgb.early_stopping(150, verbose=False)])
        
        oof_test_preds.append(model.predict(testX[feat_cols]))
        print(f"FOLD {fold+1}/{N_SPLITS} TAMAMLANDI")

    # =================================================================
    # ADIM 4: TAHMİNLERİ BİRLEŞTİR VE GÖNDERİM DOSYASI OLUŞTUR
    # =================================================================
    print("\n[3/4] Tüm katmanların sıralama skorları birleştiriliyor...")
    
    avg_rank_scores = np.mean(oof_test_preds, axis=0)
    
    out = testX[["session_id", "content_id_hashed"]].copy()
    out["score"] = avg_rank_scores
    
    print("[4/4] Gönderim dosyası oluşturuluyor...")
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
    parser = argparse.ArgumentParser(description="Kapsamlı LTR Pipeline (LGBMRanker + Optuna + v4 Features)")
    parser.add_argument("--data_dir", type=str, default="data", help="Veri klasörü.")
    parser.add_argument("--out", type=str, default="submission_ranker_v4.csv", help="Gönderim dosyası adı.")
    parser.add_argument("--force_rebuild", action="store_true", help="v4 özellikleri yeniden oluşturur.")
    parser.add_argument("--n_trials", type=int, default=30, help="Her katman için Optuna deneme sayısı.")
    args = parser.parse_args()
    main(args)