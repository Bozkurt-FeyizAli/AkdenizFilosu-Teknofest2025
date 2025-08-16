import argparse
from pathlib import Path
import pandas as pd
import lightgbm as lgb
import optuna
import warnings
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import GroupKFold
from optuna.integration import LightGBMPruningCallback

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

# Tek seferlik ve örneklem üzerinden hızlı hiperparametre araması
def tune_best_params(train_df: pd.DataFrame,
                     groups: pd.Series,
                     target_col: str,
                     feat_cols: list[str],
                     n_trials: int,
                     n_estimators: int,
                     timeout_min: int,
                     sample_frac: float,
                     random_state: int = 42) -> dict:
    rng = np.random.default_rng(random_state)
    # Grup bazlı örnekleme (session_id)
    unique_groups = train_df['session_id'].unique()
    sample_size = max(1, int(len(unique_groups) * sample_frac))
    sampled_groups = rng.choice(unique_groups, size=sample_size, replace=False)
    df_sub = train_df[train_df['session_id'].isin(sampled_groups)].reset_index(drop=True)
    groups_sub = df_sub['session_id']

    # Tek katman train/valid böl
    gkf = GroupKFold(n_splits=min(5, max(2, int(1 / max(1e-6, 1 - sample_frac)) + 2)))
    split = next(iter(gkf.split(df_sub, df_sub[target_col], groups=groups_sub)))
    tr_idx, va_idx = split
    tr, va = df_sub.iloc[tr_idx], df_sub.iloc[va_idx]

    def objective(trial: optuna.Trial) -> float:
        y_tr, y_va = tr[target_col], va[target_col]
        pos_weight = (y_tr == 0).sum() / max(1, (y_tr == 1).sum())
        params = {
            'objective': 'binary', 'metric': 'auc', 'random_state': random_state,
            'n_estimators': n_estimators, 'boosting_type': 'gbdt', 'n_jobs': -1, 'verbosity': -1,
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 15, 256),
            'max_depth': trial.suggest_int('max_depth', 4, 16),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 20, 200),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 2.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 2.0),
            'scale_pos_weight': float(pos_weight),
        }
        model = lgb.LGBMClassifier(**params)
        model.fit(
            tr[feat_cols], y_tr,
            eval_set=[(va[feat_cols], y_va)],
            callbacks=[
                lgb.early_stopping(100, verbose=False),
                LightGBMPruningCallback(trial, 'auc', 'valid_0')
            ],
        )
        return float(model.best_score_['valid_0']['auc'])

    study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner(n_warmup_steps=10))
    study.optimize(objective, n_trials=n_trials, timeout=timeout_min * 60)
    best_params = study.best_params
    # Sabit parametreleri enjekte et
    best_params.update({'objective': 'binary', 'metric': 'auc', 'random_state': random_state,
                        'n_estimators': n_estimators, 'boosting_type': 'gbdt', 'n_jobs': -1, 'verbosity': -1})
    return best_params

# =================================================================
# ANA ORKESTRASYON
# =================================================================
def main(args):
    DATA = Path(args.data_dir)

    # Hızlı mod ve varsayılanlar
    if args.folds is not None:
        N_SPLITS = args.folds
    else:
        N_SPLITS = 3 if args.fast else 5

    if args.trials is not None:
        N_TRIALS = args.trials
    else:
        N_TRIALS = 12 if args.fast else 25

    if args.n_estimators is not None:
        N_ESTIMATORS = args.n_estimators
    else:
        N_ESTIMATORS = 800 if args.fast else 1500

    if args.study_timeout_min is not None:
        STUDY_TIMEOUT_MIN = args.study_timeout_min
    else:
        STUDY_TIMEOUT_MIN = 10 if args.fast else 30

    if args.tune_sample_frac is not None:
        TUNE_SAMPLE_FRAC = args.tune_sample_frac
    else:
        TUNE_SAMPLE_FRAC = 0.2 if args.fast else 0.35

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
    ignore_cols = [
        "ts_hour", "search_term_normalized", "clicked", "ordered", "added_to_cart", "added_to_fav",
        "user_id_hashed", "content_id_hashed", "session_id", "content_creation_date", "cv_tags",
        "update_date", "date"
    ] + cat_cols
    num_cols = [col for col in trainX.columns if col not in ignore_cols]
    feat_cols = cat_cols + num_cols
    
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    trainX[cat_cols] = enc.fit_transform(trainX[cat_cols].astype(str))
    testX[cat_cols] = enc.transform(testX[cat_cols].astype(str))
    
    for df in [trainX, testX]:
        for col in feat_cols:
            if col not in df.columns:
                df[col] = -1.0
        df[feat_cols] = df[feat_cols].fillna(-1.0).astype("float32")

    # =================================================================
    # ADIM 3: TEK SEFER HİPERPARAMETRE ARAMASI + CV EĞİTİM
    # =================================================================
    print(f"\n[2/3] {N_SPLITS}-Katmanlı CV ve Hafif Optimizasyon Başlıyor...")
    gkf = GroupKFold(n_splits=N_SPLITS)
    groups = trainX['session_id']

    # Fold indekslerini önceden hazırla
    fold_splits = list(gkf.split(trainX, trainX['ordered'], groups=groups))

    # 3.1 Click için tek seferlik hyperparametre araması
    print("   - Click hiperparametre araması (tek sefer, örneklem üzerinde)...")
    best_params_click = tune_best_params(
        train_df=trainX,
        groups=groups,
        target_col='clicked',
        feat_cols=feat_cols,
        n_trials=N_TRIALS,
        n_estimators=N_ESTIMATORS,
        timeout_min=STUDY_TIMEOUT_MIN,
        sample_frac=TUNE_SAMPLE_FRAC,
        random_state=42,
    )

    # 3.2 OOF p_click üret ve test için click tahminlerini topla
    oof_test_click_preds = []
    trainX['p_click'] = -1.0
    for fold, (train_idx, val_idx) in enumerate(fold_splits):
        print("-" * 50)
        print(f"FOLD {fold+1}/{N_SPLITS} (Click)")
        tr, va = trainX.iloc[train_idx], trainX.iloc[val_idx]
        model_click = lgb.LGBMClassifier(**best_params_click)
        model_click.fit(
            tr[feat_cols], tr['clicked'],
            eval_set=[(va[feat_cols], va['clicked'])],
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )
        va_click = model_click.predict_proba(va[feat_cols])[:, 1]
        trainX.loc[val_idx, 'p_click'] = va_click
        oof_test_click_preds.append(model_click.predict_proba(testX[feat_cols])[:, 1])

    # 3.3 Order için tek seferlik hyperparametre araması (p_click ile)
    feat_cols_with_p_click = feat_cols + ['p_click']
    print("   - Order hiperparametre araması (tek sefer, örneklem üzerinde, p_click ile)...")
    best_params_order = tune_best_params(
        train_df=trainX,
        groups=trainX['session_id'],
        target_col='ordered',
        feat_cols=feat_cols_with_p_click,
        n_trials=max(8, N_TRIALS // 2),
        n_estimators=N_ESTIMATORS,
        timeout_min=max(6, STUDY_TIMEOUT_MIN // 2),
        sample_frac=TUNE_SAMPLE_FRAC,
        random_state=42,
    )

    # 3.4 Order modeli için CV eğitimi ve test tahminleri
    oof_test_order_preds = []
    for fold, (train_idx, val_idx) in enumerate(fold_splits):
        print("-" * 50)
        print(f"FOLD {fold+1}/{N_SPLITS} (Order)")
        tr, va = trainX.iloc[train_idx], trainX.iloc[val_idx]
        tr_with_p = tr[feat_cols_with_p_click]
        va_with_p = va[feat_cols_with_p_click]

        model_order = lgb.LGBMClassifier(**best_params_order)
        model_order.fit(
            tr_with_p, tr['ordered'],
            eval_set=[(va_with_p, va['ordered'])],
            callbacks=[lgb.early_stopping(150 if not args.fast else 80, verbose=False)],
        )
        # Test setine fold'un click tahminini p_click olarak ekle
        testX_with_p_click = testX.copy()
        testX_with_p_click['p_click'] = oof_test_click_preds[fold]
        oof_test_order_preds.append(model_order.predict_proba(testX_with_p_click[feat_cols_with_p_click])[:, 1])

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

    # Hız/kalite kontrolü için yeni argümanlar
    parser.add_argument("--fast", action="store_true", help="Hızlı mod: katman/trial/estimator sayısını otomatik düşürür.")
    parser.add_argument("--folds", type=int, default=None, help="CV katman sayısı (varsayılan: fast=3, aksi halde 5)")
    parser.add_argument("--trials", type=int, default=None, help="Optuna deneme sayısı (varsayılan: fast=12, aksi halde 25)")
    parser.add_argument("--n_estimators", type=int, default=None, help="LightGBM ağaç sayısı (varsayılan: fast=800, aksi halde 1500)")
    parser.add_argument("--study_timeout_min", type=int, default=None, help="Optuna çalışma süresi (dakika) - her study için")
    parser.add_argument("--tune_sample_frac", type=float, default=None, help="Tuning için grup örnekleme oranı (0-1)")

    args = parser.parse_args()
    main(args)
