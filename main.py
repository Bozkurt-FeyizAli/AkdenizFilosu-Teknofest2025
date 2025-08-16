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
from utils.helpers import group_sizes_by_session
from build_features_polars import generate_features
from features.advanced_features import (
    generate_content_term_features,
    generate_user_term_features,
    generate_content_recency_features,
    add_text_similarity_features,
)  # yeni

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
# SIZINTISIZ OOF TARGET ENCODING (CLICKED / ORDERED)
# =================================================================

def add_oof_target_encoding(trainX: pd.DataFrame, testX: pd.DataFrame, keys_list, targets, n_splits: int = 5, smoothing: float = 20.0):
    """
    Oturum (session_id) gruplarını bozmadan, GroupKFold ile sızıntısız target encoding.
    Her key kombinasyonu ve her hedef için OOF train ve full-fit test encodings üretir.
    """
    print("   - OOF Target Encoding başlıyor...")
    # Eksik kategorileri boş string ile doldur (stabil join için)
    all_key_cols = sorted(set([k for keys in keys_list for k in keys]))
    for col in all_key_cols:
        trainX[col] = trainX[col].fillna('').astype(str)
        testX[col] = testX[col].fillna('').astype(str)

    groups = trainX['session_id']
    gkf = GroupKFold(n_splits=n_splits)

    for keys in keys_list:
        for target in targets:
            colname = f"te_{target}__{'__'.join(keys)}"
            print(f"     • {colname} hesaplanıyor...")
            global_mean = float(trainX[target].mean())
            oof_vals = np.zeros(len(trainX), dtype=np.float32)

            for tr_idx, va_idx in gkf.split(trainX, trainX[target], groups=groups):
                tr_fold = trainX.iloc[tr_idx]
                stats = tr_fold.groupby(keys)[target].agg(['mean', 'count']).reset_index()
                stats['te'] = (stats['mean'] * stats['count'] + global_mean * smoothing) / (stats['count'] + smoothing)
                va_keys = trainX.iloc[va_idx][keys]
                va_te = va_keys.merge(stats[keys + ['te']], on=keys, how='left')['te'].fillna(global_mean).astype('float32').values
                oof_vals[va_idx] = va_te

            trainX[colname] = oof_vals

            # Test için full-fit encoding
            stats_full = trainX.groupby(keys)[target].agg(['mean', 'count']).reset_index()
            stats_full['te'] = (stats_full['mean'] * stats_full['count'] + global_mean * smoothing) / (stats_full['count'] + smoothing)
            testX[colname] = testX[keys].merge(stats_full[keys + ['te']], on=keys, how='left')['te'].fillna(global_mean).astype('float32')

    print("   - OOF Target Encoding tamamlandı.")
    return trainX, testX


# =================================================================
# ANA ORKESTRASYON
# =================================================================

def main(args):
    DATA = Path(args.data_dir)
    N_SPLITS = 4  # Çapraz validasyon için katman sayısı
    N_TRIALS = 25 # Her model için Optuna deneme sayısı

    # ADIM 1: ÖZELLİK ÜRETİMİ
    train_feature_path = DATA / "train_features.parquet"
    test_feature_path = DATA / "test_features.parquet"
    if not train_feature_path.exists() or not test_feature_path.exists() or args.force_rebuild:
        generate_features(DATA, is_train=True)
        generate_features(DATA, is_train=False)

    # ADIM 1.5: Etkileşim ve recency özellikleri üret ve cache'le
    ct_path = DATA / "content_term_features.parquet"
    if not ct_path.exists() or args.force_rebuild:
        ct_df = generate_content_term_features(DATA, prior=80.0, min_impr=0.0)
        ct_df.to_parquet(ct_path, index=False)
    else:
        ct_df = pd.read_parquet(ct_path)

    ut_path = DATA / "user_term_features.parquet"
    if not ut_path.exists() or args.force_rebuild:
        ut_df = generate_user_term_features(DATA, prior=80.0)
        ut_df.to_parquet(ut_path, index=False)
    else:
        ut_df = pd.read_parquet(ut_path)

    rec_path = DATA / "content_recency_features.parquet"
    if not rec_path.exists() or args.force_rebuild:
        rec_df = generate_content_recency_features(DATA, half_life_days=28.0)
        rec_df.to_parquet(rec_path, index=False)
    else:
        rec_df = pd.read_parquet(rec_path)

    # ADIM 2: VERİ OKUMA VE HAZIRLAMA
    print("\n[1/3] Veriler okunuyor ve hazırlanıyor...")
    trainX = pd.read_parquet(train_feature_path)
    testX = pd.read_parquet(test_feature_path)

    # İçerik x terim & kullanıcı x terim & recency özelliklerini ana tabloya ekle
    join_keys_ct = ["content_id_hashed", "search_term_normalized"]
    trainX = trainX.merge(ct_df, on=join_keys_ct, how='left')
    testX = testX.merge(ct_df, on=join_keys_ct, how='left')

    join_keys_ut = ["user_id_hashed", "search_term_normalized"]
    trainX = trainX.merge(ut_df, on=join_keys_ut, how='left')
    testX = testX.merge(ut_df, on=join_keys_ut, how='left')

    trainX = trainX.merge(rec_df, on="content_id_hashed", how='left')
    testX = testX.merge(rec_df, on="content_id_hashed", how='left')

    trainX = add_advanced_features(trainX)
    testX = add_advanced_features(testX)

    # Yeni: Char n-gram cosine benzerliği
    trainX = add_text_similarity_features(trainX)
    testX = add_text_similarity_features(testX)

    # Yeni: Sızıntısız OOF target encoding (ordered/clicked) – kategorik grup anahtarları
    te_keys = [
        ['search_term_normalized'],
        ['leaf_category_name'],
        ['level2_category_name'],
        ['search_term_normalized', 'leaf_category_name'],
    ]
    trainX, testX = add_oof_target_encoding(trainX, testX, te_keys, targets=['clicked', 'ordered'], n_splits=N_SPLITS, smoothing=25.0)

    cat_cols = ["level1_category_name", "level2_category_name", "leaf_category_name", "user_gender"]
    ignore_cols = [
        "ts_hour", "search_term_normalized", "clicked", "ordered", "added_to_cart", "added_to_fav",
        "user_id_hashed", "content_id_hashed", "session_id", "content_creation_date", "cv_tags", "update_date", "date"
    ] + cat_cols
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
    # OOF validasyon skorları (blend ağırlığını ayarlamak için)
    val_order_oof = np.zeros(len(trainX), dtype=np.float32)
    val_click_oof = np.zeros(len(trainX), dtype=np.float32)

    # Ranker için OOF/Test tahminleri
    oof_test_order_ranker_preds = []
    val_order_ranker_oof = np.zeros(len(trainX), dtype=np.float32)

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

        # --- En iyi parametrelerle sınıflandırıcı modelleri eğit ve OOF/Test tahmini yap ---
        print(f"   - Fold {fold+1}: Final sınıflandırıcı modeller eğitiliyor ve tahmin yapılıyor...")
        model_order = lgb.LGBMClassifier(**best_params_order, objective='binary', random_state=42, n_jobs=-1, n_estimators=2000)
        model_order.fit(tr[feat_cols], tr['ordered'], eval_set=[(va[feat_cols], va['ordered'])], callbacks=[lgb.early_stopping(100, verbose=False)])
        val_order = model_order.predict_proba(va[feat_cols])[:, 1]
        val_order_oof[val_idx] = val_order.astype(np.float32)
        oof_test_order_preds.append(model_order.predict_proba(testX[feat_cols])[:, 1])

        model_click = lgb.LGBMClassifier(**best_params_click, objective='binary', random_state=42, n_jobs=-1, n_estimators=2000)
        model_click.fit(tr[feat_cols], tr['clicked'], eval_set=[(va[feat_cols], va['clicked'])], callbacks=[lgb.early_stopping(100, verbose=False)])
        val_click = model_click.predict_proba(va[feat_cols])[:, 1]
        val_click_oof[val_idx] = val_click.astype(np.float32)
        oof_test_click_preds.append(model_click.predict_proba(testX[feat_cols])[:, 1])

        # --- Ranker (LGBMRanker) ile sipariş için oturum-bilinçli sıralama ---
        print(f"   - Fold {fold+1}: LGBMRanker (ordered) eğitiliyor...")
        tr_rank = tr.sort_values('session_id').reset_index(drop=True)
        va_rank = va.sort_values('session_id').reset_index(drop=True)
        # Use helper for group sizes to avoid unused import warning
        group_tr = group_sizes_by_session(tr_rank).astype(int).tolist()
        group_va = group_sizes_by_session(va_rank).astype(int).tolist()

        ranker = lgb.LGBMRanker(
            objective='lambdarank',
            metric='ndcg',
            n_estimators=2000,
            learning_rate=0.05,
            num_leaves=127,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            n_jobs=-1,
        )
        ranker.fit(
            tr_rank[feat_cols], tr_rank['ordered'],
            group=group_tr,
            eval_set=[(va_rank[feat_cols], va_rank['ordered'])],
            eval_group=[group_va],
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )
        val_ranker = ranker.predict(va_rank[feat_cols])
        val_order_ranker_oof[val_idx] = val_ranker.astype(np.float32)

        # Test için ranker tahminleri
        test_rank = testX.sort_values('session_id').reset_index(drop=True)
        test_pred_rank = ranker.predict(test_rank[feat_cols])
        # Orijinal sıraya döndür
        test_pred_rank = pd.Series(test_pred_rank, index=test_rank.index).sort_index().values
        oof_test_order_ranker_preds.append(test_pred_rank)
        
        print(f"FOLD {fold+1}/{N_SPLITS} TAMAMLANDI")

    # 2D grid: (beta) ranker vs classifier (ordered), (w) order vs click
    print("\n2D blend araması: beta (ranker vs classifier) ve w (order vs click)...")
    base_df = trainX[["session_id", "clicked", "ordered"]].copy()
    best_combo = (0.5, 0.72)
    best_score = -1.0
    for beta in np.linspace(0.2, 0.8, 7):
        s_order = beta * val_order_ranker_oof + (1.0 - beta) * val_order_oof
        s_click = val_click_oof
        for w in np.linspace(0.60, 0.85, 14):
            s = w * s_order + (1.0 - w) * s_click
            auc_click = session_auc(base_df, s, 'clicked')
            auc_order = session_auc(base_df, s, 'ordered')
            comb = 0.3 * auc_click + 0.7 * auc_order
            if comb > best_score:
                best_score = comb
                best_combo = (float(beta), float(w))
    beta, w = best_combo
    print(f"Seçilen beta={beta:.3f}, w={w:.3f} (OOF skor={best_score:.6f})")

    # =================================================================
    # ADIM 4: TAHMİNLERİ BİRLEŞTİR VE GÖNDERİM DOSYASI OLUŞTUR
    # =================================================================
    print("\n[3/3] Tüm katmanların tahminleri birleştiriliyor...")

    avg_order_preds = np.mean(oof_test_order_preds, axis=0)
    avg_click_preds = np.mean(oof_test_click_preds, axis=0)
    avg_order_ranker = np.mean(oof_test_order_ranker_preds, axis=0)

    final_order_test = beta * avg_order_ranker + (1.0 - beta) * avg_order_preds
    test_scores_blended = (w * final_order_test) + ((1.0 - w) * avg_click_preds)

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
