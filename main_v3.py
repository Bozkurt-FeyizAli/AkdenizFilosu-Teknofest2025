# main_v2.py
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna # Optuna'yı import et
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import roc_auc_score
import warnings

# --- Kendi Yardımcı Fonksiyonlarımız ---
def group_sizes_by_session(df):
    return df.groupby("session_id", observed=True).size().values

def session_auc(df_with_targets, scores, target_col):
    tmp = df_with_targets.copy()
    tmp["__score__"] = scores
    aucs = [
        roc_auc_score(g[target_col], g["__score__"])
        for _, g in tmp.groupby("session_id")
        if g[target_col].nunique() > 1
    ]
    return float(np.mean(aucs)) if aucs else float("nan")

# --- Polars Özellik Script'imizi Import Edelim ---
# Eğer build_features_polars_v2.py adıyla kaydettiyseniz:
from build_features_polars_v3 import generate_features_v3 

# Uyarıları bastır
warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 200)

def run_training(args):
    DATA = Path(args.data_dir)
    
    # 1. Gelişmiş Özellikleri Üret (Eğer dosyalar yoksa)
    # run_training fonksiyonu içinde:

    # 1. Gelişmiş Özellikleri Üret (Eğer dosyalar yoksa)
    train_feature_path = DATA / "train_features_v3.parquet" # v2'yi v3 yap
    test_feature_path = DATA / "test_features_v3.parquet"  # v2'yi v3 yap

    if not train_feature_path.exists() or not test_feature_path.exists() or args.force_rebuild:
        print("Gelişmiş özellik dosyaları (v3) bulunamadı veya yeniden oluşturma zorlandı.")
        generate_features_v3(DATA, is_train=True) # v2'yi v3 yap
        generate_features_v3(DATA, is_train=False) # v2'yi v3 yap
    else:
        print("Mevcut gelişmiş özellik dosyaları (v3) kullanılacak.")

    # 2. Önceden İşlenmiş Verileri Oku
    print("[1/5] Önceden işlenmiş train/test verileri okunuyor...")
    trainX = pd.read_parquet(train_feature_path)
    testX = pd.read_parquet(test_feature_path)

    # 3. Model için Son Hazırlıklar
    print("[2/5] Model için son hazırlıklar yapılıyor...")
    cat_cols = [c for c in trainX.columns if trainX[c].dtype == 'object' or isinstance(trainX[c].dtype, pd.CategoricalDtype)]
    cat_cols = [c for c in cat_cols if c not in ['user_id_hashed', 'content_id_hashed', 'session_id', 'search_term_normalized']]

    # OrdinalEncoder kategorik sütunlar için
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    trainX[cat_cols] = enc.fit_transform(trainX[cat_cols].astype(str))
    testX[cat_cols] = enc.transform(testX[cat_cols].astype(str))
    
    ignore_cols = ["ts_hour", "clicked", "ordered", "added_to_cart", "added_to_fav", 
                   "user_id_hashed", "content_id_hashed", "session_id", "search_term_normalized",
                   "content_creation_date", "cv_tags", "update_date", "date"]
    
    feat_cols = [col for col in trainX.columns if col not in ignore_cols]
    
    for df in [trainX, testX]:
        df[feat_cols] = df[feat_cols].fillna(-1.0).astype("float32")

    # 4. Veri Ayırma ve Hedef Belirleme
    print("[3/5] Train/validation setleri oluşturuluyor...")
    cutoff = trainX["ts_hour"].quantile(0.85) # Validasyon setini biraz daha küçültebiliriz.
    trainX["relevance"] = (
        3 * trainX.get("ordered", 0).astype(int) + 
        1 * trainX.get("clicked", 0).astype(int)
    )
    
    tr = trainX[trainX["ts_hour"] < cutoff].copy()
    va = trainX[trainX["ts_hour"] >= cutoff].copy()
    
    X_tr, y_tr = tr[feat_cols], tr["relevance"]
    X_va, y_va = va[feat_cols], va["relevance"]
    g_tr = group_sizes_by_session(tr)
    g_va = group_sizes_by_session(va)

    # 5. Optuna ile Hiperparametre Optimizasyonu ve Model Eğitimi
    def objective(trial):
        params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "boosting_type": "gbdt",
        "n_estimators": 5000,
        "random_state": 42,
        "n_jobs": -1,
        # ARAMA ALANLARINI GÜNCELLEYELİM
        "learning_rate": trial.suggest_float("learning_rate", 5e-3, 5e-2, log=True), # Aralığı biraz aşağı çektik
        "num_leaves": trial.suggest_int("num_leaves", 100, 500), # Daha basit modellere de şans verelim
        "max_depth": trial.suggest_int("max_depth", 5, 10), # En önemli değişiklik! Aralığı 5-10 arasına çektik.
        "subsample": trial.suggest_float("subsample", 0.6, 0.9),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.9),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 1.0, log=True), # Aralığı aşağı çektik
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }

        ranker = lgb.LGBMRanker(**params)
        ranker.fit(
            X_tr, y_tr, group=g_tr,
            eval_set=[(X_va, y_va)], eval_group=[g_va], eval_at=[20],
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )
        
        va_scores = ranker.predict(va[feat_cols])
        click_auc = session_auc(va[["session_id", "clicked"]], va_scores, "clicked")
        order_auc = session_auc(va[["session_id", "ordered"]], va_scores, "ordered")
        final_score = 0.7 * order_auc + 0.3 * click_auc
        
        print(f"TRIAL Sonucu -> Click AUC: {click_auc:.6f} | Order AUC: {order_auc:.6f} | Final Score: {final_score:.6f}")
        return final_score

    print("[4/5] Optuna ile hiperparametre optimizasyonu başlatılıyor...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=args.n_trials) # n_trials'ı 20-50 arası bir değerle başlatabilirsiniz

    print("\nEn iyi parametreler bulundu:", study.best_params)
    print("En iyi validasyon skoru:", study.best_value)

    # En iyi parametrelerle final modelini eğit
    final_ranker = lgb.LGBMRanker(**study.best_params, n_estimators=5000, random_state=42, n_jobs=-1)
    final_ranker.fit(
        X_tr, y_tr, group=g_tr,
        eval_set=[(X_va, y_va)], eval_group=[g_va], eval_at=[20],
        callbacks=[lgb.early_stopping(150, verbose=False), lgb.log_evaluation(200)],
    )

    # 6. Test Seti Tahmini ve Gönderim Dosyası
    print("[5/5] Test seti skorlanıyor ve gönderim dosyası oluşturuluyor...")
    test_scores = final_ranker.predict(testX[feat_cols])
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
    parser = argparse.ArgumentParser(description="V2: Optuna ve Gelişmiş Özelliklerle Eğitim Pipeline'ı")
    parser.add_argument("--data_dir", type=str, default="data", help="Veri setlerinin bulunduğu klasör.")
    parser.add_argument("--out", type=str, default="submission_v2_optuna.csv", help="Oluşturulacak gönderim dosyasının adı.")
    parser.add_argument("--force_rebuild", action="store_true", help="Mevcut olsalar bile özellik dosyalarını yeniden oluşturur.")
    parser.add_argument("--n_trials", type=int, default=30, help="Optuna optimizasyon deneme sayısı.") # Komut satırından ayarlanabilir
    args = parser.parse_args()
    run_training(args)