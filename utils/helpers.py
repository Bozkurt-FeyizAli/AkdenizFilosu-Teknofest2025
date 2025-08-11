import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score # Bu importu ekleyin

def safe_div(a, b):
    """Sıfıra bölme hatasını önleyen güvenli bölme işlemi."""
    a = a.astype(float)
    b = b.astype(float)
    out = np.zeros_like(a, dtype=float)
    mask = b > 0
    # Pandas serileri için .values kullanarak numpy dizisine erişim sağlayalım
    a_vals = a.values if isinstance(a, pd.Series) else a
    b_vals = b.values if isinstance(b, pd.Series) else b
    mask_vals = b_vals > 0
    out[mask_vals] = a_vals[mask_vals] / b_vals[mask_vals]
    
    if isinstance(a, pd.Series):
        return pd.Series(out, index=a.index)
    return out

def season_from_month(m):
    """Ay bilgisinden mevsimi döndürür (0:Kış, 1:İlkbahar, 2:Yaz, 3:Sonbahar)."""
    if m in [12, 1, 2]: return 0
    if m in [3, 4, 5]: return 1
    if m in [6, 7, 8]: return 2
    return 3

def add_time_feats(df, col="ts_hour"):
    """DataFrame'e zaman damgasından saat, hafta günü, ay gibi özellikler ekler."""
    dt = pd.to_datetime(df[col])
    df["hour"] = dt.dt.hour
    df["dow"] = dt.dt.dayofweek
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["month"] = dt.dt.month
    df["season"] = df["month"].apply(season_from_month)
    return df

def latest_by_key_fast(df, key, timecol):
    """
    Verilen bir anahtara göre gruplayıp her grup için zaman sütunundaki
    en son kaydı verimli bir şekilde alır.
    """
    idx = df.groupby(key)[timecol].idxmax()
    return df.loc[idx].reset_index(drop=True)

def group_sizes_by_session(df):
    """LGBMRanker için her bir session_id'nin boyutunu (grup boyutu) hesaplar."""
    return df.groupby("session_id", observed=True).size().values

def chunked_merge(left_df, right_df, on_cols, how="left", chunk_size=500_000):
    """
    Büyük bir DataFrame'i (left_df) küçük parçalara bölerek birleştirir.
    Bu, RAM kullanımını azaltır.
    """
    parts = []
    for start in range(0, len(left_df), chunk_size):
        chunk = left_df.iloc[start:start + chunk_size]
        parts.append(chunk.merge(right_df, on=on_cols, how=how))
    return pd.concat(parts, ignore_index=True)


# ## YENİ EKLENEN FONKSİYON ##
def session_auc(df_with_targets, scores, target_col):
    """
    Oturum bazında (session-wise) AUC skorunu hesaplar.
    """
    # Fonksiyonun Pandas DataFrame'leri ile çalıştığından emin olalım
    if not isinstance(df_with_targets, pd.DataFrame):
        df_with_targets = df_with_targets.to_pandas()

    tmp = df_with_targets.copy()
    tmp["__score__"] = scores
    
    # Oturumda en az bir pozitif ve bir negatif örnek varsa AUC hesapla
    aucs = [
        roc_auc_score(g[target_col], g["__score__"])
        for _, g in tmp.groupby("session_id")
        if g[target_col].nunique() > 1
    ]
    
    # Eğer hiçbir oturum AUC hesaplamaya uygun değilse NaN dön, aksi halde ortalamayı al
    return float(np.mean(aucs)) if aucs else float("nan")