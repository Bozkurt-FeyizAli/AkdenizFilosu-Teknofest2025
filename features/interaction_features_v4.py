# features/interaction_features.py v4
import pandas as pd
import numpy as np
from pathlib import Path
from utils.helpers import safe_div

# =================================================================
# YENİ GÖREV (v4): Kullanıcı Etkileşim Geçmişi Özellikleri (Sızıntısız)
# =================================================================
def generate_user_history_features(main_df: pd.DataFrame, content_df: pd.DataFrame) -> pd.DataFrame:
    """
    Her bir oturum için, o ana kadar olan kullanıcı etkileşim geçmişini (sızıntısız) hesaplar.
    - Kullanıcı-İçerik: Bu kullanıcı bu ürünü daha önce kaç kere tıkladı/sipariş etti?
    - Kullanıcı-Kategori: Bu kullanıcı bu kategoriden daha önce kaç kere sipariş verdi?

    Args:
        main_df (pd.DataFrame): train_sessions veya test_sessions'dan gelen ana tablo.
        content_df (pd.DataFrame): content özelliklerini içeren tablo.

    Returns:
        pd.DataFrame: Yeni geçmiş özellikleri eklenmiş DataFrame.
    """
    print("    [Interaction] Sızıntısız kullanıcı-içerik geçmişi özellikleri üretiliyor...")
    
    # Geçmiş hesaplaması için gerekli kolonları birleştir
    cols = ["ts_hour", "user_id_hashed", "content_id_hashed", "clicked", "ordered"]
    hist_df = main_df[cols].merge(content_df[["leaf_category_name"]], on="content_id_hashed", how="left")
    hist_df = hist_df.sort_values("ts_hour").reset_index(drop=True)

    # 1. Kullanıcı-İçerik Etkileşim Geçmişi
    # cumcount(), o ana kadar aynı gruptan kaç tane olduğunu sayar. Bu, sızıntıyı önler.
    grouped_user_content = hist_df.groupby(["user_id_hashed", "content_id_hashed"])
    hist_df["user_content_view_count"] = grouped_user_content.cumcount()
    hist_df["user_content_click_sum"] = grouped_user_content["clicked"].cumsum() - hist_df["clicked"]
    hist_df["user_content_order_sum"] = grouped_user_content["ordered"].cumsum() - hist_df["ordered"]
    
    # 2. Kullanıcı-Kategori Etkileşim Geçmişi
    grouped_user_cat = hist_df.groupby(["user_id_hashed", "leaf_category_name"])
    hist_df["user_cat_view_count"] = grouped_user_cat.cumcount()
    hist_df["user_cat_click_sum"] = grouped_user_cat["clicked"].cumsum() - hist_df["clicked"]
    hist_df["user_cat_order_sum"] = grouped_user_cat["ordered"].cumsum() - hist_df["ordered"]
    
    # Oranlar
    hist_df["user_content_ctr"] = safe_div(hist_df["user_content_click_sum"], hist_df["user_content_view_count"])
    hist_df["user_cat_cvr"] = safe_div(hist_df["user_cat_order_sum"], hist_df["user_cat_click_sum"])

    feature_cols = [
        "user_content_view_count", "user_content_click_sum", "user_content_order_sum",
        "user_cat_view_count", "user_cat_click_sum", "user_cat_order_sum",
        "user_content_ctr", "user_cat_cvr"
    ]
    return hist_df[feature_cols]

# =================================================================
# GÖREV 1: Ürün ve Arama Terimi Etkileşimi (DEĞİŞİKLİK YOK)
# =================================================================
def generate_top_terms_features(data_path: Path) -> pd.DataFrame:
    """Mevcut haliyle iyi, değişiklik yapılmadı."""
    # ... (kodunuzun orijinal hali buraya gelecek)
    print("    [Interaction] Top terms verisi okunuyor ve işleniyor...")
    top_terms = pd.read_parquet(
        data_path / "content" / "top_terms_log.parquet",
        columns=["content_id_hashed", "search_term_normalized", "total_search_impression", "total_search_click"],
    )
    for c in ["content_id_hashed", "search_term_normalized"]:
        top_terms[c] = top_terms[c].astype("category")

    top_terms_small = (
        top_terms.groupby(["content_id_hashed", "search_term_normalized"], as_index=False, observed=True)
        .agg(
            tt_search_impr=("total_search_impression", "sum"),
            tt_search_click=("total_search_click", "sum")
        )
    )

    if len(top_terms_small) > 1_000_000:
        q = top_terms_small["tt_search_impr"].quantile(0.05)
        top_terms_small = top_terms_small[top_terms_small["tt_search_impr"] >= q]

    top_terms_small["term_ctr"] = safe_div(
        top_terms_small["tt_search_click"].astype("float32"),
        top_terms_small["tt_search_impr"].astype("float32")
    )
    top_terms_small["tt_search_impr"] = top_terms_small["tt_search_impr"].astype("float32")
    top_terms_small["tt_search_click"] = top_terms_small["tt_search_click"].astype("float32")
    return top_terms_small


# =================================================================
# GÖREV 2: Oturum İçi Bağlamsal Özellikler (GELİŞTİRİLDİ)
# =================================================================
def add_session_context_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ana tabloya, bir ürünün kendi oturumu içindeki bağlamsal özelliklerini ekler.
    
    v4 Değişiklikleri:
    - Sadece fiyata değil, aynı zamanda ürünün genel popülerliğine (c_total_click)
      ve puanına (content_rate_avg) göre de oturum içi sıralama ve z-skor özellikleri eklendi.
    
    Args:
        df (pd.DataFrame): Ana özellik tablosu.

    Returns:
        pd.DataFrame: Yeni oturum-içi özellikler eklenmiş DataFrame.
    """
    print("    [Interaction] Oturum içi bağlamsal özellikler ekleniyor...")
    
    features_to_rank = {
        "discounted_price": "price",
        "c_total_click": "pop",       # content_features'dan geldiğini varsayıyoruz
        "content_rate_avg": "rate"  # content_features'dan geldiğini varsayıyoruz
    }
    
    tmp = df[["session_id"] + list(features_to_rank.keys())].copy()
    
    for col, prefix in features_to_rank.items():
        if col not in tmp.columns: continue
        
        grouped_session = tmp.groupby("session_id", observed=True)[col]
        
        # Rank (sıralama)
        tmp[f"session_{prefix}_rank"] = grouped_session.rank(method="average", pct=True) # pct=True ile normalize rank
        
        # Z-Score (standart sapmaya göre konumu)
        mean_val = grouped_session.transform("mean")
        std_val = grouped_session.transform("std").replace(0, np.nan)
        tmp[f"session_{prefix}_z"] = safe_div((tmp[col] - mean_val), std_val)

    # Eklenecek kolonları seçip ana DataFrame'e ekle
    new_cols = [col for col in tmp.columns if col not in features_to_rank and col != "session_id"]
    return df.join(tmp[new_cols])

# =================================================================
# GÖREV 3: Gelişmiş Fiyat Tercihi Özellikleri (DEĞİŞİKLİK YOK)
# =================================================================
# Bu bölümdeki fonksiyonlar (_robust_pref_table, _merge_pref_with_fallback, 
# add_price_preference_features) zaten çok iyi ve sızıntısız tasarlandığı için
# olduğu gibi bırakılmıştır. Onları buraya tekrar kopyalamıyorum.
# ... (kodunuzun orijinal hali buraya gelecek)

# =================================================================
# GÖREV 1: Ürün ve Arama Terimi Etkileşimi
# =================================================================
def generate_top_terms_features(data_path: Path) -> pd.DataFrame:
    """
    content/top_terms_log.parquet dosyasını okur, işler ve her bir (içerik, arama terimi)
    ikilisi için toplam gösterim, tıklama ve tıklanma oranı (CTR) gibi özellikler üretir.

    Args:
        data_path (Path): Ana 'data' klasörünün yolu.

    Returns:
        pd.DataFrame: 'content_id_hashed' ve 'search_term_normalized' bazında özetlenmiş
                      özellikleri içeren DataFrame.
    """
    print("    [Interaction] Top terms verisi okunuyor ve işleniyor...")
    top_terms = pd.read_parquet(
        data_path / "content" / "top_terms_log.parquet",
        columns=["content_id_hashed", "search_term_normalized", "total_search_impression", "total_search_click"],
    )
    for c in ["content_id_hashed", "search_term_normalized"]:
        top_terms[c] = top_terms[c].astype("category")

    top_terms_small = (
        top_terms.groupby(["content_id_hashed", "search_term_normalized"], as_index=False, observed=True)
        .agg(
            tt_search_impr=("total_search_impression", "sum"),
            tt_search_click=("total_search_click", "sum")
        )
    )

    # Düşük gösterime sahip olanları kırparak RAM'den tasarruf et (opsiyonel)
    if len(top_terms_small) > 1_000_000:
        q = top_terms_small["tt_search_impr"].quantile(0.05)
        top_terms_small = top_terms_small[top_terms_small["tt_search_impr"] >= q]

    top_terms_small["term_ctr"] = safe_div(
        top_terms_small["tt_search_click"].astype("float32"),
        top_terms_small["tt_search_impr"].astype("float32")
    )
    top_terms_small["tt_search_impr"] = top_terms_small["tt_search_impr"].astype("float32")
    top_terms_small["tt_search_click"] = top_terms_small["tt_search_click"].astype("float32")

    return top_terms_small


# =================================================================
# GÖREV 2: Oturum İçi Bağlamsal Özellikler
# =================================================================
def add_session_context_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ana tabloya, bir ürünün kendi oturumu içindeki bağlamsal özelliklerini ekler.
    Örneğin, o oturumdaki diğer ürünlere göre fiyat sıralaması.

    Args:
        df (pd.DataFrame): Ana özellik tablosu (train veya test).

    Returns:
        pd.DataFrame: Yeni oturum-içi özellikler eklenmiş DataFrame.
    """
    print("    [Interaction] Oturum içi fiyat pozisyonu özellikleri ekleniyor...")
    tmp = df[["session_id", "discounted_price"]].copy()
    
    # observed=True, sadece mevcut session_id'ler için işlem yaparak performansı artırır.
    tmp["price_rank"] = tmp.groupby("session_id", observed=True)["discounted_price"].rank(method="average")
    tmp["price_mean"] = tmp.groupby("session_id", observed=True)["discounted_price"].transform("mean")
    tmp["price_std"] = tmp.groupby("session_id", observed=True)["discounted_price"].transform("std").replace(0, np.nan)
    tmp["price_z"] = safe_div((tmp["discounted_price"] - tmp["price_mean"]), tmp["price_std"])
    
    return df.join(tmp[["price_rank", "price_z"]])


# =================================================================
# GÖREV 3: Gelişmiş Fiyat Tercihi Özellikleri (Sızıntısız)
# =================================================================
def _robust_pref_table(df, group_keys, price_col="discounted_price"):
    """Yardımcı fonksiyon: Verilen gruplar için sağlam fiyat istatistikleri hesaplar."""
    if df.empty:
        return pd.DataFrame(columns=group_keys + ["price_median", "q25", "q75", "band_width"])
    gb = df.groupby(group_keys)[price_col]
    out = gb.agg(
        price_median="median",
        q25=lambda s: s.quantile(0.25),
        q75=lambda s: s.quantile(0.75),
    ).reset_index()
    out["band_width"] = (out["q75"] - out["q25"]).replace(0, np.nan)
    return out

def _merge_pref_with_fallback(base_df, pref_tables, keys_list):
    """Yardımcı fonksiyon: Hiyerarşik bir şekilde fiyat tercihlerini ana tabloya ekler."""
    X = base_df.copy()
    for c in ["price_median", "q25", "q75", "band_width"]:
        if c not in X.columns: X[c] = np.nan
    filled = pd.Series(False, index=X.index)
    for pref, keys in zip(pref_tables, keys_list):
        if pref is None or pref.empty: continue
        need = ~filled
        if need.any():
            cols_to_merge = keys + ["price_median", "q25", "q75", "band_width"]
            tmp = X.loc[need, keys].merge(pref[cols_to_merge], on=keys, how="left")
            idx = X.index[need]
            for c in ["price_median", "q25", "q75", "band_width"]:
                vals = tmp[c].values
                write_mask = ~pd.isna(vals)
                X.loc[idx[write_mask], c] = vals[write_mask]
            filled = ~X["price_median"].isna()

    bw = X["band_width"]
    if bw.isna().all(): bw = pd.Series(0.1, index=X.index)
    bw = bw.fillna(bw.median() if not np.isnan(bw.median()) else 0.1)
    bw = bw.replace(0, bw.median() if not np.isnan(bw.median()) else 0.1)

    X["price_delta"] = X["discounted_price"] - X["price_median"]
    X.loc[X["price_median"].isna(), "price_delta"] = np.nan
    X["abs_price_delta"] = X["price_delta"].abs()
    X["z_price_delta"] = X["price_delta"] / (bw + 1e-6)
    X["in_band"] = ((X["discounted_price"] >= X["q25"]) & (X["discounted_price"] <= X["q75"])).astype(float)
    X["price_gauss_affinity"] = np.exp(-((X["abs_price_delta"]) / (bw + 1e-6)) ** 2)

    for col in ["price_delta", "abs_price_delta", "z_price_delta", "in_band", "price_gauss_affinity"]:
        X[col] = X[col].fillna(0.0)
    return X


def add_price_preference_features(train_df, test_df, cutoff):
    """
    Veri sızıntısını önleyerek, kullanıcıların geçmiş siparişlerine dayalı fiyat tercihlerini
    hesaplar ve bu özellikleri train ve test setlerine ekler.

    Args:
        train_df (pd.DataFrame): Tüm birleştirilmiş train seti.
        test_df (pd.DataFrame): Tüm birleştirilmiş test seti.
        cutoff (datetime): Train ve validasyon setini ayıran zaman damgası.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Yeni özellikler eklenmiş train ve test DataFrame'leri.
    """
    print("    [Interaction] Sızıntısız fiyat-tercih özellikleri hesaplanıyor...")
    # 1. Sadece geçmiş verilerden (ordered) tercih tablolarını oluştur.
    hist = train_df[train_df["ts_hour"] < cutoff].copy()
    pref_base = hist[hist["ordered"] == 1].copy()
    if pref_base.empty: pref_base = hist[hist["clicked"] == 1].copy()

    # Hiyerarşik tercih tabloları (spesifikten genele)
    pref_usr_season_cat = _robust_pref_table(pref_base, ["user_id_hashed", "season", "leaf_category_name"])
    pref_usr_season = _robust_pref_table(pref_base, ["user_id_hashed", "season"])
    pref_term_season = _robust_pref_table(pref_base, ["search_term_normalized", "season"])
    pref_global_season = _robust_pref_table(pref_base, ["season"])
    
    pref_tables_hist = [pref_usr_season_cat, pref_usr_season, pref_term_season, pref_global_season]
    keys_list_hist = [
        ["user_id_hashed", "season", "leaf_category_name"],
        ["user_id_hashed", "season"],
        ["search_term_normalized", "season"],
        ["season"]
    ]

    # Bu tercihleri train setinin parçalarına uygula
    train_part = _merge_pref_with_fallback(hist, pref_tables_hist, keys_list_hist)
    valid_part = train_df[train_df["ts_hour"] >= cutoff].copy()
    valid_part_enh = _merge_pref_with_fallback(valid_part, pref_tables_hist, keys_list_hist)
    train_df_enh = pd.concat([train_part, valid_part_enh], axis=0).sort_index()

    # 2. Test seti için, TÜM train verisini kullanarak tercih tablolarını yeniden oluştur.
    print("    [Interaction] Test seti için tercih tabloları yeniden oluşturuluyor...")
    pref_base_all = train_df[train_df["ordered"] == 1].copy()
    if pref_base_all.empty: pref_base_all = train_df[train_df["clicked"] == 1].copy()
    
    pref_usr_season_cat_all = _robust_pref_table(pref_base_all, ["user_id_hashed", "season", "leaf_category_name"])
    pref_usr_season_all = _robust_pref_table(pref_base_all, ["user_id_hashed", "season"])
    pref_term_season_all = _robust_pref_table(pref_base_all, ["search_term_normalized", "season"])
    pref_global_season_all = _robust_pref_table(pref_base_all, ["season"])

    pref_tables_all = [pref_usr_season_cat_all, pref_usr_season_all, pref_term_season_all, pref_global_season_all]
    
    test_df_enh = _merge_pref_with_fallback(test_df, pref_tables_all, keys_list_hist) # keys_list aynı

    return train_df_enh, test_df_enh