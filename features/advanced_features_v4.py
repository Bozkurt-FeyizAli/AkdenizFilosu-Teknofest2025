# features/advanced_features_v4.py
import pandas as pd
from pathlib import Path
from utils.helpers import safe_div

def generate_advanced_user_features(data_path: Path) -> pd.DataFrame:
    """
    Kullanılmayan kullanıcı loglarından (moda odaklı ve genel arama) ek ve daha
    sağlam özellikler üretir.

    v4 Değişiklikleri:
    - Kritik İyileştirme: Sadece son kaydı almak yerine, kullanıcının TÜM moda
      ve arama log geçmişini `groupby` ile özetleyerek genel davranış profilini
      (toplam, ortalama) çıkarır.
    
    Args:
        data_path (Path): Ana 'data' klasörünün yolu.

    Returns:
        pd.DataFrame: Gelişmiş kullanıcı özelliklerini içeren DataFrame.
    """
    print("    [Advanced User] Moda ve arama logları okunuyor...")
    
    # 1. Kullanıcının moda kategorisindeki genel davranışları
    u_fashion_log = pd.read_parquet(data_path / "user" / "fashion_sitewide_log.parquet")
    u_fashion_agg = u_fashion_log.groupby("user_id_hashed").agg(
        user_fashion_click_sum=('total_click', 'sum'),
        user_fashion_cart_sum=('total_cart', 'sum'),
        user_fashion_order_sum=('total_order', 'sum'),
        user_fashion_order_mean=('total_order', 'mean'),
    )
    u_fashion_agg["user_fashion_click_to_order"] = safe_div(u_fashion_agg["user_fashion_order_sum"], u_fashion_agg["user_fashion_click_sum"])
    
    # 2. Kullanıcının genel arama istatistikleri (zaten user_features_v4'te var, ama burada daha detaylı olabilir)
    u_search_log = pd.read_parquet(data_path / "user" / "search_log.parquet")
    u_search_agg = u_search_log.groupby("user_id_hashed").agg(
        user_search_impr_sum=('total_search_impression', 'sum'),
        user_search_click_sum=('total_search_click', 'sum'),
    )
    u_search_agg["user_global_search_ctr"] = safe_div(u_search_agg["user_search_click_sum"], u_search_agg["user_search_impr_sum"])

    # İki tabloyu birleştir
    final_df = u_fashion_agg.merge(u_search_agg, on="user_id_hashed", how="outer")
    
    print(f"    [Advanced User] Özellik üretimi tamamlandı. Shape: {final_df.shape}")
    return final_df


def generate_term_features(data_path: Path) -> pd.DataFrame:
    """
    Arama terimlerinin genel popülerliğini ve performansını hesaplar.
    
    v4 Değişiklikleri:
    - 'sum' ve 'mean' agregasyonları ile terim popülerliğinin hem toplam gücünü
      hem de zaman içindeki ortalama aktivitesini yakalar.
    
    Args:
        data_path (Path): Ana 'data' klasörünün yolu.

    Returns:
        pd.DataFrame: Arama terimi bazında özetlenmiş DataFrame.
    """
    print("    [Term] Arama terimi logları okunuyor...")
    term_log = pd.read_parquet(data_path / "term" / "search_log.parquet")
    
    term_agg = term_log.groupby("search_term_normalized").agg(
        term_total_impr_sum=("total_search_impression", "sum"),
        term_total_click_sum=("total_search_click", "sum"),
        term_total_click_mean=("total_search_click", "mean"),
    ).reset_index()
    
    term_agg["term_global_ctr"] = safe_div(term_agg["term_total_click_sum"], term_agg["term_total_impr_sum"])
    
    print(f"    [Term] Özellik üretimi tamamlandı. Shape: {term_agg.shape}")
    return term_agg.set_index("search_term_normalized")


def add_search_context_features(main_df: pd.DataFrame, user_features: pd.DataFrame, term_features: pd.DataFrame) -> pd.DataFrame:
    """
    (YENİ FONKSİYON - v4)
    Arama anındaki bağlamsal özellikleri ekler. Örneğin, kullanıcının kişisel
    CTR'sinin, aradığı terimin genel CTR'sine oranı. Bu, modelin "Bu kullanıcı,
    bu popüler terimde ortalamadan daha mı tıklayıcı?" sorusunu yanıtlamasını sağlar.
    
    Args:
        main_df (pd.DataFrame): Ana tablo (train veya test).
        user_features (pd.DataFrame): Kullanıcı özellikleri tablosu.
        term_features (pd.DataFrame): Terim özellikleri tablosu.

    Returns:
        pd.DataFrame: Yeni bağlamsal özellikler eklenmiş DataFrame.
    """
    print("    [Advanced Context] Arama-bağlamı özellikleri ekleniyor...")
    
    # Gerekli özellikleri ana tabloya ekle
    df = main_df.merge(user_features[['user_global_search_ctr']], on="user_id_hashed", how="left")
    df = df.merge(term_features[['term_global_ctr']], on="search_term_normalized", how="left")

    # Karşılaştırma özelliklerini üret
    df["user_ctr_vs_term_ctr"] = safe_div(df["user_global_search_ctr"], df["term_global_ctr"])
    
    # Ana tabloya sadece yeni üretilen özellikleri ekle, join maliyetini düşür
    main_df["user_ctr_vs_term_ctr"] = df["user_ctr_vs_term_ctr"].values
    
    return main_df