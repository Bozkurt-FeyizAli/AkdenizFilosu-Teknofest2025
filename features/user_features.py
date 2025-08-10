# features/user_features.py

import pandas as pd
from pathlib import Path
from utils.helpers import latest_by_key_fast, safe_div

def generate_user_features(data_path: Path) -> pd.DataFrame:
    """
    Tüm user (kullanıcı) verilerini okur, işler ve kullanıcı bazında özet bir tablo döndürür.
    Bu tablo, ana pipeline tarafından kullanılacaktır.

    Args:
        data_path (Path): Ana 'data' klasörünün yolu.

    Returns:
        pd.DataFrame: Her bir 'user_id_hashed' için birleştirilmiş ve özetlenmiş
                      tüm kullanıcı özelliklerini içeren DataFrame.
    """
    print("    [User] Meta ve log verileri okunuyor...")
    user_meta = pd.read_parquet(
        data_path / "user" / "metadata.parquet",
        columns=["user_id_hashed", "user_gender", "user_birth_year", "user_tenure_in_days"],
    )
    u_site = pd.read_parquet(
        data_path / "user" / "sitewide_log.parquet",
        columns=["user_id_hashed", "ts_hour", "total_click", "total_cart", "total_fav", "total_order"],
    )

    print("    [User] Kullanıcı tabloları özetleniyor...")
    # Temel demografik bilgiler
    user_basic = user_meta.copy()

    # Her kullanıcı için en güncel site-geneli log'ları al ve dönüşüm oranlarını hesapla
    user_last = latest_by_key_fast(u_site, "user_id_hashed", "ts_hour")[
        ["user_id_hashed", "total_click", "total_cart", "total_fav", "total_order"]
    ].copy()
    user_last["user_click_to_order"] = safe_div(user_last["total_order"], user_last["total_click"])
    user_last["user_cart_to_order"] = safe_div(user_last["total_order"], user_last["total_cart"])

    # İşlenmiş kullanıcı tablolarını birleştir
    final_user_df = user_basic.merge(user_last, on="user_id_hashed", how="left")
    
    print("    [User] Özellik üretimi tamamlandı.")
    return final_user_df