# features/user_features.py v4

import pandas as pd
from pathlib import Path
from utils.helpers import safe_div

def generate_user_features(data_path: Path) -> pd.DataFrame:
    """
    Tüm user (kullanıcı) verilerini okur, işler ve kullanıcı bazında zenginleştirilmiş
    özet bir tablo döndürür. Bu tablo, ana pipeline tarafından kullanılacaktır.

    v4 Değişiklikleri:
    - Sadece son log'u almak yerine, kullanıcının tüm site-geneli ve arama log
      geçmişini groupby ile özetleyerek daha zengin istatistikler (toplam, ortalama, std)
      üretir. Bu, kullanıcının genel davranış profiline dair daha derin bir anlayış sağlar.

    Args:
        data_path (Path): Ana 'data' klasörünün yolu.

    Returns:
        pd.DataFrame: Her bir 'user_id_hashed' için birleştirilmiş ve özetlenmiş
                      tüm kullanıcı özelliklerini içeren DataFrame.
    """
    print("    [User] Meta, sitewide ve search log verileri okunuyor...")
    user_meta = pd.read_parquet(
        data_path / "user" / "metadata.parquet",
        columns=["user_id_hashed", "user_gender", "user_birth_year", "user_tenure_in_days"],
    )
    u_site = pd.read_parquet(
        data_path / "user" / "sitewide_log.parquet",
    )
    u_search = pd.read_parquet(
        data_path / "user" / "search_log.parquet",
        columns=["user_id_hashed", "total_search_impression", "total_search_click"],
    )

    print("    [User] Kullanıcı tabloları özetleniyor...")
    # Temel demografik bilgiler
    user_meta["user_age"] = 2024 - user_meta["user_birth_year"]
    user_meta = user_meta.drop("user_birth_year", axis=1)

    # 1. Site-geneli log'ları özetle
    agg_funcs = ["sum", "mean", "std"]
    user_site_agg = u_site.groupby("user_id_hashed")[
        ["total_click", "total_cart", "total_fav", "total_order"]
    ].agg(agg_funcs)
    user_site_agg.columns = ["_".join(col) for col in user_site_agg.columns.values]

    # Kullanıcının genel dönüşüm oranları
    user_site_agg["user_global_ctr"] = safe_div(user_site_agg["total_order_sum"], user_site_agg["total_click_sum"])
    user_site_agg["user_global_cart_rate"] = safe_div(user_site_agg["total_cart_sum"], user_site_agg["total_click_sum"])

    # 2. Arama log'larını özetle
    user_search_agg = u_search.groupby("user_id_hashed")[
        ["total_search_impression", "total_search_click"]
    ].sum()
    user_search_agg.columns = ["user_search_impression_sum", "user_search_click_sum"]
    user_search_agg["user_search_ctr"] = safe_div(
        user_search_agg["user_search_click_sum"],
        user_search_agg["user_search_impression_sum"]
    )

    # İşlenmiş kullanıcı tablolarını birleştir
    final_user_df = user_meta.merge(user_site_agg, on="user_id_hashed", how="left")
    final_user_df = final_user_df.merge(user_search_agg, on="user_id_hashed", how="left")
    
    final_user_df = final_user_df.set_index("user_id_hashed")

    print(f"    [User] Özellik üretimi tamamlandı. Shape: {final_user_df.shape}")
    return final_user_df