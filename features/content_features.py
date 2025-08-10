# features/content_features.py

import pandas as pd
from pathlib import Path
from utils.helpers import latest_by_key_fast, safe_div

def generate_content_features(data_path: Path) -> pd.DataFrame:
    """
    Tüm content (ürün) verilerini okur, işler ve ürün bazında özet bir tablo döndürür.
    Bu tablo, ana pipeline tarafından kullanılacaktır.

    Args:
        data_path (Path): Ana 'data' klasörünün yolu.

    Returns:
        pd.DataFrame: Her bir 'content_id_hashed' için birleştirilmiş ve özetlenmiş
                      tüm ürün özelliklerini içeren DataFrame.
    """
    print("    [Content] Meta, fiyat ve log verileri okunuyor...")
    meta = pd.read_parquet(
        data_path / "content" / "metadata.parquet",
        columns=["content_id_hashed", "level1_category_name", "level2_category_name", "leaf_category_name",
                 "attribute_type_count", "total_attribute_option_count", "merchant_count", "filterable_label_count",
                 "content_creation_date"],
    )
    price = pd.read_parquet(
        data_path / "content" / "price_rate_review_data.parquet",
        columns=["content_id_hashed", "update_date", "original_price", "selling_price", "discounted_price",
                 "content_review_count", "content_review_wth_media_count", "content_rate_count", "content_rate_avg"],
    )
    c_search_log = pd.read_parquet(
        data_path / "content" / "search_log.parquet",
        columns=["content_id_hashed", "date", "total_search_impression", "total_search_click"],
    )
    c_sitewide_log = pd.read_parquet(
        data_path / "content" / "sitewide_log.parquet",
        columns=["content_id_hashed", "date", "total_click", "total_cart", "total_fav", "total_order"],
    )

    print("    [Content] Ürün tabloları özetleniyor...")
    # Her ürün için en güncel fiyat, yorum ve puan bilgilerini al
    content_price = latest_by_key_fast(price, "content_id_hashed", "update_date")[
        ["content_id_hashed", "original_price", "selling_price", "discounted_price",
         "content_review_count", "content_review_wth_media_count", "content_rate_count", "content_rate_avg"]
    ].copy()

    # Her ürün için en güncel arama log'larını al ve CTR hesapla
    content_search = latest_by_key_fast(c_search_log, "content_id_hashed", "date")[
        ["content_id_hashed", "total_search_impression", "total_search_click"]
    ].copy()
    content_search["content_search_ctr"] = safe_div(
        content_search["total_search_click"], content_search["total_search_impression"]
    )

    # Her ürün için en güncel site-geneli log'ları al ve dönüşüm oranlarını hesapla
    content_site = latest_by_key_fast(c_sitewide_log, "content_id_hashed", "date")[
        ["content_id_hashed", "total_click", "total_cart", "total_fav", "total_order"]
    ].copy()
    content_site["content_click_to_order"] = safe_div(content_site["total_order"], content_site["total_click"])
    content_site["content_cart_to_order"] = safe_div(content_site["total_order"], content_site["total_cart"])

    # Tüm işlenmiş content tablolarını birleştirerek nihai ürün tablosunu oluştur
    final_content_df = (meta.merge(content_price, on="content_id_hashed", how="left")
                            .merge(content_search, on="content_id_hashed", how="left")
                            .merge(content_site, on="content_id_hashed", how="left"))

    print("    [Content] Özellik üretimi tamamlandı.")
    return final_content_df