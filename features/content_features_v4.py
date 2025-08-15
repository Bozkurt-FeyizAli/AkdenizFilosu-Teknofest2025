# features/content_features_v4.py

import pandas as pd
from pathlib import Path
from utils.helpers import latest_by_key_fast, safe_div

def generate_content_features(data_path: Path) -> pd.DataFrame:
    """
    Tüm content (ürün) verilerini okur, işler ve ürün bazında zenginleştirilmiş
    özet bir tablo döndürür. Bu tablo, ana pipeline tarafından kullanılacaktır.

    v4 Değişiklikleri:
    - Kritik İyileştirme: `search_log` ve `sitewide_log` için sadece en son kaydı
      almak yerine, `groupby` ile tüm geçmiş verinin toplamını, ortalamasını ve
      standart sapmasını alarak çok daha sağlam popülerlik ve davranış metrikleri
      üretir.
    - `content_age_days`: Ürünün ne kadar süredir platformda olduğunu gösteren
      bir özellik eklendi.
    - `num_cv_tags`: Ürün için üretilen yapay zeka etiketlerinin sayısı,
      ürünün ne kadar "anlaşılır" olduğuna dair bir proxy olabilir.
    - Fiyat ve yorum verileri için en güncel bilgiyi kullanmaya devam eder,
      çünkü bu değerler anlık durumu yansıtır.

    Args:
        data_path (Path): Ana 'data' klasörünün yolu.

    Returns:
        pd.DataFrame: Her bir 'content_id_hashed' için birleştirilmiş ve özetlenmiş
                      tüm ürün özelliklerini içeren DataFrame.
    """
    print("    [Content] Meta, fiyat ve log verileri okunuyor...")
    meta = pd.read_parquet(data_path / "content" / "metadata.parquet")
    price = pd.read_parquet(data_path / "content" / "price_rate_review_data.parquet")
    c_search_log = pd.read_parquet(data_path / "content" / "search_log.parquet")
    c_sitewide_log = pd.read_parquet(data_path / "content" / "sitewide_log.parquet")

    print("    [Content] Ürün tabloları özetleniyor...")

    # 1. Meta verilerinden yaş ve tag özellikleri
    meta['content_creation_date'] = pd.to_datetime(meta['content_creation_date'])
    meta['content_age_days'] = (pd.to_datetime('now', utc=True) - meta['content_creation_date']).dt.days
    meta['cv_tags'] = meta['cv_tags'].fillna('')
    meta['num_cv_tags'] = meta['cv_tags'].apply(lambda x: len(x.split(',')) if x else 0)

    # 2. Her ürün için en güncel fiyat, yorum ve puan bilgilerini al (Bu doğru bir yaklaşım)
    content_price = latest_by_key_fast(price, "content_id_hashed", "update_date")[
        ["content_id_hashed", "original_price", "selling_price", "discounted_price",
         "content_review_count", "content_review_wth_media_count", "content_rate_count", "content_rate_avg"]
    ].copy()
    content_price["discount_ratio"] = safe_div(
        content_price["original_price"] - content_price["selling_price"], content_price["original_price"]
    )

    # 3. (V4 İYİLEŞTİRMESİ) Arama log'larının TÜMÜNÜ özetle
    search_agg = c_search_log.groupby("content_id_hashed").agg(
        c_search_impr_sum=('total_search_impression', 'sum'),
        c_search_click_sum=('total_search_click', 'sum'),
        c_search_click_mean=('total_search_click', 'mean'),
        c_search_click_std=('total_search_click', 'std')
    )
    search_agg["c_global_search_ctr"] = safe_div(search_agg["c_search_click_sum"], search_agg["c_search_impr_sum"])

    # 4. (V4 İYİLEŞTİRMESİ) Site-geneli log'ların TÜMÜNÜ özetle
    sitewide_agg = c_sitewide_log.groupby("content_id_hashed").agg(
        c_total_click_sum=('total_click', 'sum'),
        c_total_cart_sum=('total_cart', 'sum'),
        c_total_fav_sum=('total_fav', 'sum'),
        c_total_order_sum=('total_order', 'sum'),
        c_total_order_mean=('total_order', 'mean'),
        c_total_order_std=('total_order', 'std')
    )
    sitewide_agg["c_click_to_order_rate"] = safe_div(sitewide_agg["c_total_order_sum"], sitewide_agg["c_total_click_sum"])
    sitewide_agg["c_cart_to_order_rate"] = safe_div(sitewide_agg["c_total_order_sum"], sitewide_agg["c_total_cart_sum"])

    # Tüm işlenmiş content tablolarını birleştirerek nihai ürün tablosunu oluştur
    final_content_df = (meta.merge(content_price, on="content_id_hashed", how="left")
                            .merge(search_agg, on="content_id_hashed", how="left")
                            .merge(sitewide_agg, on="content_id_hashed", how="left"))
    
    final_content_df = final_content_df.set_index("content_id_hashed")

    print(f"    [Content] Özellik üretimi tamamlandı. Shape: {final_content_df.shape}")
    return final_content_df