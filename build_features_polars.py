import polars as pl
from pathlib import Path
import time

print(f"Polars versiyonu: {pl.__version__}")

def generate_features(data_dir: Path, is_train: bool):
    """
    Polars kullanarak train veya test seti için tüm özellikleri üretir ve tek bir parquet dosyası olarak kaydeder.
    """
    
    # ---------------------------------
    # 1. Gerekli Tüm Verileri Lazy Oku
    # ---------------------------------
    # .scan_parquet() veriyi hemen belleğe yüklemez, sadece bir "plan" oluşturur.
    # Bu, bellek kullanımını minimumda tutar ve Polars'ın optimizasyon yapmasını sağlar.
    
    print(f"\n[{'TRAIN' if is_train else 'TEST'}] için özellik üretimi başlıyor...")
    
    # Ana Oturum Dosyası
    if is_train:
        sessions_lazy = pl.scan_parquet(data_dir / "train_sessions.parquet")
    else:
        sessions_lazy = pl.scan_parquet(data_dir / "test_sessions.parquet")

    # Ürün (Content) Verileri
    content_meta_lazy = pl.scan_parquet(data_dir / "content" / "metadata.parquet").select(
        "content_id_hashed", "level1_category_name", "level2_category_name", 
        "leaf_category_name", "attribute_type_count", "total_attribute_option_count", 
        "merchant_count", "filterable_label_count", "content_creation_date", "cv_tags"
    )
    content_price_lazy = pl.scan_parquet(data_dir / "content" / "price_rate_review_data.parquet")
    content_search_log_lazy = pl.scan_parquet(data_dir / "content" / "search_log.parquet")
    content_sitewide_log_lazy = pl.scan_parquet(data_dir / "content" / "sitewide_log.parquet")
    content_top_terms_lazy = pl.scan_parquet(data_dir / "content" / "top_terms_log.parquet")

    # Kullanıcı (User) Verileri
    user_meta_lazy = pl.scan_parquet(data_dir / "user" / "metadata.parquet")
    user_sitewide_log_lazy = pl.scan_parquet(data_dir / "user" / "sitewide_log.parquet")
    # YENİ EKLENEN KULLANICI VERİLERİ
    user_fashion_sitewide_lazy = pl.scan_parquet(data_dir / "user" / "fashion_sitewide_log.parquet")
    user_fashion_search_lazy = pl.scan_parquet(data_dir / "user" / "fashion_search_log.parquet")
    
    # Terim (Term) Verileri
    term_search_log_lazy = pl.scan_parquet(data_dir / "term" / "search_log.parquet")

    # ---------------------------------
    # 2. Özellikleri Paralel Olarak Üret
    # ---------------------------------
    print("   - Özellik tabloları hazırlanıyor...")

    # latest_by_key_fast fonksiyonunun Polars versiyonu
    def latest_by(df_lazy, key, timecol):
        return df_lazy.sort(timecol, descending=True).group_by(key, maintain_order=True).first()

    # --- Content Features ---
    content_price_feats = latest_by(content_price_lazy, "content_id_hashed", "update_date").select(
        "content_id_hashed", "original_price", "selling_price", "discounted_price",
        "content_review_count", "content_review_wth_media_count", "content_rate_count", "content_rate_avg"
    )
    content_search_feats = latest_by(content_search_log_lazy, "content_id_hashed", "date").with_columns(
        content_search_ctr = pl.col("total_search_click") / pl.col("total_search_impression")
    ).select("content_id_hashed", "total_search_impression", "total_search_click", "content_search_ctr")
    
    content_sitewide_feats = latest_by(content_sitewide_log_lazy, "content_id_hashed", "date").with_columns(
        content_click_to_order = pl.col("total_order") / pl.col("total_click"),
        content_cart_to_order = pl.col("total_order") / pl.col("total_cart")
    ).select("content_id_hashed", "total_click", "total_cart", "total_fav", "total_order", "content_click_to_order", "content_cart_to_order")

    content_base_feats = content_meta_lazy.join(content_price_feats, on="content_id_hashed", how="left") \
                                          .join(content_search_feats, on="content_id_hashed", how="left") \
                                          .join(content_sitewide_feats, on="content_id_hashed", how="left")

    # --- User Features ---
    user_sitewide_feats = latest_by(user_sitewide_log_lazy, "user_id_hashed", "ts_hour").with_columns(
        user_click_to_order = pl.col("total_order") / pl.col("total_click"),
        user_cart_to_order = pl.col("total_order") / pl.col("total_cart")
    ).select("user_id_hashed", "total_click", "total_cart", "total_fav", "total_order", "user_click_to_order", "user_cart_to_order") \
     .rename({"total_click": "user_total_click", "total_cart": "user_total_cart", "total_fav": "user_total_fav", "total_order": "user_total_order"})

    user_fashion_feats = latest_by(user_fashion_sitewide_lazy, "user_id_hashed", "ts_hour").with_columns(
        user_fashion_click_to_order = pl.col("total_order") / pl.col("total_click")
    ).select("user_id_hashed", "total_click", "user_fashion_click_to_order").rename({"total_click": "user_fashion_total_click"})
    
    user_fashion_search_feats = latest_by(user_fashion_search_lazy, "user_id_hashed", "ts_hour").with_columns(
        user_fashion_search_ctr = pl.col("total_search_click") / pl.col("total_search_impression")
    ).select("user_id_hashed", "user_fashion_search_ctr")

    user_base_feats = user_meta_lazy.join(user_sitewide_feats, on="user_id_hashed", how="left") \
                                    .join(user_fashion_feats, on="user_id_hashed", how="left") \
                                    .join(user_fashion_search_feats, on="user_id_hashed", how="left")

    # --- Interaction & Term Features ---
    top_terms_feats = content_top_terms_lazy.group_by("content_id_hashed", "search_term_normalized").agg(
        tt_search_impr = pl.col("total_search_impression").sum(),
        tt_search_click = pl.col("total_search_click").sum()
    ).with_columns(term_ctr = pl.col("tt_search_click") / pl.col("tt_search_impr"))

    term_feats = term_search_log_lazy.group_by("search_term_normalized").agg(
        term_total_impr = pl.col("total_search_impression").sum(),
        term_total_click = pl.col("total_search_click").sum()
    ).with_columns(term_global_ctr = pl.col("term_total_click") / pl.col("term_total_impr"))

    # ---------------------------------
    # 3. Tüm Verileri Birleştir (Join)
    # ---------------------------------
    print("   - Ana tablo birleştiriliyor...")
    
    main_df_lazy = sessions_lazy.join(content_base_feats, on="content_id_hashed", how="left") \
                                .join(user_base_feats, on="user_id_hashed", how="left") \
                                .join(term_feats, on="search_term_normalized", how="left") \
                                .join(top_terms_feats, on=["content_id_hashed", "search_term_normalized"], how="left")
    
    # ---------------------------------
    # 4. Final Özellikleri Oluştur (Feature Engineering)
    # ---------------------------------
    # .with_columns() ile zincirleme ve paralel özellik üretimi
    print("   - Son özellik mühendisliği adımları yapılıyor...")
    
    final_lazy = main_df_lazy.with_columns(
        # Zaman Özellikleri
        ts_hour_dt = pl.col("ts_hour"),
        hour = pl.col("ts_hour").dt.hour(),
        dow = pl.col("ts_hour").dt.weekday(),
        is_weekend = pl.when(pl.col("ts_hour").dt.weekday() > 5).then(1).otherwise(0),
        month = pl.col("ts_hour").dt.month(),
    ).with_columns(
        season = pl.when(pl.col("month").is_in([12, 1, 2])).then(0)
                 .when(pl.col("month").is_in([3, 4, 5])).then(1)
                 .when(pl.col("month").is_in([6, 7, 8])).then(2)
                 .otherwise(3)
    ).with_columns(
        # Yaş ve İndirim Oranı
        content_age_days = (pl.col("ts_hour_dt") - pl.col("content_creation_date")).dt.total_days(),
        age = pl.col("ts_hour_dt").dt.year() - pl.col("user_birth_year"),
        discount_rate = (1 - pl.col("discounted_price") / pl.col("original_price")).fill_nan(0),
        
        # !! EN ÖNEMLİ PERFORMANS İYİLEŞTİRMESİ !!
        # .str.contains() ile yavaş .apply() yerine hızlı bir operasyon
        search_term_in_cv_tags = pl.col("cv_tags").str.contains(pl.col("search_term_normalized")).cast(pl.Int8).fill_null(0)
    ).drop("ts_hour_dt")

    # ---------------------------------
    # 5. Planı Çalıştır ve Kaydet
    # ---------------------------------
    print("   - Sorgu planı çalıştırılıyor ve sonuçlar diske yazılıyor...")
    
    start_time = time.time()
    # .collect() komutu ile tüm tembel işlemleri çalıştır
    final_df = final_lazy.collect(streaming=True) 
    
    output_path = data_dir / f"{'train' if is_train else 'test'}_features.parquet"
    final_df.write_parquet(output_path)
    
    end_time = time.time()
    print(f"   - Bitti! {len(final_df):,} satır işlendi ve '{output_path.name}' olarak kaydedildi. Süre: {end_time - start_time:.2f} saniye.")

def add_sequential_features(lazy_df: pl.LazyFrame) -> pl.LazyFrame:
    """
    Oturum içi sıralı özellikleri ekler.
    Her bir eylemin, aynı oturumdaki bir önceki eyleme göre değişimini hesaplar.
    """
    print("   - Sıralı (sequential) özellikler ekleniyor...")
    
    # Oturumları zamana göre sırala
    df_sorted = lazy_df.sort("ts_hour")
    
    # shift() operasyonu ile bir önceki satırdaki değeri al
    sequential_features = df_sorted.with_columns(
        # Bir önceki ürünün fiyatı
        prev_price = pl.col("discounted_price").shift(1).over("session_id"),
        
        # Bir önceki ürünün puanı
        prev_rate_avg = pl.col("content_rate_avg").shift(1).over("session_id"),
        
        # Bir önceki ürünün tıklanma oranı (CTR)
        prev_content_search_ctr = pl.col("content_search_ctr").shift(1).over("session_id"),
        
        # Oturum başlangıcından bu yana geçen süre
        time_since_session_start = (pl.col("ts_hour") - pl.col("ts_hour").first().over("session_id")).dt.total_seconds(),
        
        # Oturumdaki kaçıncı ürün olduğu
        item_rank_in_session = pl.col("ts_hour").rank("ordinal").over("session_id")
    )
    
    # Fark ve oran özellikleri üret
    final_with_seq = sequential_features.with_columns(
        price_diff_from_prev = pl.col("discounted_price") - pl.col("prev_price"),
        rate_avg_diff_from_prev = pl.col("content_rate_avg") - pl.col("prev_rate_avg"),
        ctr_ratio_from_prev = pl.col("content_search_ctr") / pl.col("prev_content_search_ctr")
    )
    
    return final_with_seq

if __name__ == "__main__":
    # Bu scripti doğrudan çalıştırarak özellik dosyalarını oluşturabilirsiniz.
    DATA_DIR = Path("data") # Veri klasörünüzün yolunu buraya yazın
    
    # Train ve Test için özellikleri üret
    generate_features(DATA_DIR, is_train=True)
    generate_features(DATA_DIR, is_train=False)
    
    print("\nTüm özellik dosyaları başarıyla oluşturuldu!")