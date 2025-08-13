# build_features_polars_v2.py

import polars as pl
from pathlib import Path
import time

print(f"Polars versiyonu: {pl.__version__}")

def generate_features_v2(data_dir: Path, is_train: bool):
    """
    Polars kullanarak train veya test seti için TÜM özellikleri üretir ve tek bir parquet dosyası olarak kaydeder.
    Versiyon 2: Daha fazla veri kaynağı, oturum-içi özellikler ve gelişmiş etkileşimler içerir.
    """
    
    print(f"\n[{'TRAIN' if is_train else 'TEST'}] için Gelişmiş Özellik Üretimi (v2) başlıyor...")
    
    # ---------------------------------
    # 1. Gerekli Tüm Verileri Lazy Oku
    # ---------------------------------
    if is_train:
        sessions_lazy = pl.scan_parquet(data_dir / "train_sessions.parquet")
    else:
        sessions_lazy = pl.scan_parquet(data_dir / "test_sessions.parquet")

    # Content Data
    content_meta_lazy = pl.scan_parquet(data_dir / "content" / "metadata.parquet")
    content_price_lazy = pl.scan_parquet(data_dir / "content" / "price_rate_review_data.parquet")
    content_search_log_lazy = pl.scan_parquet(data_dir / "content" / "search_log.parquet")
    content_sitewide_log_lazy = pl.scan_parquet(data_dir / "content" / "sitewide_log.parquet")
    content_top_terms_lazy = pl.scan_parquet(data_dir / "content" / "top_terms_log.parquet")

    # User Data
    user_meta_lazy = pl.scan_parquet(data_dir / "user" / "metadata.parquet")
    user_sitewide_log_lazy = pl.scan_parquet(data_dir / "user" / "sitewide_log.parquet")
    user_fashion_sitewide_lazy = pl.scan_parquet(data_dir / "user" / "fashion_sitewide_log.parquet")
    user_fashion_search_lazy = pl.scan_parquet(data_dir / "user" / "fashion_search_log.parquet")
    # --- YENİ EKLENEN KULLANICI VERİLERİ ---
    user_search_log_lazy = pl.scan_parquet(data_dir / "user" / "search_log.parquet")
    user_top_terms_lazy = pl.scan_parquet(data_dir / "user" / "top_terms_log.parquet")
    
    # Term Data
    term_search_log_lazy = pl.scan_parquet(data_dir / "term" / "search_log.parquet")

    # ---------------------------------
    # 2. Yardımcı Fonksiyon ve Özellik Tabloları
    # ---------------------------------
    print("   - Özellik tabloları hazırlanıyor...")
    def latest_by(df_lazy, key, timecol):
        return df_lazy.sort(timecol, descending=True).group_by(key, maintain_order=True).first()

    # --- Content Features ---
    content_price_feats = latest_by(content_price_lazy, "content_id_hashed", "update_date").select(
        "content_id_hashed", "original_price", "selling_price", "discounted_price",
        "content_review_count", "content_rate_count", "content_rate_avg"
    )
    content_search_feats = latest_by(content_search_log_lazy, "content_id_hashed", "date").select(
        "content_id_hashed", "total_search_impression", "total_search_click"
    )
    content_sitewide_feats = latest_by(content_sitewide_log_lazy, "content_id_hashed", "date").select(
        "content_id_hashed", "total_click", "total_cart", "total_order"
    )
    content_base_feats = content_meta_lazy.join(content_price_feats, on="content_id_hashed", how="left") \
                                          .join(content_search_feats, on="content_id_hashed", how="left") \
                                          .join(content_sitewide_feats, on="content_id_hashed", how="left")

    # --- User Features ---
    user_sitewide_feats = latest_by(user_sitewide_log_lazy, "user_id_hashed", "ts_hour").select(
        "user_id_hashed", "total_click", "total_cart", "total_order"
    ).rename({"total_click": "user_total_click", "total_cart": "user_total_cart", "total_order": "user_total_order"})
    
    user_fashion_feats = latest_by(user_fashion_sitewide_lazy, "user_id_hashed", "ts_hour").select(
        "user_id_hashed", "total_click"
    ).rename({"total_click": "user_fashion_total_click"})
    
    # --- YENİ: Genel Kullanıcı Arama İstatistikleri ---
    user_search_feats = latest_by(user_search_log_lazy, "user_id_hashed", "ts_hour").select(
        "user_id_hashed", "total_search_impression", "total_search_click"
    ).rename({
        "total_search_impression": "user_gen_search_impr",
        "total_search_click": "user_gen_search_click"
    })

    user_base_feats = user_meta_lazy.join(user_sitewide_feats, on="user_id_hashed", how="left") \
                                    .join(user_fashion_feats, on="user_id_hashed", how="left") \
                                    .join(user_search_feats, on="user_id_hashed", how="left")

    # --- Interaction & Term Features ---
    term_feats = term_search_log_lazy.group_by("search_term_normalized").agg(
        term_total_impr = pl.col("total_search_impression").sum(),
        term_total_click = pl.col("total_search_click").sum()
    )

    # ---------------------------------
    # 3. Tüm Verileri Birleştir (Join)
    # ---------------------------------
    print("   - Ana tablo birleştiriliyor...")
    main_df_lazy = sessions_lazy.join(content_base_feats, on="content_id_hashed", how="left") \
                                .join(user_base_feats, on="user_id_hashed", how="left") \
                                .join(term_feats, on="search_term_normalized", how="left")
    
    # ---------------------------------
    # 4. Final Özellikleri Oluştur (Feature Engineering)
    # ---------------------------------
    print("   - Son özellik mühendisliği adımları yapılıyor...")
    
    final_lazy = main_df_lazy.with_columns(
        # Zaman Özellikleri
        hour = pl.col("ts_hour").dt.hour(),
        dow = pl.col("ts_hour").dt.weekday(),
        month = pl.col("ts_hour").dt.month(),
        
        # Etkileşim Oranları (Daha güvenli bölme için)
        user_click_to_order = pl.col("user_total_order") / (pl.col("user_total_click") + 1e-6),
        user_gen_search_ctr = pl.col("user_gen_search_click") / (pl.col("user_gen_search_impr") + 1e-6),
        content_click_to_order = pl.col("total_order") / (pl.col("total_click") + 1e-6),
        content_search_ctr = pl.col("total_search_click") / (pl.col("total_search_impression") + 1e-6),
        term_global_ctr = pl.col("term_total_click") / (pl.col("term_total_impr") + 1e-6),
        
        # Diğer Temel Özellikler
        content_age_days = (pl.col("ts_hour") - pl.col("content_creation_date")).dt.total_days(),
        user_age = pl.col("ts_hour").dt.year() - pl.col("user_birth_year"),
        discount_rate = (1 - pl.col("discounted_price") / pl.col("original_price")).fill_nan(0),
        search_term_in_cv_tags = pl.col("cv_tags").str.contains(pl.col("search_term_normalized"), literal=True).cast(pl.Int8).fill_null(0)
    
    ).with_columns(
        # --- YENİ: Oturum İçi (In-Session) Özellikler ---
        # Polars'ın window fonksiyonu `over()` bu işlem için mükemmeldir.
        # Her bir ürünün fiyatını, o oturumdaki ortalama fiyata bölerek normalize et
        price_vs_session_mean = pl.col("discounted_price") / pl.col("discounted_price").mean().over("session_id"),
        # Oturum içindeki fiyat sıralaması (ucuzdan pahalıya)
        price_rank_in_session = pl.col("discounted_price").rank(method='ordinal').over("session_id"),
        # Oturum içindeki popülerlik (tıklanma) sıralaması
        popularity_rank_in_session = pl.col("total_click").rank(method='ordinal', descending=True).over("session_id"),
        # Oturum içindeki puan sıralaması
        rating_rank_in_session = pl.col("content_rate_avg").rank(method='ordinal', descending=True).over("session_id"),
    )

    # ---------------------------------
    # 5. Planı Çalıştır ve Kaydet
    # ---------------------------------
    print("   - Sorgu planı çalıştırılıyor ve sonuçlar diske yazılıyor...")
    start_time = time.time()
    final_df = final_lazy.collect(streaming=True) 
    
    output_path = data_dir / f"{'train' if is_train else 'test'}_features_v2.parquet"
    final_df.write_parquet(output_path)
    
    end_time = time.time()
    print(f"   - Bitti! {len(final_df):,} satır işlendi ve '{output_path.name}' olarak kaydedildi. Süre: {end_time - start_time:.2f} saniye.")