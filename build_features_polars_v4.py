# build_features_polars_v4.py

import polars as pl
from pathlib import Path
import time

print(f"Polars versiyonu: {pl.__version__}")

def generate_features(data_dir: Path, is_train: bool):
    """
    Polars kullanarak train veya test seti için v4 seviyesinde tüm özellikleri üretir
    ve tek bir parquet dosyası olarak kaydeder.

    v4 Değişiklikleri:
    - Sadece son log kaydı ('latest') yerine, tüm geçmiş veriyi (.group_by().agg())
      kullanarak çok daha sağlam kullanıcı ve ürün profilleri oluşturulmuştur.
    - Polars'ın window fonksiyonları (.over()) kullanılarak kritik oturum-içi
      bağlamsal özellikler (örn. fiyata göre rank, popülerliğe göre z-score) eklenmiştir.
    - Sızıntısız kullanıcı-içerik ve kullanıcı-kategori etkileşim geçmişi özellikleri
      (.cumulative_sum().over()) eklenmiştir. Bu, kişiselleştirme için en önemli adımlardan biridir.
    """
    
    # ---------------------------------
    # 1. Gerekli Tüm Verileri Lazy Oku
    # ---------------------------------
    print(f"\n[{'TRAIN' if is_train else 'TEST'}] için v4 özellik üretimi başlıyor...")
    
    # Ana Oturum Dosyası
    sessions_lazy = pl.scan_parquet(data_dir / f"{'train' if is_train else 'test'}_sessions.parquet")

    # Diğer tüm veri setlerini lazy olarak yükle
    content_meta_lazy = pl.scan_parquet(data_dir / "content" / "metadata.parquet")
    content_price_lazy = pl.scan_parquet(data_dir / "content" / "price_rate_review_data.parquet")
    content_search_log_lazy = pl.scan_parquet(data_dir / "content" / "search_log.parquet")
    content_sitewide_log_lazy = pl.scan_parquet(data_dir / "content" / "sitewide_log.parquet")
    content_top_terms_lazy = pl.scan_parquet(data_dir / "content" / "top_terms_log.parquet")
    user_meta_lazy = pl.scan_parquet(data_dir / "user" / "metadata.parquet")
    user_sitewide_log_lazy = pl.scan_parquet(data_dir / "user" / "sitewide_log.parquet")
    user_fashion_sitewide_lazy = pl.scan_parquet(data_dir / "user" / "fashion_sitewide_log.parquet")
    term_search_log_lazy = pl.scan_parquet(data_dir / "term" / "search_log.parquet")
    
    # ---------------------------------
    # 2. Ana Özellik Tablolarını Üret
    # ---------------------------------
    print("   - (v4) Ana özellik tabloları (tüm geçmiş kullanılarak) hazırlanıyor...")

    # --- Content Features (v4) ---
    # Fiyat/puan için son bilgi en mantıklısı
    content_price_feats = content_price_lazy.sort("update_date", descending=True).group_by("content_id_hashed").first().with_columns(
        discount_rate = (1 - pl.col("discounted_price") / pl.col("original_price")).fill_nan(0),
    )
    # Loglar için tüm geçmişi özetle
    content_search_agg = content_search_log_lazy.group_by("content_id_hashed").agg(
        c_search_impr_sum = pl.col("total_search_impression").sum(),
        c_search_click_sum = pl.col("total_search_click").sum(),
        c_search_click_mean = pl.col("total_search_click").mean()
    ).with_columns(c_global_search_ctr = pl.col("c_search_click_sum") / pl.col("c_search_impr_sum"))

    content_sitewide_agg = content_sitewide_log_lazy.group_by("content_id_hashed").agg(
        c_click_sum = pl.col("total_click").sum(), c_cart_sum = pl.col("total_cart").sum(),
        c_fav_sum = pl.col("total_fav").sum(), c_order_sum = pl.col("total_order").sum(),
        c_order_mean = pl.col("total_order").mean(), c_order_std = pl.col("total_order").std()
    ).with_columns(c_click_to_order_rate = pl.col("c_order_sum") / pl.col("c_click_sum"))

    content_base_feats = content_meta_lazy.join(content_price_feats, on="content_id_hashed", how="left") \
                                          .join(content_search_agg, on="content_id_hashed", how="left") \
                                          .join(content_sitewide_agg, on="content_id_hashed", how="left") \
                                          .with_columns(num_cv_tags = pl.col("cv_tags").str.count_matches(",") + 1)
    
    # --- User Features (v4) ---
    user_sitewide_agg = user_sitewide_log_lazy.group_by("user_id_hashed").agg(
        u_click_sum = pl.col("total_click").sum(), u_order_sum = pl.col("total_order").sum(),
        u_cart_sum = pl.col("total_cart").sum(), u_order_mean = pl.col("total_order").mean()
    ).with_columns(u_click_to_order_rate = pl.col("u_order_sum") / pl.col("u_click_sum"))

    user_fashion_agg = user_fashion_sitewide_lazy.group_by("user_id_hashed").agg(
        u_fashion_click_sum = pl.col("total_click").sum(),
        u_fashion_order_sum = pl.col("total_order").sum()
    ).with_columns(u_fashion_ctr = pl.col("u_fashion_order_sum") / pl.col("u_fashion_click_sum"))
    
    user_base_feats = user_meta_lazy.join(user_sitewide_agg, on="user_id_hashed", how="left") \
                                    .join(user_fashion_agg, on="user_id_hashed", how="left")
    
    # --- Term & Interaction Features (v4) ---
    top_terms_feats = content_top_terms_lazy.group_by("content_id_hashed", "search_term_normalized").agg(
        term_content_impr_sum = pl.col("total_search_impression").sum(),
        term_content_click_sum = pl.col("total_search_click").sum()
    ).with_columns(term_content_ctr = pl.col("term_content_click_sum") / pl.col("term_content_impr_sum"))

    term_feats = term_search_log_lazy.group_by("search_term_normalized").agg(
        term_global_impr = pl.col("total_search_impression").sum(),
        term_global_click = pl.col("total_search_click").sum()
    ).with_columns(term_global_ctr = pl.col("term_global_click") / pl.col("term_global_impr"))

    # ---------------------------------
    # 3. Tüm Verileri Birleştir ve Temel Özellikleri Üret
    # ---------------------------------
    print("   - Ana tablo birleştiriliyor ve temel özellikler üretiliyor...")
    
    main_df_lazy = sessions_lazy.join(content_base_feats, on="content_id_hashed", how="left") \
                                .join(user_base_feats, on="user_id_hashed", how="left") \
                                .join(term_feats, on="search_term_normalized", how="left") \
                                .join(top_terms_feats, on=["content_id_hashed", "search_term_normalized"], how="left") \
                                .with_columns(
                                    # Zaman Özellikleri
                                    hour = pl.col("ts_hour").dt.hour(), dow = pl.col("ts_hour").dt.weekday(),
                                    is_weekend = pl.col("ts_hour").dt.weekday().is_in([6, 7]).cast(pl.Int8),
                                    # Diğer temel özellikler
                                    content_age_days = (pl.col("ts_hour") - pl.col("content_creation_date")).dt.total_days(),
                                    age = pl.col("ts_hour").dt.year() - pl.col("user_birth_year"),
                                    search_term_in_cv_tags = pl.col("cv_tags").str.contains(pl.col("search_term_normalized")).cast(pl.Int8).fill_null(0)
                                )
    
    # ----------------------------------------------------
    # 4. (YENİ - v4) Oturum İçi & Geçmiş Özellikleri
    # ----------------------------------------------------
    print("   - (v4) Oturum-içi ve kullanıcı geçmişi özellikleri ekleniyor...")
    
    if is_train:
        # Sadece eğitim setinde target değişkenleri vardır. Bunları sızıntısız geçmiş oluşturmak için kullanacağız.
        history_cols = ["clicked", "ordered"]
        main_df_lazy = main_df_lazy.with_columns(
            (pl.col(c).cum_sum().over(["user_id_hashed", "content_id_hashed"]) - pl.col(c)).alias(f"hist_user_content_{c}") for c in history_cols
        ).with_columns(
            (pl.col(c).cum_sum().over(["user_id_hashed", "leaf_category_name"]) - pl.col(c)).alias(f"hist_user_cat_{c}") for c in history_cols
        )
    final_lazy = main_df_lazy.with_columns(
    # Oturum içi (session-context) sıralama ve z-skor özellikleri
    session_price_rank = pl.col("discounted_price").rank().over("session_id"),
    session_pop_rank = pl.col("c_click_sum").rank().over("session_id"),
    session_price_z = (pl.col("discounted_price") - pl.col("discounted_price").mean().over("session_id")) / (pl.col("discounted_price").std().over("session_id") + 1e-6),
    session_pop_z = (pl.col("c_click_sum") - pl.col("c_click_sum").mean().over("session_id")) / (pl.col("c_click_sum").std().over("session_id") + 1e-6)
    )
    # ---------------------------------
    # 5. Planı Çalıştır ve Kaydet
    # ---------------------------------
    print("   - Sorgu planı çalıştırılıyor ve sonuçlar diske yazılıyor...")
    start_time = time.time()
    
    # streaming=True büyük veri setlerinde belleği daha verimli kullanır
    final_df = final_lazy.collect(streaming=True) 
    
    output_path = data_dir / f"{'train' if is_train else 'test'}_features_v4.parquet"
    final_df.write_parquet(output_path)
    
    end_time = time.time()
    print(f"   - Bitti! {len(final_df):,} satır işlendi ve '{output_path.name}' olarak kaydedildi. Süre: {end_time - start_time:.2f} saniye.")


if __name__ == "__main__":
    # Bu scripti doğrudan çalıştırarak özellik dosyalarını oluşturabilirsiniz.
    # Örnek kullanım: python build_features_polars_v4.py
    DATA_DIR = Path("./") # Gerçek veri klasörünüzün yoluyla değiştirin
    
    generate_features(DATA_DIR, is_train=True)
    generate_features(DATA_DIR, is_train=False)
    
    print("\nTüm v4 özellik dosyaları başarıyla oluşturuldu!")