# features/advanced_features.py
import pandas as pd
import numpy as np
from pathlib import Path
from utils.helpers import latest_by_key_fast, safe_div

def generate_advanced_user_features(data_path: Path) -> pd.DataFrame:
    """
    Kullanılmayan kullanıcı loglarından (moda odaklı ve genel arama) ek özellikler üretir.
    - user/fashion_sitewide_log.parquet: Kullanıcının moda kategorisindeki genel davranışları.
    - user/search_log.parquet: Kullanıcının tüm arama geçmişi.
    """
    print("    [Advanced User] Moda ve arama logları okunuyor...")
    
    # 1. Kullanıcının moda kategorisindeki genel davranışları
    u_fashion_log = pd.read_parquet(
        data_path / "user" / "fashion_sitewide_log.parquet",
        columns=["user_id_hashed", "ts_hour", "total_click", "total_cart", "total_order"]
    )
    
    # En güncel logları al
    u_fashion_latest = latest_by_key_fast(u_fashion_log, "user_id_hashed", "ts_hour")
    u_fashion_latest = u_fashion_latest.rename(columns={
        "total_click": "user_fashion_total_click",
        "total_cart": "user_fashion_total_cart",
        "total_order": "user_fashion_total_order"
    })
    
    # Moda odaklı dönüşüm oranları
    u_fashion_latest["user_fashion_click_to_order"] = safe_div(
        u_fashion_latest["user_fashion_total_order"], u_fashion_latest["user_fashion_total_click"]
    )
    u_fashion_latest["user_fashion_cart_to_order"] = safe_div(
        u_fashion_latest["user_fashion_total_order"], u_fashion_latest["user_fashion_total_cart"]
    )
    
    # 2. Kullanıcının genel arama istatistikleri
    u_search_log = pd.read_parquet(
        data_path / "user" / "search_log.parquet",
        columns=["user_id_hashed", "ts_hour", "total_search_impression", "total_search_click"]
    )
    
    u_search_latest = latest_by_key_fast(u_search_log, "user_id_hashed", "ts_hour")
    u_search_latest = u_search_latest.rename(columns={
        "total_search_impression": "user_search_total_impr",
        "total_search_click": "user_search_total_click"
    })
    
    # Kullanıcının kişisel arama CTR'ı
    u_search_latest["user_search_ctr"] = safe_div(
        u_search_latest["user_search_total_click"], u_search_latest["user_search_total_impr"]
    )
    
    # İki tabloyu birleştir
    final_df = u_fashion_latest.merge(
        u_search_latest, on="user_id_hashed", how="outer",
        suffixes=('', '_drop')
    ).filter(regex='^(?!.*_drop)')
    
    print("    [Advanced User] Özellik üretimi tamamlandı.")
    return final_df.drop(columns=["ts_hour_x", "ts_hour_y"], errors="ignore")


def generate_term_features(data_path: Path) -> pd.DataFrame:
    """
    Arama terimlerinin genel popülerliğini ve performansını hesaplar.
    - term/search_log.parquet: Her terimin genel gösterim ve tıklanma sayıları.
    """
    print("    [Term] Arama terimi logları okunuyor...")
    term_log = pd.read_parquet(
        data_path / "term" / "search_log.parquet"
    )
    
    # Zaman içinde biriken istatistikler yerine genel bir toplama yapalım
    term_agg = term_log.groupby("search_term_normalized").agg(
        term_total_impr=("total_search_impression", "sum"),
        term_total_click=("total_search_click", "sum")
    ).reset_index()
    
    # Terim bazında genel CTR
    term_agg["term_global_ctr"] = safe_div(
        term_agg["term_total_click"], term_agg["term_total_impr"]
    )
    
    print("    [Term] Özellik üretimi tamamlandı.")
    return term_agg