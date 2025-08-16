# features/advanced_features.py
import pandas as pd
import numpy as np
from pathlib import Path
from utils.helpers import latest_by_key_fast, safe_div
from sklearn.feature_extraction.text import HashingVectorizer  # yeni

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


# === YENİ: İçerik x Terim ilişki özellikleri (Bayes yumuşatmalı) ===

def generate_content_term_features(data_path: Path, prior: float = 50.0, min_impr: float = 0.0) -> pd.DataFrame:
    """
    content/top_terms_log.parquet ve term/search_log.parquet dosyalarından
    (content_id_hashed, search_term_normalized) ikilisi için CTR ve Bayes
    yumuşatmalı CTR üretir. Bu, aranan terim ile ürün ilişkisini güçlendirir
    ve sıralama metriklerine doğrudan etki eder.

    Args:
        data_path: Ana 'data' klasörünün yolu.
        prior: Bayes karışımı için kullanılan öncelik gücü (alpha).
        min_impr: Çok düşük gösterimleri kırpmak için alt sınır.

    Returns:
        pd.DataFrame: Kolonlar: [content_id_hashed, search_term_normalized,
            tt_search_impr, tt_search_click, term_ctr, smoothed_term_ctr, tt_log_impr]
    """
    print("    [Interaction] Content x Term özellikleri (Bayes) hesaplanıyor...")

    # Top terms: içerik x terim bazlı gösterim/tıklama
    top_terms = pd.read_parquet(
        data_path / "content" / "top_terms_log.parquet",
        columns=[
            "content_id_hashed", "search_term_normalized",
            "total_search_impression", "total_search_click"
        ]
    )
    # Kategorik sütunları string'e çevirerek join stabilitesi sağlayalım
    top_terms["content_id_hashed"] = top_terms["content_id_hashed"].astype(str)
    top_terms["search_term_normalized"] = top_terms["search_term_normalized"].astype(str)

    ct = (
        top_terms
        .groupby(["content_id_hashed", "search_term_normalized"], as_index=False, observed=True)
        .agg(
            tt_search_impr=("total_search_impression", "sum"),
            tt_search_click=("total_search_click", "sum")
        )
    )
    if min_impr > 0:
        ct = ct[ct["tt_search_impr"] >= float(min_impr)]

    # Terimlerin global CTR'ı (ön bilgi)
    term_agg = generate_term_features(data_path)[[
        "search_term_normalized", "term_total_impr", "term_total_click", "term_global_ctr"
    ]]

    # Birleştir ve Bayes yumuşatmalı CTR hesapla
    ct = ct.merge(term_agg, on="search_term_normalized", how="left")
    ct["term_ctr"] = safe_div(ct["tt_search_click"].astype("float32"), ct["tt_search_impr"].astype("float32"))

    # smoothed = (click + prior * prior_mean) / (impr + prior)
    prior_mean = ct["term_global_ctr"].fillna(ct["term_global_ctr"].median() if not np.isnan(ct["term_global_ctr"].median()) else 0.05)
    ct["smoothed_term_ctr"] = (ct["tt_search_click"].astype("float32") + prior * prior_mean.astype("float32")) / (
        ct["tt_search_impr"].astype("float32") + prior
    )

    ct["tt_log_impr"] = np.log1p(ct["tt_search_impr"].astype("float32"))

    # Tipleri sıkıştır
    for c in ["tt_search_impr", "tt_search_click", "term_total_impr", "term_total_click", "term_ctr", "smoothed_term_ctr", "tt_log_impr", "term_global_ctr"]:
        if c in ct.columns:
            ct[c] = ct[c].astype("float32")

    return ct[[
        "content_id_hashed", "search_term_normalized",
        "tt_search_impr", "tt_search_click", "term_ctr", "smoothed_term_ctr", "tt_log_impr"
    ]]


def generate_user_term_features(data_path: Path, prior: float = 50.0) -> pd.DataFrame:
    """
    user/top_terms_log.parquet'ten (user_id_hashed, search_term_normalized)
    düzeyinde gösterim, tıklama ve Bayes yumuşatmalı CTR üretir.
    """
    print("    [User-Term] Kullanıcı x Terim özellikleri hesaplanıyor...")
    df = pd.read_parquet(
        data_path / "user" / "top_terms_log.parquet",
        columns=["user_id_hashed", "search_term_normalized", "total_search_impression", "total_search_click", "ts_hour"],
    )
    df["user_id_hashed"] = df["user_id_hashed"].astype(str)
    df["search_term_normalized"] = df["search_term_normalized"].astype(str)

    ut = (
        df.groupby(["user_id_hashed", "search_term_normalized"], as_index=False, observed=True)
        .agg(
            ut_impr=("total_search_impression", "sum"),
            ut_click=("total_search_click", "sum"),
        )
    )

    # Ön bilgi: terimlerin global CTR'ı
    term_agg = generate_term_features(data_path)[["search_term_normalized", "term_global_ctr"]]
    ut = ut.merge(term_agg, on="search_term_normalized", how="left")

    ut["ut_term_ctr"] = safe_div(ut["ut_click"].astype("float32"), ut["ut_impr"].astype("float32"))
    prior_mean = ut["term_global_ctr"].fillna(ut["term_global_ctr"].median() if not np.isnan(ut["term_global_ctr"].median()) else 0.05)
    ut["ut_smoothed_ctr"] = (ut["ut_click"].astype("float32") + prior * prior_mean.astype("float32")) / (
        ut["ut_impr"].astype("float32") + prior
    )
    ut["ut_log_impr"] = np.log1p(ut["ut_impr"].astype("float32"))

    for c in ["ut_impr", "ut_click", "ut_term_ctr", "ut_smoothed_ctr", "ut_log_impr", "term_global_ctr"]:
        if c in ut.columns:
            ut[c] = ut[c].astype("float32")

    return ut[["user_id_hashed", "search_term_normalized", "ut_impr", "ut_click", "ut_term_ctr", "ut_smoothed_ctr", "ut_log_impr"]]


def generate_content_recency_features(data_path: Path, half_life_days: float = 30.0) -> pd.DataFrame:
    """
    İçerik arama ve sitewide log'larından zaman bozunmalı (exponential decay)
    popülerlik ve dönüşüm oranları üretir.
    """
    print("    [Content-Recency] Zaman bozunmalı içerik özellikleri hesaplanıyor...")

    # Search log (content-level)
    s = pd.read_parquet(
        data_path / "content" / "search_log.parquet",
        columns=["content_id_hashed", "date", "total_search_impression", "total_search_click"],
    )
    s["date"] = pd.to_datetime(s["date"])  # güvenli
    maxd = s["date"].max()
    age = (maxd - s["date"]).dt.days.astype("float32")
    w = np.power(0.5, age / float(half_life_days))
    s["w"] = w
    s["w_impr"] = s["total_search_impression"].astype("float32") * s["w"]
    s["w_click"] = s["total_search_click"].astype("float32") * s["w"]
    s_agg = s.groupby("content_id_hashed", as_index=False, observed=True).agg(
        decayed_search_impr=("w_impr", "sum"),
        decayed_search_click=("w_click", "sum"),
    )
    s_agg["decayed_search_ctr"] = safe_div(s_agg["decayed_search_click"], s_agg["decayed_search_impr"])

    # Sitewide log (content-level)
    sw = pd.read_parquet(
        data_path / "content" / "sitewide_log.parquet",
        columns=["content_id_hashed", "date", "total_click", "total_cart", "total_fav", "total_order"],
    )
    sw["date"] = pd.to_datetime(sw["date"])
    maxd_sw = sw["date"].max()
    age_sw = (maxd_sw - sw["date"]).dt.days.astype("float32")
    w2 = np.power(0.5, age_sw / float(half_life_days))
    sw["w"] = w2
    for col in ["total_click", "total_cart", "total_fav", "total_order"]:
        sw[f"w_{col}"] = sw[col].astype("float32") * sw["w"]
    sw_agg = sw.groupby("content_id_hashed", as_index=False, observed=True).agg(
        decayed_click=("w_total_click", "sum"),
        decayed_cart=("w_total_cart", "sum"),
        decayed_fav=("w_total_fav", "sum"),
        decayed_order=("w_total_order", "sum"),
    )
    sw_agg["decayed_click_to_order"] = safe_div(sw_agg["decayed_order"], sw_agg["decayed_click"])
    sw_agg["decayed_cart_to_order"] = safe_div(sw_agg["decayed_order"], sw_agg["decayed_cart"])

    # Birleştir
    out = s_agg.merge(sw_agg, on="content_id_hashed", how="left")
    for c in out.columns:
        if c != "content_id_hashed":
            out[c] = out[c].astype("float32")
    return out


def add_text_similarity_features(df: pd.DataFrame, term_col: str = 'search_term_normalized', tag_col: str = 'cv_tags') -> pd.DataFrame:
    """
    search_term_normalized ile cv_tags arasındaki karakter n-gram (3-5) tabanlı
    HashingVectorizer kullanarak kosinüs benzerliğini hesaplar.
    Bu, Jaccard'a göre gürültüye daha dayanıklı ve yüksek-ayrıştırıcı bir sinyal sağlar.
    """
    print("    [TextSim] Char n-gram cosine (term↔cv_tags) hesaplanıyor...")
    s1 = df[term_col].fillna('').astype(str).values
    s2 = df[tag_col].fillna('').astype(str).values
    vec = HashingVectorizer(analyzer='char_wb', ngram_range=(3, 5), n_features=2**18, alternate_sign=False, norm='l2')
    A = vec.transform(s1)
    B = vec.transform(s2)
    cos = (A.multiply(B)).sum(axis=1).A1.astype('float32')
    df['term_tag_char_cosine'] = cos
    return df