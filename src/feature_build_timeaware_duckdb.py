# -*- coding: utf-8 -*-
"""
feature_build_timeaware_duckdb.py  (HIZLI & ZAMAN-FARKINDALI)
- 7/30g rolling (sitewide content & term×content)  [leakage-free, as-of join]
- Ek sinyaller:
    * term_ctr_30d                  (terim genel CTR, 30g)
    * user_term_ctr_30d             (kullanıcı × terim CTR, 30g) [vardı, devam]
    * user_l1_clk_30d, user_l1_ord_30d, user_l1_ctr_30d  (kullanıcı × L1 kategori)
    * discount_pct, rating_avg, rating_log_cnt (price/rating)
    * jaccard_term_cv, term_in_cv_overlap     (sorgu ↔ cv_tags benzerliği)
    * discount_rank_in_cat                     (kategori içi indirim sırası)
    * days_since_creation                      (ürün yaşı)
- RAM ve hız için: yalnız oturumlarda görülen content’leri ve tarih aralığını işler.
"""

from __future__ import annotations
import os, re
import duckdb
import pandas as pd
from typing import Tuple
from .utils import reduce_memory_df


def _mk_roll(win: int, alias_prefix: str, col: str, part_cols: str) -> str:
    return (
        f"SUM({col}) OVER (PARTITION BY {part_cols} ORDER BY d "
        f"RANGE BETWEEN INTERVAL {win} DAY PRECEDING AND CURRENT ROW) "
        f"AS {alias_prefix}_{win}d"
    )


def assemble_timeaware_features(
    sessions: pd.DataFrame,
    windows: Tuple[int, ...] = (7, 30),
) -> pd.DataFrame:
    print("[TA] DuckDB builder -> start")

    # ---- giriş DF (yalnız gerekli kolonlar) ----
    need_cols = [
        "ts_hour", "search_term_normalized", "content_id_hashed", "session_id",
        "clicked", "ordered", "added_to_cart", "added_to_fav", "user_id_hashed"
    ]
    need_cols = [c for c in need_cols if c in sessions.columns]
    sess = sessions[need_cols].copy()
    sess["ts_hour"] = pd.to_datetime(sess["ts_hour"], utc=False)
    sess["session_date"] = sess["ts_hour"].dt.floor("D")

    con = duckdb.connect()
    try:
        n = max(1, min(os.cpu_count() or 4, 8))
        con.execute(f"PRAGMA threads={n};")
    except Exception:
        pass
    con.register("sessions_df", sess)

    # ---- tarih aralığı + anahtar setleri ----
    min_d, max_d = con.execute("""
        SELECT MIN(session_date)::DATE, MAX(session_date)::DATE
        FROM sessions_df
    """).fetchone()
    con.execute(
        "CREATE OR REPLACE TEMP VIEW sess_bounds AS "
        f"SELECT DATE '{min_d}' AS min_d, DATE '{max_d}' AS max_d"
    )
    con.execute("""CREATE OR REPLACE TEMP VIEW sess_keys_content AS
                   SELECT DISTINCT content_id_hashed FROM sessions_df""")
    con.execute("""CREATE OR REPLACE TEMP VIEW sess_keys_ct AS
                   SELECT DISTINCT content_id_hashed, search_term_normalized
                   FROM sessions_df""")
    con.execute("""CREATE OR REPLACE TEMP VIEW sess_keys_ut AS
                   SELECT DISTINCT user_id_hashed, search_term_normalized
                   FROM sessions_df
                   WHERE user_id_hashed IS NOT NULL AND search_term_normalized IS NOT NULL""")

    # ---- sitewide rolling (content × date) ----
    sw_roll_exprs = []
    for w in windows:
        sw_roll_exprs += [
            _mk_roll(w, "click", "total_click", "content_id_hashed"),
            _mk_roll(w, "order", "total_order", "content_id_hashed"),
        ]
    con.execute(f"""
        CREATE OR REPLACE TEMP VIEW sw_roll AS
        WITH sw AS (
            SELECT sw.content_id_hashed,
                   CAST(sw.date AS DATE) AS d,
                   CAST(sw.total_click AS DOUBLE) AS total_click,
                   CAST(sw.total_order AS DOUBLE) AS total_order
            FROM read_parquet('data/content/sitewide_log.parquet') sw
            JOIN sess_keys_content k USING (content_id_hashed)
            WHERE CAST(sw.date AS DATE) BETWEEN
                (SELECT min_d - INTERVAL 90 DAY FROM sess_bounds) AND
                (SELECT max_d FROM sess_bounds)
        )
        SELECT content_id_hashed, d, {", ".join(sw_roll_exprs)}
        FROM sw
        ORDER BY content_id_hashed, d
    """)

    # ---- top terms rolling (content × term × date) ----
    tt_roll_exprs = []
    for w in windows:
        tt_roll_exprs += [
            _mk_roll(w, "imp", "total_search_impression", "content_id_hashed, search_term_normalized"),
            _mk_roll(w, "clk", "total_search_click", "content_id_hashed, search_term_normalized"),
        ]
    con.execute(f"""
        CREATE OR REPLACE TEMP VIEW tt_roll AS
        WITH ctt AS (
            SELECT c.content_id_hashed, c.search_term_normalized,
                   CAST(c.date AS DATE) AS d,
                   CAST(c.total_search_impression AS DOUBLE) AS total_search_impression,
                   CAST(c.total_search_click AS DOUBLE) AS total_search_click
            FROM read_parquet('data/content/top_terms_log.parquet') c
            JOIN sess_keys_ct k
              ON k.content_id_hashed = c.content_id_hashed
             AND k.search_term_normalized = c.search_term_normalized
            WHERE CAST(c.date AS DATE) BETWEEN
                (SELECT min_d - INTERVAL 90 DAY FROM sess_bounds) AND
                (SELECT max_d FROM sess_bounds)
        )
        SELECT content_id_hashed, search_term_normalized, d, {", ".join(tt_roll_exprs)}
        FROM ctt
        ORDER BY content_id_hashed, search_term_normalized, d
    """)

    # ---- price/rating zamanlı kayıtlar ----
    con.execute("""
        CREATE OR REPLACE TEMP VIEW prr AS
        SELECT p.content_id_hashed,
               CAST(p.update_date AS DATE) AS d,
               CAST(p.original_price   AS DOUBLE) AS original_price,
               CAST(p.selling_price    AS DOUBLE) AS selling_price,
               CAST(p.discounted_price AS DOUBLE) AS discounted_price,
               CAST(p.content_rate_avg AS DOUBLE) AS rate_avg,
               CAST(p.content_rate_count AS DOUBLE) AS rate_cnt
        FROM read_parquet('data/content/price_rate_review_data.parquet') p
        JOIN sess_keys_content k USING (content_id_hashed)
        WHERE CAST(p.update_date AS DATE) BETWEEN
              (SELECT min_d - INTERVAL 90 DAY FROM sess_bounds) AND
              (SELECT max_d FROM sess_bounds)
    """)

    # ---- user × term rolling (30g) ----
    con.execute("""
        CREATE OR REPLACE TEMP VIEW ut_roll AS
        WITH ut AS (
            SELECT u.user_id_hashed, u.search_term_normalized,
                   CAST(u.ts_hour AS DATE) AS d,
                   CAST(u.total_search_impression AS DOUBLE) AS imp,
                   CAST(u.total_search_click     AS DOUBLE) AS clk
            FROM read_parquet('data/user/top_terms_log.parquet') u
            JOIN sess_keys_ut k
              ON k.user_id_hashed = u.user_id_hashed
             AND k.search_term_normalized = u.search_term_normalized
            WHERE CAST(u.ts_hour AS DATE) BETWEEN
                  (SELECT min_d - INTERVAL 90 DAY FROM sess_bounds) AND
                  (SELECT max_d FROM sess_bounds)
        )
        SELECT user_id_hashed, search_term_normalized, d,
               SUM(imp) OVER (PARTITION BY user_id_hashed, search_term_normalized
                              ORDER BY d RANGE BETWEEN INTERVAL 30 DAY PRECEDING AND CURRENT ROW) AS u_imp_30d,
               SUM(clk) OVER (PARTITION BY user_id_hashed, search_term_normalized
                              ORDER BY d RANGE BETWEEN INTERVAL 30 DAY PRECEDING AND CURRENT ROW) AS u_clk_30d
        FROM ut
        ORDER BY user_id_hashed, search_term_normalized, d
    """)

    # ---- term prior CTR (30g)  [dosya yoksa atla] ----
    term_roll_ok = True
    try:
        con.execute("""
            CREATE OR REPLACE TEMP VIEW term_roll AS
            WITH tt AS (
                SELECT
                  CAST(ts_hour AS DATE) AS d,
                  LOWER(search_term_normalized) AS term,
                  CAST(total_search_impression AS DOUBLE) AS imp,
                  CAST(total_search_click     AS DOUBLE) AS clk
                FROM read_parquet('data/term/search_log.parquet')
                WHERE CAST(ts_hour AS DATE) BETWEEN
                    (SELECT min_d - INTERVAL 90 DAY FROM sess_bounds) AND
                    (SELECT max_d FROM sess_bounds)
            )
            SELECT
              term, d,
              SUM(imp) OVER (PARTITION BY term ORDER BY d RANGE BETWEEN INTERVAL 30 DAY PRECEDING AND CURRENT ROW) AS term_imp_30d,
              SUM(clk) OVER (PARTITION BY term ORDER BY d RANGE BETWEEN INTERVAL 30 DAY PRECEDING AND CURRENT ROW) AS term_clk_30d
            FROM tt
            ORDER BY term, d
        """)
    except Exception as e:
        term_roll_ok = False
        print(f"[TA] term_roll skipped: {e}")

    # ---- user × L1 kategori (30g)  [metadata üzerinden L1]  [dosya yoksa atla] ----
    user_l1_ok = True
    try:
        con.execute("""
            CREATE OR REPLACE TEMP VIEW user_l1_roll AS
            WITH uf AS (
              SELECT
                u.user_id_hashed,
                CAST(u.ts_hour AS DATE) AS d,
                u.content_id_hashed,
                CAST(u.total_click AS DOUBLE) AS clk,
                CAST(u.total_order AS DOUBLE) AS ord
              FROM read_parquet('data/user/fashion_sitewide_log.parquet') u
              JOIN sess_keys_content k USING (content_id_hashed)
              WHERE CAST(u.ts_hour AS DATE) BETWEEN
                    (SELECT min_d - INTERVAL 90 DAY FROM sess_bounds) AND
                    (SELECT max_d FROM sess_bounds)
            ),
            ufc AS (
              SELECT
                uf.user_id_hashed,
                m.level1_category_name AS l1,
                uf.d, uf.clk, uf.ord
              FROM uf
              JOIN read_parquet('data/content/metadata.parquet') m
                ON m.content_id_hashed = uf.content_id_hashed
            )
            SELECT
              user_id_hashed, l1, d,
              SUM(clk) OVER (PARTITION BY user_id_hashed, l1 ORDER BY d RANGE BETWEEN INTERVAL 30 DAY PRECEDING AND CURRENT ROW) AS u_l1_clk_30d,
              SUM(ord) OVER (PARTITION BY user_id_hashed, l1 ORDER BY d RANGE BETWEEN INTERVAL 30 DAY PRECEDING AND CURRENT ROW) AS u_l1_ord_30d
            FROM ufc
            ORDER BY user_id_hashed, l1, d
        """)
    except Exception as e:
        user_l1_ok = False
        print(f"[TA] user_l1_roll skipped: {e}")

    # ---- meta (cv_tags ve L1 dahil) ----
    con.execute("""
        CREATE OR REPLACE TEMP VIEW meta AS
        SELECT content_id_hashed,
               CAST(content_creation_date AS DATE) AS creation_date,
               level1_category_name,
               cv_tags
        FROM read_parquet('data/content/metadata.parquet')
    """)

    # ---- oran/ctr sütun adları + lateral joinler ----
    rate_cols, ctr_cols, sel_sw, sel_tt = [], [], [], []
    for w in windows:
        sel_sw += [f"r.click_{w}d AS click_{w}d", f"r.order_{w}d AS order_{w}d"]
        sel_tt += [f"r.imp_{w}d AS imp_{w}d", f"r.clk_{w}d AS clk_{w}d"]
        rate_cols += [
            f"(COALESCE(swf.click_{w}d,0)+1.0)/(COALESCE(swf.click_{w}d,0)+1.0+3.0) AS click_rate_{w}d",
            f"(COALESCE(swf.order_{w}d,0)+1.0)/(COALESCE(swf.click_{w}d,0)+1.0+3.0) AS order_rate_{w}d",
        ]
        ctr_cols += [
            f"(COALESCE(ttf.clk_{w}d,0)+1.0)/(COALESCE(ttf.imp_{w}d,0)+1.0+3.0) AS tc_ctr_{w}d",
        ]

    # dinamik ek kolonlar & joinler
    extra_select = []
    extra_joins  = []

    # user × term 30g
    extra_select.append("(COALESCE(utf.u_clk_30d,0)+1.0)/(COALESCE(utf.u_imp_30d,0)+1.0+3.0) AS user_term_ctr_30d")
    extra_joins.append("""
        LEFT JOIN LATERAL (
            SELECT r.d, r.u_imp_30d, r.u_clk_30d
            FROM ut_roll r
            WHERE r.user_id_hashed = s.user_id_hashed
              AND r.search_term_normalized = s.search_term_normalized
              AND r.d <= s.sdate
            ORDER BY r.d DESC
            LIMIT 1
        ) AS utf ON TRUE
    """)

    # term prior 30g
    if term_roll_ok:
        extra_select.append("(COALESCE(trf.term_clk_30d,0)+1.0)/(COALESCE(trf.term_imp_30d,0)+1.0+3.0) AS term_ctr_30d")
        extra_joins.append("""
            LEFT JOIN LATERAL (
                SELECT r.d, r.term_imp_30d, r.term_clk_30d
                FROM term_roll r
                WHERE r.term = LOWER(s.search_term_normalized)
                  AND r.d <= s.sdate
                ORDER BY r.d DESC
                LIMIT 1
            ) AS trf ON TRUE
        """)

    # user × L1 30g
    if user_l1_ok:
        extra_select += [
            "COALESCE(ul1f.u_l1_clk_30d,0) AS user_l1_clk_30d_raw",
            "COALESCE(ul1f.u_l1_ord_30d,0) AS user_l1_ord_30d_raw",
            "(COALESCE(ul1f.u_l1_clk_30d,0)+1.0)/(COALESCE(ul1f.u_l1_clk_30d,0)+COALESCE(ul1f.u_l1_ord_30d,0)+3.0) AS user_l1_ctr_30d",
        ]
        extra_joins.append("""
            LEFT JOIN LATERAL (
                SELECT r.d, r.u_l1_clk_30d, r.u_l1_ord_30d
                FROM user_l1_roll r
                JOIN meta mm
                  ON mm.level1_category_name = r.l1
                WHERE r.user_id_hashed = s.user_id_hashed
                  AND mm.content_id_hashed = s.content_id_hashed
                  AND r.d <= s.sdate
                ORDER BY r.d DESC
                LIMIT 1
            ) AS ul1f ON TRUE
        """)

    sql = f"""
        WITH s AS (
            SELECT *, CAST(session_date AS DATE) AS sdate
            FROM sessions_df
        )
        SELECT
            s.*,
            {", ".join(rate_cols)},
            {", ".join(ctr_cols)},
            -- price/rating
            CASE
              WHEN prrf.original_price IS NOT NULL AND prrf.original_price > 0
              THEN (prrf.original_price - COALESCE(prrf.discounted_price, prrf.selling_price)) / prrf.original_price
              ELSE NULL
            END AS discount_pct,
            prrf.rate_avg      AS rating_avg,
            LOG(1 + COALESCE(prrf.rate_cnt, 0)) AS rating_log_cnt,
            -- extra (dinamik)
            {", ".join(extra_select) if extra_select else "0.0 AS user_term_ctr_30d"},
            COALESCE(DATEDIFF('day', m.creation_date, s.sdate), -1) AS days_since_creation,
            m.level1_category_name,
            m.cv_tags
        FROM s
        LEFT JOIN LATERAL (
            SELECT r.d, {", ".join(sel_sw)}
            FROM sw_roll r
            WHERE r.content_id_hashed = s.content_id_hashed
              AND r.d <= s.sdate
            ORDER BY r.d DESC
            LIMIT 1
        ) AS swf ON TRUE
        LEFT JOIN LATERAL (
            SELECT r.d, {", ".join(sel_tt)}
            FROM tt_roll r
            WHERE r.content_id_hashed = s.content_id_hashed
              AND r.search_term_normalized = s.search_term_normalized
              AND r.d <= s.sdate
            ORDER BY r.d DESC
            LIMIT 1
        ) AS ttf ON TRUE
        LEFT JOIN LATERAL (
            SELECT r.d,
                   r.selling_price, r.discounted_price, r.original_price,
                   r.rate_avg, r.rate_cnt
            FROM prr r
            WHERE r.content_id_hashed = s.content_id_hashed
              AND r.d <= s.sdate
            ORDER BY r.d DESC
            LIMIT 1
        ) AS prrf ON TRUE
        {" ".join(extra_joins)}
        LEFT JOIN meta m
          ON m.content_id_hashed = s.content_id_hashed
    """

    out = con.execute(sql).df()

    # ---- python tarafı ek sinyaller ----
    # cv_tags ↔ term benzerliği
    def _tok(s):
        if s is None or pd.isna(s):
            return set()
        return set(re.findall(r"[a-z0-9]+", str(s).lower()))

    term_tokens = out["search_term_normalized"].map(_tok)
    cv_tokens = out["cv_tags"].map(_tok) if "cv_tags" in out.columns else [set()]*len(out)

    def _jacc(a, b):
        if not a or not b: return 0.0
        inter = len(a & b); uni = len(a | b)
        return inter / (uni + 1e-9)

    out["term_in_cv_overlap"] = [len(a & b) if a and b else 0 for a, b in zip(term_tokens, cv_tokens)]
    out["jaccard_term_cv"]    = [_jacc(a, b) for a, b in zip(term_tokens, cv_tokens)]

    # kategori içi indirim sırası
    if "level1_category_name" in out.columns and "discount_pct" in out.columns:
        out["discount_rank_in_cat"] = (
            out.groupby("level1_category_name")["discount_pct"]
               .rank(method="average", pct=True)
               .fillna(0.0).astype("float32")
        )
    else:
        out["discount_rank_in_cat"] = 0.0

    # trend_click_delta (7g - 30g)
    if {"click_rate_7d", "click_rate_30d"}.issubset(out.columns):
        out["trend_click_delta"] = (out["click_rate_7d"].fillna(0) - out["click_rate_30d"].fillna(0)).astype("float32")
    else:
        out["trend_click_delta"] = 0.0

    # gereksiz yardımcı kolonlar
    drop_cols = [c for c in out.columns if c.endswith(".d") or c.endswith("_d") or c in {"sdate"}]
    keep = [c for c in out.columns if c not in set(drop_cols)]
    out = out[keep]

    out = reduce_memory_df(out)
    print("[TA] DuckDB builder -> done")
    return out
