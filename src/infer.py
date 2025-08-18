# -*- coding: utf-8 -*-
"""
infer.py
Test için sıralama ve submission dosyası üretimi.
"""

import pandas as pd

# src/infer.py

# def make_submission(df, out_path: str, expected_sessions: int | None = None):
#     """
#     df: en az ['session_id','content_id_hashed','pred_final'] kolonları olan skorlanmış tablo
#     expected_sessions: beklenen benzersiz session sayısı (verilmezse df içinden hesaplar)
#     """
#     use = df[["session_id", "content_id_hashed", "pred_final"]].copy()
#     use["content_id_hashed"] = use["content_id_hashed"].astype(str)

#     # oturum içi azalan skorla sırala
#     use = use.sort_values(["session_id", "pred_final"], ascending=[True, False])

#     # tek satıra sıkıştır
#     sub = (use.groupby("session_id", sort=False)["content_id_hashed"]
#                 .apply(lambda s: " ".join(s.tolist()))
#                 .reset_index()
#                 .rename(columns={"content_id_hashed": "prediction"}))

#     # beklenen satır sayısını doğrula
#     if expected_sessions is None:
#         expected_sessions = use["session_id"].nunique()

#     if len(sub) != expected_sessions:
#         # eksik/artan varsa, df içindeki tüm session'lara reindex et
#         all_sessions = use["session_id"].drop_duplicates()
#         sub = sub.set_index("session_id").reindex(all_sessions).reset_index()
#         sub["prediction"] = sub["prediction"].fillna("")
#         print(f"[FIX] submission rows adjusted -> {len(sub)} (expected {expected_sessions})")

#     sub.to_csv(out_path, index=False)
#     print(f"[OK] Submission written -> {out_path} (rows={len(sub)})")


def make_submission(df, out_path: str, expected_sessions: int | None = None, session_index: list[str] | None = None):
    """
    df: ['session_id','content_id_hashed','pred_final'] içeren skorlanmış tablo
    session_index: sample_submission.csv'den gelen doğru ve tam session_id sırası.
    expected_sessions: güvenlik için beklenen satır sayısı (opsiyonel).
    """
    use = df[["session_id", "content_id_hashed", "pred_final"]].copy()
    use["session_id"] = use["session_id"].astype(str)
    use["content_id_hashed"] = use["content_id_hashed"].astype(str)

    # oturum içinde azalan skorla sırala
    use = use.sort_values(["session_id", "pred_final"], ascending=[True, False])

    # oturum -> tek satır
    sub = (use.groupby("session_id", sort=False)["content_id_hashed"]
              .apply(lambda s: " ".join(s.tolist()))
              .reset_index()
              .rename(columns={"content_id_hashed": "prediction"}))

    # sample_submission indexine reindex et (varsa)
    if session_index is not None:
        all_sessions = pd.Series(session_index, name="session_id")
        sub = sub.set_index("session_id").reindex(all_sessions).reset_index()
        sub["prediction"] = sub["prediction"].fillna("")
    else:
        # en azından satır sayısını zorla
        if expected_sessions is None:
            expected_sessions = use["session_id"].nunique()
        if len(sub) != expected_sessions:
            all_sessions = use["session_id"].drop_duplicates()
            sub = sub.set_index("session_id").reindex(all_sessions).reset_index()
            sub["prediction"] = sub["prediction"].fillna("")
            print(f"[FIX] submission rows adjusted -> {len(sub)} (expected {expected_sessions})")

    sub.to_csv(out_path, index=False)
    print(f"[OK] Submission written -> {out_path} (rows={len(sub)})")
