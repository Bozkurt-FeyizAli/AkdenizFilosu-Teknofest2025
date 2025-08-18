# -*- coding: utf-8 -*-
"""
main.py
Tek giriş noktası.

Komutlar:
  --prepare            : Smoke test + dummy submission (random-per-session sırası)
  --baseline           : Baseline özellikleri, offline validasyon ve test submission
  --baseline_timeaware : DuckDB time-aware baseline + submission
  --train_ltr          : LambdaMART (click & order), valid ve model kaydet
  --infer_ltr          : Kaydedilmiş LTR modelleriyle submission
  --infer_blend        : LTR ⊕ time-aware blend submission
  --validate           : Sadece baseline offline validasyonu
  --score_submission   : (submission.csv + train parquet) offline ölçüm
"""

import argparse
import numpy as np
import pandas as pd

from src.utils import set_seed, timer
from src.dataio import load_train_sessions, load_test_sessions, load_sample_submission_session_ids
from src.feature_build import assemble_baseline_features
from src.train_rank import score_baseline, score_timeaware_baseline
from src.postprocess import normalize_in_session
from src.metric import evaluate_sessions
from src.infer import make_submission
from src.train_rank_ltr import train_lambdarank_graded, save_lgb_model_graded, load_lgb_model_graded

from src.train_rank_ltr import (
    get_feature_cols, train_lambdarank, predict_on_df,
    save_lgb_model, load_lgb_model
)

from src.feature_build_timeaware_duckdb import assemble_timeaware_features


def split_time_holdout(train_df: pd.DataFrame, holdout_days: int = 7, fallback_q: float = 0.8):
    ts = pd.to_datetime(train_df["ts_hour"])
    cutoff = ts.max().normalize() - pd.Timedelta(days=holdout_days-1)
    tr = train_df[ts < cutoff].copy()
    va = train_df[ts >= cutoff].copy()
    if len(tr) == 0 or len(va) == 0:
        q = ts.quantile(fallback_q)
        tr = train_df[ts < q].copy()
        va = train_df[ts >= q].copy()
        print(f"[SPLIT] Fallback quantile used at {q}")
    else:
        print(f"[SPLIT] cutoff={cutoff.date()}  train={len(tr):,} rows  valid={len(va):,} rows")
    return tr, va


def run_prepare():
    set_seed(42)
    with timer("prepare: dummy submission"):
        test = load_test_sessions()
        rng = np.random.default_rng(42)
        test["_rnd"] = rng.random(len(test))
        test = test.sort_values(["session_id", "_rnd"], ascending=[True, True])
        make_submission(test.rename(columns={"_rnd": "pred_final"}), "outputs/dummy_submission.csv")


def run_validate():
    set_seed(42)
    with timer("validate: baseline (no model)"):
        train = load_train_sessions()
        feats = assemble_baseline_features(train)
        scored = score_baseline(feats)
        tr, va = split_time_holdout(scored, holdout_days=7)
        va = normalize_in_session(va, "pred_final")
        res = evaluate_sessions(va, click_score_col="pred_click", order_score_col="pred_order", w_order=0.75)
        print(f"[AUC] click={res['auc_click']:.6f}  order={res['auc_order']:.6f}  FINAL={res['final']:.6f}")


def run_baseline(out_path: str):
    set_seed(42)
    with timer("baseline: train/valid metric + test submission"):
        train = load_train_sessions()
        feats_tr = assemble_baseline_features(train)
        scored_tr = score_baseline(feats_tr)
        tr, va = split_time_holdout(scored_tr, holdout_days=7)
        va = normalize_in_session(va, "pred_final")
        res = evaluate_sessions(va, click_score_col="pred_click", order_score_col="pred_order", w_order=0.75)
        print(f"[AUC] click={res['auc_click']:.6f}  order={res['auc_order']:.6f}  FINAL={res['final']:.6f}")

        test = load_test_sessions()
        feats_te = assemble_baseline_features(test)
        scored_te = score_baseline(feats_te)
        scored_te = normalize_in_session(scored_te, "pred_final")

        idx = load_sample_submission_session_ids()
        make_submission(scored_te, out_path, session_index=idx)


def run_baseline_timeaware(out_path: str):
    set_seed(42)
    with timer("baseline_timeaware: valid metric + test submission"):
        train = load_train_sessions()
        feats_tr = assemble_timeaware_features(train, windows=(7, 30,90))
        scored_tr = score_timeaware_baseline(feats_tr)
        tr, va = split_time_holdout(scored_tr, holdout_days=7)
        va = normalize_in_session(va, "pred_final")
        res = evaluate_sessions(va, click_score_col="pred_click", order_score_col="pred_order", w_order=0.75)
        print(f"[AUC] TA click={res['auc_click']:.6f}  order={res['auc_order']:.6f}  FINAL={res['final']:.6f}")

        test = load_test_sessions()
        feats_te = assemble_timeaware_features(test, windows=(7, 30,90))
        scored_te = score_timeaware_baseline(feats_te)
        scored_te = normalize_in_session(scored_te, "pred_final")

        idx = load_sample_submission_session_ids()
        make_submission(scored_te, out_path, session_index=idx)


def tune_alpha_on_valid(va: pd.DataFrame, alphas=(0.70, 0.72, 0.74, 0.76, 0.78, 0.80, 0.82, 0.85)):
    best = (None, -1.0)
    for a in alphas:
        tmp = va.copy()
        tmp["pred_final"] = a*tmp["pred_order"] + (1.0-a)*tmp["pred_click"]
        res = evaluate_sessions(tmp, click_score_col="pred_click", order_score_col="pred_order", w_order=a)
        print(f"[ALPHA] a={a:.2f}  click={res['auc_click']:.6f}  order={res['auc_order']:.6f}  FINAL={res['final']:.6f}")
        if res["final"] > best[1]:
            best = (a, res["final"])
    print(f"[ALPHA] best={best[0]:.2f}  FINAL={best[1]:.6f}")
    return best


def run_train_ltr(alpha: float = 0.75):
    set_seed(42)
    with timer("LTR train (time-aware features)"):
        train = load_train_sessions()
        feats_tr = assemble_timeaware_features(train, windows=(7, 30,90))

        feat_cols = get_feature_cols(feats_tr)
        tr, va = split_time_holdout(feats_tr, holdout_days=7)

        click_model = train_lambdarank(tr, va, label_col="clicked", feat_cols=feat_cols)
        order_model = train_lambdarank(tr, va, label_col="ordered", feat_cols=feat_cols)

        va = va.copy()
        va["pred_click"] = predict_on_df(va, click_model, feat_cols)
        va["pred_order"] = predict_on_df(va, order_model, feat_cols)

        tune_alpha_on_valid(va)

        va["pred_final"] = alpha*va["pred_order"] + (1.0 - alpha)*va["pred_click"]
        res = evaluate_sessions(va, click_score_col="pred_click", order_score_col="pred_order", w_order=alpha)
        print(f"[LTR] VALID  click={res['auc_click']:.6f}  order={res['auc_order']:.6f}  FINAL={res['final']:.6f}")

        save_lgb_model(click_model, "models/lgb_click.txt")
        save_lgb_model(order_model, "models/lgb_order.txt")
        print("[LTR] models saved -> models/lgb_click.txt & models/lgb_order.txt")


def run_infer_ltr(out_path: str, alpha: float = 0.78):
    set_seed(42)
    with timer("LTR infer (time-aware features)"):
        test = load_test_sessions()
        feats_te = assemble_timeaware_features(test, windows=(7, 30,90))

        click_model = load_lgb_model("models/lgb_click.txt")
        order_model = load_lgb_model("models/lgb_order.txt")
        feat_cols = list(click_model.feature_name())

        scored = feats_te.copy()
        scored["pred_click"] = predict_on_df(scored, click_model, feat_cols)
        scored["pred_order"] = predict_on_df(scored, order_model, feat_cols)
        scored["pred_final"] = (1.0 - alpha) * scored["pred_click"] + alpha * scored["pred_order"]
        scored = normalize_in_session(scored, "pred_final")

        idx = load_sample_submission_session_ids()
        make_submission(scored, out_path, session_index=idx)

def run_train_ltr_strong():
    """
    Multi-seed graded LambdaMART (daha agresif parametreler + ensembling).
    """
    set_seed(42)
    with timer("LTR STRONG train (graded, multi-seed)"):
        train = load_train_sessions()
        feats = assemble_timeaware_features(train, windows=(7, 30))

        feat_cols = get_feature_cols(feats)
        tr, va = split_time_holdout(feats, holdout_days=7)

        from src.train_rank_ltr import (
            train_lambdarank_graded_multi, predict_ensemble, save_lgb_models
        )

        models = train_lambdarank_graded_multi(
            tr, va, feat_cols,
            seeds=(42, 1337, 2025),
            num_boost_round=3500,
            learning_rate=0.035,
            truncation_level=50,
            feature_fraction=0.90,
            bagging_fraction=0.85,
            min_data_in_leaf=20,
            lambda_l2=5.0,
        )

        # valid ensemble skoru
        va = va.copy()
        va["pred_final"] = predict_ensemble(models, va, feat_cols)
        res = evaluate_sessions(va, click_score_col="pred_final", order_score_col="pred_final", w_order=0.85)
        print(f"[LTR-STRONG] VALID  click={res['auc_click']:.6f}  order={res['auc_order']:.6f}  FINAL={res['final']:.6f}")

        save_lgb_models(models, "models/lgb_graded_s{seed}.txt")
        print("[LTR-STRONG] models saved -> models/lgb_graded_s*.txt")


def run_infer_ltr_strong(out_path: str, alpha: float = 0.85):
    """
    Multi-seed ensembled graded model ile test tahmini.
    (alpha paramı graded modelde kullanılmıyor; API uyumu için var.)
    """
    set_seed(42)
    with timer("LTR STRONG infer (graded ensemble)"):
        test = load_test_sessions()
        feats_te = assemble_timeaware_features(test, windows=(7, 30))

        from src.train_rank_ltr import load_lgb_models, predict_ensemble
        models = load_lgb_models("models/lgb_graded_s*.txt")
        feat_cols = get_feature_cols(feats_te)

        scored = feats_te.copy()
        scored["pred_final"] = predict_ensemble(models, scored, feat_cols)

        scored = normalize_in_session(scored, "pred_final")
        idx = load_sample_submission_session_ids("data/sample_submission.csv")
        make_submission(scored, out_path, session_index=idx)

def run_infer_blend(out_path: str, alpha: float = 0.82, beta: float = 0.40):
    """
    alpha: order ağırlığı (LTR içindeki blend)
    beta : LTR (1-beta)  ⊕  time-aware (beta)
    """
    set_seed(42)
    with timer("Blend infer (LTR ⊕ time-aware)"):
        test = load_test_sessions()
        feats_te = assemble_timeaware_features(test, windows=(7, 30,90))

        # --- LTR ---
        click_model = load_lgb_model("models/lgb_click.txt")
        order_model = load_lgb_model("models/lgb_order.txt")
        feat_cols   = list(click_model.feature_name())

        ltr = feats_te.copy()
        ltr["pred_click"] = predict_on_df(ltr, click_model, feat_cols)
        ltr["pred_order"] = predict_on_df(ltr, order_model, feat_cols)
        ltr["pred_final"] = (1.0 - alpha)*ltr["pred_click"] + alpha*ltr["pred_order"]
        ltr = normalize_in_session(ltr, "pred_final")
        ltr = ltr[["session_id","content_id_hashed","pred_final"]].rename(columns={"pred_final":"pred_ltr"})

        # --- Time-aware baseline ---
        ta = score_timeaware_baseline(feats_te)
        ta = normalize_in_session(ta, "pred_final")
        ta = ta[["session_id","content_id_hashed","pred_final"]].rename(columns={"pred_final":"pred_ta"})

        # --- Blend ---
        df = ltr.merge(ta, on=["session_id","content_id_hashed"], how="inner")
        df["pred_final"] = (1.0 - beta)*df["pred_ltr"] + beta*df["pred_ta"]

        idx = load_sample_submission_session_ids()
        make_submission(df[["session_id","content_id_hashed","pred_final"]], out_path, session_index=idx)

def run_train_ltr_graded():
    """
    Tek başlı graded LTR (relevance = 2*ordered + 1*clicked).
    VALID üzerinde AUC'leri tek skorla ölçüyoruz (pred_graded'i hem click hem order için kullanıyoruz).
    """
    set_seed(42)
    with timer("LTR graded train (time-aware features)"):
        train = load_train_sessions()
        feats = assemble_timeaware_features(train, windows=(7, 30))

        feat_cols = get_feature_cols(feats)

        tr, va = split_time_holdout(feats, holdout_days=7)
        model = train_lambdarank_graded(tr, va, feat_cols=feat_cols)

        # valid üzerinde tek skorla ölç (hem click hem order AUC aynı skorla)
        va = va.copy()
        va = va.sort_values(["session_id"]).reset_index(drop=True)
        pred = model.predict(va[feat_cols], num_iteration=getattr(model, "best_iteration_", None))
        va["pred_click"] = pred
        va["pred_order"] = pred
        va["pred_final"] = pred
        res = evaluate_sessions(va, click_score_col="pred_click", order_score_col="pred_order", w_order=0.85)
        print(f"[LTR-GRADED] VALID  click={res['auc_click']:.6f}  order={res['auc_order']:.6f}  FINAL={res['final']:.6f}")

        save_lgb_model_graded(model, "models/lgb_graded.txt")
        print("[LTR-GRADED] model saved -> models/lgb_graded.txt")

def run_infer_ltr_graded(out_path: str):
    set_seed(42)
    with timer("LTR graded infer (time-aware features)"):
        test = load_test_sessions()
        feats_te = assemble_timeaware_features(test, windows=(7, 30))

        model = load_lgb_model_graded("models/lgb_graded.txt")
        feat_cols = list(model.feature_name())

        scored = feats_te.copy()
        scored["pred_final"] = model.predict(scored[feat_cols], num_iteration=getattr(model, "best_iteration_", None))
        scored = normalize_in_session(scored, "pred_final")

        idx = load_sample_submission_session_ids("data/sample_submission.csv")
        make_submission(scored, out_path, session_index=idx)



if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--baseline", action="store_true")
    ap.add_argument("--baseline_timeaware", action="store_true", help="Time-aware baseline + submission")
    ap.add_argument("--train_ltr", action="store_true", help="İki başlı LambdaMART eğitim")
    ap.add_argument("--infer_ltr", action="store_true", help="Kaydedilmiş LTR modelleriyle submission")
    # ap.add_argument("--infer_blend", action="store_true", help="LTR ⊕ time-aware blend submission")
    ap.add_argument("--alpha", type=float, default=0.75, help="Blend: order ağırlığı (0-1)")
    # ap.add_argument("--beta", type=float, default=0.40, help="Blend oranı (LTR vs TA)")  # <<< EKLENDİ
    ap.add_argument("--out", type=str, default="outputs/submission_baseline.csv")
    ap.add_argument("--train_ltr_graded", action="store_true", help="Graded (single-head) LambdaMART eğit")
    ap.add_argument("--infer_ltr_graded", action="store_true", help="Kaydedilmiş graded LTR ile submission")
    ap.add_argument("--train_ltr_strong", action="store_true", help="Graded LambdaMART (multi-seed ensemble)")
    ap.add_argument("--infer_ltr_strong", action="store_true", help="Graded ensemble inference")
    ap.add_argument("--score_submission", type=str, help="Değerlendirilecek submission CSV yolu")
    ap.add_argument("--ground", type=str, default="data/train_sessions.parquet", help="Etiketli ground parquet yolu")
    ap.add_argument("--w_order", type=float, default=0.75, help="Final skorunda order ağırlığı (0-1)")
    # ... mevcut importlarınız ...
    # en altta parser’a ekleyin:
    ap.add_argument("--infer_blend", action="store_true",
                    help="LTR ⊕ time-aware blend submission (beta küçük) ")
    ap.add_argument("--beta", type=float, default=0.35,
                    help="Blend oranı: final = (1-beta)*LTR + beta*TA")
    args = ap.parse_args()

    if args.prepare:
        run_prepare()
    elif args.validate:
        run_validate()
    elif args.baseline:
        run_baseline(args.out)
    elif args.baseline_timeaware:
        run_baseline_timeaware(args.out)
    elif args.train_ltr:
        run_train_ltr(alpha=args.alpha)
    elif args.train_ltr_graded:
        run_train_ltr_graded()
    elif args.infer_ltr_graded:
        run_infer_ltr_graded(args.out)
    elif args.train_ltr_strong:
        run_train_ltr_strong()
    elif args.infer_ltr_strong:
        run_infer_ltr_strong(args.out, alpha=args.alpha)
    elif args.infer_ltr:
        run_infer_ltr(args.out, alpha=args.alpha)
    elif args.infer_blend:
        run_infer_blend(args.out, alpha=args.alpha, beta=args.beta)  # <<< beta CLI’dan geçiyor
    elif args.score_submission:
        from src.evaluator import evaluate_submission
        with timer("score_submission"):
            res = evaluate_submission(args.score_submission, args.ground, args.w_order)
            print(f"[EVAL] AUC_click={res['auc_click']:.6f}  AUC_order={res['auc_order']:.6f}  FINAL={res['final']:.6f}")
            print(f"[EVAL] sessions_covered={res['sessions_covered']:,}/{res['sessions_total']:,}  coverage={res['coverage']:.2%}")
    # ... if-elif zincirinde:
    elif args.infer_blend:
        from src.feature_build_timeaware_duckdb import assemble_timeaware_features
        from src.train_rank_ltr import load_lgb_model, predict_on_df
        from src.postprocess import normalize_in_session
        from src.dataio import load_sample_submission_session_ids
        from src.train_rank import score_timeaware_baseline

        test = load_test_sessions()
        feats_te = assemble_timeaware_features(test, windows=(7,30))

        click_model = load_lgb_model("models/lgb_click.txt")
        order_model = load_lgb_model("models/lgb_order.txt")
        feat_cols   = list(click_model.feature_name())

        # LTR
        ltr = feats_te.copy()
        ltr["pred_click"] = predict_on_df(ltr, click_model, feat_cols)
        ltr["pred_order"] = predict_on_df(ltr, order_model, feat_cols)
        ltr["pred_final"] = (1.0 - args.alpha)*ltr["pred_click"] + args.alpha*ltr["pred_order"]
        ltr = normalize_in_session(ltr, "pred_final")
        ltr = ltr[["session_id","content_id_hashed","pred_final"]].rename(columns={"pred_final":"pred_ltr"})

        # TA
        ta = score_timeaware_baseline(feats_te)
        ta = normalize_in_session(ta, "pred_final")
        ta = ta[["session_id","content_id_hashed","pred_final"]].rename(columns={"pred_final":"pred_ta"})

        # BLEND
        df = ltr.merge(ta, on=["session_id","content_id_hashed"], how="inner")
        df["pred_final"] = (1.0 - args.beta)*df["pred_ltr"] + args.beta*df["pred_ta"]

        idx = load_sample_submission_session_ids("data/sample_submission.csv")
        make_submission(df[["session_id","content_id_hashed","pred_final"]], args.out, session_index=idx)
    
    
    else:
        print("Hiçbir komut verilmedi. Örnek: python -m src.main --baseline --out outputs/submission.csv")
