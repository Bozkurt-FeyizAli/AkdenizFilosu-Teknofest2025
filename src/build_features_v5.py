import argparse
from pathlib import Path
from .features.features_v5 import build_features_v5

def main(args):
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    tr = build_features_v5(args.data_dir, is_train=True)
    te = build_features_v5(args.data_dir, is_train=False)

    tr.to_parquet(out / "features_v5_train.parquet")
    te.to_parquet(out / "features_v5_test.parquet")
    print("[OK] features_v5_train.parquet & features_v5_test.parquet yazıldı.")

    # Feather cache
    tr.reset_index(drop=True).to_feather(out / "features_v5_train.feather")
    te.reset_index(drop=True).to_feather(out / "features_v5_test.feather")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    main(ap.parse_args())

def add_item_ctr_cvr(df, logs_df):
    """
    logs_df: tüm event logları (item_id, clicked, ordered)
    df: train/test tablosu
    """
    stats = logs_df.groupby("item_id").agg(
        clicks=("clicked","sum"),
        views=("item_id","count"),
        orders=("ordered","sum")
    ).reset_index()
    stats["ctr"] = stats["clicks"] / (stats["views"] + 1e-6)
    stats["cvr"] = stats["orders"] / (stats["clicks"] + 1e-6)
    return df.merge(stats[["item_id","ctr","cvr"]], on="item_id", how="left").fillna(0)

def add_session_features(df):
    # session-level counts
    session_stats = df.groupby("session_id").agg(
        session_items=("item_id","count"),
        session_clicks=("clicked","sum"),
        session_orders=("ordered","sum")
    ).reset_index()
    df = df.merge(session_stats, on="session_id", how="left")

    # item position within session
    df["pos_in_session"] = df.groupby("session_id").cumcount()
    df["rel_pos_in_session"] = df["pos_in_session"] / df["session_items"].clip(lower=1)

    return df

def add_term_item_affinity(df, logs_df):
    """
    df: ana tablo (session_id, item_id, term_id)
    logs_df: geçmiş loglar
    """
    affinity = logs_df.groupby(["term_id","item_id"]).agg(
        term_item_clicks=("clicked","sum"),
        term_item_orders=("ordered","sum"),
        term_item_views=("item_id","count")
    ).reset_index()
    affinity["term_item_ctr"] = affinity["term_item_clicks"] / (affinity["term_item_views"]+1e-6)
    affinity["term_item_cvr"] = affinity["term_item_orders"] / (affinity["term_item_clicks"]+1e-6)

    return df.merge(
        affinity[["term_id","item_id","term_item_ctr","term_item_cvr"]],
        on=["term_id","item_id"], how="left"
    ).fillna(0)
