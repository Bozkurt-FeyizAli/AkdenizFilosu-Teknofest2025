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
