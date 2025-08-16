import argparse
from pathlib import Path
from src.features.features_v6 import build_features_v6

def main(args):
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    tr = build_features_v6(args.data_dir, is_train=True)
    te = build_features_v6(args.data_dir, is_train=False)
    tr.to_parquet(out/"features_v6_train.parquet")
    te.to_parquet(out/"features_v6_test.parquet")
    print("[OK] v6 feature dosyaları yazıldı.")
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    main(ap.parse_args())
