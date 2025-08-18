# -*- coding: utf-8 -*-
"""
utils.py
Küçük yardımcılar: rastgelelik sabitleme, zamanlayıcı, dtype küçültme.
"""

import os
import random
import time
from contextlib import contextmanager
import numpy as np

def set_seed(seed: int = 42):
    """Tüm kütüphaneler için deterministik tohum."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

@contextmanager
def timer(name: str):
    """Kod bloklarının süresini loglamak için basit zamanlayıcı."""
    t0 = time.time()
    print(f"[TIMER] {name} ...", flush=True)
    yield
    dt = time.time() - t0
    print(f"[TIMER] {name} done in {dt:.2f}s", flush=True)

def reduce_memory_df(df):
    import pandas as pd
    for col in df.columns:
        t = df[col].dtype
        if pd.api.types.is_float_dtype(t):
            df.loc[:, col] = df[col].astype("float32")
        elif pd.api.types.is_integer_dtype(t):
            df.loc[:, col] = pd.to_numeric(df[col], downcast="integer")
    return df

