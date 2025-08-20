#!/usr/bin/env python3
"""
Bellek optimizasyonu test scripti
"""

import sys
import gc
import pandas as pd
import numpy as np

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("[WARNING] psutil not available, limited memory monitoring")

def check_memory():
    """Mevcut bellek kullanımını kontrol et"""
    if PSUTIL_AVAILABLE:
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        memory_pct = process.memory_percent()
        print(f"[MEMORY] Current usage: {memory_mb:.1f} MB ({memory_pct:.1f}%)")
        return memory_mb, memory_pct
    else:
        print("[MEMORY] psutil not available, performing garbage collection")
        gc.collect()
        return 0, 0

def simulate_large_dataset():
    """Büyük dataset simülasyonu"""
    print("Creating large dataset simulation...")
    
    # Büyük DataFrame oluştur
    n_rows = 500000
    data = {
        'ts_hour': pd.date_range('2024-01-01', periods=n_rows, freq='H'),
        'user_id': np.random.randint(1, 10000, n_rows),
        'content_id': np.random.randint(1, 50000, n_rows),
        'search_term': ['term_' + str(i) for i in np.random.randint(1, 1000, n_rows)],
        'clicked': np.random.choice([0, 1], n_rows, p=[0.9, 0.1]),
        'ordered': np.random.choice([0, 1], n_rows, p=[0.98, 0.02]),
    }
    
    df = pd.DataFrame(data)
    check_memory()
    print(f"Dataset created: {len(df):,} rows")
    
    return df

def test_memory_limits():
    """Bellek sınırlarını test et"""
    print("Testing memory limits...")
    
    # Başlangıç bellek durumu
    check_memory()
    
    # Büyük dataset oluştur
    df = simulate_large_dataset()
    
    # Bellek kullanımını kontrol et
    mem_mb, mem_pct = check_memory()
    
    if mem_pct > 80:
        print(f"[WARNING] High memory usage detected: {mem_pct:.1f}%")
        
    # Bellek temizliği test et
    del df
    gc.collect()
    print("After cleanup:")
    check_memory()

if __name__ == "__main__":
    print("=== Memory Optimization Test ===")
    test_memory_limits()
    print("=== Test Complete ===")
