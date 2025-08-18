# run.py (kök dizinde)
import os, sys, runpy

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "src")

# 1) Kök ve src klasörlerini import yoluna ekle
for p in (ROOT, SRC):
    if p not in sys.path:
        sys.path.insert(0, p)

# 2) Sağlık kontrolü
assert os.path.isdir(SRC), f"src klasörü yok: {SRC}"
assert os.path.isfile(os.path.join(SRC, "__init__.py")), "src/__init__.py yok!"
main_path = os.path.join(SRC, "main.py")
assert os.path.isfile(main_path), f"main.py bulunamadı: {main_path}"

print("[RUN] ROOT =", ROOT)
print("[RUN] SRC  =", SRC)
print("[RUN] sys.path[0:3] =", sys.path[0:3])
print("[RUN] running:", main_path)

# 3) src/main.py'yi dosya olarak çalıştır (absolute import'lar için yol hazır)
runpy.run_path(main_path, run_name="__main__")
