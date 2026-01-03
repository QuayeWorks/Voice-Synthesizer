# dump_debug_mel.py
import numpy as np
from pathlib import Path
# --- Add project root to Python path ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.insert(0, PROJECT_ROOT)

path = Path("debug_mel.npy")
if not path.exists():
    print("debug_mel.npy not found in current directory.")
    raise SystemExit(1)

m = np.load(path)
print("shape:", m.shape)

flat = m.flatten()
# print first 90 numbers but keep it readable
print("first 90 values:")
for i, v in enumerate(flat[:90]):
    print(f"{i:02d}: {v:.6f}")
