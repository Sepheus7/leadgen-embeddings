from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
ARTIFACTS_DIR = BASE_DIR / "artifacts"

TEXT_MODEL_NAME_PRIMARY = "intfloat/e5-small-v2"
TEXT_MODEL_NAME_FALLBACK = "sentence-transformers/all-MiniLM-L6-v2"
TABULAR_PCA_COMPONENTS = 16
TOPK_DEFAULT = 20

for d in [DATA_DIR, ARTIFACTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

