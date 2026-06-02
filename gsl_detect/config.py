from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw/isolated_GSL_corpus"
GLOSSES = RAW_DATA_DIR / "Glosses_videos"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"
CSV = PROCESSED_DATA_DIR / "isolated_GSL_corpus.csv"
MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Feature dimensions (no face landmarks)
# Pose: 15 pts × 4 coords (x, y, z, visibility) = 60
# Left hand: 21 pts × 3 coords = 63
# Right hand: 21 pts × 3 coords = 63
# Total: 186 features
POSE_SELECTED_INDICES = list(range(15))  # 15 upper body pose landmarks

MAX_SEQUENCE_LENGTH = 60
FRAME_WIDTH = 848
FRAME_HEIGHT = 480

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
