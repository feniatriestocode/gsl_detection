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

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

LIPS_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291] # 11 points x 3 coords = 33 features
LIPS_INNER = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308] # 11 points x 3 coords = 33 features
RIGHT_EYE = [33, 160, 158, 133, 153, 144] # 6 points x 3 coords = 18 features
LEFT_EYE = [362, 385, 387, 263, 373, 380] # 6 points x 3 coords = 18 features
EYEBROWS = [70, 63, 105, 66, 107, 336, 296, 334, 293, 300] # 10 points x 3 coords = 30 features

FACE_SELECTED_INDICES = list(set(LIPS_OUTER + LIPS_INNER + RIGHT_EYE + LEFT_EYE + EYEBROWS)) # 44 unique points x 3 coords = 132 features

POSE_SELECTED_INDICES = list(range(15))  # 15 upper body pose landmarks x 4 coords = 60 features
# UPPER_BODY_POSE = [i for i in UPPER_BODY_POSE if i not in FACE_SELECTED_INDICES]


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

