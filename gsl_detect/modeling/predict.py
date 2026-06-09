from pathlib import Path
from loguru import logger
from tqdm import tqdm
import typer

from gsl_detect.config import MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer()
