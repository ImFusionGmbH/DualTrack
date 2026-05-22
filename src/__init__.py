import os
import dotenv
from pathlib import Path

dotenv.load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"), verbose=True)

EXTERNALS_DIR = os.path.join(
    os.path.dirname(__file__),
    "..",
    "external",
)
DATA_DIR = Path(__file__).parent.parent / 'data' 