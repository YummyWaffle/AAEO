"""Keep all test output inside this standalone distribution."""
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
TEMP = ROOT / ".work" / "tmp"
TEMP.mkdir(parents=True, exist_ok=True)
tempfile.tempdir = str(TEMP)
os.environ["TEMP"] = os.environ["TMP"] = str(TEMP)
