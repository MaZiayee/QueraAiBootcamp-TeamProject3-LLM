import os
import re
import time
import unicodedata
from pathlib import Path
from functools import wraps


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
INDEX_DIR = DATA_DIR / "indexes"

PRODUCTS_PATH = DATA_DIR / "products.parquet"
REVIEWS_PATH = DATA_DIR / "reviews.parquet"

INDEX_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Persian Text Normalization
# ============================================================

def normalize_persian_text(text):
    """
    Normalize Persian/Arabic text for search and matching.

    - Converts Arabic ی / ك to Persian ی / ک
    - Removes ZWNJ
    - Normalizes Unicode
    - Normalizes whitespace
    """

    if text is None:
        return ""

    text = str(text)

    # Unicode normalization
    text = unicodedata.normalize("NFKC", text)

    # Arabic -> Persian characters
    replacements = {
        "ي": "ی",
        "ى": "ی",
        "ك": "ک",
        "ؤ": "و",
        "ۀ": "ه",
        "ة": "ه",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    # Remove zero-width non-joiner and zero-width characters
    text = text.replace("\u200c", " ")
    text = text.replace("\u200b", "")
    text = text.replace("\ufeff", "")

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ============================================================
# Text Cleaning
# ============================================================

def clean_text(text):
    """
    Basic text cleaning while preserving the original meaning.
    """

    text = normalize_persian_text(text)

    # Remove URLs
    text = re.sub(
        r"https?://\S+|www\.\S+",
        " ",
        text
    )

    # Normalize repeated punctuation
    text = re.sub(r"[!]{2,}", "!", text)
    text = re.sub(r"[?]{2,}", "?", text)
    text = re.sub(r"[.]{2,}", ".", text)

    # Normalize whitespace again
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ============================================================
# Number Utilities
# ============================================================

def to_float(value, default=None):
    """
    Safely convert a value to float.
    """

    if value is None:
        return default

    try:
        if isinstance(value, str):
            value = value.replace(",", "").strip()

        return float(value)

    except (ValueError, TypeError):
        return default


def to_int(value, default=None):
    """
    Safely convert a value to integer.
    """

    if value is None:
        return default

    try:
        if isinstance(value, str):
            value = value.replace(",", "").strip()

        return int(float(value))

    except (ValueError, TypeError):
        return default


# ============================================================
# General Helpers
# ============================================================

def safe_str(value):
    """
    Convert any value to a clean string.
    """

    if value is None:
        return ""

    return str(value).strip()


def is_empty(value):
    """
    Check whether a value is None or an empty string.
    """

    if value is None:
        return True

    if isinstance(value, str):
        return not value.strip()

    return False


# ============================================================
# Timing
# ============================================================

def timer():
    """
    Simple timer.

    Example:
        start = timer()
        ...
        elapsed = start()
    """

    start_time = time.perf_counter()

    def elapsed():
        return time.perf_counter() - start_time

    return elapsed


def measure_time(func):
    """
    Decorator for measuring function execution time.

    The wrapped function returns:

        result, latency_ms
    """

    @wraps(func)
    def wrapper(*args, **kwargs):

        start = time.perf_counter()

        result = func(*args, **kwargs)

        latency_ms = (time.perf_counter() - start) * 1000

        return result, latency_ms

    return wrapper


# ============================================================
# Environment Variables
# ============================================================

def get_env(name, default=None, required=False):
    """
    Read an environment variable.

    Args:
        name: Environment variable name
        default: Default value
        required: Raise an error if variable doesn't exist
    """

    value = os.getenv(name, default)

    if required and not value:
        raise ValueError(
            f"Required environment variable '{name}' is not set."
        )

    return value


# ============================================================
# Directory Utilities
# ============================================================

def ensure_dir(path):
    """
    Create a directory if it doesn't exist.
    """

    path = Path(path)

    path.mkdir(
        parents=True,
        exist_ok=True
    )

    return path


# ============================================================
# List Utilities
# ============================================================

def unique_list(items):
    """
    Remove duplicates while preserving order.
    """

    seen = set()
    result = []

    for item in items:

        if item not in seen:
            seen.add(item)
            result.append(item)

    return result


# ============================================================
# Debugging
# ============================================================

def print_section(title):
    """
    Print a readable section title in notebooks.
    """

    print()
    print("=" * 60)
    print(title)
    print("=" * 60)


def print_dataframe_info(df, name="DataFrame"):
    """
    Print basic information about a pandas DataFrame.

    Pandas is intentionally imported inside the function
    so utils.py does not require pandas just to be imported.
    """

    print_section(name)

    print(f"Rows    : {len(df):,}")
    print(f"Columns : {len(df.columns):,}")

    print("\nColumns:")
    for column in df.columns:
        print(f"  - {column}")


# ============================================================
# File Utilities
# ============================================================

def file_exists(path):
    """
    Check whether a file exists.
    """

    return Path(path).is_file()


def get_file_size_mb(path):
    """
    Return file size in MB.
    """

    path = Path(path)

    if not path.exists():
        return None

    return path.stat().st_size / (1024 ** 2)

