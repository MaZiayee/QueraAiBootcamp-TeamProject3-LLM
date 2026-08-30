import os
import json
import hashlib
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv
import openai

from src.utils import get_env


# ============================================================
# Environment
# ============================================================

load_dotenv(override=True)


# ============================================================
# Metis Configuration
# ============================================================

METIS_BASE_URL = (
    "https://api.metisai.ir/api/v1/wrapper/google"
)

API_KEY = get_env(
    "METIS_API_KEY",
    default=""
).strip()

LLM_MODELS = [
    model.strip()
    for model in os.getenv(
        "LLM_MODELS",
        "gemini-2.5-flash"
    ).split(",")
    if model.strip()
]

DEFAULT_MODEL = LLM_MODELS[0]


# ============================================================
# Generation Settings
# ============================================================

# For normal assistant responses.
MAX_OUTPUT_TOKENS = int(
    os.getenv(
        "LLM_MAX_OUTPUT_TOKENS",
        "500"
    )
)

TEMPERATURE = float(
    os.getenv(
        "LLM_TEMPERATURE",
        "0.1"
    )
)


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

CACHE_DIR = (
    PROJECT_ROOT
    / "data"
    / "indexes"
)

CACHE_FILE = (
    CACHE_DIR
    / "llm_cache.json"
)


# ============================================================
# Client
# ============================================================

client = None
LIVE_API = False
AVAILABLE_MODELS = []


def initialize_client():
    """
    Initialize Metis gateway client.
    """

    global client
    global LIVE_API
    global AVAILABLE_MODELS

    client = None
    LIVE_API = False
    AVAILABLE_MODELS = []

    if not API_KEY:
        print(
            "METIS_API_KEY is not set."
        )
        return None, False

    try:

        client = openai.OpenAI(
            base_url=METIS_BASE_URL,
            api_key=API_KEY,
            timeout=120.0,
        )

        models = client.models.list().data

        AVAILABLE_MODELS = sorted(
            model.id
            for model in models
        )

        LIVE_API = True

        print(
            f"{len(AVAILABLE_MODELS)} models reachable."
        )

        return client, True

    except Exception as error:

        client = None
        LIVE_API = False

        print(
            "Could not reach Metis gateway:"
        )

        print(
            f"{type(error).__name__} — "
            f"{str(error)[:200]}"
        )

        print(
            "Running in cache-only mode."
        )

        return None, False


initialize_client()


# ============================================================
# Cache
# ============================================================

def ensure_cache_dir():
    CACHE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


def load_cache():
    ensure_cache_dir()

    if not CACHE_FILE.exists():
        return {}

    try:

        with open(
            CACHE_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            data = json.load(file)

        if isinstance(data, dict):
            return data

    except (
        json.JSONDecodeError,
        OSError
    ):
        pass

    return {}


def save_cache(cache):
    ensure_cache_dir()

    temp_file = CACHE_FILE.with_suffix(
        ".tmp"
    )

    with open(
        temp_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            cache,
            file,
            ensure_ascii=False,
            indent=2
        )

    temp_file.replace(
        CACHE_FILE
    )


def clear_cache():
    if CACHE_FILE.exists():
        CACHE_FILE.unlink()


def cache_size():
    return len(
        load_cache()
    )


# ============================================================
# Cache Key
# ============================================================

def create_cache_key(
    query,
    evidence_text,
    model,
    structured
):
    payload = {
        "query": query,
        "evidence": evidence_text,
        "model": model,
        "structured": structured,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
    }

    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True
    )

    return hashlib.sha256(
        serialized.encode("utf-8")
    ).hexdigest()


# ============================================================
# Prompts
# ============================================================

SYSTEM_PROMPT = """
You are a Persian shopping assistant.

Use ONLY the supplied evidence.

Rules:
- Never invent information.
- Never invent reviews or Review IDs.
- Separate product facts from user experiences.
- Every review-based claim must be supported by the provided evidence.
- If evidence is insufficient, say so.
- Answer directly.
- Keep the answer concise.
- Usually answer in 2 to 5 sentences.
- Finish every sentence completely.
""".strip()


def build_user_prompt(
    query,
    evidence_text
):
    return f"""
QUESTION:
{query}

EVIDENCE:
{evidence_text}

Answer directly in Persian using ONLY the evidence.

Important:
- Do not invent anything.
- Do not repeat all evidence.
- Mention Review IDs when making claims based on reviews.
- If the evidence is insufficient, say that clearly.
- Keep the answer short: normally 2 to 5 sentences.
- Every sentence must be complete.
- Return ONLY the final answer.
""".strip()


def build_structured_prompt(
    query,
    evidence_text
):
    return f"""
QUESTION:
{query}

EVIDENCE:
{evidence_text}

Return ONLY compact valid JSON:

{{
  "answer": "حداکثر 100 کلمه",
  "review_findings": [
    {{
      "claim": "کوتاه",
      "review_ids": [123]
    }}
  ],
  "recommendation": "کوتاه یا null",
  "evidence_sufficient": true
}}

Rules:
- Be extremely concise.
- Use only supplied evidence.
- Never invent Review IDs.
- Do not add unnecessary explanation.
- Return valid JSON only.
""".strip()


# ============================================================
# Response Extraction
# ============================================================

def extract_text_response(response):
    """
    Extract text from Metis/OpenAI-compatible response.
    """

    try:

        content = (
            response
            .choices[0]
            .message
            .content
        )

        if content:
            return content

    except (
        AttributeError,
        IndexError,
        TypeError
    ):
        pass

    try:

        return response.output_text

    except AttributeError:
        pass

    return str(response)


# ============================================================
# JSON Parsing
# ============================================================

def parse_json_response(text):
    """
    Parse JSON safely.
    """

    if not text:

        return {
            "answer": "",
            "review_findings": [],
            "recommendation": None,
            "evidence_sufficient": False,
        }

    text = text.strip()

    if text.startswith("```"):

        lines = text.splitlines()

        if lines:
            lines = lines[1:]

        if (
            lines
            and lines[-1].strip() == "```"
        ):
            lines = lines[:-1]

        text = "\n".join(
            lines
        ).strip()

    try:

        result = json.loads(
            text
        )

        if isinstance(result, dict):
            return result

    except json.JSONDecodeError:
        pass

    return {
        "answer": text,
        "review_findings": [],
        "recommendation": None,
        "evidence_sufficient": True,
    }


# ============================================================
# Usage
# ============================================================

def extract_usage(response):
    usage = getattr(
        response,
        "usage",
        None
    )

    if usage is None:

        return {
            "input_tokens": None,
            "output_tokens": None,
            "total_tokens": None,
        }

    input_tokens = getattr(
        usage,
        "input_tokens",
        None
    )

    output_tokens = getattr(
        usage,
        "output_tokens",
        None
    )

    total_tokens = getattr(
        usage,
        "total_tokens",
        None
    )

    if input_tokens is None:
        input_tokens = getattr(
            usage,
            "prompt_tokens",
            None
        )

    if output_tokens is None:
        output_tokens = getattr(
            usage,
            "completion_tokens",
            None
        )

    if (
        total_tokens is None
        and input_tokens is not None
        and output_tokens is not None
    ):

        total_tokens = (
            input_tokens
            + output_tokens
        )

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }


# ============================================================
# Cost
# ============================================================

def estimate_cost(
    usage,
    input_price_per_million=0.0,
    output_price_per_million=0.0
):
    input_tokens = (
        usage.get(
            "input_tokens"
        ) or 0
    )

    output_tokens = (
        usage.get(
            "output_tokens"
        ) or 0
    )

    input_cost = (
        input_tokens
        / 1_000_000
        * input_price_per_million
    )

    output_cost = (
        output_tokens
        / 1_000_000
        * output_price_per_million
    )

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": (
            input_tokens
            + output_tokens
        ),
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": (
            input_cost
            + output_cost
        ),
    }


# ============================================================
# API Request
# ============================================================

def request_generation(
    query,
    evidence_text,
    model=None,
    structured=False
):
    """
    Make exactly one API request.
    """

    if not LIVE_API or client is None:

        raise RuntimeError(
            "Metis API is unavailable."
        )

    model = model or DEFAULT_MODEL

    if model not in AVAILABLE_MODELS:

        raise ValueError(
            f"Model '{model}' is not available."
        )

    if structured:

        prompt = build_structured_prompt(
            query,
            evidence_text
        )

    else:

        prompt = build_user_prompt(
            query,
            evidence_text
        )

    response = client.chat.completions.create(
        model=model,

        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],

        temperature=TEMPERATURE,

    )

    text = extract_text_response(
        response
    )

    usage = extract_usage(
        response
    )

    return {
        "text": text,
        "usage": usage,
        "model": model,
        "raw_response": response,
    }


# ============================================================
# Main Cached LLM Call
# ============================================================

def call_llm(
    query,
    evidence_text,
    model=None,
    structured=False,
    use_cache=True,
    allow_api=True,
):
    """
    Main LLM entry point.

    Default mode is normal concise text.

    Cache is checked before API.
    """

    model = model or DEFAULT_MODEL

    if not query:
        raise ValueError(
            "Query cannot be empty."
        )

    if not evidence_text:
        raise ValueError(
            "Evidence cannot be empty."
        )

    cache_key = create_cache_key(
        query=query,
        evidence_text=evidence_text,
        model=model,
        structured=structured,
    )

    # ========================================================
    # Cache
    # ========================================================

    if use_cache:

        cache = load_cache()

        cached = cache.get(
            cache_key
        )

        if cached:

            return {
                "result": cached["result"],

                "usage": cached.get(
                    "usage",
                    {
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "total_tokens": 0,
                    }
                ),

                "model": cached.get(
                    "model",
                    model
                ),

                "cache_hit": True,

                "cache_key": cache_key,
            }

    # ========================================================
    # API
    # ========================================================

    if (
        not allow_api
        or not LIVE_API
        or client is None
    ):

        raise RuntimeError(
            "Cache miss and Metis API is unavailable. "
            "No API request was made."
        )

    response = request_generation(
        query=query,
        evidence_text=evidence_text,
        model=model,
        structured=structured,
    )

    raw_text = response["text"]

    # --------------------------------------------------------
    # Structured output
    # --------------------------------------------------------

    if structured:

        parsed = parse_json_response(
            raw_text
        )

        # If the model still returns malformed/truncated JSON,
        # preserve the text instead of throwing away the answer.
        result = parsed

    # --------------------------------------------------------
    # Normal output
    # --------------------------------------------------------

    else:

        result = {
            "answer": raw_text.strip()
        }

    usage = response["usage"]

    # ========================================================
    # Save Cache
    # ========================================================

    if use_cache:

        cache = load_cache()

        cache[cache_key] = {
            "result": result,
            "usage": usage,
            "model": model,
            "created_at": datetime.now().isoformat(),
        }

        save_cache(
            cache
        )

    return {
        "result": result,

        "usage": usage,

        "model": model,

        "cache_hit": False,

        "cache_key": cache_key,
    }