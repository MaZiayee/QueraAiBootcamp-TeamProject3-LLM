
import time
import numpy as np
import pandas as pd

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
)


# ============================================================
# Generic Helpers
# ============================================================

def safe_divide(
    numerator,
    denominator
):
    """
    Safe division.
    """

    if denominator == 0:
        return 0.0

    return numerator / denominator


def normalize_id(value):
    """
    Normalize IDs for comparison.
    """

    if value is None:
        return None

    try:
        return int(value)

    except (
        ValueError,
        TypeError
    ):
        return str(value)


# ============================================================
# Retrieval Metrics
# ============================================================

def precision_at_k(
    retrieved_ids,
    relevant_ids,
    k=None
):
    """
    Precision@K.
    """

    if k is not None:
        retrieved_ids = retrieved_ids[:k]

    if not retrieved_ids:
        return 0.0

    relevant_set = {
        normalize_id(item)
        for item in relevant_ids
    }

    hits = sum(
        normalize_id(item) in relevant_set
        for item in retrieved_ids
    )

    return safe_divide(
        hits,
        len(retrieved_ids)
    )


def recall_at_k(
    retrieved_ids,
    relevant_ids,
    k=None
):
    """
    Recall@K.
    """

    if k is not None:
        retrieved_ids = retrieved_ids[:k]

    if not relevant_ids:
        return 0.0

    relevant_set = {
        normalize_id(item)
        for item in relevant_ids
    }

    hits = sum(
        normalize_id(item) in relevant_set
        for item in retrieved_ids
    )

    return safe_divide(
        hits,
        len(relevant_set)
    )


def hit_at_k(
    retrieved_ids,
    relevant_ids,
    k=10
):
    """
    Hit@K.

    Returns 1 if at least one relevant item appears
    in the first K retrieved items.
    """

    retrieved_ids = retrieved_ids[:k]

    relevant_set = {
        normalize_id(item)
        for item in relevant_ids
    }

    return int(
        any(
            normalize_id(item)
            in relevant_set
            for item in retrieved_ids
        )
    )


def reciprocal_rank(
    retrieved_ids,
    relevant_ids
):
    """
    Reciprocal Rank.
    """

    relevant_set = {
        normalize_id(item)
        for item in relevant_ids
    }

    for rank, item in enumerate(
        retrieved_ids,
        start=1
    ):

        if normalize_id(item) in relevant_set:

            return 1.0 / rank

    return 0.0


def mean_reciprocal_rank(
    results
):
    """
    Mean Reciprocal Rank across queries.

    results format:

        [
            {
                "retrieved_ids": [...],
                "relevant_ids": [...]
            },
            ...
        ]
    """

    if not results:
        return 0.0

    scores = []

    for item in results:

        scores.append(
            reciprocal_rank(
                item.get(
                    "retrieved_ids",
                    []
                ),
                item.get(
                    "relevant_ids",
                    []
                )
            )
        )

    return float(
        np.mean(scores)
    )


def evaluate_retrieval(
    evaluation_data,
    k=10
):
    """
    Evaluate product/review retrieval.

    evaluation_data format:

        [
            {
                "retrieved_ids": [...],
                "relevant_ids": [...]
            }
        ]
    """

    if not evaluation_data:

        return {
            "precision_at_k": 0.0,
            "recall_at_k": 0.0,
            "hit_at_k": 0.0,
            "mrr": 0.0,
            "query_count": 0,
        }

    precision_scores = []
    recall_scores = []
    hit_scores = []

    for item in evaluation_data:

        retrieved = item.get(
            "retrieved_ids",
            []
        )

        relevant = item.get(
            "relevant_ids",
            []
        )

        precision_scores.append(
            precision_at_k(
                retrieved,
                relevant,
                k
            )
        )

        recall_scores.append(
            recall_at_k(
                retrieved,
                relevant,
                k
            )
        )

        hit_scores.append(
            hit_at_k(
                retrieved,
                relevant,
                k
            )
        )

    return {
        "precision_at_k": float(
            np.mean(
                precision_scores
            )
        ),

        "recall_at_k": float(
            np.mean(
                recall_scores
            )
        ),

        "hit_at_k": float(
            np.mean(
                hit_scores
            )
        ),

        "mrr": mean_reciprocal_rank(
            evaluation_data
        ),

        "query_count": len(
            evaluation_data
        ),
    }


# ============================================================
# Grounding
# ============================================================

def calculate_grounding_score(
    claims
):
    """
    Calculate grounding score.

    claims format:

        [
            {
                "claim": "...",
                "supported": True
            },
            ...
        ]
    """

    if not claims:
        return 0.0

    supported = sum(
        bool(
            item.get(
                "supported",
                False
            )
        )
        for item in claims
    )

    return safe_divide(
        supported,
        len(claims)
    )


def evaluate_grounding(
    grounding_data
):
    """
    Evaluate grounding over multiple answers.

    grounding_data format:

        [
            {
                "claims": [
                    {
                        "claim": "...",
                        "supported": True
                    }
                ]
            }
        ]
    """

    all_claims = []

    for answer in grounding_data:

        all_claims.extend(
            answer.get(
                "claims",
                []
            )
        )

    score = calculate_grounding_score(
        all_claims
    )

    supported = sum(
        bool(
            item.get(
                "supported",
                False
            )
        )
        for item in all_claims
    )

    return {
        "grounding_score": score,
        "total_claims": len(
            all_claims
        ),
        "supported_claims": supported,
        "unsupported_claims": (
            len(all_claims)
            - supported
        ),
    }


# ============================================================
# Answer Quality
# ============================================================

def calculate_average_quality(
    evaluations
):
    """
    Calculate average human/LLM evaluation score.

    Expected score range:
        1-5
    """

    if not evaluations:
        return 0.0

    scores = [
        float(item)
        for item in evaluations
        if item is not None
    ]

    if not scores:
        return 0.0

    return float(
        np.mean(scores)
    )


def evaluate_answer_quality(
    answers
):
    """
    Evaluate answer quality.

    Expected format:

        [
            {
                "relevance": 4,
                "usefulness": 5,
                "correctness": 4
            }
        ]
    """

    if not answers:

        return {
            "answer_count": 0,
            "relevance": 0.0,
            "usefulness": 0.0,
            "correctness": 0.0,
            "overall": 0.0,
        }

    relevance = []
    usefulness = []
    correctness = []

    for item in answers:

        if item.get("relevance") is not None:
            relevance.append(
                item["relevance"]
            )

        if item.get("usefulness") is not None:
            usefulness.append(
                item["usefulness"]
            )

        if item.get("correctness") is not None:
            correctness.append(
                item["correctness"]
            )

    values = [
        calculate_average_quality(
            relevance
        ),
        calculate_average_quality(
            usefulness
        ),
        calculate_average_quality(
            correctness
        ),
    ]

    available_values = [
        value
        for value in values
        if value > 0
    ]

    overall = (
        float(
            np.mean(
                available_values
            )
        )
        if available_values
        else 0.0
    )

    return {
        "answer_count": len(
            answers
        ),

        "relevance": values[0],

        "usefulness": values[1],

        "correctness": values[2],

        "overall": overall,
    }


# ============================================================
# Recommendation Prediction
# ============================================================

def evaluate_recommendation_prediction(
    y_true,
    y_pred
):
    """
    Evaluate status_recommendation prediction.

    Main metric:
        Macro F1
    """

    y_true = list(
        y_true
    )

    y_pred = list(
        y_pred
    )

    if not y_true:
        return {
            "macro_f1": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "accuracy": 0.0,
        }

    return {
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0
            )
        ),

        "macro_precision": float(
            precision_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0
            )
        ),

        "macro_recall": float(
            recall_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0
            )
        ),

        "accuracy": float(
            np.mean(
                np.array(y_true)
                == np.array(y_pred)
            )
        ),
    }


# ============================================================
# Latency
# ============================================================

def measure_latency(
    func,
    *args,
    **kwargs
):
    """
    Measure a single function call.

    Returns:
        result
        latency_ms
    """

    start = time.perf_counter()

    result = func(
        *args,
        **kwargs
    )

    latency_ms = (
        time.perf_counter()
        - start
    ) * 1000

    return result, latency_ms


def summarize_latency(
    latencies
):
    """
    Summarize latency measurements.
    """

    if not latencies:

        return {
            "count": 0,
            "mean_ms": 0.0,
            "median_ms": 0.0,
            "p95_ms": 0.0,
            "min_ms": 0.0,
            "max_ms": 0.0,
        }

    values = np.array(
        latencies,
        dtype=float
    )

    return {
        "count": len(values),

        "mean_ms": float(
            np.mean(values)
        ),

        "median_ms": float(
            np.median(values)
        ),

        "p95_ms": float(
            np.percentile(
                values,
                95
            )
        ),

        "min_ms": float(
            np.min(values)
        ),

        "max_ms": float(
            np.max(values)
        ),
    }


# ============================================================
# Cost
# ============================================================

def calculate_request_cost(
    input_tokens=0,
    output_tokens=0,
    input_price_per_million=0.0,
    output_price_per_million=0.0
):
    """
    Calculate the cost of one API request.
    """

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


def summarize_costs(
    usage_records,
    input_price_per_million=0.0,
    output_price_per_million=0.0
):
    """
    Aggregate token usage and estimated cost.

    usage_records format:

        [
            {
                "input_tokens": 100,
                "output_tokens": 50
            }
        ]
    """

    total_input = 0
    total_output = 0

    request_count = len(
        usage_records
    )

    cache_hits = 0

    for record in usage_records:

        total_input += (
            record.get(
                "input_tokens"
            )
            or 0
        )

        total_output += (
            record.get(
                "output_tokens"
            )
            or 0
        )

        if record.get(
            "cache_hit",
            False
        ):
            cache_hits += 1

    cost = calculate_request_cost(
        input_tokens=total_input,
        output_tokens=total_output,
        input_price_per_million=(
            input_price_per_million
        ),
        output_price_per_million=(
            output_price_per_million
        ),
    )

    cache_hit_rate = safe_divide(
        cache_hits,
        request_count
    )

    return {
        **cost,

        "request_count": request_count,

        "cache_hits": cache_hits,

        "cache_hit_rate": cache_hit_rate,
    }


# ============================================================
# Failure Analysis
# ============================================================

def build_failure_record(
    query,
    expected,
    actual,
    reason,
    category="unknown"
):
    """
    Create a standardized failure-analysis record.
    """

    return {
        "query": query,

        "expected": expected,

        "actual": actual,

        "reason": reason,

        "category": category,
    }


def summarize_failures(
    failures
):
    """
    Summarize failure cases by category.
    """

    if not failures:

        return {
            "total_failures": 0,
            "by_category": {},
        }

    category_counts = {}

    for failure in failures:

        category = failure.get(
            "category",
            "unknown"
        )

        category_counts[category] = (
            category_counts.get(
                category,
                0
            )
            + 1
        )

    return {
        "total_failures": len(
            failures
        ),

        "by_category": (
            category_counts
        ),
    }


# ============================================================
# Full Evaluation Report
# ============================================================

def build_evaluation_report(
    retrieval=None,
    grounding=None,
    answer_quality=None,
    recommendation=None,
    latency=None,
    cost=None,
    failures=None
):
    """
    Combine all evaluation components into one report.
    """

    return {
        "retrieval": retrieval or {},
        "grounding": grounding or {},
        "answer_quality": (
            answer_quality or {}
        ),
        "recommendation": (
            recommendation or {}
        ),
        "latency": latency or {},
        "cost": cost or {},
        "failure_analysis": (
            summarize_failures(
                failures or []
            )
        ),
    }


# ============================================================
# Save Evaluation Report
# ============================================================

def save_evaluation_report(
    report,
    path
):
    """
    Save evaluation report as JSON.
    """

    import json

    path = str(path)

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            report,
            file,
            ensure_ascii=False,
            indent=2,
            default=str
        )

    return path
