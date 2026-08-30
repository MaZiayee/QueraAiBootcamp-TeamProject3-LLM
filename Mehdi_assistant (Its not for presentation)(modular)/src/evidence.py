import ast
import math
from collections import Counter

import pandas as pd

from src.utils import normalize_persian_text


# ============================================================
# Basic Helpers
# ============================================================

def safe_value(value):
    """
    Convert pandas missing values to None.
    """

    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    return value


def safe_float(value, default=None):
    """
    Safely convert a value to float.
    """

    value = safe_value(value)

    if value is None:
        return default

    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_list_value(value):
    """
    Convert list-like values into Python lists.

    Handles:
        actual lists
        string representations of lists
        comma-separated strings
        None / NA
    """

    value = safe_value(value)

    if value is None:
        return []

    if isinstance(value, list):
        return [
            str(item).strip()
            for item in value
            if str(item).strip()
        ]

    text = str(value).strip()

    if not text:
        return []

    # Example:
    # "['بدون چربی', 'سبک']"
    if (
        text.startswith("[")
        and text.endswith("]")
    ):
        try:
            parsed = ast.literal_eval(text)

            if isinstance(parsed, list):
                return [
                    str(item).strip()
                    for item in parsed
                    if str(item).strip()
                ]

        except (ValueError, SyntaxError):
            pass

    # Fallback
    return [
        item.strip()
        for item in text.split(",")
        if item.strip()
    ]


# ============================================================
# Product Evidence
# ============================================================

def build_product_evidence(
    product
):
    """
    Extract factual product information.

    This information comes directly from the product data.
    """

    if isinstance(product, pd.Series):
        product = product.to_dict()

    evidence = {
        "type": "product",
        "id": safe_value(
            product.get("id")
        ),
        "title": safe_value(
            product.get("title_fa")
        ),
        "brand": safe_value(
            product.get("Brand")
        ),
        "category": safe_value(
            product.get("Category1")
        ),
        "sub_category": safe_value(
            product.get("Category2")
        ),
        "price": safe_float(
            product.get("Price")
        ),
        "rating": safe_float(
            product.get("mean_rate")
        ),
        "rating_count": safe_value(
            product.get("Rate_cnt")
        ),
        "recommendation_ratio": safe_float(
            product.get("rec_ratio")
        ),
        "recommendation_score": safe_float(
            product.get("rec_score")
        ),
        "rating_score": safe_float(
            product.get("rate_score")
        ),
        "review_count": safe_value(
            product.get("n_reviews")
        ),
        "buyer_count": safe_value(
            product.get("n_buyers")
        ),
    }

    return evidence


# ============================================================
# Review Evidence
# ============================================================

def build_review_evidence(
    review,
    fallback_id=None
):
    """
    Convert a retrieved review into a compact evidence object.

    The original review text is preserved.
    """

    if isinstance(review, pd.Series):
        review = review.to_dict()

    review_id = safe_value(
        review.get("id")
    )

    if review_id is None:
        review_id = fallback_id

    advantages = parse_list_value(
        review.get("advantages")
    )

    disadvantages = parse_list_value(
        review.get("disadvantages")
    )

    text = safe_value(
        review.get("body")
    )

    if text is None:
        text = safe_value(
            review.get("body_norm")
        )

    evidence = {
        "type": "review",

        "review_id": review_id,

        "product_id": safe_value(
            review.get("product_id")
        ),

        "text": (
            str(text).strip()
            if text is not None
            else ""
        ),

        "rating": safe_float(
            review.get("rate")
        ),

        "recommendation": safe_value(
            review.get(
                "recommendation_status"
            )
        ),

        "is_buyer": safe_value(
            review.get("is_buyer")
        ),

        "likes": safe_value(
            review.get("likes")
        ),

        "is_substantive": safe_value(
            review.get(
                "is_substantive"
            )
        ),

        "advantages": advantages,

        "disadvantages": disadvantages,

        "retrieval_score": safe_float(
            review.get(
                "_review_score"
            ),
            default=0.0
        ),

        "final_score": safe_float(
            review.get(
                "_final_review_score"
            ),
            default=0.0
        ),
    }

    return evidence


# ============================================================
# Review Quality
# ============================================================

def calculate_review_quality(
    review
):
    """
    Estimate evidence quality.

    Signals:

        retrieval relevance
        substantive review
        buyer status
        likes
        review length
    """

    if isinstance(review, pd.Series):
        review = review.to_dict()

    retrieval = safe_float(
        review.get(
            "_review_score"
        ),
        0.0
    )

    # Semantic similarity is the strongest signal.
    score = 0.60 * retrieval

    # --------------------------------------------------------
    # Substantive review
    # --------------------------------------------------------

    substantive = review.get(
        "is_substantive"
    )

    if substantive is True:
        score += 0.15

    # --------------------------------------------------------
    # Verified buyer
    # --------------------------------------------------------

    is_buyer = review.get(
        "is_buyer"
    )

    if is_buyer is True:
        score += 0.15

    # --------------------------------------------------------
    # Likes
    # --------------------------------------------------------

    likes = safe_float(
        review.get("likes"),
        0.0
    )

    if likes > 0:
        score += min(
            0.05,
            math.log1p(likes) / 100
        )

    # --------------------------------------------------------
    # Review length
    # --------------------------------------------------------

    text = str(
        review.get(
            "body",
            ""
        )
    )

    text_length = len(
        normalize_persian_text(text)
    )

    if text_length >= 50:
        score += 0.05

    elif text_length >= 20:
        score += 0.025

    return min(
        score,
        1.0
    )


# ============================================================
# Evidence Deduplication
# ============================================================

def deduplicate_evidence(
    reviews,
    max_items=None
):
    """
    Remove duplicate or almost identical review texts.
    """

    if not reviews:
        return []

    seen = set()
    result = []

    for review in reviews:

        text = normalize_persian_text(
            review.get("text", "")
        ).lower()

        # Collapse whitespace.
        text = " ".join(
            text.split()
        )

        if not text:
            key = (
                "review_id",
                review.get("review_id")
            )
        else:
            key = (
                "text",
                text
            )

        if key in seen:
            continue

        seen.add(key)
        result.append(
            review
        )

        if (
            max_items is not None
            and len(result) >= max_items
        ):
            break

    return result


# ============================================================
# Evidence Ranking
# ============================================================

def rank_review_evidence(
    reviews
):
    """
    Rank review evidence by evidence quality.
    """

    scored = []

    for review in reviews:

        quality = calculate_review_quality(
            review
        )

        item = dict(
            review
        )

        item["evidence_quality"] = round(
            quality,
            4
        )

        scored.append(
            item
        )

    scored.sort(
        key=lambda item: (
            item["evidence_quality"],
            item.get(
                "final_score",
                0
            )
        ),
        reverse=True
    )

    return scored


# ============================================================
# Evidence Diversity
# ============================================================

def select_diverse_evidence(
    reviews,
    max_items=10
):
    """
    Select evidence while avoiding over-representation
    from identical texts.

    Also tries to preserve a mixture of ratings when possible.
    """

    if not reviews:
        return []

    selected = []
    seen_texts = set()
    rating_counts = Counter()

    # First pass: highest-quality unique evidence.
    for review in reviews:

        text = normalize_persian_text(
            review.get("text", "")
        ).lower()

        text = " ".join(
            text.split()
        )

        if text in seen_texts:
            continue

        rating = review.get(
            "rating"
        )

        rating_key = (
            int(rating)
            if rating is not None
            and isinstance(
                rating,
                (int, float)
            )
            and not isinstance(
                rating,
                bool
            )
            else None
        )

        # Avoid taking too many reviews with exactly
        # the same rating when sufficient alternatives exist.
        if (
            rating_key is not None
            and rating_counts[rating_key] >= 5
            and len(reviews) > max_items
        ):
            continue

        seen_texts.add(text)
        selected.append(review)

        if rating_key is not None:
            rating_counts[rating_key] += 1

        if len(selected) >= max_items:
            break

    # Fallback in case diversity filtering was too strict.
    if len(selected) < min(
        max_items,
        len(reviews)
    ):

        selected_ids = {
            review.get("review_id")
            for review in selected
        }

        for review in reviews:

            if review.get(
                "review_id"
            ) in selected_ids:
                continue

            selected.append(
                review
            )

            if len(selected) >= max_items:
                break

    return selected


# ============================================================
# Review Statistics
# ============================================================

def calculate_review_statistics(
    reviews
):
    """
    Calculate basic statistics over retrieved reviews.

    These are descriptive statistics only.
    """

    if not reviews:

        return {
            "count": 0,
            "average_rating": None,
            "recommended_count": 0,
            "not_recommended_count": 0,
            "buyer_count": 0,
            "substantive_count": 0,
        }

    ratings = []

    recommended_count = 0
    not_recommended_count = 0
    buyer_count = 0
    substantive_count = 0

    for review in reviews:

        rating = safe_float(
            review.get("rating")
        )

        if rating is not None:
            ratings.append(
                rating
            )

        recommendation = normalize_persian_text(
            str(
                review.get(
                    "recommendation",
                    ""
                )
            )
        ).lower()

        if recommendation in {
            "recommended",
            "recommend",
            "بله",
            "true",
            "1",
        }:
            recommended_count += 1

        elif recommendation in {
            "not_recommended",
            "not recommended",
            "خیر",
            "false",
            "0",
        }:
            not_recommended_count += 1

        if review.get(
            "is_buyer"
        ) is True:
            buyer_count += 1

        if review.get(
            "is_substantive"
        ) is True:
            substantive_count += 1

    average_rating = (
        sum(ratings) / len(ratings)
        if ratings
        else None
    )

    return {
        "count": len(reviews),

        "average_rating": (
            round(
                average_rating,
                3
            )
            if average_rating is not None
            else None
        ),

        "recommended_count": (
            recommended_count
        ),

        "not_recommended_count": (
            not_recommended_count
        ),

        "buyer_count": buyer_count,

        "substantive_count": (
            substantive_count
        ),
    }


# ============================================================
# Main Evidence Builder
# ============================================================

def build_evidence(
    product=None,
    reviews=None,
    max_reviews=8
):
    """
    Build the complete evidence package.

    Output:

        {
            "product": {...},
            "reviews": [...],
            "review_statistics": {...}
        }

    Important:
    This function does NOT generate any answer.
    It only prepares factual evidence.
    """

    product_evidence = None

    if product is not None:

        product_evidence = (
            build_product_evidence(
                product
            )
        )

    review_evidence = []

    if reviews is not None:

        if isinstance(
            reviews,
            pd.DataFrame
        ):

            for index, row in reviews.iterrows():

                review_evidence.append(
                    build_review_evidence(
                        row,
                        fallback_id=index
                    )
                )

        else:

            for item in reviews:

                if isinstance(
                    item,
                    dict
                ):

                    review_evidence.append(
                        build_review_evidence(
                            item
                        )
                    )

                elif isinstance(
                    item,
                    pd.Series
                ):

                    review_evidence.append(
                        build_review_evidence(
                            item
                        )
                    )

    # --------------------------------------------------------
    # Deduplicate
    # --------------------------------------------------------

    review_evidence = deduplicate_evidence(
        review_evidence
    )

    # --------------------------------------------------------
    # Quality ranking
    # --------------------------------------------------------

    review_evidence = rank_review_evidence(
        review_evidence
    )

    # --------------------------------------------------------
    # Diversity
    # --------------------------------------------------------

    review_evidence = select_diverse_evidence(
        review_evidence,
        max_items=max_reviews
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    statistics = calculate_review_statistics(
        review_evidence
    )

    return {
        "product": product_evidence,
        "reviews": review_evidence,
        "review_statistics": statistics,
    }


# ============================================================
# LLM Context Formatter
# ============================================================

def format_evidence_for_llm(
    evidence
):
    """
    Convert evidence into compact text for the LLM.

    The LLM will receive:
        product facts
        numbered review evidence
        review statistics

    Every review has an ID so the answer can cite it.
    """

    lines = []

    # --------------------------------------------------------
    # Product
    # --------------------------------------------------------

    product = evidence.get(
        "product"
    )

    if product:

        lines.append(
            "PRODUCT FACTS:"
        )

        fields = [
            ("ID", "id"),
            ("Title", "title"),
            ("Brand", "brand"),
            ("Category", "category"),
            ("Sub-category", "sub_category"),
            ("Price", "price"),
            ("Rating", "rating"),
            ("Rating count", "rating_count"),
            ("Recommendation ratio", "recommendation_ratio"),
            ("Review count", "review_count"),
            ("Buyer count", "buyer_count"),
        ]

        for label, key in fields:

            value = product.get(
                key
            )

            if value is not None:

                lines.append(
                    f"- {label}: {value}"
                )

    # --------------------------------------------------------
    # Review statistics
    # --------------------------------------------------------

    statistics = evidence.get(
        "review_statistics",
        {}
    )

    lines.append(
        "\nREVIEW STATISTICS:"
    )

    for key, value in statistics.items():

        if value is not None:

            lines.append(
                f"- {key}: {value}"
            )

    # --------------------------------------------------------
    # Reviews
    # --------------------------------------------------------

    reviews = evidence.get(
        "reviews",
        []
    )

    lines.append(
        "\nREVIEW EVIDENCE:"
    )

    for i, review in enumerate(
        reviews,
        start=1
    ):

        review_id = review.get(
            "review_id"
        )

        lines.append(
            f"\n[Evidence {i} | Review ID: {review_id}]"
        )

        text = review.get(
            "text"
        )

        if text:

            lines.append(
                f"Text: {text}"
            )

        rating = review.get(
            "rating"
        )

        if rating is not None:

            lines.append(
                f"Rating: {rating}"
            )

        recommendation = review.get(
            "recommendation"
        )

        if recommendation is not None:

            lines.append(
                f"Recommendation: {recommendation}"
            )

        advantages = review.get(
            "advantages"
        )

        if advantages:

            lines.append(
                "Advantages: "
                + ", ".join(
                    map(
                        str,
                        advantages
                    )
                )
            )

        disadvantages = review.get(
            "disadvantages"
        )

        if disadvantages:

            lines.append(
                "Disadvantages: "
                + ", ".join(
                    map(
                        str,
                        disadvantages
                    )
                )
            )

        lines.append(
            f"Evidence quality: "
            f"{review.get('evidence_quality', 0):.3f}"
        )

    return "\n".join(
        lines
    )
