import pandas as pd

from src.evidence import (
    build_product_evidence,
    build_evidence,
    format_evidence_for_llm,
)
from src.review_search import search_reviews


# ============================================================
# Product Lookup
# ============================================================

def get_products_by_ids(
    products,
    product_ids
):
    """
    Return products matching the requested IDs.
    """

    if not product_ids:
        return products.iloc[0:0].copy()

    return products.loc[
        products["id"].isin(product_ids)
    ].copy()


# ============================================================
# Product Facts
# ============================================================

def build_comparison_product_data(
    products,
    product_ids
):
    """
    Extract direct product facts for comparison.
    """

    selected = get_products_by_ids(
        products,
        product_ids
    )

    result = []

    for _, product in selected.iterrows():

        result.append(
            build_product_evidence(
                product
            )
        )

    return result


# ============================================================
# Review Retrieval
# ============================================================

def retrieve_comparison_reviews(
    product_ids,
    query,
    reviews,
    review_index,
    reviews_per_product=5
):
    """
    Retrieve review evidence separately for every product.
    """

    all_reviews = {}

    for product_id in product_ids:

        result = search_reviews(
            query=query,
            product_id=product_id,
            reviews=reviews,
            review_index=review_index,
            top_k=reviews_per_product,
        )

        all_reviews[product_id] = result

    return all_reviews


# ============================================================
# Product Review Evidence
# ============================================================

def build_product_review_evidence(
    product_id,
    review_results,
    max_reviews=5
):
    """
    Build evidence package for one product.
    """

    product_reviews = review_results.get(
        product_id
    )

    if product_reviews is None:
        return {
            "product_id": product_id,
            "reviews": [],
            "review_statistics": {
                "count": 0,
                "average_rating": None,
                "recommended_count": 0,
                "not_recommended_count": 0,
                "buyer_count": 0,
                "substantive_count": 0,
            },
        }

    evidence = build_evidence(
        product=None,
        reviews=product_reviews,
        max_reviews=max_reviews,
    )

    return {
        "product_id": product_id,
        "reviews": evidence["reviews"],
        "review_statistics": evidence[
            "review_statistics"
        ],
    }


# ============================================================
# Complete Comparison Evidence
# ============================================================

def build_comparison_evidence(
    products,
    reviews,
    review_index,
    product_ids,
    query="مقایسه این محصولات بر اساس قیمت، کیفیت و نظر کاربران",
    reviews_per_product=5,
):
    """
    Build complete evidence for comparing products.

    Output contains:

        product facts
        review evidence
        review statistics
    """

    product_ids = list(
        dict.fromkeys(
            product_ids
        )
    )

    # --------------------------------------------------------
    # Product facts
    # --------------------------------------------------------

    product_data = (
        build_comparison_product_data(
            products,
            product_ids
        )
    )

    found_ids = {
        item["id"]
        for item in product_data
    }

    missing_ids = [
        product_id
        for product_id in product_ids
        if product_id not in found_ids
    ]

    # --------------------------------------------------------
    # Reviews
    # --------------------------------------------------------

    review_results = (
        retrieve_comparison_reviews(
            product_ids=product_ids,
            query=query,
            reviews=reviews,
            review_index=review_index,
            reviews_per_product=reviews_per_product,
        )
    )

    review_evidence = {}

    for product_id in product_ids:

        review_evidence[product_id] = (
            build_product_review_evidence(
                product_id,
                review_results,
                max_reviews=reviews_per_product,
            )
        )

    return {
        "product_ids": product_ids,
        "products": product_data,
        "reviews": review_evidence,
        "missing_product_ids": missing_ids,
        "query": query,
    }


# ============================================================
# Comparison Table
# ============================================================

def build_comparison_table(
    products,
    product_ids
):
    """
    Build a simple DataFrame suitable for display.
    """

    selected = get_products_by_ids(
        products,
        product_ids
    )

    if selected.empty:
        return pd.DataFrame()

    columns = [
        "id",
        "title_fa",
        "Brand",
        "Category1",
        "Category2",
        "Price",
        "Rate",
        "Rate_cnt",
        "mean_rate",
        "rec_ratio",
        "n_reviews",
        "n_buyers",
    ]

    available = [
        column
        for column in columns
        if column in selected.columns
    ]

    return selected[
        available
    ].copy()


# ============================================================
# Comparison Summary Statistics
# ============================================================

def calculate_comparison_scores(
    comparison_evidence
):
    """
    Create normalized comparison scores.

    These scores are descriptive and are NOT themselves
    a final recommendation.
    """

    rows = []

    for product in comparison_evidence[
        "products"
    ]:

        product_id = product["id"]

        review_info = comparison_evidence[
            "reviews"
        ].get(
            product_id,
            {}
        )

        stats = review_info.get(
            "review_statistics",
            {}
        )

        recommendation_ratio = (
            product.get(
                "recommendation_ratio"
            )
        )

        rating = product.get(
            "rating"
        )

        price = product.get(
            "price"
        )

        rows.append({
            "product_id": product_id,

            "title": product.get(
                "title"
            ),

            "price": price,

            "rating": rating,

            "recommendation_ratio": (
                recommendation_ratio
            ),

            "review_count": product.get(
                "review_count"
            ),

            "buyer_count": product.get(
                "buyer_count"
            ),

            "retrieved_review_count": (
                stats.get(
                    "count",
                    0
                )
            ),

            "retrieved_average_rating": (
                stats.get(
                    "average_rating"
                )
            ),

            "retrieved_recommended": (
                stats.get(
                    "recommended_count",
                    0
                )
            ),

            "retrieved_not_recommended": (
                stats.get(
                    "not_recommended_count",
                    0
                )
            ),
        })

    return pd.DataFrame(
        rows
    )


# ============================================================
# Comparison LLM Context
# ============================================================

def format_comparison_for_llm(
    comparison_evidence
):
    """
    Convert comparison evidence into compact text.

    The text explicitly separates:

        PRODUCT FACTS
        REVIEW EVIDENCE
        REVIEW STATISTICS
    """

    lines = []

    products = comparison_evidence[
        "products"
    ]

    reviews = comparison_evidence[
        "reviews"
    ]

    # --------------------------------------------------------
    # Product facts
    # --------------------------------------------------------

    lines.append(
        "=== PRODUCT FACTS ==="
    )

    for i, product in enumerate(
        products,
        start=1
    ):

        lines.append(
            f"\nPRODUCT {i}"
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
            (
                "Recommendation ratio",
                "recommendation_ratio"
            ),
            (
                "Review count",
                "review_count"
            ),
            (
                "Buyer count",
                "buyer_count"
            ),
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
    # Reviews
    # --------------------------------------------------------

    lines.append(
        "\n=== REVIEW EVIDENCE ==="
    )

    for product_id, review_info in reviews.items():

        lines.append(
            f"\nPRODUCT ID: {product_id}"
        )

        statistics = review_info.get(
            "review_statistics",
            {}
        )

        lines.append(
            "Statistics:"
        )

        for key, value in statistics.items():

            if value is not None:

                lines.append(
                    f"- {key}: {value}"
                )

        product_reviews = review_info.get(
            "reviews",
            []
        )

        for review in product_reviews:

            review_id = review.get(
                "review_id"
            )

            lines.append(
                f"\n[Review ID: {review_id}]"
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
                "Evidence quality: "
                f"{review.get('evidence_quality', 0):.3f}"
            )

    # --------------------------------------------------------
    # Missing products
    # --------------------------------------------------------

    missing = comparison_evidence.get(
        "missing_product_ids",
        []
    )

    if missing:

        lines.append(
            "\n=== MISSING PRODUCTS ==="
        )

        lines.append(
            ", ".join(
                map(
                    str,
                    missing
                )
            )
        )

    return "\n".join(
        lines
    )


# ============================================================
# Main Comparison Function
# ============================================================

def compare_products(
    products,
    reviews,
    review_index,
    product_ids,
    query="مقایسه این محصولات بر اساس قیمت، کیفیت و نظر کاربران",
    reviews_per_product=5,
):
    """
    Main comparison function.

    Returns:

        evidence
        comparison_table
        comparison_scores
        llm_context
    """

    if len(product_ids) < 2:

        raise ValueError(
            "At least two product IDs are required."
        )

    evidence = build_comparison_evidence(
        products=products,
        reviews=reviews,
        review_index=review_index,
        product_ids=product_ids,
        query=query,
        reviews_per_product=reviews_per_product,
    )

    table = build_comparison_table(
        products,
        product_ids
    )

    scores = calculate_comparison_scores(
        evidence
    )

    llm_context = format_comparison_for_llm(
        evidence
    )

    return {
        "evidence": evidence,
        "table": table,
        "scores": scores,
        "llm_context": llm_context,
    }
