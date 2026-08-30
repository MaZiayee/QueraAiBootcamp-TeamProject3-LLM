import ast
import re
import numpy as np
import pandas as pd

from src.utils import normalize_persian_text


# ============================================================
# Constants
# ============================================================

PRODUCT_ID = "id"
TITLE = "title_fa"
CATEGORY_1 = "Category1"
CATEGORY_2 = "Category2"
BRAND = "Brand"
PRICE = "Price"

RATING = "mean_rate"
REC_RATIO = "rec_ratio"

REVIEW_COUNT = "n_reviews"
BUYER_COUNT = "n_buyers"

PROS_COUNT = "n_pros"
CONS_COUNT = "n_cons"


# ============================================================
# Helpers
# ============================================================

def safe_numeric(
    df,
    column,
    default=np.nan
):
    """
    Safely convert a column to numeric.
    """

    if column not in df.columns:

        return pd.Series(
            default,
            index=df.index,
            dtype=float
        )

    return pd.to_numeric(
        df[column],
        errors="coerce"
    )


def parse_list_value(value):
    """
    Convert list-like values into a Python list.
    """

    if value is None:
        return []

    try:
        if pd.isna(value):
            return []
    except (TypeError, ValueError):
        pass

    if isinstance(value, list):
        return [
            str(item).strip()
            for item in value
            if str(item).strip()
        ]

    text = str(value).strip()

    if not text:
        return []

    if (
        text.startswith("[")
        and text.endswith("]")
    ):

        try:

            parsed = ast.literal_eval(
                text
            )

            if isinstance(
                parsed,
                list
            ):

                return [
                    str(item).strip()
                    for item in parsed
                    if str(item).strip()
                ]

        except (
            ValueError,
            SyntaxError
        ):
            pass

    return [
        item.strip()
        for item in text.split(",")
        if item.strip()
    ]


# ============================================================
# Recommendation Statistics
# ============================================================

def calculate_recommendation_statistics(
    products
):
    """
    Calculate overall recommendation statistics.
    """

    rec = safe_numeric(
        products,
        REC_RATIO
    )

    rating = safe_numeric(
        products,
        RATING
    )

    review_count = safe_numeric(
        products,
        REVIEW_COUNT,
        default=0
    )

    buyer_count = safe_numeric(
        products,
        BUYER_COUNT,
        default=0
    )

    return {
        "product_count": len(products),

        "mean_recommendation_ratio": (
            round(
                rec.mean(),
                4
            )
            if rec.notna().any()
            else None
        ),

        "median_recommendation_ratio": (
            round(
                rec.median(),
                4
            )
            if rec.notna().any()
            else None
        ),

        "mean_rating": (
            round(
                rating.mean(),
                4
            )
            if rating.notna().any()
            else None
        ),

        "median_rating": (
            round(
                rating.median(),
                4
            )
            if rating.notna().any()
            else None
        ),

        "total_reviews": int(
            review_count.fillna(0).sum()
        ),

        "total_buyers": int(
            buyer_count.fillna(0).sum()
        ),
    }


# ============================================================
# Category Analysis
# ============================================================

def analyze_categories(
    products,
    min_products=10
):
    """
    Analyze product performance by Category1.
    """

    if CATEGORY_1 not in products.columns:

        return pd.DataFrame()

    df = products.copy()

    df[REC_RATIO] = safe_numeric(
        df,
        REC_RATIO
    )

    df[RATING] = safe_numeric(
        df,
        RATING
    )

    df[REVIEW_COUNT] = safe_numeric(
        df,
        REVIEW_COUNT,
        default=0
    )

    grouped = (
        df.groupby(
            CATEGORY_1,
            dropna=False
        )
        .agg(
            product_count=(
                PRODUCT_ID,
                "count"
            ),
            average_rating=(
                RATING,
                "mean"
            ),
            average_recommendation=(
                REC_RATIO,
                "mean"
            ),
            total_reviews=(
                REVIEW_COUNT,
                "sum"
            ),
            median_price=(
                PRICE,
                "median"
            ),
        )
        .reset_index()
    )

    grouped = grouped[
        grouped["product_count"]
        >= min_products
    ]

    grouped[
        "average_rating"
    ] = grouped[
        "average_rating"
    ].round(3)

    grouped[
        "average_recommendation"
    ] = grouped[
        "average_recommendation"
    ].round(3)

    grouped[
        "median_price"
    ] = grouped[
        "median_price"
    ].round(0)

    return grouped.sort_values(
        "product_count",
        ascending=False
    ).reset_index(
        drop=True
    )


# ============================================================
# Brand Analysis
# ============================================================

def analyze_brands(
    products,
    min_products=10
):
    """
    Analyze feedback by brand.
    """

    if BRAND not in products.columns:

        return pd.DataFrame()

    df = products.copy()

    df[REC_RATIO] = safe_numeric(
        df,
        REC_RATIO
    )

    df[RATING] = safe_numeric(
        df,
        RATING
    )

    df[REVIEW_COUNT] = safe_numeric(
        df,
        REVIEW_COUNT,
        default=0
    )

    grouped = (
        df.groupby(
            BRAND,
            dropna=False
        )
        .agg(
            product_count=(
                PRODUCT_ID,
                "count"
            ),
            average_rating=(
                RATING,
                "mean"
            ),
            average_recommendation=(
                REC_RATIO,
                "mean"
            ),
            total_reviews=(
                REVIEW_COUNT,
                "sum"
            ),
        )
        .reset_index()
    )

    grouped = grouped[
        grouped["product_count"]
        >= min_products
    ]

    grouped[
        "average_rating"
    ] = grouped[
        "average_rating"
    ].round(3)

    grouped[
        "average_recommendation"
    ] = grouped[
        "average_recommendation"
    ].round(3)

    return grouped.sort_values(
        "total_reviews",
        ascending=False
    ).reset_index(
        drop=True
    )


# ============================================================
# Common Advantages
# ============================================================

def extract_common_features(
    reviews,
    column,
    top_k=20,
    min_frequency=2
):
    """
    Extract meaningful frequently mentioned features.

    Empty / no-problem expressions are ignored.
    """

    if column not in reviews.columns:

        return pd.DataFrame(
            columns=[
                "feature",
                "count"
            ]
        )

    # Expressions that should NOT be interpreted as
    # complaints or advantages.
    ignored_terms = {
        "ندارد",
        "نداره",
        "نداشت",
        "ندیدم",
        "هیچی",
        "هیچ",
        "ندید",
        "نداشتم",
        "موردی نیست",
        "مشکلی نیست",
        "مشکل ندارد",
        "مشکل نداره",
        "ایرادی نیست",
        "ایراد ندارد",
        "ایرادی نداره",
        "نداشته",
        "هیچ مشکلی",
    }

    counter = {}

    for value in reviews[column]:

        items = parse_list_value(
            value
        )

        for item in items:

            item = normalize_persian_text(
                item
            ).strip()

            if not item:
                continue

            key = item.lower()

            # Ignore empty/no-problem statements
            if key in ignored_terms:
                continue

            # Ignore very short generic tokens
            if len(key) < 2:
                continue

            counter[key] = (
                counter.get(
                    key,
                    0
                )
                + 1
            )

    rows = [
        {
            "feature": feature,
            "count": count
        }
        for feature, count in counter.items()
        if count >= min_frequency
    ]

    if not rows:

        return pd.DataFrame(
            columns=[
                "feature",
                "count"
            ]
        )

    result = pd.DataFrame(
        rows
    )

    return result.sort_values(
        "count",
        ascending=False
    ).head(
        top_k
    ).reset_index(
        drop=True
    )

def normalize_review_ratings(
    reviews
):
    """
    Normalize review ratings to the nearest 0.5 step.

    Example:
        4.25 -> 4.0
        4.30 -> 4.5
        3.14 -> 3.0

    This is used only for reporting/distribution,
    not for changing the original data.
    """

    if "rate" not in reviews.columns:
        return pd.Series(
            dtype="float64"
        )

    ratings = pd.to_numeric(
        reviews["rate"],
        errors="coerce"
    ).dropna()

    if ratings.empty:
        return ratings

    return (
        np.round(
            ratings * 2
        ) / 2
    )


def find_common_positive_features(
    reviews,
    top_k=20,
    min_frequency=2
):
    """
    Most common positive features from advantages.
    """

    return extract_common_features(
        reviews,
        column="advantages",
        top_k=top_k,
        min_frequency=min_frequency
    )


def find_common_negative_features(
    reviews,
    top_k=20,
    min_frequency=2
):
    """
    Most common complaints/problems from disadvantages.
    """

    return extract_common_features(
        reviews,
        column="disadvantages",
        top_k=top_k,
        min_frequency=min_frequency
    )


# ============================================================
# Category Complaint Analysis
# ============================================================

def analyze_category_complaints(
    products,
    reviews,
    category,
    top_k=20
):
    """
    Find common complaints for a specific category.
    """

    if CATEGORY_1 not in products.columns:

        return pd.DataFrame()

    product_ids = set(
        products.loc[
            products[CATEGORY_1]
            == category,
            PRODUCT_ID
        ]
    )

    if not product_ids:

        return pd.DataFrame()

    if "product_id" not in reviews.columns:

        return pd.DataFrame()

    category_reviews = reviews[
        reviews["product_id"].isin(
            product_ids
        )
    ].copy()

    return find_common_negative_features(
        category_reviews,
        top_k=top_k
    )


# ============================================================
# Category Positive Features
# ============================================================

def analyze_category_positive_features(
    products,
    reviews,
    category,
    top_k=20
):
    """
    Find commonly praised features for a category.
    """

    if CATEGORY_1 not in products.columns:

        return pd.DataFrame()

    product_ids = set(
        products.loc[
            products[CATEGORY_1]
            == category,
            PRODUCT_ID
        ]
    )

    if not product_ids:

        return pd.DataFrame()

    if "product_id" not in reviews.columns:

        return pd.DataFrame()

    category_reviews = reviews[
        reviews["product_id"].isin(
            product_ids
        )
    ].copy()

    return find_common_positive_features(
        category_reviews,
        top_k=top_k
    )


# ============================================================
# Low Recommendation Products
# ============================================================

def find_low_recommendation_products(
    products,
    min_reviews=10,
    max_recommendation=0.50,
    top_k=20
):
    """
    Find products with enough review evidence but low
    recommendation ratio.
    """

    df = products.copy()

    df[REC_RATIO] = safe_numeric(
        df,
        REC_RATIO
    )

    df[REVIEW_COUNT] = safe_numeric(
        df,
        REVIEW_COUNT,
        default=0
    )

    mask = (
        df[REC_RATIO].notna()
        & (
            df[REC_RATIO]
            <= max_recommendation
        )
        & (
            df[REVIEW_COUNT]
            >= min_reviews
        )
    )

    result = df.loc[
        mask
    ].copy()

    if result.empty:
        return result

    columns = [
        PRODUCT_ID,
        TITLE,
        BRAND,
        CATEGORY_1,
        PRICE,
        RATING,
        REC_RATIO,
        REVIEW_COUNT,
        BUYER_COUNT,
    ]

    columns = [
        column
        for column in columns
        if column in result.columns
    ]

    return result[
        columns
    ].sort_values(
        REC_RATIO,
        ascending=True
    ).head(
        top_k
    )


# ============================================================
# High Rating / Low Recommendation
# ============================================================

def find_high_rating_low_recommendation(
    products,
    min_rating=4.0,
    max_recommendation=0.60,
    min_reviews=10,
    top_k=20
):
    """
    Find products that have high ratings but relatively
    low recommendation ratios.

    This can reveal a useful business signal:
    users rate the product well, but recommendation behavior
    is weaker.
    """

    df = products.copy()

    df[RATING] = safe_numeric(
        df,
        RATING
    )

    df[REC_RATIO] = safe_numeric(
        df,
        REC_RATIO
    )

    df[REVIEW_COUNT] = safe_numeric(
        df,
        REVIEW_COUNT,
        default=0
    )

    mask = (
        (df[RATING] >= min_rating)
        & (
            df[REC_RATIO]
            <= max_recommendation
        )
        & (
            df[REVIEW_COUNT]
            >= min_reviews
        )
    )

    result = df.loc[
        mask
    ].copy()

    columns = [
        PRODUCT_ID,
        TITLE,
        BRAND,
        CATEGORY_1,
        CATEGORY_2,
        PRICE,
        RATING,
        REC_RATIO,
        REVIEW_COUNT,
        BUYER_COUNT,
    ]

    columns = [
        column
        for column in columns
        if column in result.columns
    ]

    return result[
        columns
    ].sort_values(
        REC_RATIO,
        ascending=True
    ).head(
        top_k
    )


# ============================================================
# Products With High Review Volume
# ============================================================

def find_most_reviewed_products(
    products,
    top_k=20
):
    """
    Find products with the largest amount of review data.
    """

    if REVIEW_COUNT not in products.columns:

        return pd.DataFrame()

    df = products.copy()

    df[REVIEW_COUNT] = safe_numeric(
        df,
        REVIEW_COUNT,
        default=0
    )

    columns = [
        PRODUCT_ID,
        TITLE,
        BRAND,
        CATEGORY_1,
        PRICE,
        RATING,
        REC_RATIO,
        REVIEW_COUNT,
        BUYER_COUNT,
    ]

    columns = [
        column
        for column in columns
        if column in df.columns
    ]

    return df[
        columns
    ].sort_values(
        REVIEW_COUNT,
        ascending=False
    ).head(
        top_k
    )


# ============================================================
# Review Distribution
# ============================================================

def review_rating_distribution(
    reviews
):
    """
    Distribution of individual review ratings.
    """

    if "rate" not in reviews.columns:

        return pd.DataFrame(
            columns=[
                "rating",
                "count",
                "percentage"
            ]
        )

    ratings = normalize_review_ratings(
    reviews
)

    if ratings.empty:

        return pd.DataFrame(
            columns=[
                "rating",
                "count",
                "percentage"
            ]
        )

    result = (
        ratings.value_counts(
            sort=False
        )
        .sort_index()
        .rename_axis(
            "rating"
        )
        .reset_index(
            name="count"
        )
    )

    result["percentage"] = (
        result["count"]
        / result["count"].sum()
        * 100
    ).round(2)

    return result


# ============================================================
# Recommendation Distribution
# ============================================================

def recommendation_status_distribution(
    reviews
):
    """
    Distribution of recommendation statuses.
    """

    column = "recommendation_status"

    if column not in reviews.columns:

        return pd.DataFrame(
            columns=[
                "status",
                "count",
                "percentage"
            ]
        )

    result = (
        reviews[column]
        .fillna("unknown")
        .astype(str)
        .value_counts()
        .rename_axis(
            "status"
        )
        .reset_index(
            name="count"
        )
    )

    result["percentage"] = (
        result["count"]
        / result["count"].sum()
        * 100
    ).round(2)

    return result


# ============================================================
# Category Dashboard
# ============================================================

def build_category_dashboard(
    products,
    reviews,
    category,
    min_review_count=1
):
    """
    Build a complete analytical summary for a category.
    """

    if CATEGORY_1 not in products.columns:

        raise ValueError(
            "Category1 column is required."
        )

    category_products = products.loc[
        products[CATEGORY_1]
        == category
    ].copy()

    if category_products.empty:

        return {
            "category": category,
            "product_count": 0,
            "product_statistics": {},
            "common_positive_features": pd.DataFrame(),
            "common_negative_features": pd.DataFrame(),
            "rating_distribution": pd.DataFrame(),
            "recommendation_distribution": pd.DataFrame(),
        }

    product_ids = set(
        category_products[
            PRODUCT_ID
        ]
    )

    if "product_id" in reviews.columns:

        category_reviews = reviews[
            reviews[
                "product_id"
            ].isin(product_ids)
        ].copy()

    else:

        category_reviews = reviews.iloc[
            0:0
        ].copy()

    result = {
        "category": category,

        "product_count": len(
            category_products
        ),

        "review_count": len(
            category_reviews
        ),

        "product_statistics": (
            calculate_recommendation_statistics(
                category_products
            )
        ),

        "common_positive_features": (
            find_common_positive_features(
                category_reviews
            )
        ),

        "common_negative_features": (
            find_common_negative_features(
                category_reviews
            )
        ),

        "rating_distribution": (
            review_rating_distribution(
                category_reviews
            )
        ),

        "recommendation_distribution": (
            recommendation_status_distribution(
                category_reviews
            )
        ),
    }

    return result


# ============================================================
# Manager Query Router
# ============================================================

def detect_analytics_query(
    query
):
    """
    Detect manager analytics intent.

    Returns:

        complaints
        positive_features
        low_recommendation
        category
        brand
        overview
    """

    query = normalize_persian_text(
        query
    ).lower()

    if any(
        phrase in query
        for phrase in [
            "شکایت",
            "ایراد پرتکرار",
            "مشکل پرتکرار",
            "معایب",
            "نارضایتی",
        ]
    ):
        return "complaints"

    if any(
        phrase in query
        for phrase in [
            "نقاط قوت",
            "ویژگی مثبت",
            "مزایا",
            "بیشتر راضی",
            "ویژگی هایی که راضی",
        ]
    ):
        return "positive_features"

    if any(
        phrase in query
        for phrase in [
            "پیشنهاد خرید پایین",
            "پیشنهاد خرید کم",
            "رضایت کم",
            "recommendation پایین",
            "recommendation کم",
        ]
    ):
        return "low_recommendation"

    if any(
        phrase in query
        for phrase in [
            "برند",
            "برندها",
        ]
    ):
        return "brand"

    if any(
        phrase in query
        for phrase in [
            "دسته",
            "دسته بندی",
            "دسته‌بندی",
        ]
    ):
        return "category"

    return "overview"


# ============================================================
# General Overview
# ============================================================

def build_overview(
    products,
    reviews
):
    """
    Build a general manager-level overview.
    """

    return {
        "products": (
            calculate_recommendation_statistics(
                products
            )
        ),

        "reviews": {
            "review_count": len(
                reviews
            ),

            "rating_distribution": (
                review_rating_distribution(
                    reviews
                )
                .to_dict(
                    orient="records"
                )
            ),

            "recommendation_distribution": (
                recommendation_status_distribution(
                    reviews
                )
                .to_dict(
                    orient="records"
                )
            ),
        },

        "top_reviewed_products": (
            find_most_reviewed_products(
                products,
                top_k=10
            )
            .to_dict(
                orient="records"
            )
        ),
    }


# ============================================================
# Main Analytics Function
# ============================================================

def run_analytics(
    products,
    reviews,
    query=None,
    category=None,
    top_k=20
):
    """
    Main entry point for manager analytics.

    This function performs deterministic data analysis.
    LLM can later convert the output into natural language.
    """

    if query:
        intent = detect_analytics_query(
            query
        )
    else:
        intent = "overview"

    # --------------------------------------------------------
    # Complaints
    # --------------------------------------------------------

    if intent == "complaints":

        if category:

            result = analyze_category_complaints(
                products,
                reviews,
                category=category,
                top_k=top_k
            )

            return {
                "intent": intent,
                "category": category,
                "result": result,
            }

        result = find_common_negative_features(
            reviews,
            top_k=top_k
        )

        return {
            "intent": intent,
            "result": result,
        }

    # --------------------------------------------------------
    # Positive features
    # --------------------------------------------------------

    if intent == "positive_features":

        if category:

            result = analyze_category_positive_features(
                products,
                reviews,
                category=category,
                top_k=top_k
            )

            return {
                "intent": intent,
                "category": category,
                "result": result,
            }

        result = find_common_positive_features(
            reviews,
            top_k=top_k
        )

        return {
            "intent": intent,
            "result": result,
        }

    # --------------------------------------------------------
    # Low recommendation
    # --------------------------------------------------------

    if intent == "low_recommendation":

        result = find_high_rating_low_recommendation(
            products,
            top_k=top_k
        )

        return {
            "intent": intent,
            "result": result,
        }

    # --------------------------------------------------------
    # Brand
    # --------------------------------------------------------

    if intent == "brand":

        result = analyze_brands(
            products
        )

        return {
            "intent": intent,
            "result": result,
        }

    # --------------------------------------------------------
    # Category
    # --------------------------------------------------------

    if intent == "category":

        result = analyze_categories(
            products
        )

        return {
            "intent": intent,
            "result": result,
        }

    # --------------------------------------------------------
    # Overview
    # --------------------------------------------------------

    return {
        "intent": "overview",
        "result": build_overview(
            products,
            reviews
        ),
    }
