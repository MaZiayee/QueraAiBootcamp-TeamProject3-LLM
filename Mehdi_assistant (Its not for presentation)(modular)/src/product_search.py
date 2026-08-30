import re
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer

from src.utils import normalize_persian_text


# ============================================================
# Column Names
# ============================================================

PRODUCT_ID = "id"
TITLE = "title_fa"
BRAND = "Brand"
CATEGORY_1 = "Category1"
CATEGORY_2 = "Category2"
PRICE = "Price"
RATING = "mean_rate"
SEARCH_TEXT = "search_text"
REC_RATIO = "rec_ratio"


# ============================================================
# Validation
# ============================================================

def validate_product_columns(products):
    required = [
        PRODUCT_ID,
        TITLE,
        PRICE,
    ]

    missing = [
        column
        for column in required
        if column not in products.columns
    ]

    if missing:
        raise ValueError(
            f"Missing product columns: {missing}"
        )

    return True


# ============================================================
# Numeric
# ============================================================

def numeric_series(
    df,
    column,
    default=np.nan
):
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


# ============================================================
# Search Text
# ============================================================

def get_search_text(products):
    """
    Use prepared search_text when available.
    """

    if SEARCH_TEXT in products.columns:

        return (
            products[SEARCH_TEXT]
            .fillna("")
            .astype(str)
            .apply(normalize_persian_text)
        )

    columns = [
        TITLE,
        BRAND,
        CATEGORY_1,
        CATEGORY_2,
    ]

    columns = [
        column
        for column in columns
        if column in products.columns
    ]

    return (
        products[columns]
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
        .apply(normalize_persian_text)
    )


# ============================================================
# Product Index
# ============================================================

def build_product_index(
    products,
    max_features=100_000,
    min_df=2
):
    """
    Build reusable TF-IDF index.

    Important:
    The TF-IDF matrix is built ONCE.
    It is never rebuilt per query.
    """

    validate_product_columns(
        products
    )

    text = get_search_text(
        products
    )

    vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=min_df,
        max_features=max_features,
        sublinear_tf=True,
    )

    matrix = vectorizer.fit_transform(
        text
    )

    return {
        "vectorizer": vectorizer,
        "matrix": matrix,
        "text": text,
    }


# ============================================================
# Text Match
# ============================================================

def text_contains(
    series,
    text
):
    """
    Normalized substring matching.
    """

    text = normalize_persian_text(
        str(text)
    ).strip().lower()

    if not text:
        return pd.Series(
            False,
            index=series.index
        )

    values = (
        series
        .fillna("")
        .astype(str)
        .apply(normalize_persian_text)
        .str.lower()
    )

    return values.str.contains(
        re.escape(text),
        regex=True,
        na=False
    )


# ============================================================
# Concept Filter
# ============================================================

def filter_by_concept(
    products,
    concept,
    concept_terms=None
):
    """
    Fast concept filtering using pre-normalized columns.

    Uses:
        title_norm
        cat1_norm
        cat2_norm

    No normalization is performed per query.
    """

    if not concept:
        return products.copy()

    if not concept_terms:
        concept_terms = [
            concept
        ]

    columns = [
        "title_norm",
        "cat1_norm",
        "cat2_norm",
    ]

    available = [
        column
        for column in columns
        if column in products.columns
    ]

    if not available:

        return products.iloc[
            0:0
        ].copy()

    mask = pd.Series(
        False,
        index=products.index
    )

    for term in concept_terms:

        term = normalize_persian_text(
            term
        ).strip().lower()

        if not term:
            continue

        escaped = re.escape(
            term
        )

        for column in available:

            mask |= (
                products[column]
                .fillna("")
                .astype(str)
                .str.contains(
                    escaped,
                    regex=True,
                    na=False
                )
            )

    return products.loc[
        mask
    ].copy()


# ============================================================
# Brand Filter
# ============================================================

def filter_by_brand(
    products,
    brand
):
    if not brand or BRAND not in products.columns:
        return products.copy()

    target = normalize_persian_text(
        brand
    ).strip().lower()

    values = (
        products[BRAND]
        .fillna("")
        .astype(str)
        .apply(normalize_persian_text)
        .str.strip()
        .str.lower()
    )

    return products.loc[
        values == target
    ].copy()


# ============================================================
# Price Filter
# ============================================================

def filter_by_price(
    products,
    min_price=None,
    max_price=None
):
    if PRICE not in products.columns:
        return products.copy()

    prices = numeric_series(
        products,
        PRICE
    )

    mask = pd.Series(
        True,
        index=products.index
    )

    if min_price is not None:

        mask &= (
            prices >= min_price
        )

    if max_price is not None:

        mask &= (
            prices <= max_price
        )

    return products.loc[
        mask
    ].copy()


# ============================================================
# Sub-category Filter
# ============================================================

def filter_by_sub_category(
    products,
    sub_category
):
    """
    Apply the sub-category only if an actual match exists.
    """

    if (
        not sub_category
        or CATEGORY_2 not in products.columns
    ):
        return products.copy()

    mask = text_contains(
        products[CATEGORY_2],
        sub_category
    )

    # Natural-language usage hints such as "روزمره"
    # may not exist in the dataset.
    if not mask.any():
        return products.copy()

    return products.loc[
        mask
    ].copy()


# ============================================================
# Structured Filters
# ============================================================

def filter_products(
    products,
    plan
):
    """
    Apply structured constraints.

    Order:

        price
        brand
        concept
        sub-category
    """

    validate_product_columns(
        products
    )

    result = products

    result = filter_by_price(
        result,
        min_price=plan.get(
            "min_price"
        ),
        max_price=plan.get(
            "max_price"
        ),
    )

    result = filter_by_brand(
        result,
        plan.get("brand")
    )

    result = filter_by_concept(
        result,
        plan.get("concept"),
        plan.get("concept_terms")
    )

    result = filter_by_sub_category(
        result,
        plan.get("sub_category")
    )

    return result.copy()


# ============================================================
# Query Complexity
# ============================================================

def has_meaningful_text_constraints(plan):
    """
    Decide whether TF-IDF is actually needed.

    Fast path:
        If the query is mainly composed of a known concept
        plus structured constraints, do NOT use TF-IDF.

    Examples:
        "یک کیف زیر ۲ میلیون"
            -> False

        "یک کیف برای استفاده روزمره میخوام که گرون نباشه"
            -> False

        "یک کیف جادار برای سفر با جیب‌های زیاد"
            -> True
    """

    query = normalize_persian_text(
        plan.get("query", "")
    ).strip().lower()

    # --------------------------------------------------------
    # 1. Remove concept terms
    # --------------------------------------------------------

    for term in plan.get(
        "concept_terms",
        []
    ):
        term = normalize_persian_text(
            term
        ).strip().lower()

        if term:
            query = query.replace(
                term,
                " "
            )

    # --------------------------------------------------------
    # 2. Remove numbers / price expressions
    # --------------------------------------------------------

    query = re.sub(
        r"\d+(?:\.\d+)?",
        " ",
        query
    )

    # --------------------------------------------------------
    # 3. Remove common structured phrases
    # --------------------------------------------------------

    phrases = [
        # Request
        "میخوام",
        "می خواهم",
        "می‌خوام",
        "می‌خواهم",
        "میخام",

        "پیدا کن",
        "پیدا کنم",
        "معرفی کن",
        "نشان بده",
        "نشون بده",

        # Price
        "خیلی گرون نباشه",
        "خیلی گران نباشد",
        "گرون نباشه",
        "گران نباشه",
        "ارزون باشه",
        "ارزان باشه",
        "قیمت مناسب",
        "قیمتش مناسب باشه",
        "مقرون به صرفه",
        "مقرون بصرفه",
        "زیر",
        "کمتر از",
        "بیشتر از",
        "حداقل",
        "حداکثر",
        "بالای",

        # Satisfaction
        "راضی باشه",
        "راضی باشن",
        "راضی باشند",
        "راضی باشیم",
        "راضی بودن",
        "رضایت بالا",
        "رضایت خوب",
        "نظر خوب",
        "نظرات خوب",
        "بازخورد خوب",

        # Usage
        "برای استفاده روزمره",
        "استفاده روزمره",
        "روزمره",
        "برای سفر",
        "سفر",
        "برای دانشگاه",
        "دانشگاه",
        "برای دانشجو",
        "دانشجویی",
        "برای استفاده رسمی",
        "رسمی",
        "برای استفاده ورزشی",
        "ورزشی",

        # Gender
        "زنانه",
        "مردانه",
        "بچگانه",

        # Generic
        "یک",
        "یه",
        "محصول",
        "کالا",
        "چند",
        "چندتا",
        "چند تا",
        "خیلی",
        "مناسب",
        "خوب",
        "باشه",
        "باشن",
        "باشد",
        "باشند",
        "نباشه",
        "نباشد",
        "که",
        "و",
        "با",
        "برای",
        "از",
        "در",
        "هم",
        "این",
        "آن",
        "را",
        "رو",
    ]

    # Long phrases first
    phrases = sorted(
        phrases,
        key=len,
        reverse=True
    )

    for phrase in phrases:

        phrase = normalize_persian_text(
            phrase
        ).strip().lower()

        query = query.replace(
            phrase,
            " "
        )

    # --------------------------------------------------------
    # 4. Remove punctuation
    # --------------------------------------------------------

    query = re.sub(
        r"[^\w\sآ-ی]",
        " ",
        query
    )

    # --------------------------------------------------------
    # 5. Token cleanup
    # --------------------------------------------------------

    remaining_tokens = [
        token
        for token in query.split()
        if len(token) > 1
    ]

    remaining = " ".join(
        remaining_tokens
    )

    # Debug-friendly:
    # print("Remaining semantic text:", remaining)

    # --------------------------------------------------------
    # 6. No meaningful text remains
    # --------------------------------------------------------

    if not remaining:
        return False

    # --------------------------------------------------------
    # 7. Only tiny leftovers -> still structured
    # --------------------------------------------------------

    if len(remaining) < 4:
        return False

    return True


# ============================================================
# Fast Structured Ranking
# ============================================================

def fast_product_ranking(
    products,
    plan
):
    """
    Fast ranking for queries that do not need semantic retrieval.

    Uses deterministic signals:

        satisfaction
        evidence
        price
        rating
    """

    if products.empty:
        return products.copy()

    result = products.copy()

    # --------------------------------------------------------
    # Satisfaction
    # --------------------------------------------------------

    rec = numeric_series(
        result,
        REC_RATIO
    )

    rating = numeric_series(
        result,
        RATING
    )

    rating_normalized = (
        (rating - 1) / 4
    ).clip(
        0,
        1
    )

    rec = (
        rec.clip(
            0,
            1
        )
    )

    satisfaction = (
        0.7 * rec.fillna(
            rating_normalized
        )
        +
        0.3 * rating_normalized.fillna(
            0
        )
    )

    # --------------------------------------------------------
    # Evidence
    # --------------------------------------------------------

    evidence_columns = [
        "n_reviews",
        "n_substantive",
        "n_buyers",
    ]

    available = [
        column
        for column in evidence_columns
        if column in result.columns
    ]

    if available:

        evidence_values = result[
            available
        ].apply(
            pd.to_numeric,
            errors="coerce"
        )

        evidence_total = (
            evidence_values.sum(
                axis=1,
                min_count=1
            )
            .fillna(0)
        )

        max_evidence = max(
            evidence_total.max(),
            1
        )

        evidence_score = (
            np.log1p(
                evidence_total
            )
            /
            np.log1p(
                max_evidence
            )
        ).clip(
            0,
            1
        )

    else:

        evidence_score = pd.Series(
            0.5,
            index=result.index
        )

    # --------------------------------------------------------
    # Price
    # --------------------------------------------------------

    prices = numeric_series(
        result,
        PRICE
    )

    max_price = prices.max()

    if (
        pd.notna(max_price)
        and max_price > 0
    ):

        price_score = (
            1
            - prices / max_price
        ).clip(
            0,
            1
        ).fillna(0.5)

    else:

        price_score = pd.Series(
            0.5,
            index=result.index
        )

    # --------------------------------------------------------
    # Weights
    # --------------------------------------------------------

    satisfaction_weight = 0.50
    evidence_weight = 0.20
    price_weight = 0.15
    rating_weight = 0.15

    if plan.get(
        "require_satisfaction",
        False
    ):

        satisfaction_weight = 0.60
        evidence_weight = 0.15
        price_weight = 0.10
        rating_weight = 0.15

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    result["_semantic_score"] = 0.0

    result["_satisfaction_score"] = (
        satisfaction
    )

    result["_evidence_score"] = (
        evidence_score
    )

    result["_price_score"] = (
        price_score
    )

    result["_final_score"] = (
        satisfaction_weight
        * result[
            "_satisfaction_score"
        ]

        +

        evidence_weight
        * result[
            "_evidence_score"
        ]

        +

        price_weight
        * result[
            "_price_score"
        ]

        +

        rating_weight
        * rating_normalized.fillna(
            0
        )
    )

    return result.sort_values(
        "_final_score",
        ascending=False
    )


# ============================================================
# Candidate TF-IDF Retrieval
# ============================================================

def retrieve_from_candidates(
    query,
    products,
    product_index,
    candidate_indices,
    top_k=100
):
    """
    TF-IDF retrieval restricted to candidate rows.

    IMPORTANT:
    We DO NOT compute cosine similarity against the entire
    product matrix.
    """

    if products.empty:
        return products.copy()

    vectorizer = product_index[
        "vectorizer"
    ]

    matrix = product_index[
        "matrix"
    ]

    # --------------------------------------------------------
    # Map dataframe indices to original matrix positions.
    # Since the product index was built in the exact order of
    # the original products DataFrame, positions correspond
    # to iloc positions.
    # --------------------------------------------------------

    positions = np.array([
        products.index.get_loc(index)
        for index in candidate_indices
        if index in products.index
    ])

    if len(positions) == 0:
        return products.iloc[
            0:0
        ].copy()

    query = normalize_persian_text(
        query
    )

    query_vector = vectorizer.transform(
        [query]
    )

    # --------------------------------------------------------
    # IMPORTANT OPTIMIZATION:
    #
    # Slice only candidate rows from sparse matrix.
    # --------------------------------------------------------

    candidate_matrix = matrix[
        positions
    ]

    # Sparse matrix multiplication:
    #
    # query_vector @ candidate_matrix.T
    #
    # Since vectors are TF-IDF normalized, this is cosine
    # similarity.
    similarities = (
        query_vector
        @ candidate_matrix.T
    ).toarray().ravel()

    order = np.argsort(
        similarities
    )[::-1][:top_k]

    selected_positions = (
        positions[order]
    )

    selected_indices = [
        products.index[
            position
        ]
        for position in [
            products.index.get_loc(index)
            for index in products.index
            if index in products.index
        ]
    ]

    # Simpler direct mapping
    candidate_index_list = list(
        candidate_indices
    )

    selected_indices = [
        candidate_index_list[
            i
        ]
        for i in order
    ]

    result = products.loc[
        selected_indices
    ].copy()

    result["_semantic_score"] = (
        similarities[order]
    )

    return result


# ============================================================
# Semantic Retrieval
# ============================================================

def semantic_product_search(
    query,
    products,
    product_index,
    top_k=100,
    candidate_indices=None
):
    """
    Semantic search.

    If candidate_indices are provided, only those rows
    participate in retrieval.
    """

    if products.empty:
        return products.copy()

    if candidate_indices is None:

        candidate_indices = (
            products.index
        )

    return retrieve_from_candidates(
        query=query,
        products=products,
        product_index=product_index,
        candidate_indices=candidate_indices,
        top_k=top_k
    )


# ============================================================
# Satisfaction
# ============================================================

def calculate_satisfaction_score(
    products
):
    """
    Satisfaction score.

    Primary:
        rec_ratio

    Secondary:
        mean_rate
    """

    rec = numeric_series(
        products,
        REC_RATIO
    )

    rating = numeric_series(
        products,
        RATING
    )

    rec = rec.clip(
        0,
        1
    )

    rating_normalized = (
        (rating - 1) / 4
    ).clip(
        0,
        1
    )

    score = pd.Series(
        np.nan,
        index=products.index,
        dtype=float
    )

    rec_valid = rec.notna()

    score.loc[
        rec_valid
    ] = (
        0.7 * rec.loc[
            rec_valid
        ]
        +
        0.3 * rating_normalized.loc[
            rec_valid
        ]
    )

    rating_only = (
        ~rec_valid
        & rating.notna()
    )

    score.loc[
        rating_only
    ] = rating_normalized.loc[
        rating_only
    ]

    return score.fillna(
        0
    ).clip(
        0,
        1
    )


# ============================================================
# Evidence Score
# ============================================================

def calculate_evidence_score(
    products
):
    """
    Evidence strength based on review activity.
    """

    columns = [
        "n_substantive",
        "n_rated",
        "n_buyers",
    ]

    available = [
        column
        for column in columns
        if column in products.columns
    ]

    if not available:

        return pd.Series(
            0.5,
            index=products.index
        )

    values = products[
        available
    ].apply(
        pd.to_numeric,
        errors="coerce"
    )

    total = values.sum(
        axis=1,
        min_count=1
    ).fillna(0)

    max_total = max(
        total.max(),
        1
    )

    return (
        np.log1p(total)
        /
        np.log1p(max_total)
    ).clip(
        0,
        1
    )


# ============================================================
# Price Score
# ============================================================

def calculate_price_score(
    products
):
    """Normalize lower price to a higher score."""

    prices = numeric_series(
        products,
        PRICE
    )

    max_price = prices.max()

    if (
        pd.isna(max_price)
        or max_price <= 0
    ):

        return pd.Series(
            0.5,
            index=products.index
        )

    return (
        1
        - prices / max_price
    ).clip(
        0,
        1
    ).fillna(0.5)


# ============================================================
# Final Ranking
# ============================================================

def rank_products(
    products,
    plan
):
    """
    Final ranking for semantically retrieved candidates.
    """

    if products.empty:
        return products.copy()

    result = products.copy()

    result["_semantic_score"] = pd.to_numeric(
        result.get(
            "_semantic_score",
            0.0
        ),
        errors="coerce"
    ).fillna(0.0)

    result["_satisfaction_score"] = (
        calculate_satisfaction_score(
            result
        )
    )

    result["_evidence_score"] = (
        calculate_evidence_score(
            result
        )
    )

    result["_price_score"] = (
        calculate_price_score(
            result
        )
    )

    semantic_weight = 0.65
    satisfaction_weight = 0.20
    evidence_weight = 0.0
    price_weight = 0.15

    if plan.get(
        "require_satisfaction",
        False
    ):

        semantic_weight = 0.50
        satisfaction_weight = 0.35
        evidence_weight = 0.10
        price_weight = 0.05

    result["_final_score"] = (
        semantic_weight
        * result["_semantic_score"]

        +

        satisfaction_weight
        * result["_satisfaction_score"]

        +

        evidence_weight
        * result["_evidence_score"]

        +

        price_weight
        * result["_price_score"]
    )

    return result.sort_values(
        "_final_score",
        ascending=False
    )


# ============================================================
# Main Search
# ============================================================

def search_products(
    products,
    product_index,
    plan,
    top_k=10
):
    """
    Complete optimized product-search pipeline.

    Strategy:

        1. Structured filtering.
        2. If query is simple:
               deterministic ranking only.
        3. If query contains meaningful text:
               TF-IDF only on filtered candidates.
        4. Final ranking.
    """

    validate_product_columns(
        products
    )

    if products.empty:
        return products.copy()

    # --------------------------------------------------------
    # Structured filtering
    # --------------------------------------------------------

    filtered = filter_products(
        products,
        plan
    )

    # --------------------------------------------------------
    # Explicit concept with zero matches:
    # NEVER search unrelated products.
    # --------------------------------------------------------

    if (
        plan.get("concept")
        and filtered.empty
    ):
        return products.iloc[
            0:0
        ].copy()

    # --------------------------------------------------------
    # No candidate after other filters.
    # --------------------------------------------------------

    if filtered.empty:
        return products.iloc[
            0:0
        ].copy()

    # --------------------------------------------------------
    # Decide if semantic retrieval is needed.
    # --------------------------------------------------------

    need_semantic = has_meaningful_text_constraints(
        plan
    )

    # --------------------------------------------------------
    # FAST PATH
    # --------------------------------------------------------

    if not need_semantic:

        ranked = fast_product_ranking(
            filtered,
            plan
        )

        return ranked.head(
            top_k
        ).copy()

    # --------------------------------------------------------
    # SEMANTIC PATH
    # --------------------------------------------------------

    candidate_count = max(
        top_k * 20,
        100
    )

    candidates = semantic_product_search(
        query=plan.get(
            "query",
            ""
        ),
        products=products,
        product_index=product_index,
        top_k=candidate_count,
        candidate_indices=filtered.index
    )

    ranked = rank_products(
        candidates,
        plan
    )

    return ranked.head(
        top_k
    ).copy()


# ============================================================
# Result Formatting
# ============================================================

def format_product_results(
    results
):
    """Convert product results to dictionaries."""

    if results.empty:
        return []

    formatted = []

    for _, row in results.iterrows():

        formatted.append({

            "id": row.get(
                PRODUCT_ID
            ),

            "title": row.get(
                TITLE
            ),

            "brand": row.get(
                BRAND
            ),

            "category": row.get(
                CATEGORY_1
            ),

            "sub_category": row.get(
                CATEGORY_2
            ),

            "price": row.get(
                PRICE
            ),

            "rating": row.get(
                RATING
            ),

            "recommendation_ratio": row.get(
                REC_RATIO
            ),

            "semantic_score": round(
                float(
                    row.get(
                        "_semantic_score",
                        0
                    )
                ),
                4
            ),

            "satisfaction_score": round(
                float(
                    row.get(
                        "_satisfaction_score",
                        0
                    )
                ),
                4
            ),

            "evidence_score": round(
                float(
                    row.get(
                        "_evidence_score",
                        0
                    )
                ),
                4
            ),

            "final_score": round(
                float(
                    row.get(
                        "_final_score",
                        0
                    )
                ),
                4
            ),
        })

    return formatted
