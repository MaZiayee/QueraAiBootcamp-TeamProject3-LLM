import re
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.utils import normalize_persian_text


# ============================================================
# Column Detection
# ============================================================

def find_column(df, candidates):
    """
    Return the first existing column from candidates.
    """

    for column in candidates:
        if column in df.columns:
            return column

    return None


def detect_review_columns(reviews):
    """
    Detect important review columns automatically.
    """

    columns = {
        "product_id": find_column(
            reviews,
            [
                "product_id",
                "productId",
                "ProductId",
                "product_ID",
            ],
        ),

        "review_id": find_column(
            reviews,
            [
                "review_id",
                "comment_id",
                "id",
                "ID",
            ],
        ),

        "text": find_column(
            reviews,
            [
                "comment",
                "comment_body",
                "body",
                "review",
                "review_text",
                "text",
                "Comment",
                "CommentText",
                "content",
            ],
        ),

        "rating": find_column(
            reviews,
            [
                "rate",
                "rating",
                "Rate",
                "Rating",
            ],
        ),

        "recommendation": find_column(
            reviews,
            [
                "status_recommendation",
                "recommendation",
                "Recommendation",
                "recommended",
            ],
        ),

        "pros": find_column(
            reviews,
            [
                "pros",
                "Pros",
                "advantages",
            ],
        ),

        "cons": find_column(
            reviews,
            [
                "cons",
                "Cons",
                "disadvantages",
            ],
        ),
    }

    if columns["product_id"] is None:
        raise ValueError(
            "Could not find product_id column in reviews."
        )

    if columns["text"] is None:
        raise ValueError(
            "Could not find a review text column."
        )

    return columns


# ============================================================
# Text Preparation
# ============================================================

def build_review_text(reviews, columns):
    """
    Build searchable text from available review fields.
    """

    parts = []

    if columns["text"]:
        parts.append(
            reviews[columns["text"]]
            .fillna("")
            .astype(str)
        )

    if columns["pros"]:
        parts.append(
            reviews[columns["pros"]]
            .fillna("")
            .astype(str)
        )

    if columns["cons"]:
        parts.append(
            reviews[columns["cons"]]
            .fillna("")
            .astype(str)
        )

    if not parts:
        return pd.Series(
            "",
            index=reviews.index,
            dtype="object",
        )

    text = parts[0]

    for part in parts[1:]:
        text = text + " " + part

    return text.apply(normalize_persian_text)


# ============================================================
# Review Index
# ============================================================

def build_review_index(
    reviews,
    max_features=100_000,
    min_df=2,
):
    """
    Build TF-IDF index for reviews.
    """

    columns = detect_review_columns(
        reviews
    )

    text = build_review_text(
        reviews,
        columns
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
        "columns": columns,
    }


# ============================================================
# Product Reviews
# ============================================================

def get_product_reviews(
    reviews,
    product_id,
    columns=None,
):
    """
    Return reviews belonging to one product.
    """

    if columns is None:
        columns = detect_review_columns(
            reviews
        )

    product_column = columns[
        "product_id"
    ]

    return reviews.loc[
        reviews[product_column] == product_id
    ].copy()


# ============================================================
# Review Intent
# ============================================================

def detect_review_intent(query):
    """
    Detect the main intent of a review question.

    Returns:

        positive
        negative
        quality
        recommendation
        general
    """

    query = normalize_persian_text(
        query
    ).lower()

    negative_words = [
        "ایراد",
        "مشکل",
        "نقص",
        "بد",
        "ضعف",
        "خراب",
        "نارضای",
        "شکایت",
        "معایب",
        "نقاط ضعف",
    ]

    positive_words = [
        "مزیت",
        "خوب",
        "راضی",
        "نکته مثبت",
        "نقاط قوت",
        "قوت",
        "پسند",
    ]

    quality_words = [
        "کیفیت",
        "دوام",
        "جنس",
        "ساخت",
        "عملکرد",
        "کیفیت ساخت",
    ]

    recommendation_words = [
        "ارزش خرید",
        "پیشنهاد",
        "توصیه",
        "می ارزه",
        "میارزه",
        "دوباره میخرم",
        "خریدش",
    ]

    if any(
        word in query
        for word in negative_words
    ):
        return "negative"

    if any(
        word in query
        for word in positive_words
    ):
        return "positive"

    if any(
        word in query
        for word in quality_words
    ):
        return "quality"

    if any(
        word in query
        for word in recommendation_words
    ):
        return "recommendation"

    return "general"


# ============================================================
# Recommendation Metadata
# ============================================================

def normalize_recommendation_value(value):
    """
    Convert recommendation values to:

        True
        False
        None
    """

    if pd.isna(value):
        return None

    text = normalize_persian_text(
        str(value)
    ).strip().lower()

    true_values = {
        "true",
        "1",
        "yes",
        "recommended",
        "recommend",
        "بله",
        "مثبت",
        "پیشنهاد می کنم",
        "پیشنهاد میکنم",
    }

    false_values = {
        "false",
        "0",
        "no",
        "not_recommended",
        "not recommended",
        "خیر",
        "منفی",
        "پیشنهاد نمی کنم",
        "پیشنهاد نمیکنم",
    }

    if text in true_values:
        return True

    if text in false_values:
        return False

    return None


# ============================================================
# Intent Filtering
# ============================================================

def filter_reviews_by_intent(
    reviews,
    intent,
    columns,
):
    """
    Use metadata when available.

    Never returns an empty set only because metadata
    is unavailable.
    """

    if reviews.empty:
        return reviews.copy()

    recommendation_column = columns[
        "recommendation"
    ]

    # --------------------------------------------------------
    # Positive
    # --------------------------------------------------------

    if (
        intent == "positive"
        and recommendation_column
    ):

        values = reviews[
            recommendation_column
        ].map(
            normalize_recommendation_value
        )

        mask = values == True

        if mask.any():
            return reviews.loc[
                mask
            ].copy()

    # --------------------------------------------------------
    # Negative
    # --------------------------------------------------------

    if (
        intent == "negative"
        and recommendation_column
    ):

        values = reviews[
            recommendation_column
        ].map(
            normalize_recommendation_value
        )

        mask = values == False

        if mask.any():
            return reviews.loc[
                mask
            ].copy()

    # --------------------------------------------------------
    # Recommendation
    # --------------------------------------------------------

    if (
        intent == "recommendation"
        and recommendation_column
    ):

        values = reviews[
            recommendation_column
        ].map(
            normalize_recommendation_value
        )

        mask = values.notna()

        if mask.any():
            return reviews.loc[
                mask
            ].copy()

    return reviews.copy()


# ============================================================
# Semantic Retrieval
# ============================================================

def retrieve_reviews(
    query,
    reviews,
    review_index,
    top_k=20,
):
    """
    Retrieve the most relevant reviews from the supplied
    review subset.
    """

    if reviews.empty:
        return reviews.copy()

    vectorizer = review_index[
        "vectorizer"
    ]

    matrix = review_index[
        "matrix"
    ]

    query = normalize_persian_text(
        query
    )

    query_vector = vectorizer.transform(
        [query]
    )

    # --------------------------------------------------------
    # Map dataframe rows to matrix positions
    # --------------------------------------------------------

    positions = np.array([
        reviews.index.get_loc(index)
        for index in reviews.index
        if index in reviews.index
    ])

    if len(positions) == 0:
        return reviews.iloc[0:0].copy()

    similarities = cosine_similarity(
        query_vector,
        matrix
    ).ravel()

    scores = similarities[
        positions
    ]

    order = np.argsort(
        scores
    )[::-1][:top_k]

    selected_positions = positions[
        order
    ]

    result = reviews.iloc[
        [
            reviews.index.get_loc(index)
            for index in reviews.index[
                np.isin(
                    positions,
                    selected_positions
                )
            ]
        ]
    ].copy()

    # Safer and simpler reconstruction
    selected_indices = [
        reviews.index[position]
        for position in selected_positions
    ]

    result = reviews.loc[
        selected_indices
    ].copy()

    result["_review_score"] = [
        similarities[position]
        for position in selected_positions
    ]

    return result


# ============================================================
# Rating Score
# ============================================================

def calculate_rating_score(
    reviews,
    columns,
):
    """
    Normalize ratings into [0, 1].
    """

    rating_column = columns[
        "rating"
    ]

    if rating_column is None:
        return pd.Series(
            0.5,
            index=reviews.index
        )

    ratings = pd.to_numeric(
        reviews[rating_column],
        errors="coerce"
    )

    valid = ratings.notna()

    if not valid.any():
        return pd.Series(
            0.5,
            index=reviews.index
        )

    # Most review datasets use 1-5 ratings.
    score = (
        (ratings - 1) / 4
    ).clip(
        0,
        1
    )

    return score.fillna(0.5)


# ============================================================
# Review Ranking
# ============================================================

def rank_reviews(
    reviews,
    intent,
    columns,
):
    """
    Rank reviews by relevance.

    Rating is used only as a secondary signal.
    """

    if reviews.empty:
        return reviews.copy()

    result = reviews.copy()

    result["_review_score"] = pd.to_numeric(
        result.get(
            "_review_score",
            0.0
        ),
        errors="coerce"
    ).fillna(0.0)

    result["_rating_score"] = (
        calculate_rating_score(
            result,
            columns
        )
    )

    # --------------------------------------------------------
    # Default weights
    # --------------------------------------------------------

    semantic_weight = 0.85
    rating_weight = 0.15

    # For negative questions, relevance matters more
    # than rating.
    if intent == "negative":

        semantic_weight = 0.95
        rating_weight = 0.05

    # Positive questions can benefit from rating.
    elif intent == "positive":

        semantic_weight = 0.75
        rating_weight = 0.25

    result["_final_review_score"] = (
        semantic_weight
        * result["_review_score"]
        +
        rating_weight
        * result["_rating_score"]
    )

    return result.sort_values(
        "_final_review_score",
        ascending=False
    )


# ============================================================
# Diversity
# ============================================================

def select_diverse_reviews(
    reviews,
    top_k=10,
):
    """
    Remove near-duplicate reviews.

    This is intentionally simple and based on text uniqueness.
    """

    if reviews.empty:
        return reviews.copy()

    if top_k <= 0:
        return reviews.iloc[0:0].copy()

    seen = set()
    selected = []

    text_column = None

    if "text" in reviews.columns:
        text_column = "text"

    for index, row in reviews.iterrows():

        if text_column:
            text = normalize_persian_text(
                row[text_column]
            )
        else:
            text = str(
                row.to_dict()
            )

        # Compact duplicate key
        key = re.sub(
            r"\s+",
            " ",
            text.lower()
        ).strip()

        if key in seen:
            continue

        seen.add(key)
        selected.append(index)

        if len(selected) >= top_k:
            break

    return reviews.loc[
        selected
    ].copy()


# ============================================================
# Main Search
# ============================================================

def search_reviews(
    query,
    product_id,
    reviews,
    review_index,
    top_k=10,
):
    """
    Main review retrieval pipeline.

        Product
          ↓
        Product reviews
          ↓
        Intent detection
          ↓
        Metadata filtering
          ↓
        Semantic retrieval
          ↓
        Ranking
          ↓
        Diversity
          ↓
        Top-K
    """

    columns = review_index[
        "columns"
    ]

    # --------------------------------------------------------
    # Product reviews
    # --------------------------------------------------------

    product_reviews = get_product_reviews(
        reviews,
        product_id,
        columns
    )

    if product_reviews.empty:
        return product_reviews.copy()

    # --------------------------------------------------------
    # Intent
    # --------------------------------------------------------

    intent = detect_review_intent(
        query
    )

    # --------------------------------------------------------
    # Intent filtering
    # --------------------------------------------------------

    filtered = filter_reviews_by_intent(
        product_reviews,
        intent,
        columns
    )

    # --------------------------------------------------------
    # Retrieval
    # --------------------------------------------------------

    candidate_count = max(
        top_k * 5,
        30
    )

    retrieved = retrieve_reviews(
        query=query,
        reviews=filtered,
        review_index=review_index,
        top_k=candidate_count,
    )

    # --------------------------------------------------------
    # Ranking
    # --------------------------------------------------------

    ranked = rank_reviews(
        retrieved,
        intent,
        columns,
    )

    # --------------------------------------------------------
    # Diversity
    # --------------------------------------------------------

    result = select_diverse_reviews(
        ranked,
        top_k=top_k
    )

    return result.head(
        top_k
    ).copy()


# ============================================================
# Formatting
# ============================================================

def format_review_results(
    reviews,
    review_index,
):
    """
    Convert review results into dictionaries.

    This structure will later be used by evidence.py.
    """

    if reviews.empty:
        return []

    columns = review_index[
        "columns"
    ]

    results = []

    for _, row in reviews.iterrows():

        item = {
            "review_id": (
                row.get(
                    columns["review_id"]
                )
                if columns["review_id"]
                else None
            ),

            "product_id": row.get(
                columns["product_id"]
            ),

            "text": str(
                row.get(
                    columns["text"],
                    ""
                )
            ),

            "rating": (
                row.get(
                    columns["rating"]
                )
                if columns["rating"]
                else None
            ),

            "recommendation": (
                row.get(
                    columns["recommendation"]
                )
                if columns["recommendation"]
                else None
            ),

            "pros": (
                row.get(
                    columns["pros"]
                )
                if columns["pros"]
                else None
            ),

            "cons": (
                row.get(
                    columns["cons"]
                )
                if columns["cons"]
                else None
            ),

            "retrieval_score": round(
                float(
                    row.get(
                        "_review_score",
                        0
                    )
                ),
                4
            ),

            "final_score": round(
                float(
                    row.get(
                        "_final_review_score",
                        0
                    )
                ),
                4
            ),
        }

        results.append(
            item
        )

    return results
