import re

from src.utils import normalize_persian_text, to_float


# ============================================================
# Persian Number Conversion
# ============================================================

PERSIAN_DIGITS = "۰۱۲۳۴۵۶۷۸۹"
ARABIC_DIGITS = "٠١٢٣٤٥٦٧٨٩"
ENGLISH_DIGITS = "0123456789"

DIGIT_MAP = {}

for fa, en in zip(PERSIAN_DIGITS, ENGLISH_DIGITS):
    DIGIT_MAP[fa] = en

for ar, en in zip(ARABIC_DIGITS, ENGLISH_DIGITS):
    DIGIT_MAP[ar] = en


def normalize_digits(text):
    """Convert Persian/Arabic digits to English digits."""

    return "".join(
        DIGIT_MAP.get(char, char)
        for char in str(text)
    )


# ============================================================
# Query Normalization
# ============================================================

def normalize_query(query):
    """Normalize user query."""

    query = normalize_persian_text(query)
    query = normalize_digits(query)

    return query


# ============================================================
# Price Parsing
# ============================================================

PRICE_MULTIPLIERS = {
    "هزار": 1_000,
    "میلیون": 1_000_000,
    "میلیارد": 1_000_000_000,
}


def parse_price_value(
    number,
    unit=None
):
    """Convert numeric price expression into a number."""

    value = to_float(number)

    if value is None:
        return None

    if unit:
        value *= PRICE_MULTIPLIERS.get(
            unit.strip(),
            1
        )

    return value


def extract_price_range(query):
    """Extract minimum and maximum price."""

    query = normalize_query(query)

    min_price = None
    max_price = None

    # --------------------------------------------------------
    # Between X and Y
    # --------------------------------------------------------

    pattern = re.search(
        r"(?:بین|از)\s*"
        r"([\d.]+)\s*(میلیارد|میلیون|هزار)?"
        r"\s*(?:تا|-)\s*"
        r"([\d.]+)\s*(میلیارد|میلیون|هزار)?",
        query
    )

    if pattern:

        min_price = parse_price_value(
            pattern.group(1),
            pattern.group(2)
        )

        max_price = parse_price_value(
            pattern.group(3),
            pattern.group(4)
        )

        return min_price, max_price

    # --------------------------------------------------------
    # Maximum
    # --------------------------------------------------------

    pattern = re.search(
        r"(?:زیر|کمتر از|حداکثر|تا)\s*"
        r"([\d.]+)\s*(میلیارد|میلیون|هزار)?",
        query
    )

    if pattern:

        max_price = parse_price_value(
            pattern.group(1),
            pattern.group(2)
        )

    # --------------------------------------------------------
    # Minimum
    # --------------------------------------------------------

    pattern = re.search(
        r"(?:بالای|بیشتر از|حداقل|از)\s*"
        r"([\d.]+)\s*(میلیارد|میلیون|هزار)?",
        query
    )

    if pattern:

        min_price = parse_price_value(
            pattern.group(1),
            pattern.group(2)
        )

    # --------------------------------------------------------
    # Natural language
    # --------------------------------------------------------

    cheap_phrases = [
        "گرون نباشه",
        "گران نباشه",
        "خیلی گرون نباشه",
        "خیلی گران نباشه",
        "ارزون باشه",
        "ارزان باشه",
        "قیمت مناسب",
        "قیمتش مناسب باشه",
        "مقرون به صرفه",
        "مقرون بصرفه",
    ]

    if (
        max_price is None
        and any(
            phrase in query
            for phrase in cheap_phrases
        )
    ):
        max_price = 5_000_000

    return min_price, max_price


# ============================================================
# Product Concept Aliases
# ============================================================

CONCEPT_ALIASES = {
    "موبایل": [
        "موبایل",
        "گوشی",
        "گوشی موبایل",
        "تلفن همراه",
        "اسمارت فون",
        "smartphone",
    ],

    "لپ تاپ": [
        "لپ تاپ",
        "لپتاپ",
        "laptop",
        "نوت بوک",
    ],

    "هدفون": [
        "هدفون",
        "هدست",
        "headphone",
        "headset",
    ],

    "هندزفری": [
        "هندزفری",
        "ایرباد",
        "ایرپاد",
        "earbuds",
        "airpods",
    ],

    "کیف": [
        "کیف",
        "کیف دستی",
        "کیف دوشی",
        "کیف رودوشی",
        "کیف زنانه",
    ],

    "کفش": [
        "کفش",
        "کفش ورزشی",
        "کفش پیاده روی",
        "کفش اسپرت",
        "کتانی",
        "کتونی",
    ],

    "تیشرت": [
        "تیشرت",
        "تی شرت",
        "تی‌شرت",
        "tshirt",
    ],

    "لباس": [
        "لباس",
        "پوشاک",
    ],

    "شومیز": [
        "شومیز",
    ],

    "مانتو": [
        "مانتو",
    ],

    "دامن": [
        "دامن",
    ],

    "شلوار": [
        "شلوار",
    ],

    "پیراهن": [
        "پیراهن",
    ],

    "کاپشن": [
        "کاپشن",
        "پالتو",
    ],

    "عطر": [
        "عطر",
        "ادکلن",
        "ادوپرفیوم",
    ],

    "ساعت": [
        "ساعت",
        "ساعت مچی",
    ],

    "کتاب": [
        "کتاب",
    ],
}


def extract_concept(
    query,
    known_concepts=None
):
    """
    Extract canonical product concept.
    """

    query = normalize_query(
        query
    ).lower()

    for canonical, aliases in CONCEPT_ALIASES.items():

        aliases_sorted = sorted(
            aliases,
            key=len,
            reverse=True
        )

        for alias in aliases_sorted:

            alias = normalize_query(
                alias
            ).lower()

            if re.search(
                rf"(?<!\w){re.escape(alias)}(?!\w)",
                query
            ):
                return canonical

    if known_concepts:

        normalized_concepts = {
            normalize_query(concept): concept
            for concept in known_concepts
            if concept
        }

        concepts_sorted = sorted(
            normalized_concepts.keys(),
            key=len,
            reverse=True
        )

        for concept in concepts_sorted:

            if re.search(
                rf"(?<!\w){re.escape(concept)}(?!\w)",
                query
            ):
                return normalized_concepts[concept]

    return None


def get_concept_terms(concept):
    """Return all aliases for a concept."""

    if not concept:
        return []

    return CONCEPT_ALIASES.get(
        concept,
        [concept]
    )


# ============================================================
# Brand Extraction
# ============================================================

def extract_brand(
    query,
    known_brands=None
):
    """Extract brand from query."""

    if not known_brands:
        return None

    query = normalize_query(
        query
    ).lower()

    normalized_brands = {
        normalize_query(brand).lower(): brand
        for brand in known_brands
        if brand
    }

    brands = sorted(
        normalized_brands.keys(),
        key=len,
        reverse=True
    )

    for brand in brands:

        if re.search(
            rf"(?<!\w){re.escape(brand)}(?!\w)",
            query
        ):
            return normalized_brands[brand]

    return None


# ============================================================
# Satisfaction
# ============================================================

SATISFACTION_WORDS = [
    "راضی",
    "رضایت",
    "خوب باشه",
    "خوب باشن",
    "خریدارها راضی",
    "خریدار راضی",
    "مشتری ها راضی",
    "مشتریها راضی",
    "نظر خوب",
    "نظرات خوب",
    "بازخورد خوب",
    "امتیاز خوب",
]


def requires_satisfaction(query):
    """Detect satisfaction requirement."""

    query = normalize_query(
        query
    )

    return any(
        phrase in query
        for phrase in SATISFACTION_WORDS
    )


# ============================================================
# Sub-category
# ============================================================

def extract_sub_category(query):
    """Extract usage/category hint."""

    query = normalize_query(
        query
    )

    sub_categories = [
        "روزمره",
        "رسمی",
        "ورزشی",
        "مجلسی",
        "سفر",
        "دانشجویی",
        "اداری",
        "کودک",
        "زنانه",
        "مردانه",
        "بچگانه",
    ]

    for item in sub_categories:

        if item in query:
            return item

    return None


# ============================================================
# Intent Detection
# ============================================================


def detect_query_type(query):
    """
    Detect the primary intent.

    Priority:
        comparison
        analytics
        review
        search
    """

    query = normalize_query(
        query
    )

    # --------------------------------------------------------
    # Comparison
    # --------------------------------------------------------

    comparison_words = [
        "مقایسه",
        "مقایسه کن",
        "کدوم بهتره",
        "کدام بهتر است",
        "با هم مقایسه",
        "فرق این دو",
        "تفاوت این دو",
        "بین این دو",
        "مقایسه‌شون",
        "مقایسه شان",
        "مقایسه‌شان",
    ]

    if any(
        word in query
        for word in comparison_words
    ):
        return "comparison"

    # --------------------------------------------------------
    # Analytics
    # --------------------------------------------------------

    analytics_words = [
        "پرتکرارترین شکایت",
        "پرتکرارترین مشکل",
        "تحلیل",
        "تحلیل دسته",
        "تحلیل محصولات",
        "تحلیل بازار",
        "آمار",
        "گزارش فروش",
        "کدام دسته",
        "کدام برند",
        "برندها",
        "دسته بندی",
        "دسته‌بندی",
    ]

    if any(
        word in query
        for word in analytics_words
    ):
        return "analytics"

    # --------------------------------------------------------
    # Review
    # --------------------------------------------------------

    review_words = [
        "نظر کاربران",
        "نظرات کاربران",
        "نظر خریداران",
        "نظرات خریداران",
        "نظر مردم",
        "نظرات مردم",
        "ایراد",
        "ایرادهای",
        "مشکل این محصول",
        "مشکلات این محصول",
        "بازخورد کاربران",
        "تجربه کاربران",
        "کیفیت این محصول",
        "کیفیتش چطوره",
        "کیفیتش چطوره",
        "جنس این محصول",
        "جنسش چطوره",
        "دوام این محصول",
        "دوامش چطوره",
        "ارزش خرید این محصول",
        "ارزش خریدش",
        "پیشنهاد میشه",
        "پیشنهاد می‌شود",
        "راضی هستند",
        "راضی بودن",
    ]

    if any(
        word in query
        for word in review_words
    ):
        return "review"

    return "search"



# ============================================================
# Multi-Intent Detection
# ============================================================

def detect_requested_capabilities(query):
    """
    Detect all capabilities required by the query.

    Important distinction:

        "کاربران راضی باشند"
            -> search constraint

        "نظر کاربران درباره محصول چیست؟"
            -> review

        "درباره دوام محصول چه می‌گویند؟"
            -> review
    """

    query = normalize_query(
        query
    )

    capabilities = []

    primary = detect_query_type(
        query
    )

    # --------------------------------------------------------
    # Primary intent
    # --------------------------------------------------------

    if primary == "search":
        capabilities.append("search")

    elif primary == "review":
        capabilities.append("review")

    elif primary == "comparison":
        capabilities.extend([
            "search",
            "comparison"
        ])

    elif primary == "analytics":
        capabilities.append(
            "analytics"
        )

    # --------------------------------------------------------
    # Explicit search signals
    # --------------------------------------------------------

    search_signals = [
        "میخوام",
        "می خواهم",
        "می‌خوام",
        "می‌خواهم",
        "پیدا کن",
        "پیدا کنم",
        "معرفی کن",
        "معرفی",
        "نشون بده",
        "نشان بده",
        "چند تا",
        "چندتا",
        "دنبال",
        "میگردم",
        "می‌گردم",
    ]

    has_search_signal = any(
        signal in query
        for signal in search_signals
    )

    if has_search_signal:

        if "search" not in capabilities:

            capabilities.insert(
                0,
                "search"
            )

    # --------------------------------------------------------
    # Explicit REVIEW signals
    #
    # IMPORTANT:
    # "راضی" and "رضایت" are intentionally NOT here.
    # They are usually search constraints.
    # --------------------------------------------------------

    review_signals = [
        "نظر کاربران",
        "نظرات کاربران",
        "نظر خریداران",
        "نظرات خریداران",
        "نظر مردم",
        "نظرات مردم",
        "بازخورد کاربران",
        "بازخورد خریداران",
        "تجربه کاربران",
        "تجربه خریداران",

        "ایراد",
        "ایرادها",
        "مشکل این محصول",
        "مشکلات این محصول",

        "کیفیت این محصول",
        "کیفیتش چطوره",
        "جنس این محصول",
        "جنسش چطوره",

        "دوام این محصول",
        "دوامش چطوره",

        "ارزش خرید این محصول",
        "ارزش خریدش",

        "پیشنهاد میشه",
        "پیشنهاد می‌شود",

        "چه نظری دارن",
        "چی میگن",
        "چه می‌گویند",
    ]

    has_review_signal = any(
        signal in query
        for signal in review_signals
    )

    if has_review_signal:

        if "review" not in capabilities:

            capabilities.append(
                "review"
            )

    # --------------------------------------------------------
    # Detect review topic words only when they are being
    # ASKED ABOUT, not merely used as constraints.
    #
    # Example:
    # "بگو مردم درباره دوامش چی گفتن"
    # -> review
    #
    # Example:
    # "یک کیف با دوام میخوام"
    # -> search
    # --------------------------------------------------------

    review_question_patterns = [
        r"درباره.*دوام",
        r"درباره.*کیفیت",
        r"درباره.*جنس",
        r"درباره.*عملکرد",
        r"درباره.*بازخورد",
        r"درباره.*نظر",
        r"مردم.*چی.*گفتن",
        r"کاربران.*چی.*گفتن",
        r"خریداران.*چی.*گفتن",
        r"کاربران.*چه.*میگن",
        r"مردم.*چه.*میگن",
        r"خریداران.*چه.*میگن",
    ]

    has_review_question = any(
        re.search(
            pattern,
            query
        )
        for pattern in review_question_patterns
    )

    if has_review_question:

        if "review" not in capabilities:

            capabilities.append(
                "review"
            )

    # --------------------------------------------------------
    # Comparison
    # --------------------------------------------------------

    comparison_signals = [
        "مقایسه",
        "با هم مقایسه",
        "فرق",
        "تفاوت",
        "کدوم بهتر",
        "کدام بهتر",
    ]

    if any(
        signal in query
        for signal in comparison_signals
    ):

        if "comparison" not in capabilities:

            capabilities.append(
                "comparison"
            )

        if "search" not in capabilities:

            capabilities.insert(
                0,
                "search"
            )

        # Comparison can use review evidence.
        if "review" not in capabilities:

            capabilities.append(
                "review"
            )

    # --------------------------------------------------------
    # Analytics
    # --------------------------------------------------------

    analytics_signals = [
        "تحلیل",
        "پرتکرارترین",
        "آمار",
        "گزارش",
        "کدام دسته",
        "کدام برند",
        "دسته بندی",
        "دسته‌بندی",
    ]

    if any(
        signal in query
        for signal in analytics_signals
    ):

        if "analytics" not in capabilities:

            capabilities.append(
                "analytics"
            )

        # Analytics does not imply product search.
        if primary == "analytics":

            if "search" in capabilities:
                capabilities.remove(
                    "search"
                )

    # --------------------------------------------------------
    # Remove unnecessary review capability from pure
    # satisfaction-based search.
    # --------------------------------------------------------

    if (
        primary == "search"
        and not has_review_signal
        and not has_review_question
    ):

        capabilities = [
            capability
            for capability in capabilities
            if capability != "review"
        ]

    # --------------------------------------------------------
    # Remove duplicates while preserving order
    # --------------------------------------------------------

    return list(
        dict.fromkeys(
            capabilities
        )
    )

# ============================================================
# Execution Plan
# ============================================================

def build_execution_plan(
    query,
    query_type=None,
):
    """
    Convert capabilities into an executable plan.
    """

    if query_type is None:

        query_type = detect_query_type(
            query
        )

    capabilities = detect_requested_capabilities(
        query
    )

    # --------------------------------------------------------
    # Search
    # --------------------------------------------------------

    need_product_search = (
        "search" in capabilities
        or "comparison" in capabilities
    )

    # --------------------------------------------------------
    # Review
    # --------------------------------------------------------

    need_review_search = (
        "review" in capabilities
        or "comparison" in capabilities
    )

    # --------------------------------------------------------
    # Comparison
    # --------------------------------------------------------

    need_comparison = (
        "comparison" in capabilities
    )

    # --------------------------------------------------------
    # Analytics
    # --------------------------------------------------------

    need_analytics = (
        "analytics" in capabilities
    )

    # --------------------------------------------------------
    # LLM
    # --------------------------------------------------------

    need_llm = (
        "review" in capabilities
        or "comparison" in capabilities
        or "analytics" in capabilities
    )

    # --------------------------------------------------------
    # Return
    # --------------------------------------------------------

    return {
        "intent": query_type,

        "capabilities": capabilities,

        "need_product_search": (
            need_product_search
        ),

        "need_review_search": (
            need_review_search
        ),

        "need_comparison": (
            need_comparison
        ),

        "need_analytics": (
            need_analytics
        ),

        "need_llm": (
            need_llm
        ),
    }


# ============================================================
# Search Plan
# ============================================================

def build_search_plan(
    query,
    known_brands=None,
    known_concepts=None,
):
    """Build structured product search plan."""

    query = normalize_query(
        query
    )

    min_price, max_price = extract_price_range(
        query
    )

    concept = extract_concept(
        query,
        known_concepts=known_concepts
    )

    return {
        "query": query,

        "concept": concept,

        "concept_terms": get_concept_terms(
            concept
        ),

        "brand": extract_brand(
            query,
            known_brands=known_brands
        ),

        "min_price": min_price,

        "max_price": max_price,

        "require_satisfaction": (
            requires_satisfaction(
                query
            )
        ),

        "sub_category": extract_sub_category(
            query
        ),
    }


# ============================================================
# Main Parser
# ============================================================

def parse_query(
    query,
    known_brands=None,
    known_concepts=None,
):
    """Main parser."""

    if not query or not str(query).strip():

        raise ValueError(
            "Query cannot be empty."
        )

    return build_search_plan(
        query=query,
        known_brands=known_brands,
        known_concepts=known_concepts,
    )


# ============================================================
# Full Query Analysis
# ============================================================

def analyze_query(
    query,
    known_brands=None,
    known_concepts=None,
):
    """
    Full query analysis.

    Returns:

        query
        type
        plan
        execution_plan
    """

    normalized = normalize_query(
        query
    )

    query_type = detect_query_type(
        normalized
    )

    return {
        "query": normalized,

        "type": query_type,

        "plan": parse_query(
            normalized,
            known_brands=known_brands,
            known_concepts=known_concepts,
        ),

        "execution_plan": build_execution_plan(
            normalized,
            query_type=query_type,
        ),
    }

