import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

COMMENTS_PATH = "data/comments_clean.csv"

# تابعی برای خواندن فایل نظرات
def load_comments() -> pd.DataFrame:
    print("در حال بارگذاری فایل نظرات ...")
    return pd.read_csv(COMMENTS_PATH)


# پیدا کردن تمام نظرای مربوط به یک محصول خاص
def get_reviews_for_product(product_id: int, comments_df: pd.DataFrame) -> pd.DataFrame:
    return comments_df[comments_df["product_id"] == product_id].reset_index(drop=True)

# پیدا کردن مشابه ترین نظرها نسبت به سوالی که کاربر پرسیده
def find_evidence(question: str, product_reviews: pd.DataFrame, top_k: int = 5) -> pd.DataFrame:

    # اگر که محصول هیچ نظری نداشت...
    if len(product_reviews) == 0:
        print("هیچ نظری برای این محصول پیدا نشد.")
        return product_reviews

    # متن هر نظر رو از ترکیب عنوانش و متن اصلی میسازیم
    review_texts = (
        product_reviews["title"].fillna("") + " " + product_reviews["body"].fillna("")
    )

    vectorizer = TfidfVectorizer()

    # تبدیل همه ی نظرا به بردار
    review_matrix = vectorizer.fit_transform(review_texts)
    # تبدیل سوال کاربر به بردار و اینکه از وکتورایزر استفاده میکنیم که فضای برداری یکی باشه
    question_vector = vectorizer.transform([question])

    # محاسبه شباهت کسینوسی سوال با همه نظرا
    similarities = cosine_similarity(question_vector, review_matrix).flatten()

    top_k = min(top_k, len(product_reviews))
    top_idx = similarities.argsort()[::-1][:top_k]

    evidence = product_reviews.iloc[top_idx][
        ["id", "title", "body", "rate", "recommendation_status", "likes"]
    ].copy()

    # میزان ارتباط هر نظر با سوالش رو به نتیجه ها اضافه میکنیم
    evidence["relevance_score"] = similarities[top_idx].round(3)

    return evidence


def summarize_product_satisfaction(product_reviews: pd.DataFrame) -> None:
    print(f"\nتعداد کل نظرات این محصول: {len(product_reviews):,}")
    print("توزیع وضعیت پیشنهاد خرید:")
    print(product_reviews["recommendation_status"].value_counts(dropna=False))


def main():
    comments_df = load_comments()

    # پیدا کردن یه محصولی که بیشترین نظر رو داره
    top_product_id = comments_df["product_id"].value_counts().idxmax()
    print(f"\nمحصول انتخاب‌شده برای تست (بیشترین تعداد نظر): product_id = {top_product_id}")

    # جدا کردن همه ی نظرای مربوط به محصول انتخاب شده
    product_reviews = get_reviews_for_product(top_product_id, comments_df)
    summarize_product_satisfaction(product_reviews)

    test_questions = [
        "کیفیت این محصول چطوره؟",
        "ایرادهای این محصول چیه؟",
    ]

    for question in test_questions:
        print(f"سؤال: {question}")
        evidence = find_evidence(question, product_reviews, top_k=5)
        print(evidence.to_string(index=False))


if __name__ == "__main__":
    main()
