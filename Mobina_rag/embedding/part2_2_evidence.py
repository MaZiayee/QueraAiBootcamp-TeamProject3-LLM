import pandas as pd
from sentence_transformers import SentenceTransformer

COMMENTS_PATH = "data/comments_clean.csv"
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"


def load_comments() -> pd.DataFrame:
    print("در حال بارگذاری فایل نظرات ...")
    return pd.read_csv(COMMENTS_PATH)

# نظرای مربوط به یه محصول خاص رو پیدا میکنه
def get_reviews_for_product(product_id: int, comments_df: pd.DataFrame) -> pd.DataFrame:

    return comments_df[comments_df["product_id"] == product_id].reset_index(drop=True)

# تابع زیر در واقع از بین نظرای یه محصول اون نظری که حیلی نزدیک سوال کاربره رو پیدا میکنه
def find_evidence(
    question: str,
    product_reviews: pd.DataFrame,
    model: SentenceTransformer,
    top_k: int = 5,
) -> pd.DataFrame:

    # اگر نظری وجود نداشت بریا اون محصول
    if len(product_reviews) == 0:
        print("هیچ نظری برای این محصول پیدا نشد.")
        return product_reviews

    # متن هر نظر رو از عنوان + بدنه می‌سازیم 
    review_texts = (
        product_reviews["title"].fillna("") + " " + product_reviews["body"].fillna(""))
    # تبدیل نظرا به ایمبدینگ 
    review_embeddings = model.encode(review_texts.tolist(), convert_to_numpy=True)
    # تبدیل سوال کاربر به ایمبدینگ
    question_embedding = model.encode([question], convert_to_numpy=True)

    # محاسبه شباهت سوالا و نظرا
    similarities = model.similarity(question_embedding, review_embeddings).numpy().flatten()
    # پنج تا نظر رو نشون بده ولی اگر کمتر بود همونو نشون بده
    top_k = min(top_k, len(product_reviews))

    top_idx = similarities.argsort()[::-1][:top_k]

    # اطلاعاتی ک همیخوایم رو نگه میداریم
    evidence = product_reviews.iloc[top_idx][
        ["id", "title", "body", "rate", "recommendation_status", "likes"]
    ].copy()
    # سه ستون اظافه میکنیم که نشون میده هر نظر چقدر به سوال کاربر مرتبط بوده
    evidence["relevance_score"] = similarities[top_idx].round(3)

    return evidence

# کار این تابع اینه که یه خلاصه اماری از نظرای یه محصول بسازه
def summarize_product_satisfaction(product_reviews: pd.DataFrame) -> None:
    print(f"\nتعداد کل نظرات این محصول: {len(product_reviews):,}")
    print("توزیع وضعیت پیشنهاد خرید:")
    # تعداد اینکه چند نفر محصول رو پیشنهاد کردن چند نفر رد کردن
    print(product_reviews["recommendation_status"].value_counts(dropna=False))


def main():
    comments_df = load_comments()

    # برای دمو، محصولی را انتخاب می‌کنیم که بیشترین تعداد نظر را دارد
    top_product_id = comments_df["product_id"].value_counts().idxmax()
    print(f"\nمحصول انتخاب‌شده برای تست (بیشترین تعداد نظر): product_id = {top_product_id}")

    # جدا کردن همه نظرای اون محصول
    product_reviews = get_reviews_for_product(top_product_id, comments_df)

    # تعداد نظرا و وضعیت کسایی که خریدش رو پیشنهاد میدن رو میگه
    summarize_product_satisfaction(product_reviews)

    model = SentenceTransformer(MODEL_NAME)

    test_questions = [
        "کیفیت این محصول چطوره؟",
        "ایرادهای این محصول چیه؟",
    ]

    for question in test_questions:
        print(f"سؤال: {question}")
        evidence = find_evidence(question, product_reviews, model, top_k=5)
        print(evidence.to_string(index=False))


if __name__ == "__main__":
    main()
