import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

PRODUCTS_PATH = "data/products_clean.csv"


def search_products(query: str, products_df: pd.DataFrame, vectorizer, tfidf_matrix, top_k: int = 5):
    #"یک سؤال می‌گیرد و شبیه‌ترین محصولات را بر اساس عنوان برمی‌گرداند.

    # سؤال کاربر رو با همون vectorizer که برای عنوان‌ها آموزش دیدیم، عددی می‌کنیم
    query_vector = vectorizer.transform([query])

    # شباهت کسینوسی بین سؤال و تمام عنوان‌ های محصولات
    similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()

    # ایندکس بهترین نتایج (بیشترین شباهت)
    top_indices = similarities.argsort()[::-1][:top_k]

    results = products_df.iloc[top_indices].copy()
    results["similarity_score"] = similarities[top_indices]

    return results[["id", "title_fa", "Price", "Brand", "similarity_score"]]


def main():
    print("در حال خواندن فایل محصولات تمیزشده...")
    products_df = pd.read_csv(PRODUCTS_PATH)

    print(f"تعداد محصولات: {len(products_df):,}")

    # ساخت TF-IDF فقط بر اساس عنوان محصول
    print("در حال ساخت مدل TF-IDF بر اساس عنوان محصولات...")
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(products_df["title_fa"].fillna(""))

    # چند سؤال نمونه برای تست
    test_queries = [
        "کیف چرم مردانه",
        "گوشی سامسونگ",
        "کرم ضد آفتاب",
    ]

    for query in test_queries:
        print("\n" + "=" * 60)
        print(f"سؤال: {query}")
        print("=" * 60)
        results = search_products(query, products_df, vectorizer, tfidf_matrix, top_k=5)
        print(results.to_string(index=False))


if __name__ == "__main__":
    main()
