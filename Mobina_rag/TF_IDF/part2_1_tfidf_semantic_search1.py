import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

PRODUCTS_PATH = "data/products_clean.csv"

# تابعی برای خواندن فایل محصولات
def load_products() -> pd.DataFrame:
    return pd.read_csv(PRODUCTS_PATH)

# مشخص می‌کنیم چه اطلاعاتی از هر محصول وارد مدل بشه
def build_search_text(df: pd.DataFrame) -> pd.Series:
    return df["title_fa"].fillna("") + " " + df["Brand"].fillna("")


# تابع اصلی جست‌وجو بر اساس شباهت
def search_products(query: str, vectorizer, tfidf_matrix, df: pd.DataFrame, top_k: int = 5) -> pd.DataFrame:
    # سوال کاربر رو به بردار تبدیل میکنیم
    query_vector = vectorizer.transform([query])
    # شباهت کسینوسی سوال رو با همه محصولا محاسبه میکنیم
    similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()

    # محصولا رو براساس بیشترین شباهتشون مرتب میکنیم
    top_indices = similarities.argsort()[::-1][:top_k]

    # اطلاعاتی که مهمن رو نگه میداریم
    results = df.iloc[top_indices][["id", "title_fa", "Brand", "Price"]].copy()

    # امتیاز شباهت محصول رو به نتیجه ها اضافه میکنیم
    results["similarity_score"] = similarities[top_indices].round(3)

    return results


def main():
    print("در حال بارگذاری محصولات ...")
    df = load_products()

    # ساخت متن جستوجو از برند و عنوان هرکدوم از محصولا
    search_texts = build_search_text(df)

    print("در حال ساخت TF-IDF ...")
    vectorizer = TfidfVectorizer()
    # تبدیل متن همه ی  محصولا به ماتریس
    tfidf_matrix = vectorizer.fit_transform(search_texts)

    test_queries = [
        "کیف چرم مردانه",
        "کرم ضد آفتاب",
    ]

    # تست هر سوال
    for query in test_queries:
        print(f"سؤال: {query}")
        # پیدا کردن 5 محصول شبیه به سوال
        results = search_products(query, vectorizer, tfidf_matrix, df, top_k=5)
        print(results.to_string(index=False))
        print()


if __name__ == "__main__":
    main()
