import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

PRODUCTS_PATH = "data/products_clean.csv"
EMBEDDINGS_PATH = "data/product_embeddings.npy"
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

# سؤال‌های توصیفی (عمداً از کلمه‌ی دقیق محصول استفاده نشده)
TEST_QUERIES = [
    "یه چیزی برای محافظت از پوست در برابر نور خورشید",
    "کیفی که هر روز بشه باهاش بیرون رفت",
    "وسیله‌ای برای نگهداری و حمل وسایل شخصی",
    "دستگاهی قابل‌حمل برای تماشای فیلم و اینترنت",
]

# تابع جست‌وجوی TF-IDF
def search_tfidf(query, df, vectorizer, tfidf_matrix, top_k=5):
    # تبدیل سؤال به TF-IDF
    query_vec = vectorizer.transform([query])
    # شباهتی که سوال با همه عنوان های محصول داره رو حساب میکنیم
    sims = cosine_similarity(query_vec, tfidf_matrix).flatten()
    # بهترینشو انتخاب میکنیم و پنج تای برترشو نمایش میدیم
    top_idx = sims.argsort()[::-1][:top_k]
    # فقط عنوان و برندو نگه میداریم
    result = df.iloc[top_idx][["title_fa", "Brand"]].copy()
    # امتیاز شباهت رو هم اضافه میکنیم
    result["score"] = sims[top_idx].round(3)
    return result


def search_embedding(query, model, df, embeddings, top_k=5):
    # سوالی که کاربر پرسیده رو به ایمبدینگ تبدیل میکنیم
    query_emb = model.encode([query], convert_to_numpy=True)
    # ایمبدینگ سوال رو با همه محصولا مقایسه میکینم
    sims = model.similarity(query_emb, embeddings).numpy().flatten()
    top_idx = sims.argsort()[::-1][:top_k]
    result = df.iloc[top_idx][["title_fa", "Brand"]].copy()
    result["score"] = sims[top_idx].round(3)
    return result

# روند اصلی برنامه
def main():
    print("در حال بارگذاری داده و مدل‌ها ...")
    df = pd.read_csv(PRODUCTS_PATH)
    embeddings = np.load(EMBEDDINGS_PATH)

    # مطمئن می‌شویم تعداد ردیف‌ها با تعداد embeddingها یکی است
    assert len(df) == len(embeddings), "تعداد محصولات با تعداد embeddingها همخوانی ندارد!"

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df["title_fa"].fillna(""))

    model = SentenceTransformer(MODEL_NAME)

    for query in TEST_QUERIES:
        print(f"سؤال: {query}")

        print("\n نتایج TF-IDF")
        print(search_tfidf(query, df, vectorizer, tfidf_matrix).to_string(index=False))

        print("\n نتایج Embedding ")
        print(search_embedding(query, model, df, embeddings).to_string(index=False))


if __name__ == "__main__":
    main()
