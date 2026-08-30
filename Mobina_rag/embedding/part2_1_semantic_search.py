import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

PRODUCTS_PATH = "data/products_clean.csv"
EMBEDDINGS_PATH = "data/product_embeddings.npy"

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

SAMPLE_SIZE = None

# یک تابع برای خواندن محصولات
def load_products() -> pd.DataFrame:
    df = pd.read_csv(PRODUCTS_PATH)

    #بررسی می‌کنیم که آیا SAMPLE_SIZE مقدار دارد یا نه.
    if SAMPLE_SIZE is not None:
        df = df.sample(n=SAMPLE_SIZE, random_state=42).reset_index(drop=True)
        print(f"حالت تست: فقط {SAMPLE_SIZE:,} محصول به‌صورت تصادفی انتخاب شد.")

    # برگردوندن لیست محصولات
    return df

# یک تابع تعریف می‌کنیم که مشخص می‌کند چه متنی از محصول رو به مدل بدهیم.
def build_search_text(df: pd.DataFrame) -> pd.Series:
        # عنوان محصول و برند را به هم می‌چسبانیم
    return df["title_fa"].fillna("") + " " + df["Brand"].fillna("")

# تابعی تعریف می‌کنیم که متن محصولات رو گرفته و برای هرکدام embedding بسازه.
def compute_embeddings(model: SentenceTransformer, texts: pd.Series) -> np.ndarray:
    print(f"در حال ساخت embedding برای {len(texts):,} محصول ...")

    # مدل متن محصولات را به embedding تبدیل می‌کند.
    embeddings = model.encode(
        texts.tolist(),
        batch_size=64, #هر بار ۶۴ متن را با هم پردازش کن
        show_progress_bar=True, # نوار پیشرفت
        convert_to_numpy=True, # خروجی رو به ارایه نامپای تبدیل کنش
    )
    return embeddings

# وظیفه‌ی اصلی جست‌وجو را دارد این تابع زیر
# به ترتیب سوال کاربر و مدل و محصولات و امبدینگ همه محصولا و تعداد نتابج ورودی های تابعن
def search_products(query: str, model: SentenceTransformer, df: pd.DataFrame,
                     embeddings: np.ndarray, top_k: int = 5) -> pd.DataFrame:
    # تبدیل سؤال کاربر به embedding
    query_embedding = model.encode([query], convert_to_numpy=True)

    # مقایسه‌ی ایمبدینگ سوال با همه ی محصولا
    similarities = model.similarity(query_embedding, embeddings).numpy().flatten()
    # بهترینشو انتخاب میکنه و از بیشترین به کمترین پنجتای اول رو نمایش میده
    top_indices = similarities.argsort()[::-1][:top_k]

    # اطلاعات آی دی و عنوان و برند و قیمت محصول رو نگه میداریم
    results = df.iloc[top_indices][["id", "title_fa", "Brand", "Price"]].copy()
    # امتیاز شباهت رو تو یه ستون میاریم
    results["similarity_score"] = similarities[top_indices].round(3)

    return results

# روند کلی برنامه
def main():
    print("در حال بارگذاری مدل embedding ...")
    model = SentenceTransformer(MODEL_NAME)

    # خوندن محصولا
    products_df = load_products()
    # ساخت متن جستجو برا هر محصول
    search_texts = build_search_text(products_df)

    # همه ی متن محصول رو به ایمبدینگ تبدیل میکنه
    embeddings = compute_embeddings(model, search_texts)
    np.save(EMBEDDINGS_PATH, embeddings)
    print(f"embeddingها ذخیره شدند در: {EMBEDDINGS_PATH}\n")

    # همون سؤال‌های قبلی رو تست می‌کنیم تا با نتیجه‌ی TF-IDF مقایسه کنیم
    test_queries = [
        "کیف چرم مردانه",
        "کرم ضد آفتاب",
    ]

    # اجرای جستجوی اون تستا
    for query in test_queries:
        print(f"سؤال: {query}")
        # پنج محصولی که شبیه بهش هستن رو پیدا میکنه
        results = search_products(query, model, products_df, embeddings, top_k=5)
        # نمایش نتیجع
        print(results.to_string(index=False))
        print()

# اجرای برنامه
if __name__ == "__main__":
    main()
