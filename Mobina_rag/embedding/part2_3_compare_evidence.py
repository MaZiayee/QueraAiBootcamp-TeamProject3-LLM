import pandas as pd
from sentence_transformers import SentenceTransformer

# از توابعی که توی این فایل زیر ساخته بودم استفاده میکنم
from evidence import load_comments, get_reviews_for_product, find_evidence

PRODUCTS_PATH = "data/products_clean.csv"
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

# اطلاعات یه محصول خاص رو از فایل محصول پیدا میکنه
def get_product_info(product_id: int, products_df: pd.DataFrame) -> dict:
    # اون محصول رو از بین همه محصولا پیدا میکنه
    row = products_df[products_df["id"] == product_id]


    # اگر وجود نداشت....
    if len(row) == 0:
        return None

    row = row.iloc[0]
    # اطلاعات مورد نیاز محصول رو توی یه یکشنری میزاریم
    return {
        "id": product_id,
        "title": row["title_fa"],
        "brand": row["Brand"],
        "price": row["Price"],
        "rate": row["Rate"],
        "rate_cnt": row["Rate_cnt"],
    }

# در واقع این تابع برای هر محصولی که قراره مقایسه بشه اصلاعاتی که میخوایم رو پیدا یا حساب میکنه
def collect_comparison_evidence(
    product_ids: list,
    question: str,
    products_df: pd.DataFrame,
    comments_df: pd.DataFrame,
    model: SentenceTransformer,
    top_k: int = 5,
) -> dict:
    
    results = {}

    for product_id in product_ids:
        # گرفتن اطلاعات محصول
        info = get_product_info(product_id, products_df)

        # اگر پیدا نشد....
        if info is None:
            print(f"هشدار: محصول با id={product_id} در فایل محصولات پیدا نشد.")
            continue

        # از تابعی که قبلا نوشتیم استفاده میکنیم تا همه نظرهای مربوط به محصول رو پیدا کنیم
        product_reviews = get_reviews_for_product(product_id, comments_df)
        # پیدا کردن نظرای مرتبط
        evidence = find_evidence(question, product_reviews, model, top_k=top_k)

        # خلاصه‌ی آماری رضایت کاربران از روی نظرات واقعی.... چند درصد پیشنهاد میکنن چند درصد پیشنهاد نمیکنن
        satisfaction_summary = (
            product_reviews["recommendation_status"].value_counts(normalize=True).round(2).to_dict()
        )

        results[product_id] = {
            "info": info,
            "evidence": evidence,
            "satisfaction_summary": satisfaction_summary,
            "total_reviews": len(product_reviews),
        }

    return results

# نمایش نتایج
def print_comparison(results: dict):
    
    for product_id, data in results.items():
        
        print(f"محصول: {data['info']['title']} (id={product_id})")
        
        print(f"برند: {data['info']['brand']} | قیمت: {data['info']['price']:,} | "
              f"امتیاز: {data['info']['rate']} از {data['info']['rate_cnt']} رأی")
        print(f"تعداد کل نظرات: {data['total_reviews']:,}")
        print(f"توزیع رضایت (بر اساس نظرات واقعی): {data['satisfaction_summary']}")

        print("\nمرتبط‌ترین نظرات نسبت به سؤال:")
        print(data["evidence"][["id", "body", "relevance_score"]].to_string(index=False))


def main():
    products_df = pd.read_csv(PRODUCTS_PATH)
    comments_df = load_comments()
    model = SentenceTransformer(MODEL_NAME)

    #  دو محصولی که بیشترین تعداد نظر را دارند انتخاب می‌کنیم
    top_two_products = comments_df["product_id"].value_counts().head(2).index.tolist()
    print(f"دو محصول انتخابشده برای تست مقایسه: {top_two_products}")

    question = "کیفیت این محصول چطوره؟"

    results = collect_comparison_evidence(
        top_two_products, question, products_df, comments_df, model, top_k=3
    )

    print_comparison(results)


if __name__ == "__main__":
    main()
