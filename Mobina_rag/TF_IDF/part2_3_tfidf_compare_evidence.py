import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from evidence_tfidf_version import load_comments, get_reviews_for_product, find_evidence

PRODUCTS_PATH = "data/products_clean.csv"

# گرفتن اطلاعات یه محصول خاص از فایل محصولات
def get_product_info(product_id: int, products_df: pd.DataFrame) -> dict:
    # پیدا کردن محصول موردنظر بر اساس آیدی
    row = products_df[products_df["id"] == product_id]

    # اگر محصول پیدا نشد...
    if len(row) == 0:
        return None

    # گرفتن اطلاعات اولین ردیف پیدا شده
    row = row.iloc[0]
    #قرار دادن اطلاعات موردنیاز محصول
    return {
        "id": product_id,
        "title": row["title_fa"],
        "brand": row["Brand"],
        "price": row["Price"],
        "rate": row["Rate"],
        "rate_cnt": row["Rate_cnt"],
    }


def collect_comparison_evidence(
    product_ids: list,
    question: str,
    products_df: pd.DataFrame,
    comments_df: pd.DataFrame,
    top_k: int = 5,
) -> dict:
    # دیکشنری برای ذخیره نتایج همه محصولا
    results = {}

    # بررسی هر محصول به صورت جداگانه
    for product_id in product_ids:
        # گرفتن اطلاعات اصلی محصول
        info = get_product_info(product_id, products_df)

    # اگر محصول در فایل محصولا وجود نداشت...
        if info is None:
            print(f"هشدار: محصول با id={product_id} در فایل محصولات پیدا نشد.")
            continue

        # پیدا کردن همه ی نظرای مربوط به این محصول
        product_reviews = get_reviews_for_product(product_id, comments_df)

        # پیدا کردن شبیه ترین نظرا به سوال کاربر
        evidence = find_evidence(question, product_reviews, top_k=top_k)

        # محاسبه درصد هر وضعیت پیشنهاد خرید
        satisfaction_summary = (
            product_reviews["recommendation_status"].value_counts(normalize=True).round(2).to_dict()
        )

        # ذخیره تمام اطلاعات این محصول
        results[product_id] = {
            "info": info,
            "evidence": evidence,
            "satisfaction_summary": satisfaction_summary,
            "total_reviews": len(product_reviews),
        }

    return results

# نمایش نتایج مقایسه محصولات
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

    # پیدا کردن دو محصولی که بیشترین تعداد نظرها رو دارن
    top_two_products = comments_df["product_id"].value_counts().head(2).index.tolist()
    print(f"دو محصول انتخاب‌شده برای تست مقایسه: {top_two_products}")

    question = "کیفیت این محصول چطوره؟"

    results = collect_comparison_evidence(
        top_two_products, question, products_df, comments_df, top_k=3
    )

    print_comparison(results)


if __name__ == "__main__":
    main()
