#!/usr/bin/env python3
"""
Seed the database with sample beauty products
Run with: python -m data.seed_products
"""

from services.mongo_service import mongo_service
from datetime import datetime, timedelta
import random
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def seed_products():
    """Seed the database with sample beauty products"""
    
    print("🌱 Seeding beauty products database...")
    
    # Sample beauty products in Arabic
    beauty_products = [
        # Skincare
        {
            "product_id": "SKIN001",
            "name": "كريم مرطب للوجه من نيڤيا",
            "description": "كريم مرطب يومي للوجه يناسب جميع أنواع البشرة، يحتوي على فيتامين E وزيت الجوجوبا للترطيب العميق",
            "price": 45.99,
            "original_price": 59.99,
            "currency": "SAR",
            "category": "العناية بالبشرة",
            "subcategory": "الترطيب",
            "brand": "نيڤيا",
            "in_stock": True,
            "stock_quantity": 150,
            "rating": 4.5,
            "review_count": 128,
            "attributes": {
                "skin_type": ["جميع أنواع البشرة"],
                "volume_ml": 100,
                "spf": False,
                "fragrance": "خالي من العطر",
                "key_ingredients": ["فيتامين E", "زيت الجوجوبا"]
            },
            "image_url": "https://example.com/images/nivea_cream.jpg",
            "created_at": datetime.now() - timedelta(days=30)
        },
        {
            "product_id": "SKIN002",
            "name": "سيروم فيتامين سي من ذا أورديناري",
            "description": "سيروم مضاد للأكسدة بتركيز 15% فيتامين سي، يضيء البشرة ويقلل التجاعيد والبقع الداكنة",
            "price": 120.00,
            "original_price": 150.00,
            "currency": "SAR",
            "category": "العناية بالبشرة",
            "subcategory": "السيرومات",
            "brand": "ذا أورديناري",
            "in_stock": True,
            "stock_quantity": 75,
            "rating": 4.7,
            "review_count": 256,
            "attributes": {
                "skin_type": ["جميع أنواع البشرة"],
                "vitamin_c_percentage": 15,
                "volume_ml": 30,
                "cruelty_free": True,
                "vegan": True
            },
            "image_url": "https://example.com/images/vitamin_c_serum.jpg",
            "created_at": datetime.now() - timedelta(days=15)
        },
        {
            "product_id": "SKIN003",
            "name": "ماسك طين للبشرة الدهنية من لوريال",
            "description": "ماسك طين أخضر يمتص الزيوت الزائدة وينظف المسام بعمق، مناسب للبشرة الدهنية والمختلطة",
            "price": 28.75,
            "original_price": 35.00,
            "currency": "SAR",
            "category": "العناية بالبشرة",
            "subcategory": "المقشرات والماسكات",
            "brand": "لوريال",
            "in_stock": True,
            "stock_quantity": 67,
            "rating": 4.3,
            "review_count": 89,
            "attributes": {
                "skin_type": ["دهنية", "مختلطة"],
                "volume_ml": 75,
                "clay_type": "طين أخضر",
                "fragrance": "منعش"
            },
            "image_url": "https://example.com/images/clay_mask.jpg",
            "created_at": datetime.now() - timedelta(days=45)
        },
        
        # Hair Care
        {
            "product_id": "HAIR001",
            "name": "شامبو للشعر الجاف من بانين",
            "description": "شامبو مغذي للشعر الجاف والمتقصف، يحتوي على زيت الأرجان وفيتامين B5 لإصلاح الشعر التالف",
            "price": 32.50,
            "original_price": 32.50,
            "currency": "SAR",
            "category": "العناية بالشعر",
            "subcategory": "شامبو",
            "brand": "بانين",
            "in_stock": True,
            "stock_quantity": 85,
            "rating": 4.4,
            "review_count": 142,
            "attributes": {
                "hair_type": ["جاف", "متقصف"],
                "volume_ml": 400,
                "sulfate_free": True,
                "key_ingredients": ["زيت الأرجان", "فيتامين B5"]
            },
            "image_url": "https://example.com/images/pantene_shampoo.jpg",
            "created_at": datetime.now() - timedelta(days=60)
        },
        {
            "product_id": "HAIR002",
            "name": "بلسم مغذي للشعر من هيربال إيسنسز",
            "description": "بلسم يرطب الشعر ويعطيه لمعاناً طبيعياً، خالي من السلفات والسيليكون، برائحة اللافندر",
            "price": 29.99,
            "original_price": 35.00,
            "currency": "SAR",
            "category": "العناية بالشعر",
            "subcategory": "بلسم",
            "brand": "هيربال إيسنسز",
            "in_stock": True,
            "stock_quantity": 92,
            "rating": 4.6,
            "review_count": 178,
            "attributes": {
                "hair_type": ["جميع أنواع الشعر"],
                "volume_ml": 400,
                "sulfate_free": True,
                "silicone_free": True,
                "fragrance": "لافندر"
            },
            "image_url": "https://example.com/images/herbal_conditioner.jpg",
            "created_at": datetime.now() - timedelta(days=25)
        },
        
        # Makeup
        {
            "product_id": "MAKE001",
            "name": "أحمر شفاه مات من ماك",
            "description": "أحمر شفاه بتشطيب مات، لون كلاسيكي أحمر، يدوم طويلاً ولا يجفف الشفاه",
            "price": 89.00,
            "original_price": 110.00,
            "currency": "SAR",
            "category": "مكياج",
            "subcategory": "أحمر شفاه",
            "brand": "ماك",
            "in_stock": True,
            "stock_quantity": 42,
            "rating": 4.8,
            "review_count": 312,
            "attributes": {
                "color": "أحمر كلاسيكي",
                "finish": "مات",
                "weight_g": 3,
                "cruelty_free": True,
                "long_wearing": True
            },
            "image_url": "https://example.com/images/mac_lipstick.jpg",
            "created_at": datetime.now() - timedelta(days=10)
        },
        {
            "product_id": "MAKE002",
            "name": "أساس سائل من إستي لودر",
            "description": "أساس سائل بتغطية متوسطة إلى عالية، يناسب جميع أنواع البشرة، متوفر بعدة درجات لونية",
            "price": 185.00,
            "original_price": 220.00,
            "currency": "SAR",
            "category": "مكياج",
            "subcategory": "أساس",
            "brand": "إستي لودر",
            "in_stock": True,
            "stock_quantity": 38,
            "rating": 4.7,
            "review_count": 198,
            "attributes": {
                "coverage": "متوسطة إلى عالية",
                "finish": "شبه مات",
                "volume_ml": 30,
                "spf": 15,
                "skin_type": ["جميع أنواع البشرة"]
            },
            "image_url": "https://example.com/images/estee_lauder_foundation.jpg",
            "created_at": datetime.now() - timedelta(days=20)
        },
        {
            "product_id": "MAKE003",
            "name": "بلاشر وردي من بينفيت",
            "description": "بلاشر بودرة بلون وردي طبيعي يعطي الخدود تورداً جميلاً، بتشطيب مات",
            "price": 65.00,
            "original_price": 65.00,
            "currency": "SAR",
            "category": "مكياج",
            "subcategory": "بلاشر وبرونزر",
            "brand": "بينفيت",
            "in_stock": True,
            "stock_quantity": 31,
            "rating": 4.5,
            "review_count": 124,
            "attributes": {
                "color": "وردي طبيعي",
                "finish": "مات",
                "weight_g": 5,
                "cruelty_free": False
            },
            "image_url": "https://example.com/images/benefit_blush.jpg",
            "created_at": datetime.now() - timedelta(days=35)
        },
        
        # Fragrances
        {
            "product_id": "FRAG001",
            "name": "عطر فلورال من شانيل",
            "description": "عطر نسائي برائحة زهور الربيع، مزيج من الياسمين والورد والبيرغاموت، تدوم طويلاً",
            "price": 350.00,
            "original_price": 420.00,
            "currency": "SAR",
            "category": "العطور",
            "subcategory": "عطور نسائية",
            "brand": "شانيل",
            "in_stock": True,
            "stock_quantity": 23,
            "rating": 4.9,
            "review_count": 267,
            "attributes": {
                "fragrance_type": "فلورال",
                "volume_ml": 50,
                "concentration": "Eau de Parfum",
                "longevity": "8-10 ساعات"
            },
            "image_url": "https://example.com/images/chanel_perfume.jpg",
            "created_at": datetime.now() - timedelta(days=5)
        },
        {
            "product_id": "FRAG002",
            "name": "عطر خشبي للرجال من ديور",
            "description": "عطر رجالي برائحة خشبية، مزيج من خشب الصندل والفانيليا والبهارات، رائحة قوية وجذابة",
            "price": 320.00,
            "original_price": 380.00,
            "currency": "SAR",
            "category": "العطور",
            "subcategory": "عطور رجالية",
            "brand": "ديور",
            "in_stock": True,
            "stock_quantity": 41,
            "rating": 4.8,
            "review_count": 189,
            "attributes": {
                "fragrance_type": "خشبي",
                "volume_ml": 100,
                "concentration": "Eau de Toilette",
                "longevity": "6-8 ساعات"
            },
            "image_url": "https://example.com/images/dior_perfume.jpg",
            "created_at": datetime.now() - timedelta(days=12)
        },
        
        # Body Care
        {
            "product_id": "BODY001",
            "name": "لوشن مرطب للجسم من فازلين",
            "description": "لوشن مرطب سريع الامتصاص للجسم، يحتوي على فيتامين E لترطيب 24 ساعة",
            "price": 24.99,
            "original_price": 29.99,
            "currency": "SAR",
            "category": "العناية بالجسم",
            "subcategory": "مرطبات الجسم",
            "brand": "فازلين",
            "in_stock": True,
            "stock_quantity": 156,
            "rating": 4.4,
            "review_count": 231,
            "attributes": {
                "skin_type": ["جميع أنواع البشرة"],
                "volume_ml": 400,
                "fragrance": "خالي من العطر",
                "key_ingredients": ["فيتامين E"]
            },
            "image_url": "https://example.com/images/vaseline_lotion.jpg",
            "created_at": datetime.now() - timedelta(days=50)
        },
        
        # Out of stock items
        {
            "product_id": "SKIN004",
            "name": "تونر من لا روش بوزاي",
            "description": "تونر لتهدئة البشرة وموازنة درجة الحموضة، مناسب للبشرة الحساسة",
            "price": 75.00,
            "original_price": 90.00,
            "currency": "SAR",
            "category": "العناية بالبشرة",
            "subcategory": "تونر",
            "brand": "لا روش بوزاي",
            "in_stock": False,
            "stock_quantity": 0,
            "rating": 4.6,
            "review_count": 145,
            "attributes": {
                "skin_type": ["حساسة", "جميع أنواع البشرة"],
                "volume_ml": 200,
                "alcohol_free": True,
                "fragrance": "خالي من العطر"
            },
            "image_url": "https://example.com/images/la_roche_toner.jpg",
            "created_at": datetime.now() - timedelta(days=90)
        }
    ]
    
    # Current offers
    current_offers = [
        {
            "offer_id": "OFF001",
            "title": "تخفيضات العناية بالبشرة",
            "description": "خصم 20% على جميع منتجات العناية بالبشرة",
            "discount_percentage": 20,
            "category": "العناية بالبشرة",
            "starts_at": datetime.now() - timedelta(days=1),
            "expires_at": datetime.now() + timedelta(days=7),
            "active": True,
            "image_url": "https://example.com/offers/skincare_sale.jpg",
            "created_at": datetime.now() - timedelta(days=1)
        },
        {
            "offer_id": "OFF002",
            "title": "عرض العطور الفاخرة",
            "description": "خصم 15% على جميع العطور الأصلية",
            "discount_percentage": 15,
            "category": "العطور",
            "starts_at": datetime.now() - timedelta(days=2),
            "expires_at": datetime.now() + timedelta(days=5),
            "active": True,
            "image_url": "https://example.com/offers/perfume_sale.jpg",
            "created_at": datetime.now() - timedelta(days=2)
        },
        {
            "offer_id": "OFF003",
            "title": "عرض نهاية الموسم",
            "description": "تخفيضات تصل إلى 50% على منتجات مختارة",
            "discount_percentage": 50,
            "category": "all",
            "starts_at": datetime.now(),
            "expires_at": datetime.now() + timedelta(days=3),
            "active": True,
            "image_url": "https://example.com/offers/season_end.jpg",
            "created_at": datetime.now()
        },
        {
            "offer_id": "OFF004",
            "title": "عرض المكياج الجديد",
            "description": "خصم 25% على منتجات المكياج الجديدة",
            "discount_percentage": 25,
            "category": "مكياج",
            "starts_at": datetime.now() - timedelta(days=3),
            "expires_at": datetime.now() + timedelta(days=4),
            "active": True,
            "image_url": "https://example.com/offers/makeup_sale.jpg",
            "created_at": datetime.now() - timedelta(days=3)
        },
        {
            "offer_id": "OFF005",
            "title": "عرض خاص للعناية بالشعر",
            "description": "اشترِ شامبو وبلسم واحصل على مصل شعر مجاناً",
            "discount_percentage": 0,
            "category": "العناية بالشعر",
            "starts_at": datetime.now() - timedelta(days=5),
            "expires_at": datetime.now() + timedelta(days=2),
            "active": True,
            "image_url": "https://example.com/offers/haircare_bundle.jpg",
            "created_at": datetime.now() - timedelta(days=5)
        }
    ]
    
    try:
        # Clear existing data
        print("🧹 Clearing existing data...")
        mongo_service.products.delete_many({})
        mongo_service.offers.delete_many({})
        
        # Insert products
        print(f"📦 Inserting {len(beauty_products)} beauty products...")
        result_products = mongo_service.products.insert_many(beauty_products)
        
        # Insert offers
        print(f"🎁 Inserting {len(current_offers)} offers...")
        result_offers = mongo_service.offers.insert_many(current_offers)
        
        # Create indexes
        print("🔧 Creating indexes...")
        mongo_service._create_indexes()
        
        print("\n✅ Database seeded successfully!")
        print(f"   Products inserted: {len(result_products.inserted_ids)}")
        print(f"   Offers inserted: {len(result_offers.inserted_ids)}")
        
        # Show sample data
        print("\n📊 Sample data:")
        print(f"   Categories: {len(mongo_service.get_categories())}")
        print(f"   Brands: {len(mongo_service.get_brands())}")
        print(f"   In-stock products: {mongo_service.products.count_documents({'in_stock': True})}")
        
        return {
            "products_inserted": len(result_products.inserted_ids),
            "offers_inserted": len(result_offers.inserted_ids)
        }
        
    except Exception as e:
        print(f"❌ Error seeding database: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    seed_products()