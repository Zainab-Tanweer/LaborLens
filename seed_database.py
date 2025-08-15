# seed_database.py

import pandas as pd
from datetime import datetime
from werkzeug.security import generate_password_hash

from app import app, db, Company, User, classify_company

def seed_database():
    with app.app_context():
        # Load dataset
        df = pd.read_csv("true_cost_fast_fashion.csv")

        # Create government official admin user if not exists
        admin_user = User.query.filter_by(role='government_official').first()
        if not admin_user:
            admin_user = User(
                username='admin',
                email='admin@government.gov',
                password_hash=generate_password_hash('Admin@123'),
                role='government_official'
            )
            db.session.add(admin_user)
            db.session.commit()
            print("Created admin government_official user.")

        added_count = 0

        for _, row in df.iterrows():
            company_name = f"{row['Brand']}_{row['Country']}_{row['Year']}"

            # Skip if company already exists
            if Company.query.filter_by(name=company_name).first():
                continue

            # Prepare data for classification (same keys as add_company in app.py)
            company_data = {
                'brand': row['Brand'],
                'country': row['Country'],
                'year': int(row['Year']),
                'monthly_production_tonnes': float(row['Monthly_Production_Tonnes']),
                'avg_item_price_usd': float(row['Avg_Item_Price_USD']),
                'release_cycles_per_year': int(row['Release_Cycles_Per_Year']),
                'carbon_emissions_tco2e': float(row['Carbon_Emissions_tCO2e']),
                'water_usage_million_litres': float(row['Water_Usage_Million_Litres']),
                'landfill_waste_tonnes': float(row['Landfill_Waste_Tonnes']),
                'avg_worker_wage_usd': float(row['Avg_Worker_Wage_USD']),
                'working_hours_per_week': int(row['Working_Hours_Per_Week']),
                'return_rate_percent': float(row['Return_Rate_Percent']),
                'avg_spend_per_customer_usd': float(row['Avg_Spend_Per_Customer_USD']),
                'shopping_frequency_per_year': int(row['Shopping_Frequency_Per_Year']),
                # 'instagram_mentions_thousands': float(row['Instagram_Mentions_Thousands']),
                # 'tiktok_mentions_thousands': float(row['TikTok_Mentions_Thousands']),
                'sentiment_score': float(row['Sentiment_Score']),
                'social_sentiment_label': row['Social_Sentiment_Label'],
                'gdp_contribution_million_usd': float(row['GDP_Contribution_Million_USD']),
                'env_cost_index': float(row['Env_Cost_Index']),
                'sustainability_score': float(row['Sustainability_Score']),
                'transparency_index': float(row['Transparency_Index']),
                'compliance_score': float(row['Compliance_Score']),
                'ethical_rating': float(row['Ethical_Rating']),
            }

            # Classify company with ML model
            classification, confidence = classify_company(company_data)

            # Create Company entry
            company = Company(
                name=company_name,
                brand=row['Brand'],
                country=row['Country'],
                year=int(row['Year']),
                monthly_production_tonnes=company_data['monthly_production_tonnes'],
                avg_item_price_usd=company_data['avg_item_price_usd'],
                release_cycles_per_year=company_data['release_cycles_per_year'],
                carbon_emissions_tco2e=company_data['carbon_emissions_tco2e'],
                water_usage_million_litres=company_data['water_usage_million_litres'],
                landfill_waste_tonnes=company_data['landfill_waste_tonnes'],
                avg_worker_wage_usd=company_data['avg_worker_wage_usd'],
                working_hours_per_week=company_data['working_hours_per_week'],
                return_rate_percent=company_data['return_rate_percent'],
                avg_spend_per_customer_usd=company_data['avg_spend_per_customer_usd'],
                shopping_frequency_per_year=company_data['shopping_frequency_per_year'],
                instagram_mentions_thousands=float(row['Instagram_Mentions_Thousands']),
                tiktok_mentions_thousands=float(row['TikTok_Mentions_Thousands']),
                sentiment_score=company_data['sentiment_score'],
                social_sentiment_label=company_data['social_sentiment_label'],
                gdp_contribution_million_usd=company_data['gdp_contribution_million_usd'],
                env_cost_index=company_data['env_cost_index'],
                sustainability_score=company_data['sustainability_score'],
                transparency_index=company_data['transparency_index'],
                compliance_score=company_data['compliance_score'],
                ethical_rating=company_data['ethical_rating'],
                classification=classification,
                confidence_score=confidence,
                updated_by=admin_user.id
            )

            db.session.add(company)
            added_count += 1

        db.session.commit()
        print(f"Added {added_count} new companies into the database.")

if __name__ == "__main__":
    seed_database()
