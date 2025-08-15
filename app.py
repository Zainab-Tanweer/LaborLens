from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import os
import pickle
import numpy as np
from datetime import datetime
import pandas as pd

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# User model
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(120), nullable=False)
    role = db.Column(db.String(20), nullable=False, default='normal_user')  # 'government_official' or 'normal_user'
    created_at = db.Column(db.DateTime, server_default=db.func.now())

# Company model for child labor classification - Updated with all features from the dataset
class Company(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(200), unique=True, nullable=False)
    
    # Basic Information
    brand = db.Column(db.String(100), nullable=True)
    country = db.Column(db.String(100), nullable=True)
    year = db.Column(db.Integer, nullable=True)
    
    # Production & Operations
    monthly_production_tonnes = db.Column(db.Float, nullable=True)
    avg_item_price_usd = db.Column(db.Float, nullable=True)
    release_cycles_per_year = db.Column(db.Integer, nullable=True)
    
    # Environmental Impact
    carbon_emissions_tco2e = db.Column(db.Float, nullable=True)
    water_usage_million_litres = db.Column(db.Float, nullable=True)
    landfill_waste_tonnes = db.Column(db.Float, nullable=True)
    
    # Labor & Working Conditions
    avg_worker_wage_usd = db.Column(db.Float, nullable=True)
    working_hours_per_week = db.Column(db.Integer, nullable=True)
    
    # Business Metrics
    return_rate_percent = db.Column(db.Float, nullable=True)
    avg_spend_per_customer_usd = db.Column(db.Float, nullable=True)
    shopping_frequency_per_year = db.Column(db.Integer, nullable=True)
    
    # Social Media & Sentiment
    instagram_mentions_thousands = db.Column(db.Float, nullable=True)
    tiktok_mentions_thousands = db.Column(db.Float, nullable=True)
    sentiment_score = db.Column(db.Float, nullable=True)
    social_sentiment_label = db.Column(db.String(50), nullable=True)
    
    # Economic & Compliance
    gdp_contribution_million_usd = db.Column(db.Float, nullable=True)
    env_cost_index = db.Column(db.Float, nullable=True)
    sustainability_score = db.Column(db.Float, nullable=True)
    transparency_index = db.Column(db.Float, nullable=True)
    compliance_score = db.Column(db.Float, nullable=True)
    ethical_rating = db.Column(db.Float, nullable=True)
    
    # Classification Results
    classification = db.Column(db.String(50), nullable=True)  # 'complicit' or 'not_complicit'
    confidence_score = db.Column(db.Float, nullable=True)
    
    # Metadata
    last_updated = db.Column(db.DateTime, default=datetime.utcnow)
    updated_by = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    
    # Government official notes
    notes = db.Column(db.Text, nullable=True)
    evidence_sources = db.Column(db.Text, nullable=True)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Load the trained model from classifier.py
def load_model():
    try:
        # Import the classifier module
        from classifier import pipeline
        return pipeline
    except ImportError:
        print("Could not import classifier.py. Please ensure the file is in the project directory.")
        return None

# Function to classify a company using the actual model
def classify_company(company_data):
    model = load_model()
    if model is None:
        return None, 0.0
    
    try:
        # Create a DataFrame with the expected column names
        features = extract_features(company_data)
        
        # Make prediction
        prediction = model.predict(features)[0]
        
        # Get confidence score (probability of positive class)
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(features)[0]
            confidence = proba[1] if len(proba) > 1 else proba[0]  # Probability of class 1 (complicit)
        else:
            confidence = 0.8  # Default confidence if no probabilities available
        
        # Convert prediction to string labels
        classification = 'complicit' if prediction == 1 else 'not_complicit'
        
        return classification, confidence
    except Exception as e:
        print(f"Error in classification: {e}")
        return None, 0.0

# Function to extract features from company data - Updated to match the model's expected input
def extract_features(company_data):
    from classifier import X_test
    """
    Extract features in the exact order expected by the model.
    The model expects 30 features in this specific order.
    """
    features = []
    
    # 1. Brand (will be one-hot encoded)
    features.append(company_data.get('brand', 'Unknown'))
    
    # 2. Country (will be one-hot encoded)
    features.append(company_data.get('country', 'Unknown'))
    
    # 3. Year
    features.append(company_data.get('year', 2024))
    
    # 4. Monthly_Production_Tonnes
    features.append(company_data.get('monthly_production_tonnes', 0.0))
    
    # 5. Avg_Item_Price_USD
    features.append(company_data.get('avg_item_price_usd', 0.0))
    
    # 6. Release_Cycles_Per_Year
    features.append(company_data.get('release_cycles_per_year', 0))
    
    # 7. Carbon_Emissions_tCO2e
    features.append(company_data.get('carbon_emissions_tco2e', 0.0))
    
    # 8. Water_Usage_Million_Litres
    features.append(company_data.get('water_usage_million_litres', 0.0))
    
    # 9. Landfill_Waste_Tonnes
    features.append(company_data.get('landfill_waste_tonnes', 0.0))
    
    # 10. Avg_Worker_Wage_USD
    features.append(company_data.get('avg_worker_wage_usd', 0.0))
    
    # 11. Working_Hours_Per_Week
    features.append(company_data.get('working_hours_per_week', 0))
    
    # 12. Return_Rate_Percent
    features.append(company_data.get('return_rate_percent', 0.0))
    
    # 13. Avg_Spend_Per_Customer_USD
    features.append(company_data.get('avg_spend_per_customer_usd', 0.0))
    
    # 14. Shopping_Frequency_Per_Year
    features.append(company_data.get('shopping_frequency_per_year', 0))
    
    # # 15. Instagram_Mentions_Thousands
    # features.append(company_data.get('instagram_mentions_thousands', 0.0))
    
    # # 16. TikTok_Mentions_Thousands
    # features.append(company_data.get('tiktok_mentions_thousands', 0.0))
    
    # 17. Sentiment_Score
    features.append(company_data.get('sentiment_score', 0.0))
    
    # 18. Social_Sentiment_Label (will be one-hot encoded)
    features.append(company_data.get('social_sentiment_label', 'Neutral'))
    
    # 19. GDP_Contribution_Million_USD
    features.append(company_data.get('gdp_contribution_million_usd', 0.0))
    
    # 20. Env_Cost_Index
    features.append(company_data.get('env_cost_index', 0.0))
    
    # 21. Sustainability_Score
    features.append(company_data.get('sustainability_score', 0.0))
    
    # 22. Transparency_Index
    features.append(company_data.get('transparency_index', 0.0))
    
    # 23. Compliance_Score
    features.append(company_data.get('compliance_score', 0.0))
    
    # 24. Ethical_Rating
    features.append(company_data.get('ethical_rating', 0.0))
    
    # 25. Production_per_Release (engineered feature)
    monthly_prod = company_data.get('monthly_production_tonnes', 0.0)
    release_cycles = company_data.get('release_cycles_per_year', 1)
    features.append(monthly_prod / release_cycles if release_cycles > 0 else 0.0)
    
    # 26. Emissions_per_Tonne (engineered feature)
    emissions = company_data.get('carbon_emissions_tco2e', 0.0)
    features.append(emissions / monthly_prod if monthly_prod > 0 else 0.0)
    
    # 27. Waste_per_Tonne (engineered feature)
    waste = company_data.get('landfill_waste_tonnes', 0.0)
    features.append(waste / monthly_prod if monthly_prod > 0 else 0.0)
    
    # 28. Water_per_Tonne (engineered feature)
    water = company_data.get('water_usage_million_litres', 0.0)
    features.append(water / monthly_prod if monthly_prod > 0 else 0.0)
    
    # 29. Wage_per_Hour (engineered feature)
    wage = company_data.get('avg_worker_wage_usd', 0.0)
    hours = company_data.get('working_hours_per_week', 1)
    features.append(wage / hours if hours > 0 else 0.0)
    
    # 30. Social_Media_Mentions (engineered feature)
    instagram = company_data.get('instagram_mentions_thousands', 0.0)
    tiktok = company_data.get('tiktok_mentions_thousands', 0.0)
    features.append(instagram + tiktok)

    features = pd.DataFrame([features], columns=X_test.columns)
    
    return features

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        role = request.form['role']
        
        # Check if user already exists
        if User.query.filter_by(username=username).first():
            flash('Username already exists')
            return redirect(url_for('register'))
        
        if User.query.filter_by(email=email).first():
            flash('Email already registered')
            return redirect(url_for('register'))
        
        # Create new user
        user = User(
            username=username,
            email=email,
            password_hash=generate_password_hash(password),
            role=role
        )
        
        db.session.add(user)
        db.session.commit()
        
        flash('Registration successful! Please login.')
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        user = User.query.filter_by(username=username).first()
        
        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            flash(f'Welcome back, {username}!')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password')
    
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.')
    return redirect(url_for('index'))

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html')

@app.route("/donate")
def donate():
    return render_template("donate.html")


# Company search route for normal users
@app.route('/search_company', methods=['GET', 'POST'])
@login_required
def search_company():
    if request.method == 'POST':
        company_name = request.form.get('company_name', '').strip()
        
        if company_name:
            # Search for existing company
            company = Company.query.filter(Company.name.ilike(f'%{company_name}%')).first()
            
            if company:
                return render_template('company_result.html', company=company)
            else:
                flash(f'Company "{company_name}" not found in our database.')
                return render_template('search_company.html')
    
    return render_template('search_company.html')

# Company result display
@app.route('/company/<int:company_id>')
@login_required
def company_detail(company_id):
    company = Company.query.get_or_404(company_id)
    return render_template('company_detail.html', company=company)

# Government official routes for managing companies
@app.route('/manage_companies')
@login_required
def manage_companies():
    if current_user.role != 'government_official':
        flash('Access denied. Government officials only.')
        return redirect(url_for('dashboard'))
    
    companies = Company.query.order_by(Company.name).all()
    return render_template('manage_companies.html', companies=companies)

@app.route('/add_company', methods=['GET', 'POST'])
@login_required
def add_company():
    if current_user.role != 'government_official':
        flash('Access denied. Government officials only.')
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        # Extract all form data
        name = request.form.get('name', '').strip()
        brand = request.form.get('brand', '').strip()
        country = request.form.get('country', '').strip()
        year = request.form.get('year')
        monthly_production_tonnes = request.form.get('monthly_production_tonnes')
        avg_item_price_usd = request.form.get('avg_item_price_usd')
        release_cycles_per_year = request.form.get('release_cycles_per_year')
        carbon_emissions_tco2e = request.form.get('carbon_emissions_tco2e')
        water_usage_million_litres = request.form.get('water_usage_million_litres')
        landfill_waste_tonnes = request.form.get('landfill_waste_tonnes')
        avg_worker_wage_usd = request.form.get('avg_worker_wage_usd')
        working_hours_per_week = request.form.get('working_hours_per_week')
        return_rate_percent = request.form.get('return_rate_percent')
        avg_spend_per_customer_usd = request.form.get('avg_spend_per_customer_usd')
        shopping_frequency_per_year = request.form.get('shopping_frequency_per_year')
        instagram_mentions_thousands = request.form.get('instagram_mentions_thousands')
        tiktok_mentions_thousands = request.form.get('tiktok_mentions_thousands')
        sentiment_score = request.form.get('sentiment_score')
        social_sentiment_label = request.form.get('social_sentiment_label', '').strip()
        gdp_contribution_million_usd = request.form.get('gdp_contribution_million_usd')
        env_cost_index = request.form.get('env_cost_index')
        sustainability_score = request.form.get('sustainability_score')
        transparency_index = request.form.get('transparency_index')
        compliance_score = request.form.get('compliance_score')
        ethical_rating = request.form.get('ethical_rating')
        notes = request.form.get('notes', '').strip()
        evidence_sources = request.form.get('evidence_sources', '').strip()
        
        if not name:
            flash('Company name is required.')
            return render_template('add_company.html')
        
        # Check if company already exists
        if Company.query.filter_by(name=name).first():
            flash('Company already exists in database.')
            return render_template('add_company.html')
        
        # Prepare company data for classification
        company_data = {
            'brand': brand,
            'country': country,
            'year': int(year) if year else 2024,
            'monthly_production_tonnes': float(monthly_production_tonnes) if monthly_production_tonnes else 0.0,
            'avg_item_price_usd': float(avg_item_price_usd) if avg_item_price_usd else 0.0,
            'release_cycles_per_year': int(release_cycles_per_year) if release_cycles_per_year else 0,
            'carbon_emissions_tco2e': float(carbon_emissions_tco2e) if carbon_emissions_tco2e else 0.0,
            'water_usage_million_litres': float(water_usage_million_litres) if water_usage_million_litres else 0.0,
            'landfill_waste_tonnes': float(landfill_waste_tonnes) if landfill_waste_tonnes else 0.0,
            'avg_worker_wage_usd': float(avg_worker_wage_usd) if avg_worker_wage_usd else 0.0,
            'working_hours_per_week': int(working_hours_per_week) if working_hours_per_week else 0,
            'return_rate_percent': float(return_rate_percent) if return_rate_percent else 0.0,
            'avg_spend_per_customer_usd': float(avg_spend_per_customer_usd) if avg_spend_per_customer_usd else 0.0,
            'shopping_frequency_per_year': int(shopping_frequency_per_year) if shopping_frequency_per_year else 0,
            'instagram_mentions_thousands': float(instagram_mentions_thousands) if instagram_mentions_thousands else 0.0,
            'tiktok_mentions_thousands': float(tiktok_mentions_thousands) if tiktok_mentions_thousands else 0.0,
            'sentiment_score': float(sentiment_score) if sentiment_score else 0.0,
            'social_sentiment_label': social_sentiment_label,
            'gdp_contribution_million_usd': float(gdp_contribution_million_usd) if gdp_contribution_million_usd else 0.0,
            'env_cost_index': float(env_cost_index) if env_cost_index else 0.0,
            'sustainability_score': float(sustainability_score) if sustainability_score else 0.0,
            'transparency_index': float(transparency_index) if transparency_index else 0.0,
            'compliance_score': float(compliance_score) if compliance_score else 0.0,
            'ethical_rating': float(ethical_rating) if ethical_rating else 0.0
        }
        
        # Classify the company
        classification, confidence = classify_company(company_data)
        
        # Create new company
        company = Company(
            name=name,
            brand=brand,
            country=country,
            year=int(year) if year else 2024,
            monthly_production_tonnes=float(monthly_production_tonnes) if monthly_production_tonnes else None,
            avg_item_price_usd=float(avg_item_price_usd) if avg_item_price_usd else None,
            release_cycles_per_year=int(release_cycles_per_year) if release_cycles_per_year else None,
            carbon_emissions_tco2e=float(carbon_emissions_tco2e) if carbon_emissions_tco2e else None,
            water_usage_million_litres=float(water_usage_million_litres) if water_usage_million_litres else None,
            landfill_waste_tonnes=float(landfill_waste_tonnes) if landfill_waste_tonnes else None,
            avg_worker_wage_usd=float(avg_worker_wage_usd) if avg_worker_wage_usd else None,
            working_hours_per_week=int(working_hours_per_week) if working_hours_per_week else None,
            return_rate_percent=float(return_rate_percent) if return_rate_percent else None,
            avg_spend_per_customer_usd=float(avg_spend_per_customer_usd) if avg_spend_per_customer_usd else None,
            shopping_frequency_per_year=int(shopping_frequency_per_year) if shopping_frequency_per_year else None,
            # instagram_mentions_thousands=float(instagram_mentions_thousands) if instagram_mentions_thousands else None,
            # tiktok_mentions_thousands=float(tiktok_mentions_thousands) if tiktok_mentions_thousands else None,
            sentiment_score=float(sentiment_score) if sentiment_score else None,
            social_sentiment_label=social_sentiment_label,
            gdp_contribution_million_usd=float(gdp_contribution_million_usd) if gdp_contribution_million_usd else None,
            env_cost_index=float(env_cost_index) if env_cost_index else None,
            sustainability_score=float(sustainability_score) if sustainability_score else None,
            transparency_index=float(transparency_index) if transparency_index else None,
            compliance_score=float(compliance_score) if compliance_score else None,
            ethical_rating=float(ethical_rating) if ethical_rating else None,
            classification=classification,
            confidence_score=confidence,
            notes=notes,
            evidence_sources=evidence_sources,
            updated_by=current_user.id
        )
        
        db.session.add(company)
        db.session.commit()
        
        flash(f'Company "{name}" added successfully with classification: {classification}')
        return redirect(url_for('manage_companies'))
    
    return render_template('add_company.html')

@app.route('/edit_company/<int:company_id>', methods=['GET', 'POST'])
@login_required
def edit_company(company_id):
    if current_user.role != 'government_official':
        flash('Access denied. Government officials only.')
        return redirect(url_for('dashboard'))
    
    company = Company.query.get_or_404(company_id)
    
    if request.method == 'POST':
        # Update all fields
        company.brand = request.form.get('brand', '').strip()
        company.country = request.form.get('country', '').strip()
        company.year = int(request.form.get('year')) if request.form.get('year') else None
        company.monthly_production_tonnes = float(request.form.get('monthly_production_tonnes')) if request.form.get('monthly_production_tonnes') else None
        company.avg_item_price_usd = float(request.form.get('avg_item_price_usd')) if request.form.get('avg_item_price_usd') else None
        company.release_cycles_per_year = int(request.form.get('release_cycles_per_year')) if request.form.get('release_cycles_per_year') else None
        company.carbon_emissions_tco2e = float(request.form.get('carbon_emissions_tco2e')) if request.form.get('carbon_emissions_tco2e') else None
        company.water_usage_million_litres = float(request.form.get('water_usage_million_litres')) if request.form.get('water_usage_million_litres') else None
        company.landfill_waste_tonnes = float(request.form.get('landfill_waste_tonnes')) if request.form.get('landfill_waste_tonnes') else None
        company.avg_worker_wage_usd = float(request.form.get('avg_worker_wage_usd')) if request.form.get('avg_worker_wage_usd') else None
        company.working_hours_per_week = int(request.form.get('working_hours_per_week')) if request.form.get('working_hours_per_week') else None
        company.return_rate_percent = float(request.form.get('return_rate_percent')) if request.form.get('return_rate_percent') else None
        company.avg_spend_per_customer_usd = float(request.form.get('avg_spend_per_customer_usd')) if request.form.get('avg_spend_per_customer_usd') else None
        company.shopping_frequency_per_year = int(request.form.get('shopping_frequency_per_year')) if request.form.get('shopping_frequency_per_year') else None
        company.instagram_mentions_thousands = float(request.form.get('instagram_mentions_thousands')) if request.form.get('instagram_mentions_thousands') else None
        company.tiktok_mentions_thousands = float(request.form.get('tiktok_mentions_thousands')) if request.form.get('tiktok_mentions_thousands') else None
        company.sentiment_score = float(request.form.get('sentiment_score')) if request.form.get('sentiment_score') else None
        company.social_sentiment_label = request.form.get('social_sentiment_label', '').strip()
        company.gdp_contribution_million_usd = float(request.form.get('gdp_contribution_million_usd')) if request.form.get('gdp_contribution_million_usd') else None
        company.env_cost_index = float(request.form.get('env_cost_index')) if request.form.get('env_cost_index') else None
        company.sustainability_score = float(request.form.get('sustainability_score')) if request.form.get('sustainability_score') else None
        company.transparency_index = float(request.form.get('transparency_index')) if request.form.get('transparency_index') else None
        company.compliance_score = float(request.form.get('compliance_score')) if request.form.get('compliance_score') else None
        company.ethical_rating = float(request.form.get('ethical_rating')) if request.form.get('ethical_rating') else None
        company.notes = request.form.get('notes', '').strip()
        company.evidence_sources = request.form.get('evidence_sources', '').strip()
        company.last_updated = datetime.utcnow()
        company.updated_by = current_user.id
        
        # Reclassify with updated data
        company_data = {
            'brand': company.brand,
            'country': company.country,
            'year': company.year,
            'monthly_production_tonnes': company.monthly_production_tonnes,
            'avg_item_price_usd': company.avg_item_price_usd,
            'release_cycles_per_year': company.release_cycles_per_year,
            'carbon_emissions_tco2e': company.carbon_emissions_tco2e,
            'water_usage_million_litres': company.water_usage_million_litres,
            'landfill_waste_tonnes': company.landfill_waste_tonnes,
            'avg_worker_wage_usd': company.avg_worker_wage_usd,
            'working_hours_per_week': company.working_hours_per_week,
            'return_rate_percent': company.return_rate_percent,
            'avg_spend_per_customer_usd': company.avg_spend_per_customer_usd,
            'shopping_frequency_per_year': company.shopping_frequency_per_year,
            'instagram_mentions_thousands': company.instagram_mentions_thousands,
            'tiktok_mentions_thousands': company.tiktok_mentions_thousands,
            'sentiment_score': company.sentiment_score,
            'social_sentiment_label': company.social_sentiment_label,
            'gdp_contribution_million_usd': company.gdp_contribution_million_usd,
            'env_cost_index': company.env_cost_index,
            'sustainability_score': company.sustainability_score,
            'transparency_index': company.transparency_index,
            'compliance_score': company.compliance_score,
            'ethical_rating': company.ethical_rating
        }
        
        classification, confidence = classify_company(company_data)
        company.classification = classification
        company.confidence_score = confidence
        
        db.session.commit()
        flash(f'Company "{company.name}" updated successfully.')
        return redirect(url_for('manage_companies'))
    
    return render_template('edit_company.html', company=company)

@app.route('/delete_company/<int:company_id>', methods=['POST'])
@login_required
def delete_company(company_id):
    if current_user.role != 'government_official':
        flash('Access denied. Government officials only.')
        return redirect(url_for('dashboard'))
    
    company = Company.query.get_or_404(company_id)
    company_name = company.name
    
    db.session.delete(company)
    db.session.commit()
    
    flash(f'Company "{company_name}" deleted successfully.')
    return redirect(url_for('manage_companies'))

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True) 