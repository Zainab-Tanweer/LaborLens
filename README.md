# Government Portal - AI-Powered Child Labor Classification System

A comprehensive Flask web application that integrates a pre-trained scikit-learn machine learning model to classify companies based on their compliance with child labor regulations. The system provides role-based access control for government officials and citizens.

## 🚀 Features

### For Government Officials
- **Comprehensive Company Management**: Add, edit, and delete companies with 30+ data fields
- **AI-Powered Classification**: Automatic classification using the trained scikit-learn model
- **Real-time Updates**: Reclassification occurs automatically when company data is modified
- **Evidence Tracking**: Add government notes and evidence sources for each company
- **Database Administration**: Full CRUD operations on company records

### For Normal Users (Citizens)
- **Company Search**: Search for companies by name to check their classification status
- **Detailed Results**: View comprehensive company information and AI classification results
- **Transparency**: Access to all company data used in the classification process
- **Confidence Scoring**: See how certain the AI model is about each classification

### AI Model Integration
- **30 Comprehensive Features**: Production data, environmental impact, labor conditions, business metrics, social sentiment, and compliance scores
- **Automatic Feature Engineering**: Built-in calculation of derived features (e.g., emissions per tonne, waste per tonne)
- **Real-time Classification**: Instant results when adding or updating company data
- **Confidence Assessment**: Probability scores for classification reliability

## 🏗️ Project Structure

```
LaborLens/
├── app.py                          # Main Flask application
├── classifier.py                   # Trained scikit-learn model and pipeline
├── true_cost_fast_fashion.csv      # Training dataset
├── seed_database.py                # Load data from dataset in the app
├── requirements.txt                # Python dependencies
├── templates/                      # HTML templates
│   ├── base.html                   # Base template with navigation
│   ├── index.html                  # Home page
│   ├── login.html                  # User login form
│   ├── register.html               # User registration form
│   ├── dashboard.html              # Role-based dashboard
│   ├── search_company.html         # Company search interface
│   ├── company_result.html         # Company classification results
│   ├── manage_companies.html       # Government official company management
│   ├── add_company.html            # Add new company form
│   └── edit_company.html           # Edit company form
└── README.md                       # This file
```

## 🤖 Child Labor Classifier Integration

### How It Works
The system integrates with your trained scikit-learn model (`classifier.py`) that analyzes companies based on:

1. **Basic Information**: Brand, country, year
2. **Production & Operations**: Monthly production, item prices, release cycles
3. **Environmental Impact**: Carbon emissions, water usage, landfill waste
4. **Labor & Working Conditions**: Worker wages, working hours
5. **Business Metrics**: Return rates, customer spending, shopping frequency
6. **Social Media & Sentiment**: Instagram/TikTok mentions, sentiment scores
7. **Economic & Compliance**: GDP contribution, sustainability scores, ethical ratings
8. **Engineered Features**: Production per release, emissions per tonne, waste per tonne, etc.

### Model Features (30 total)
- **Original Features**: 24 features from the dataset
- **Engineered Features**: 6 additional calculated features
- **Categorical Features**: Brand, country, social sentiment label (one-hot encoded)
- **Numerical Features**: All other metrics and scores

### Classification Output
- **Binary Classification**: Complicit (1) or Not Complicit (0) in child labor
- **Confidence Score**: Probability of the predicted class
- **Real-time Updates**: Automatic reclassification when data changes

## 🛠️ Setup Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare the Model
- Ensure `classifier.py` is in the project root directory
- The file should contain a trained `pipeline` object with the scikit-learn model
- Verify `true_cost_fast_fashion.csv` is present for reference

### 3. Run the Application
```bash
python app.py
```

### 4. Access the Application
- Open your browser and go to `http://localhost:5000`
- Register a new account (choose your role)
- Login and start using the system

## 🔐 User Roles & Capabilities

### Government Official
- **Full Access**: All company management features
- **Data Entry**: Comprehensive forms with 30+ fields
- **AI Integration**: Automatic classification on data entry/update
- **Administration**: Manage entire company database

### Normal User
- **Search Access**: Look up companies by name
- **View Results**: See classification results and company details
- **Transparency**: Access to all data used in classification

## 🗄️ Database Schema

### User Model
- `id`: Primary key
- `username`: Unique username
- `email`: Unique email address
- `password_hash`: Hashed password
- `role`: 'government_official' or 'normal_user'
- `created_at`: Account creation timestamp

### Company Model
- **Basic Info**: `name`, `brand`, `country`, `year`
- **Production**: `monthly_production_tonnes`, `avg_item_price_usd`, `release_cycles_per_year`
- **Environmental**: `carbon_emissions_tco2e`, `water_usage_million_litres`, `landfill_waste_tonnes`
- **Labor**: `avg_worker_wage_usd`, `working_hours_per_week`
- **Business**: `return_rate_percent`, `avg_spend_per_customer_usd`, `shopping_frequency_per_year`
- **Social**: `instagram_mentions_thousands`, `tiktok_mentions_thousands`, `sentiment_score`, `social_sentiment_label`
- **Compliance**: `gdp_contribution_million_usd`, `env_cost_index`, `sustainability_score`, `transparency_index`, `compliance_score`, `ethical_rating`
- **Results**: `classification`, `confidence_score`
- **Metadata**: `last_updated`, `updated_by`, `notes`, `evidence_sources`

## 🛠️ Technologies Used

- **Backend**: Flask, Flask-SQLAlchemy, Flask-Login
- **Database**: SQLite with SQLAlchemy ORM
- **Frontend**: Bootstrap 5, Font Awesome, HTML5, CSS3
- **Machine Learning**: scikit-learn, numpy, pandas
- **Authentication**: Werkzeug security utilities
- **Development**: Python 3.8+

## 📱 Available Routes

- `/` - Home page
- `/register` - User registration
- `/login` - User authentication
- `/logout` - User logout
- `/dashboard` - Role-based dashboard
- `/search_company` - Company search (normal users)
- `/company/<id>` - Company details and results
- `/manage_companies` - Company management (government officials)
- `/add_company` - Add new company (government officials)
- `/edit_company/<id>` - Edit company (government officials)
- `/delete_company/<id>` - Delete company (government officials)

## 🔧 Customization

### Feature Engineering
The `extract_features()` function in `app.py` handles feature extraction. Modify this function if you need to:
- Change feature calculations
- Add new derived features
- Modify data preprocessing

### Model Integration
The system automatically imports your trained model from `classifier.py`. Ensure your model:
- Has a `predict()` method for classification
- Has a `predict_proba()` method for confidence scores
- Expects the same feature order as defined in `extract_features()`

### Database Fields
Add new fields to the `Company` model in `app.py` and update:
- HTML templates for data entry/display
- Form handling in routes
- Feature extraction function

## 🚨 Troubleshooting

### Model Loading Issues
- Ensure `classifier.py` is in the project root
- Verify the `pipeline` object is properly defined
- Check scikit-learn version compatibility

### Feature Mismatch
- Verify the feature order in `extract_features()` matches your model
- Check that categorical values match your training data
- Ensure numerical ranges are appropriate

### Database Issues
- Delete `users.db` to reset the database
- Check file permissions in the project directory
- Verify SQLite is properly installed

## 📄 License

This project is for educational and demonstration purposes.

## 🤝 Contributing

This is a demonstration project showcasing AI integration in government systems. The architecture can be adapted for various compliance and classification use cases. 