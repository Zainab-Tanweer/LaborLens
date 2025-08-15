import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Load data
df = pd.read_csv("true_cost_fast_fashion.csv")

# Feature engineering
df['Production_per_Release'] = df['Monthly_Production_Tonnes'] / df['Release_Cycles_Per_Year']
df['Emissions_per_Tonne'] = df['Carbon_Emissions_tCO2e'] / df['Monthly_Production_Tonnes']
df['Waste_per_Tonne'] = df['Landfill_Waste_Tonnes'] / df['Monthly_Production_Tonnes']
df['Water_per_Tonne'] = df['Water_Usage_Million_Litres'] / df['Monthly_Production_Tonnes']
df['Wage_per_Hour'] = df['Avg_Worker_Wage_USD'] / df['Working_Hours_Per_Week']
df['Social_Media_Mentions'] = df['Instagram_Mentions_Thousands'] + df['TikTok_Mentions_Thousands']

# Drop redundant columns
df = df.drop(columns=['Instagram_Mentions_Thousands', 'TikTok_Mentions_Thousands'])

# Create binary target variable
df['Child_Labor_Flag'] = (df['Child_Labor_Incidents'] > 0).astype(int)

# Define features and target
X = df.drop(columns=['Child_Labor_Incidents', 'Child_Labor_Flag'])
y = df['Child_Labor_Flag']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define categorical features
categorical_features = ['Brand', 'Country', 'Social_Sentiment_Label']

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough'
)

# Build pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Train model
pipeline.fit(X_train, y_train)

# Predict with default threshold (0.5)
y_pred = pipeline.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Print evaluation results
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("Confusion Matrix:\n", conf_matrix)
print(df.columns)