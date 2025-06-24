import pandas as pd

# Load dataset
df = pd.read_csv("C:\\Users\\bless\\OneDrive\\Documents\\CHRIS\\CHRIS FILES\\CSV_XLSX\\forestfires.csv")
# Normalize column names (optional, already clean)
df.columns = df.columns.str.strip().str.lower()

# Handle missing values (fill numerics with mean, if any)
numeric_cols = df.select_dtypes(include='number').columns
for col in numeric_cols:
    if df[col].isnull().any():
        df[col].fillna(df[col].mean(), inplace=True)

# Handle categorical missing values (if any)
categorical_cols = df.select_dtypes(include='object').columns
for col in categorical_cols:
    if df[col].isnull().any():
        df[col].fillna('Unknown', inplace=True)

# Example: Convert month/day to category types
df['month'] = df['month'].astype('category')
df['day'] = df['day'].astype('category')

# Save cleaned data
df.to_csv("cleaned_forestfires.csv", index=False)

print("✅ Cleaned data saved as 'cleaned_forestfires.csv'")


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import joblib

df = pd.read_csv("C:\\Users\\bless\\OneDrive\\Documents\\CHRIS\\CHRIS FILES\\CSV_XLSX\\forestfires.csv")
df.head()

# Convert 'area' into 3 bins: 0 (no fire), 1 (small), 2 (large)
df['area_binned'] = pd.cut(df['area'], bins=[-1, 0, 5, np.inf], labels=[0, 1, 2]).astype(int)

X = pd.get_dummies(df.drop(columns=['area', 'area_binned']), drop_first=True)
y = df['area_binned']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')

print("✅ Accuracy:", round(accuracy * 100, 2), "%")
print("🎯 F1 Score (Weighted):", round(f1 * 100, 2), "%")

importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]
features_sorted = X.columns[indices]

plt.figure(figsize=(10, 6))
sns.barplot(x=importances[indices][:10], y=features_sorted[:10])
plt.title("Top 10 Feature Importances")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

joblib.dump(clf, 'forestfire_classifier.pkl')
print("✅ Model saved as 'forestfire_classifier.pkl'")

"""
nlp_insights.py – run:  python nlp_insights.py
"""
import os, sys, json, pandas as pd, spacy
from transformers import pipeline

# 🔇 Suppress symlink warnings from HuggingFace
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# ✅ Load dataset
path = r"C:\Users\bless\OneDrive\Documents\CHRIS\CHRIS FILES\INTERNSHIPS WORK\ZA_INTERN_SUMMER\text_data.csv"
df   = pd.read_csv(path)

# ✅ Load models
nlp = spacy.load("en_core_web_sm")

# 🔍 Light-weight models for speed
sentiment = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
summ      = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", truncation=True)

# ✅ Process spaCy NLP
def spacy_info(text):
    doc = nlp(text)
    return {
        "lemmas": [t.lemma_ for t in doc if not t.is_punct],
        "entities": [{"text": e.text, "label": e.label_} for e in doc.ents],
    }

df["spacy"] = df["text"].apply(spacy_info)

# ✅ Run sentiment pipeline
df["sentiment"] = sentiment(df["text"].tolist(), truncation=True)

# ✅ Batch summarization (avoid RAM spikes)
def batch_summarize(texts, batch_size=4):
    summaries = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        outputs = summ(batch)
        summaries.extend([s["summary_text"] for s in outputs])
    return summaries

df["summary"] = batch_summarize(df["text"].tolist())

# ✅ Export to CSV and JSONL
df.to_csv("sentiment_summary.csv", index=False)
with open("nlp_results.jsonl", "w", encoding="utf-8") as f:
    for row in df.to_dict(orient="records"):
        f.write(json.dumps(row) + "\n")

print("✅ Done → sentiment_summary.csv, nlp_results.jsonl")

import pandas as pd
import streamlit as st
# Load data
path=r"C:\Users\bless\OneDrive\Documents\CHRIS\CHRIS FILES\INTERNSHIPS WORK\ZA_INTERN_SUMMER\timeseries.csv"
df = pd.read_csv(path)

# Parse date and sort
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date')

# Rename columns for Prophet
df_prophet = df.rename(columns={'date': 'ds', 'value': 'y'})
# Save to CSV
output_path = "C:/Users/bless/OneDrive/Documents/CHRIS/CHRIS FILES/INTERNSHIPS WORK/ZA_INTERN_SUMMER/timeseries_prophet.csv"
df_prophet.to_csv(output_path, index=False)
print(f"Processed data saved to {output_path}")
# Display the first few rows of the processed DataFrame
print(df_prophet.head())
# Display the last few rows of the processed DataFrame
print(df_prophet.tail())
# Display the shape of the DataFrame
print(f"DataFrame shape: {df_prophet.shape}")
# Display the data types of the DataFrame
print(f"DataFrame dtypes:\n{df_prophet.dtypes}")

# Display basic statistics of the DataFrame
print(f"DataFrame statistics:\n{df_prophet.describe()}")
# Display the number of missing values in each column
print(f"Missing values in each column:\n{df_prophet.isnull().sum()}")
# Display the unique values in the 'ds' column
print(f"Unique dates in 'ds' column: {df_prophet['ds'].nunique()}")
# Display the range of dates in the 'ds' column
date_range = df_prophet['ds'].min(), df_prophet['ds'].max()
print(f"Date range in 'ds' column: {date_range}")

from prophet import Prophet
import matplotlib.pyplot as plt

# Initialize and fit model
model = Prophet()
model.fit(df_prophet)

# Create future dataframe
future = model.make_future_dataframe(periods=30)  # Forecast 30 days ahead

# Predict
forecast = model.predict(future)

# Plot forecast
fig1 = model.plot(forecast)
plt.title("Prophet Forecast")
st.pyplot(fig1)


# Optional: plot components
model.plot_components(forecast)



import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from datetime import date


# Load data
path=r"C:\Users\bless\OneDrive\Documents\CHRIS\CHRIS FILES\INTERNSHIPS WORK\ZA_INTERN_SUMMER\timeseries.csv"
df = pd.read_csv(path)
df['date'] = pd.to_datetime(df['date'])

# ---- Page Config ----
st.set_page_config(page_title="📈 Forecast Dashboard", layout="wide")

# ---- Header ----
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>📊 AI Forecast Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Powered by Streamlit & Prophet | Clean | Interactive | Beautiful</p>", unsafe_allow_html=True)
st.markdown("---")

# ---- Date Range Slider ----
min_date = df['date'].min()
max_date = df['date'].max()
selected_range = st.slider(
    "Select Date Range",
    min_value=min_date.date(),
    max_value=max_date.date(),
    value=(min_date.date(), max_date.date())
)

# Filter data
filtered_df = df[
    (df['date'].dt.date >= selected_range[0]) & 
    (df['date'].dt.date <= selected_range[1])
]


# ---- Metrics ----
col1, col2, col3 = st.columns(3)
col1.metric("🔼 Max Value", f"{filtered_df['value'].max():.2f}")
col2.metric("🔽 Min Value", f"{filtered_df['value'].min():.2f}")
col3.metric("📍 Latest Value", f"{filtered_df['value'].iloc[-1]:.2f}")

# ---- Line Chart ----
st.subheader("📉 Time Series Line Chart")
st.line_chart(filtered_df.set_index("date")["value"])

# ---- Matplotlib Styled Chart ----
st.subheader("🎨 Custom Styled Matplotlib Chart")

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(filtered_df["date"], filtered_df["value"], label="Forecast", color="#FF5733", linewidth=2)
ax.set_facecolor("#f0f2f6")
ax.set_title("Styled Forecast", fontsize=16)
ax.set_xlabel("Date")
ax.set_ylabel("Value")
ax.grid(True, linestyle="--", alpha=0.4)
st.pyplot(fig)


import io

# Create an in-memory Excel file
output = io.BytesIO()
with pd.ExcelWriter(output, engine='openpyxl') as writer:
    filtered_df.to_excel(writer, sheet_name='Data', index=False)

    # Add summary sheet
    summary_df = pd.DataFrame({
        "Metric": ["Max Value", "Min Value", "Latest Value"],
        "Value": [
            filtered_df['value'].max(),
            filtered_df['value'].min(),
            filtered_df['value'].iloc[-1]
        ]
    })
    summary_df.to_excel(writer, sheet_name='Summary', index=False)

output.seek(0)

# Streamlit download button
st.download_button(
    label="📥 Download Excel Report",
    data=output,
    file_name="light_insight_report.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)
