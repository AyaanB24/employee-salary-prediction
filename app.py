import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

st.set_page_config(page_title="Employee Salary Prediction", layout="centered")
st.title("💼 Employee Salary Prediction App")


# ========== Caching ==========
@st.cache_resource
def load_model():
    try:
        model = joblib.load("salary_model.pkl")
        le_gender = joblib.load("le_gender.pkl")
        le_edu = joblib.load("le_edu.pkl")
        le_job = joblib.load("le_job.pkl")
        return model, le_gender, le_edu, le_job
    except Exception as e:
        st.error(f"❌ Failed to load model or encoders: {e}")
        st.stop()

@st.cache_data
def load_data(_le_gender, _le_edu, _le_job):
    try:
        df = pd.read_csv("Salary Data.csv")
        df.dropna(inplace=True)
        df["Gender"] = _le_gender.transform(df["Gender"])
        df["Education Level"] = _le_edu.transform(df["Education Level"])
        df["Job Title"] = _le_job.transform(df["Job Title"])
        return df
    except Exception as e:
        st.error(f"❌ Failed to load or transform data: {e}")
        st.stop()

model, le_gender, le_edu, le_job = load_model()
st.sidebar.success("✅ Model and encoders loaded successfully.")

# Sidebar navigation
st.sidebar.title("📊 Navigation")
page = st.sidebar.radio("Choose a Page", [
    "🔮 Predict Salary", 
    "📁 Bulk CSV Upload", 
    "📊 Visualize Data", 
    "📈 Model Performance Dashboard", 
    "🤖 Chatbot Assistant"
])

# ---------- Predict Salary ----------
if page == "🔮 Predict Salary":
    st.header("🔮 Predict Employee Salary")

    age = st.slider("Age", 18, 65, 30)
    gender = st.selectbox("Gender", le_gender.classes_)
    education = st.selectbox("Education Level", le_edu.classes_)
    job = st.selectbox("Job Title", le_job.classes_)
    experience = st.slider("Years of Experience", 0, 40, 5)

    input_data = np.array([[age,
                            le_gender.transform([gender])[0],
                            le_edu.transform([education])[0],
                            le_job.transform([job])[0],
                            experience]])

    if st.button("🚀 Predict Salary"):
        try:
            prediction = model.predict(input_data)[0]
            st.markdown("---")
            st.markdown("### 🪙 **Estimated Annual Salary**")
            st.markdown(f"## 💰 **₹{int(prediction):,}**")
            st.markdown("---")
        except Exception as e:
            st.error(f"❌ Prediction error: {e}")

# ---------- Bulk Upload ----------
elif page == "📁 Bulk CSV Upload":
    st.header("📂 Upload CSV to Predict Salaries")
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            required_cols = ['Age', 'Gender', 'Education Level', 'Job Title', 'Years of Experience']
            if not all(col in data.columns for col in required_cols):
                st.error("❌ Uploaded file must contain all required columns.")
            else:
                data["Gender"] = le_gender.transform(data["Gender"])
                data["Education Level"] = le_edu.transform(data["Education Level"])
                data["Job Title"] = le_job.transform(data["Job Title"])
                data["Predicted Salary"] = model.predict(data)
                st.dataframe(data)
                csv = data.to_csv(index=False).encode()
                st.download_button("📥 Download Predicted CSV", csv, "predicted_salaries.csv", "text/csv")
        except Exception as e:
            st.error(f"❌ File processing error: {e}")

# ---------- Visualizations ----------
elif page == "📊 Visualize Data":
    st.header("📊 Data Visualizations")
    df = load_data(le_gender, le_edu, le_job)

    st.subheader("1. Salary Distribution")
    fig1, ax1 = plt.subplots()
    sns.histplot(df["Salary"], kde=True, ax=ax1, color='skyblue')
    st.pyplot(fig1)

    st.subheader("2. Salary by Education Level")
    fig2, ax2 = plt.subplots()
    sns.boxplot(x="Education Level", y="Salary", data=df, ax=ax2)
    ax2.set_xticklabels(le_edu.inverse_transform(sorted(df["Education Level"].unique())))
    st.pyplot(fig2)

    st.subheader("3. Experience vs Salary")
    fig3, ax3 = plt.subplots()
    sns.scatterplot(x="Years of Experience", y="Salary", hue=le_gender.inverse_transform(df["Gender"]), data=df, ax=ax3)
    st.pyplot(fig3)

    st.subheader("4. Correlation Heatmap")
    fig4, ax4 = plt.subplots()
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax4)
    st.pyplot(fig4)

# ---------- Performance Metrics ----------
elif page == "📈 Model Performance Dashboard":
    st.header("📈 Model Performance")
    df = load_data(le_gender, le_edu, le_job)
    X = df.drop("Salary", axis=1)
    y_true = df["Salary"]
    y_pred = model.predict(X)

    st.metric("Mean Absolute Error", f"{mean_absolute_error(y_true, y_pred):,.2f}")
    st.metric("RMSE", f"{np.sqrt(mean_squared_error(y_true, y_pred)):,.2f}")
    st.metric("R² Score", f"{r2_score(y_true, y_pred):.2f}")

# ---------- Chatbot ----------
elif page == "🤖 Chatbot Assistant":
    st.header("💬 Career Chatbot Assistant")
    query = st.text_input("Ask about salary, careers, remote jobs, etc.")

    if query:
        query = query.lower()
        responses = {
            ("increase", "raise"): "💡 Upskill, switch industries, or relocate to improve salary.",
            ("low", "why"): "📉 Salary could be low due to experience or education level.",
            ("career", "suggest"): "📚 Consider roles in Data Science, Cloud, DevOps, etc.",
            ("experience",): "🔁 Experience impacts salary — growth compounds over time.",
            ("education",): "🎓 Higher education often leads to better pay.",
            ("remote",): "🌐 Remote jobs can offer global pay if your skills match."
        }

        for keywords, response in responses.items():
            if any(k in query for k in keywords):
                st.success(response)
                break
        else:
            st.info("🤔 I didn't catch that. Try asking about salary tips or career growth.")
