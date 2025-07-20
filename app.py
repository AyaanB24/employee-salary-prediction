import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ========== Caching ========== 
@st.cache_resource
def load_model():
    model = joblib.load("salary_model.pkl")
    le_gender = joblib.load("le_gender.pkl")
    le_edu = joblib.load("le_edu.pkl")
    le_job = joblib.load("le_job.pkl")
    return model, le_gender, le_edu, le_job

@st.cache_data
def load_data(_le_gender, _le_edu, _le_job):
    df = pd.read_csv("Salary Data.csv")
    df.dropna(inplace=True)
    df["Gender"] = _le_gender.transform(df["Gender"])
    df["Education Level"] = _le_edu.transform(df["Education Level"])
    df["Job Title"] = _le_job.transform(df["Job Title"])
    return df

model, le_gender, le_edu, le_job = load_model()
st.sidebar.success("✅ Model and encoders loaded successfully.")

st.sidebar.title("📊 Navigation")
page = st.sidebar.radio("Choose a Page", ["🔮 Predict Salary", "📁 Bulk CSV Upload", "📊 Visualize Data", "📈 Model Performance Dashboard", "🤖 Chatbot Assistant"])

# ---------- Predict Salary ----------
if page == "🔮 Predict Salary":
    st.title("💼 Employee Salary Prediction")

    # Input fields
    age = st.slider("Age", 18, 65, 30)
    gender = st.selectbox("Gender", le_gender.classes_)
    education = st.selectbox("Education Level", le_edu.classes_)
    job = st.selectbox("Job Title", le_job.classes_)
    experience = st.slider("Years of Experience", 0, 40, 5)

    # Prepare input
    input_data = np.array([[age,
                            le_gender.transform([gender])[0],
                            le_edu.transform([education])[0],
                            le_job.transform([job])[0],
                            experience]])

    # Predict button
    if st.button("🚀 Predict Salary"):
        prediction = model.predict(input_data)[0]
        salary = int(prediction)

        # Classy display using markdown and formatting
        st.markdown("---")
        st.markdown("### 🪙 **Estimated Annual Salary**")
        st.markdown(f"## 💰 **₹{salary:,.2f}**")
        st.markdown("---")

        # Additional notes
        st.info("📌 This prediction is based on your provided inputs and the model's training data.")
        st.info("✅ Accuracy may vary depending on job type, location, and experience level.")

# ---------- Bulk Upload ----------
elif page == "📁 Bulk CSV Upload":
    st.title("📂 Predict Salary from Uploaded CSV")
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            required_cols = ['Age', 'Gender', 'Education Level', 'Job Title', 'Years of Experience']
            if not all(col in data.columns for col in required_cols):
                st.error("❌ Uploaded file must contain the required columns")
            else:
                st.subheader("📋 Input Preview")
                st.dataframe(data.head())
                data["Gender"] = le_gender.transform(data["Gender"])
                data["Education Level"] = le_edu.transform(data["Education Level"])
                data["Job Title"] = le_job.transform(data["Job Title"])
                result = model.predict(data)
                data["Predicted Salary"] = result
                st.subheader("✅ Predictions")
                st.dataframe(data)
                csv = data.to_csv(index=False).encode()
                st.download_button("📥 Download Predicted CSV", csv, "predicted_salaries.csv", "text/csv")
        except Exception as e:
            st.error(f"❌ Error processing file: {e}")

# ---------- Visualize Data ----------
elif page == "📊 Visualize Data":
    st.title("📊 Salary Data Visualizations")
    df = load_data(le_gender, le_edu, le_job)

    st.subheader("1. Salary Distribution")
    fig1, ax1 = plt.subplots()
    sns.histplot(df["Salary"], kde=True, color="skyblue", ax=ax1)
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

# ---------- Model Performance Dashboard ----------
elif page == "📈 Model Performance Dashboard":
    st.title("📈 Model Performance Dashboard")
    df = load_data(le_gender, le_edu, le_job)
    X = df.drop("Salary", axis=1)
    y_true = df["Salary"]
    y_pred = model.predict(X)

    st.write("**Mean Absolute Error (MAE):**", round(mean_absolute_error(y_true, y_pred), 2))
    st.write("**Root Mean Squared Error (RMSE):**", round(np.sqrt(mean_squared_error(y_true, y_pred)), 2))
    st.write("**R² Score:**", round(r2_score(y_true, y_pred), 2))


elif page == "🤖 Chatbot Assistant":
    st.title("💬 Career Assistant Chatbot")
    chat_input = st.text_input("Ask me anything about salary, career growth, remote work, etc.", key="chatbot_input")

    if chat_input:
        chat_input = chat_input.lower()
        responses = {
            ("increase", "raise"): "💡 Tip: Upskill with certifications, switch to high-paying companies, or relocate to metro cities.",
            ("low", "why"): "📉 The prediction might be low due to limited experience, lower education, or location.",
            ("career", "suggest"): "📚 Explore careers in Data Science, Software Dev, Cloud Computing for higher pay.",
            ("experience",): "🔁 Experience strongly impacts salary. Each year can bring a ~10-20% hike on average.",
            ("education",): "🎓 Higher degrees (like Masters or MBA) usually lead to better salary packages.",
            ("remote",): "🌐 Remote roles often pay globally competitive salaries if your skills match."
        }

        matched = False
        for keywords, reply in responses.items():
            if any(keyword in chat_input for keyword in keywords):
                st.success(reply)
                matched = True
                break
        if not matched:
            st.info("🤔 Sorry, I don't have an answer for that. Try asking about salary tips or career growth.")
