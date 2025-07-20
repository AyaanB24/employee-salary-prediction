# 💼 Employee Salary Prediction App

A user-friendly Streamlit web app that predicts employee salaries based on age, gender, education level, job title, and experience. It also supports bulk predictions, data visualizations, model performance evaluation, and a chatbot career assistant.

---

## 🚀 Features

- 🔮 **Predict Salary**: Estimate annual and monthly salary based on user input.
- 📁 **Bulk CSV Upload**: Upload CSV files to predict salaries for multiple employees at once.
- 📊 **Visualize Data**: Explore salary distributions, box plots by education, correlation heatmaps, and more.
- 📈 **Model Performance Dashboard**: Evaluate the model with MAE, RMSE, and R² Score.
- 🤖 **Chatbot Assistant**: Ask questions about career growth, education, remote work, and more.

---

## 🛠️ Tech Stack

- **Frontend**: [Streamlit](https://streamlit.io/)
- **Backend Model**: Scikit-learn Regression
- **Data Handling**: Pandas, NumPy
- **Visualizations**: Matplotlib, Seaborn
- **Deployment**: Streamlit Cloud / Localhost

---

## 📂 File Structure

📦 Salary Prediction App
├── app.py # Main Streamlit app
├── salary_model.pkl # Trained ML model (joblib)
├── le_gender.pkl # Label encoder for gender
├── le_edu.pkl # Label encoder for education
├── le_job.pkl # Label encoder for job titles
├── Salary Data.csv # Dataset used
├── bulk_test_employees.csv # Test bulk dataset prediction
└── README.md # Project documentation
