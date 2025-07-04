# 🧠 Autism Spectrum Disorder (ASD) Detection using Machine Learning

This repository contains the code, data, and app interface to predict Autism Spectrum Disorder using classic Machine Learning models and an interactive Streamlit app.

## 📌 Highlights

- 🔢 **Data Preprocessing** with Encoding & Cleaning
- 🧠 **Classification Models**: Logistic Regression, K-Nearest Neighbors, SVM, Decision Tree
- 📈 **Model Evaluation**: Confusion Matrix, Accuracy, Precision, Recall, F1 Score
- 🌐 **Interactive Web App** built with Streamlit
- 🧪 **Prediction Interface** to test if user input indicates ASD

---

## 📊 Dataset

**Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Autism+Screening+Adult)

Originally compiled by **Dr. Fadi Thabtah**, this dataset was collected using a mobile application for caregivers and parents to screen toddlers (ages 12–36 months) for ASD. The screening is based on responses to the AQ-10 (Autism Spectrum Quotient) test.

### ✨ Features

| Feature         | Description                                    |
| --------------- | ---------------------------------------------- |
| A1-A10          | Answers to AQ-10 Test (0 or 1)                 |
| Age             | Age of the participant                         |
| Sex             | Gender (m/f)                                   |
| Ethnicity       | Participant's ethnicity                        |
| Jaundice        | Was the child born with jaundice? (yes/no)     |
| Family\_Member  | Any family member diagnosed with ASD? (yes/no) |
| Result (Target) | ASD Positive or Negative (Yes/No)              |

---

## 🧹 Data Preprocessing

- Removed irrelevant fields: `Case_No`, `Who completed the test`, and `Qchat-10-Score`
- Encoded categorical values:
  - Binary: `Sex`, `Jaundice`, `Family_Member`, `Result` as 0/1
  - Multiclass: `Ethnicity` encoded using `LabelEncoder`
- Checked and ensured no missing values

---

## 🧠 Models Used

1. **Logistic Regression**
2. **K-Nearest Neighbors (KNN)**
3. **Support Vector Machine (SVM)**
4. **Decision Tree Classifier**

Each model was trained using scikit-learn, and performance was evaluated using confusion matrix and classification metrics.

### ✅ Best Model (Logistic Regression)

- Accuracy: **98%**
- Precision: **98%**
- Recall: **97%**
- F1 Score: **97%**

---

## 🌐 Streamlit App

Run the app locally or deploy it on Hugging Face Spaces.

### Features:

- 📥 Uploads `autismData.csv` file (or use default embedded dataset)
- 🧠 Choose among 4 ML models to train
- 🧪 Run evaluation and view:
  - Confusion Matrix
  - Classification Report
- 🔮 Predict ASD using custom input (AQ responses, age, sex, etc.)
- ✅ Final message: "Autism Detected" or "No Autism Detected"

- 💻 Hugging Face Demo:https://huggingface.co/spaces/Rakshitha2415/ASD_detection

### To Run the App

```bash
streamlit run app.py
```

---

## 📂 Project Structure

```
asd-predictor/
├── app.py       # Streamlit web app
├── autismData.csv      # Input dataset
├── requirements.txt    # Python dependencies
├── README.md           # Project overview
```

---

## ⚙️ Technologies Used

- Python 🐍
- Scikit-learn (Model Training & Evaluation)
- Pandas / NumPy (Data Handling)
- Streamlit (Web UI)

---

## 🚀 Getting Started

### 1. Clone the Repository:

```bash
git clone https://github.com/yourusername/asd-predictor.git
cd asd-predictor
```

### 2. Install Dependencies:

```bash
pip install -r requirements.txt
```

### 3. Launch the Streamlit App:

```bash
streamlit run app.py
```

---

## 🤝 Acknowledgements

- Dr. Fadi Thabtah for the dataset
- UCI Machine Learning Repository

---


