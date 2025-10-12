# Churn Guardian: An AI-Powered Customer Retention Platform

**Project Author:** Incognius
**Submission Date:** 2025-10-12

## Abstract

Churn Guardian is an end-to-end machine learning system designed to combat customer churn for auto insurance companies. It moves beyond simple predictions by providing a deeply interactive and explainable AI platform for non-technical users. The system leverages an XGBoost model to generate highly accurate churn risk scores, but its core innovation lies in its ability to translate model outputs into actionable business insights. The platform features a "Guardian Dashboard" for identifying high-risk customers and understanding the specific drivers behind their risk, and a "Strategy Sandbox"—a live simulation engine that allows strategists to model the real-time impact of retention campaigns (e.g., premium discounts) on a customer segment's churn probability. This demonstrates a full-circle approach from data processing and model training to explainable prediction and strategic business planning, all within a scalable and performant architecture.

---

## End-to-End Workflow

This project is not a single script, but a complete ML system with a defined operational workflow:

1.  **Data Pipeline (`scripts/run_data_pipeline.py`):** The raw dataset is processed through a robust, leak-proof pipeline. It performs a stratified train-validation-test split and fits a preprocessor *only* on the training data to prevent data leakage. The processed datasets and the fitted preprocessor artifact are saved.

2.  **Model Training (`scripts/create_final_model.py`):** A champion XGBoost model is trained on the processed training data using the best hyperparameters. The final trained model is saved as an artifact.

3.  **Offline Prediction & Explanation (`scripts/run_final_predictions.py`):** To ensure the main dashboard is fast, an offline batch job is run. It identifies the top 1,000 highest-risk customers, generates SHAP-based reason codes *only* for them, and saves the final, enriched results to a single CSV file.

4.  **Backend API (`backend/main.py`):** A FastAPI server provides a clean interface to the pre-computed prediction results, serving data to the frontend application.

5.  **Frontend Application (`frontend/`):** A multi-page Streamlit application provides the user interface, including the risk dashboard, strategic overview, and the live "What-If" Strategy Sandbox. The Sandbox loads the model and preprocessor artifacts directly to run its simulations client-side for maximum interactivity.

---

## Technical Stack

*   **Backend:** Python, FastAPI
*   **Frontend:** Streamlit, Plotly
*   **ML & Data Science:** Pandas, Scikit-learn, XGBoost, SHAP, Joblib
*   **Core Logic:** A custom `ChurnDataPreprocessor` class handles all feature engineering and data transformation.

---

## Executable Instructions

### 1. Prerequisites

*   Python 3.10+
*   A GitHub account to clone the repository.

### 2. Setup

**a. Clone the Repository:**
```bash
git clone <your-github-repo-url>
cd <your-repo-name>
```

**b. Create and Activate a Virtual Environment:**
```bash
# For Windows
python -m venv .venv
.\.venv\Scripts\activate

# For macOS/Linux
python3 -m venv .venv
source .venv/bin/activate
```

**c. Install Dependencies:**
```bash
pip install -r requirements.txt
```

**d. Download the Dataset:**
*   Download the "Auto Insurance churn analysis" dataset from Kaggle.
*   Find the `autoinsurance_churn.csv` file.
*   Place this file inside the `data/raw/` directory in the project. The final path should be `data/raw/auto_insurance_churn.csv`.
*(Note: The dataset is not included in the repository to adhere to size constraints.)*

### 3. Running the Full Pipeline

Execute these commands from the project's root directory in the specified order.

**a. Run the Data Processing Pipeline:**
This will create the train/val/test splits and save the preprocessor.
```bash
python -m scripts.run_data_pipeline
```

**b. Train the Champion Model:**
This will train the XGBoost model and save it.
```bash
python -m scripts.create_final_model
```

**c. Generate Offline Predictions for the Dashboard:**
This will create the `predictions_with_explanations.csv` file used by the backend.
```bash
python -m scripts.run_final_predictions
```

### 4. Launching the Application

You will need two separate terminals for this step.

**a. Terminal 1: Start the Backend API Server:**
```bash
# Navigate to the backend directory
cd backend

# Start the server
uvicorn main:app --reload
```
The server will be running at `http://localhost:8000`.

**b. Terminal 2: Start the Frontend Streamlit Application:**
```bash
# From the project root directory
streamlit run frontend/app.py
```
Your browser should open with the Churn Guardian application running.

---

## Project Structure

```
.
├── artifacts/                # Saved models, reports, and prediction results
│   ├── models/
│   ├── reports/
│   └── results/
├── backend/                  # FastAPI backend code
│   └── main.py
├── data/                     # Raw and processed data
│   ├── processed/
│   └── raw/
├── frontend/                 # Streamlit frontend application
│   ├── pages/                # Individual app pages
│   └── app.py                # Main entry point for Streamlit
├── src/                      # Source code for core logic
│   └── preprocess.py         # The custom data preprocessor class
├── scripts/                  # Standalone scripts for the ML pipeline
│   ├── run_data_pipeline.py
│   ├── create_final_model.py
│   ├── run_final_predictions.py
│   └── generate_model_report.py
├── .gitignore
├── README.md
└── requirements.txt
```