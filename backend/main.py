import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict

# --- Load the real model's output once at startup ---
try:
    # Adjust the path to be relative to the project root where you run the server
    predictions_df = pd.read_csv("../artifacts/results/predictions_with_explanations.csv")
    # Sort by churn probability descending - this is our main ranking
    predictions_df = predictions_df.sort_values(by="churn_probability", ascending=False).reset_index(drop=True)
    print("Successfully loaded and sorted prediction data.")

    predictions_df = predictions_df.head(250).reset_index(drop=True)    
    print(f"Successfully loaded and optimized prediction data. Now serving the top {len(predictions_df)} customers.")

except FileNotFoundError:
    print("ERROR: predictions_with_explanations.csv not found. Please run the prediction pipeline.")
    predictions_df = pd.DataFrame() # Create an empty DataFrame to avoid crashing

# --- API Application ---
app = FastAPI(title="Churn Guardian API - ML Integrated", version="3.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# --- CORE ENDPOINTS (POWERED BY OUR ML MODEL) ---

@app.get("/customers")
async def get_all_customers(limit: int = 1000):
    """
    Gets a list of all customers, sorted by churn risk.
    Now powered by the real model's output.
    """
    if predictions_df.empty:
        return {"customers": []}
    
    # Select relevant columns for the customer list view
    customer_list = predictions_df[[
        'customer_id', 
        'churn_probability',
        # You might need to add a 'name' column if you have one, or generate it.
        # For now, we'll use the ID as the name.
    ]].head(limit)
    
    # Rename for frontend compatibility
    customer_list = customer_list.rename(columns={
        'customer_id': 'id',
        'churn_probability': 'risk_score'
    })
    
    # Add a placeholder name column if it doesn't exist
    customer_list['name'] = 'Customer ' + customer_list['id'].astype(str)

    return {"customers": customer_list.to_dict('records')}

@app.get("/customer/{customer_id}")
async def get_customer(customer_id: int):
    """
    Gets detailed information for a single customer.
    """
    if predictions_df.empty:
        raise HTTPException(status_code=404, detail="Prediction data not loaded.")

    try:
        # customer_id from URL is a string, needs to be int for lookup if index is int
        customer_row = predictions_df[predictions_df['customer_id'] == customer_id].iloc[0]
    except (IndexError, KeyError):
        raise HTTPException(status_code=404, detail=f"Customer with ID {customer_id} not found.")

    # Convert the row to a dictionary
    customer_data = customer_row.to_dict()

    # Structure the response to match the frontend's expectations
    return {
        "customer_id": int(customer_data['customer_id']),
        "customer_name": f"Customer {int(customer_data['customer_id'])}", # Placeholder name
        "risk_score": customer_data['churn_probability'],
        "reason_codes": customer_data['reason_codes'],
        "customer_data": { # Pass along other key details
            "tenure": customer_data.get('days_tenure', 'N/A'),
            "premium": customer_data.get('curr_ann_amt', 'N/A'),
            "age": customer_data.get('age_in_years', 'N/A'),
            "income": customer_data.get('income', 'N/A')
        }
    }

# This endpoint is now simpler as the main list is already sorted by risk
@app.get("/actions/immediate")
async def get_immediate_actions(risk_threshold: float = 0.7):
    """
    Gets customers with a risk score above the threshold.
    """
    if predictions_df.empty:
        return {"actions": []}

    high_risk_df = predictions_df[predictions_df['churn_probability'] > risk_threshold]
    
    action_list = high_risk_df[[
        'customer_id',
        'churn_probability',
        'reason_codes'
    ]].copy()

    action_list = action_list.rename(columns={
        'customer_id': 'id',
        'churn_probability': 'risk_score'
    })
    action_list['name'] = 'Customer ' + action_list['id'].astype(str)
    
    # Recommend a generic action based on top reason code
    action_list['recommended_action'] = action_list['reason_codes'].apply(
        lambda x: f"Address top factor: {x.split(',')[0].strip()}"
    )

    return {"actions": action_list.to_dict('records')}


@app.get("/analytics/segment")
async def get_customer_segment(
    min_age: int = 0, max_age: int = 100,
    min_income: int = 0, max_income: int = 1000000,
    min_tenure: int = 0, max_tenure: int = 36500, # Max tenure of 100 years in days
    min_risk: float = 0.0, max_risk: float = 1.0
):
    """
    Dynamically filters customers based on provided criteria and returns
    aggregated analytics and the list of customer IDs in that segment.
    """
    if predictions_df.empty:
        raise HTTPException(status_code=503, detail="Prediction data not loaded.")

    # Create a copy to avoid modifying the global DataFrame
    df_copy = predictions_df.copy()

    # --- BUG FIX: Handle missing income values before filtering ---
    df_copy['income'] = df_copy['income'].fillna(0)

    # Apply filters to the copied DataFrame
    filtered_df = df_copy[
        (df_copy['age_in_years'].between(min_age, max_age)) &
        (df_copy['income'].between(min_income, max_income)) &
        (df_copy['days_tenure'].between(min_tenure, max_tenure)) &
        (df_copy['churn_probability'].between(min_risk, max_risk))
    ]

    # --- AGGREGATED ANALYTICS ---
    if filtered_df.empty:
        return {
            "summary": {
                "segment_size": 0,
                "average_churn_risk": 0,
            },
            "top_risk_factors": [],
            "customers": []
        }

    # Calculate summary stats for the segment
    summary = {
        "segment_size": len(filtered_df),
        "average_churn_risk": filtered_df['churn_probability'].mean(),
    }

    # Find the top risk factors FOR THIS SEGMENT
    # This is powerful - it gives context-specific reasons
    reason_counts = filtered_df['reason_codes'].str.split(',\s*', expand=True).stack().value_counts()
    top_risk_factors = reason_counts.head(5).reset_index()
    top_risk_factors.columns = ['factor', 'count']
    
    # Get the list of customers in the segment
    customer_list = filtered_df[['customer_id', 'churn_probability', 'reason_codes']].to_dict('records')

    return {
        "summary": summary,
        "top_risk_factors": top_risk_factors.to_dict('records'),
        "customers": customer_list
    }

if __name__ == "__main__":
    # To run this, navigate to the `backend` directory and run: uvicorn main:app --reload
    uvicorn.run(app, host="0.0.0.0", port=8000)