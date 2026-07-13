import os
import pickle
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from google import genai

# 1. LOAD CONFIGURATION ENVIRONMENT AT BOOT
# UPPERCASE COMMENTS FOR ACCURACY TRACKING
load_dotenv()

app = FastAPI(
    title="Real Estate Market Intelligence Engine",
    description="FastAPI Backend serving Category Classification, Price Regression, and LLM Interpretation Pipelines.",
    version="2.0.0"
)

# 2. INITIALIZE GLOBAL PRODUCTION CLIENT ASSETS
try:
    # ACCESS API KEY DYNAMICALLY FROM SECURE ENVIRONMENT
    api_key = os.getenv("GEMINI_API_KEY")
    ai_client = genai.Client(api_key=api_key) if api_key else None
except Exception as e:
    print(f"CRITICAL WARNING: GenAI SDK initialization failed: {e}")
    ai_client = None

# GLOBAL MODEL PLACEHOLDERS
classifier_pipeline = None
regressor_model = None

@app.on_event("startup")
def load_serialized_models():
    """LOAD MACHINE LEARNING MODELS FROM ARTIFACT PATHS AT SERVER STARTUP."""
    # UPPERCASE COMMENTS FOR ACCURACY TRACKING
    global classifier_pipeline, regressor_model
    try:
        # ABSOLUTE TARGET PATHS
        classifier_path = "/workspaces/graduation_project/src/models/logistic_regression_pipeline.pkl"
        regressor_path = "/workspaces/graduation_project/src/models/gradient_boosting_regression_model.pkl"
        
        models_dir = "/workspaces/graduation_project/src/models"
        
        # CHECK CLASSIFIER
        if os.path.exists(classifier_path):
            with open(classifier_path, "rb") as f:
                classifier_pipeline = pickle.load(f)
                print("SYSTEM LOG: Classifier pipeline loaded and ready.")
        else:
            print(f"CRITICAL ERROR: Could not find classifier at: {classifier_path}")
            if os.path.exists(models_dir):
                print(f"FOUND FILES IN DIRECTORY {models_dir}: {os.listdir(models_dir)}")
            classifier_pipeline = None

        # CHECK REGRESSOR
        if os.path.exists(regressor_path):
            with open(regressor_path, "rb") as f:
                regressor_model = pickle.load(f)
                print("SYSTEM LOG: Regressor engine loaded and ready.")
        else:
            print(f"CRITICAL ERROR: Could not find regressor at: {regressor_path}")
            if os.path.exists(models_dir):
                print(f"FOUND FILES IN DIRECTORY {models_dir}: {os.listdir(models_dir)}")
            regressor_model = None
            
    except Exception as e:
        print(f"CRITICAL DESERIALIZATION FAULT: {str(e)}")
        classifier_pipeline = None
        regressor_model = None
# 3. SCHEMA DEFINITION FOR VALIDATING INCOMING REQUEST PAYLOADS
class HouseFeaturesInput(BaseModel):
    bedrooms: int = Field(..., example=3, description="Number of bedrooms")
    bathrooms: float = Field(..., example=2.25, description="Number of bathrooms")
    sqft_living: int = Field(..., example=2570, description="Living space footprint area in sqft")
    sqft_lot: int = Field(..., example=7242, description="Total lot size footprint area in sqft")
    floors: float = Field(..., example=2.0, description="Number of structural floors")
    waterfront: int = Field(..., example=0, description="Waterfront view status flag (0=No, 1=Yes)")
    view: int = Field(..., example=0, description="Quality index rating of view (0-4)")
    condition: int = Field(..., example=3, description="Overall condition grading assessment score (1-5)")
    sqft_above: int = Field(..., example=2170, description="Square footage footprint above ground grade level")
    sqft_basement: int = Field(..., example=400, description="Square footage footprint below ground basement level")
    yr_built: int = Field(..., example=2005, description="Original year of construction completion")
    yr_renovated: int = Field(..., example=0, description="Year of last structural renovation (0 if never)")

# 4. SERVER HEALTH CHECK ENDPOINT
@app.get("/health", tags=["Status"])
def server_health_check():
    return {
        "status": "online",
        "models_loaded": (classifier_pipeline is not None) and (regressor_model is not None),
        "llm_client_active": ai_client is not None
    }

# 5. CORE PREDICTION AND INTERPRETATION ENDPOINT
@app.post("/predict", tags=["Analytics"])
def run_real_estate_pipeline(payload: HouseFeaturesInput):
    """
    RUN FULL PIPELINE: PREDICTS CATEGORY PORTFOLIO, CALCULATES CONTINUOUS PRICE ESTIMATE,
    AND GENERATES AN LLM COGNITIVE INTERPRETATION FROM LIVE PREDICTION ARTIFACTS.
    """
    # UPPERCASE COMMENTS FOR ACCURACY TRACKING
    global classifier_pipeline, regressor_model

    # RUNTIME SAFETY TYPE VALIDATION
    if isinstance(classifier_pipeline, str):
        print(f"CRITICAL BREAK: classifier_pipeline was overwritten with string: '{classifier_pipeline}'")
        raise HTTPException(
            status_code=500, 
            detail=f"Internal State Error: Classifier is a string description: '{classifier_pipeline}'"
        )

    if classifier_pipeline is None or regressor_model is None:
        raise HTTPException(status_code=503, detail="Machine learning inference engines are currently offline.")

    # CONVERT PAYLOAD CONTENT TO INPUT DICTIONARY FOR FEATURE INJECTION
    input_data = payload.dict()

    # CALCULATE LOG SPECS TO MATCH NOTEBOOK PIPELINE TRAINING PATTERNS
    input_data['log_sqft_living'] = np.log1p(input_data['sqft_living'])
    input_data['log_sqft_lot'] = np.log1p(input_data['sqft_lot'])
    input_data['log_sqft_above'] = np.log1p(input_data['sqft_above'])
    input_data['log_sqft_basement'] = np.log1p(input_data['sqft_basement'])

    # --- STEP 1: PORTFOLIO CATEGORY CLASSIFICATION ---
    classification_features = [
        'log_sqft_living', 'log_sqft_lot', 'log_sqft_above', 'log_sqft_basement',
        'bedrooms', 'bathrooms', 'floors', 'view', 'condition', 'yr_built', 'yr_renovated', 'waterfront'
    ]
    
    X_class = pd.DataFrame([input_data])[classification_features]
    predicted_cluster = int(classifier_pipeline.predict(X_class)[0])
    
    cluster_mapping = {0: "Budget Portfolio", 1: "Premium Portfolio", 2: "Luxury Portfolio"}
    assigned_category = cluster_mapping.get(predicted_cluster, f"Cluster {predicted_cluster}")

    # --- STEP 2: CONTINUOUS PRICE REGRESSION ---
    regression_features = [
        'log_sqft_living', 'log_sqft_lot', 'log_sqft_above', 'log_sqft_basement',
        'bedrooms', 'bathrooms', 'floors', 'view', 'condition', 'yr_built', 'yr_renovated', 'waterfront'
    ]
    
    X_reg = pd.DataFrame([input_data])[regression_features]
    
    # MANUAL ONE-HOT-ENCODING ALIGNMENT FOR CLUSTER ID DUMMIES
    X_reg['cluster_id_1'] = 1 if predicted_cluster == 1 else 0
    X_reg['cluster_id_2'] = 1 if predicted_cluster == 2 else 0
    
    predicted_log_price = regressor_model.predict(X_reg)[0]
    final_predicted_price = float(np.expm1(predicted_log_price))

    # --- STEP 3: AGENTIC LLM INTERPRETATION PIPELINE ---
    # --- STEP 3: AGENTIC LLM INTERPRETATION PIPELINE ---
    llm_interpretation = "LLM Engine offline. Missing environment credentials."

    if ai_client:
        try:
            sys_instruction = (
                "You are an expert AI Real Estate Analyst evaluating house appraisal model outputs for a graduation board. "
                "Provide a professional, clear explanation of why the asset was priced this way based on its specific size, "
                "age, and condition parameters."
            )
            
            prompt_payload = f"""
            Interpret this real estate model output:
            - Model Predicted Price: ${final_predicted_price:,.2f}
            - Assigned Group Category: {assigned_category}
            
            Property Metrics:
            - Size: {input_data['sqft_living']} sqft living area ({input_data['sqft_lot']} sqft lot)
            - Layout: {input_data['bedrooms']} beds, {input_data['bathrooms']} baths, {input_data['floors']} floors
            - Context: Built in {input_data['yr_built']}, Condition Rating: {input_data['condition']}/5, View Quality: {input_data['view']}/4
            
            Provide a tight 3-sentence summary:
            1. Analyze the core driver of this value (e.g., size vs vintage balance).
            2. Evaluate if this valuation looks realistic and justified from a market perspective.
            3. Synthesize the final investment trust outlook.
            """
            
            # UPDATE MODEL TO GEMINI-1.5-FLASH
            response = ai_client.models.generate_content(
                model='gemini-1.5-flash',
                contents=prompt_payload,
                config={'system_instruction': sys_instruction, 'temperature': 0.2}
            )
            llm_interpretation = response.text
        except Exception as api_err:
            llm_interpretation = f"Live interpretation generation suspended: {str(api_err)}"

    return {
        "classification": {
            "cluster_id": predicted_cluster,
            "portfolio_category": assigned_category
        },
        "regression": {
            "predicted_price_raw": final_predicted_price,
            "predicted_price_formatted": f"${final_predicted_price:,.2f}"
        },
        "llm_interpretation": llm_interpretation
    }