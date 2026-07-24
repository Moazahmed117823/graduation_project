
import os
import pickle
from pathlib import Path
from typing import Literal
import numpy as np
import pandas as pd
import uvicorn
from fastapi.responses import FileResponse
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openrouter import OpenRouter
from pydantic import BaseModel, Field, model_validator

# 1. LOAD CONFIGURATION ENVIRONMENT AT BOOT
# UPPERCASE COMMENTS FOR ACCURACY TRACKING
load_dotenv()

app = FastAPI(
    title="Real Estate Market Intelligence Engine",
    description="FastAPI Backend serving Category Classification, Price Regression, and LLM Interpretation Pipelines.",
    version="2.0.0",
)

# ALLOW ALL ORIGINS SO THE FRONTEND CAN REACH THE API FROM file://, DEV SERVERS, ETC.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. INITIALIZE GLOBAL PRODUCTION CLIENT ASSETS
try:
    # ACCESS API KEY DYNAMICALLY FROM SECURE ENVIRONMENT
    api_key = os.getenv("OPENROUTER_API_KEY")
    ai_client = OpenRouter(api_key=api_key) if api_key else None
except Exception as e:
    print(f"CRITICAL WARNING: OpenRouter SDK initialization failed: {e}")
    ai_client = None

# GLOBAL MODEL PLACEHOLDERS
classifier_pipeline = None
regressor_model = None

CITY_ENCODING_MAP = {
    'Shoreline': 420396.79, 'Seattle': 579837.47, 'Kent': 439492.45, 
    'Bellevue': 847180.66, 'Redmond': 667649.53, 'Maple Valley': 336582.70, 
    'North Bend': 406793.28, 'Lake Forest Park': 465859.08, 'Sammamish': 686917.56, 
    'Auburn': 299340.49, 'Des Moines': 310396.57, 'Bothell': 496545.05, 
    'Federal Way': 289888.43, 'Kirkland': 651583.59, 'Issaquah': 596163.75, 
    'Woodinville': 609560.71, 'Normandy Park': 531629.02, 'Fall City': 592637.84, 
    'Renton': 377040.97, 'Carnation': 528204.09, 'Snoqualmie': 536400.18, 
    'Duvall': 418754.09, 'Burien': 349860.06, 'Covington': 319533.51, 
    'Other': 580836.13, 'Kenmore': 448533.69, 'Newcastle': 641613.96, 
    'Mercer Island': 1123040.74, 'Black Diamond': 498928.87, 'Ravensdale': 543847.94, 
    'Clyde Hill': 774526.80, 'Algona': 489085.47, 'Tukwila': 378723.85, 
    'Vashon': 495509.27, 'SeaTac': 333934.42, 'Medina': 983976.74, 
    'Enumclaw': 383368.82, 'Pacific': 487330.60
}

# CALCULATE A DYNAMIC FALLBACK VALUE TO PREVENT UNEXPECTED API CRASHES
DEFAULT_CITY_WEIGHT = sum(CITY_ENCODING_MAP.values()) / len(CITY_ENCODING_MAP)

@app.on_event("startup")
def load_serialized_models():
    """LOAD MACHINE LEARNING MODELS FROM ARTIFACT PATHS AT SERVER STARTUP."""
    # UPPERCASE COMMENTS FOR ACCURACY TRACKING
    global classifier_pipeline, regressor_model
    try:
        # DYNAMIC ABSOLUTE PATHS — anchored to this file's location (works from any CWD)
        _models_dir = Path(__file__).resolve().parent / "models"
        classifier_path = str(_models_dir / "Gradient_Boosting_Pipeline.pkl")
        regressor_path = str(_models_dir / "gradient_boosting_regression_model.pkl")
        models_dir = str(_models_dir)

        # CHECK CLASSIFIER
        if os.path.exists(classifier_path):
            with open(classifier_path, "rb") as f:
                classifier_pipeline = pickle.load(f)
                print("SYSTEM LOG: Classifier pipeline loaded and ready.")
        else:
            print(f"CRITICAL ERROR: Could not find classifier at: {classifier_path}")
            if os.path.exists(models_dir):
                print(
                    f"FOUND FILES IN DIRECTORY {models_dir}: {os.listdir(models_dir)}"
                )
            classifier_pipeline = None

        # CHECK REGRESSOR
        if os.path.exists(regressor_path):
            with open(regressor_path, "rb") as f:
                regressor_model = pickle.load(f)
                print("SYSTEM LOG: Regressor engine loaded and ready.")
        else:
            print(f"CRITICAL ERROR: Could not find regressor at: {regressor_path}")
            if os.path.exists(models_dir):
                print(
                    f"FOUND FILES IN DIRECTORY {models_dir}: {os.listdir(models_dir)}"
                )
            regressor_model = None

    except Exception as e:
        print(f"CRITICAL DESERIALIZATION FAULT: {str(e)}")
        classifier_pipeline = None
        regressor_model = None


# 3. SCHEMA DEFINITION FOR VALIDATING INCOMING REQUEST PAYLOADS
class HouseFeaturesInput(BaseModel):
    bedrooms: int = Field(..., json_schema_extra=3, description="Number of bedrooms")
    bathrooms: float = Field(..., json_schema_extra=2.25, description="Number of bathrooms")
    sqft_living: int = Field(
        ..., json_schema_extra=2570, description="Living space footprint area in sqft"
    )
    sqft_lot: int = Field(
        ..., json_schema_extra=7242, description="Total lot size footprint area in sqft"
    )
    floors: float = Field(..., json_schema_extra=2.0, description="Number of structural floors")
    waterfront: int = Field(
        ..., json_schema_extra=0, description="Waterfront view status flag (0=No, 1=Yes)"
    )
    view: int = Field(..., json_schema_extra=0, description="Quality index rating of view (0-4)")
    condition: int = Field(
        ..., json_schema_extra=3, description="Overall condition grading assessment score (1-5)"
    )
    sqft_above: int = Field(
        ...,
        json_schema_extra=2170,
        description="Square footage footprint above ground grade level",
    )
    sqft_basement: int = Field(
        ...,
        json_schema_extra=400,
        description="Square footage footprint below ground basement level",
    )
    yr_built: int = Field(
        ..., json_schema_extra=2005, description="Original year of construction completion"
    )
    yr_renovated: int = Field(
        ..., json_schema_extra=0, description="Year of last structural renovation (0 if never)"
    )
    city: Literal[
        'Shoreline', 'Seattle', 'Kent', 'Bellevue', 'Redmond', 'Maple Valley', 
        'North Bend', 'Lake Forest Park', 'Sammamish', 'Auburn', 'Des Moines', 
        'Bothell', 'Federal Way', 'Kirkland', 'Issaquah', 'Woodinville', 
        'Normandy Park', 'Fall City', 'Renton', 'Carnation', 'Snoqualmie', 
        'Duvall', 'Burien', 'Covington', 'Other', 'Kenmore', 'Newcastle', 
        'Mercer Island', 'Black Diamond', 'Ravensdale', 'Clyde Hill', 'Algona', 
        'Tukwila', 'Vashon', 'SeaTac', 'Medina', 'Enumclaw', 'Pacific'
    ] = Field(..., json_schema_extra="Seattle", description="Select the city municipality for geographic price weighting")

    @model_validator(mode='after')
    def check_architectural_anomalies(self):
            """PREVENT ALGORITHMIC HALLUCINATIONS BY BLOCKING OUT-OF-DISTRIBUTION INPUTS."""
            
            # PREVENT DIVISION BY ZERO IF BEDROOMS = 0
            if self.bedrooms > 0:
                sqft_per_bed = self.sqft_living / self.bedrooms
                
                # IF A HOUSE HAS MORE THAN 1200 SQFT PER BEDROOM, IT IS HIGHLY ABNORMAL
                if sqft_per_bed > 2500:
                    raise ValueError(
                        f"Anomaly Detected: {sqft_per_bed:.0f} sqft per bedroom is statistically invalid for this model. "
                        "Please verify the square footage and bedroom count."
                    )
            return self

# 4. SERVER HEALTH CHECK ENDPOINT
@app.get("/health", tags=["Status"])
def server_health_check():
    return {
        "status": "online",
        "models_loaded": (classifier_pipeline is not None)
        and (regressor_model is not None),
        "llm_client_active": ai_client is not None,
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
        print(
            f"CRITICAL BREAK: classifier_pipeline was overwritten with string: '{classifier_pipeline}'"
        )
        raise HTTPException(
            status_code=500,
            detail=f"Internal State Error: Classifier is a string description: '{classifier_pipeline}'",
        )

    if classifier_pipeline is None or regressor_model is None:
        raise HTTPException(
            status_code=503,
            detail="Machine learning inference engines are currently offline.",
        )

    # CONVERT PAYLOAD CONTENT TO INPUT DICTIONARY FOR FEATURE INJECTION
    input_data = payload.model_dump()

    # CALCULATE LOG SPECS TO MATCH NOTEBOOK PIPELINE TRAINING PATTERNS
    input_data["log_sqft_living"] = np.log1p(input_data["sqft_living"])
    input_data["log_sqft_lot"] = np.log1p(input_data["sqft_lot"])
    input_data["log_sqft_above"] = np.log1p(input_data["sqft_above"])
    input_data["log_sqft_basement"] = np.log1p(input_data["sqft_basement"])

    mapped_city_encoding = CITY_ENCODING_MAP.get(payload.city, DEFAULT_CITY_WEIGHT)

    # --- STEP 1: PORTFOLIO CATEGORY CLASSIFICATION ---
    
    # ENFORCE STRICT SCIKIT-LEARN COLUMN ORDER DURING DICTIONARY INSTANTIATION
    classification_features = {
        "log_sqft_living": input_data["log_sqft_living"],
        "log_sqft_lot": input_data["log_sqft_lot"],
        "log_sqft_above": input_data["log_sqft_above"],
        "log_sqft_basement": input_data["log_sqft_basement"],
        "bedrooms": input_data["bedrooms"],
        "bathrooms": input_data["bathrooms"],
        "floors": input_data["floors"],
        "view": input_data["view"],
        "condition": input_data["condition"],
        "yr_built": input_data["yr_built"],
        "yr_renovated": input_data["yr_renovated"],
        "waterfront": input_data["waterfront"],
        "city_encoded": mapped_city_encoding
    }

    X_clf = pd.DataFrame([classification_features])

    # EXECUTE CLASSIFICATION PIPELINE
    predicted_cluster = int(classifier_pipeline.predict(X_clf)[0])

    cluster_mapping = {
        0: "Budget Portfolio",
        1: "Premium Portfolio",
        2: "Luxury Portfolio",
    }
    assigned_category = cluster_mapping.get(
        predicted_cluster, f"Cluster {predicted_cluster}"
    )

    # --- STEP 2: CONTINUOUS PRICE REGRESSION ---
    
    # ENFORCE STRICT COLUMN ORDER FOR REGRESSION ENGINE AS WELL
    # APPEND CLUSTER IDS AT THE END ASSUMING THAT WAS THE TRAINING ARCHITECTURE
    regression_features = {
    "log_sqft_living": input_data["log_sqft_living"],
    "log_sqft_lot": input_data["log_sqft_lot"],
    "log_sqft_above": input_data["log_sqft_above"],
    "log_sqft_basement": input_data["log_sqft_basement"],
    "bedrooms": input_data["bedrooms"],
    "bathrooms": input_data["bathrooms"],
    "floors": input_data["floors"],
    "view": input_data["view"],
    "condition": input_data["condition"],
    "yr_built": input_data["yr_built"],
    "yr_renovated": input_data["yr_renovated"],
    "waterfront": input_data["waterfront"],
    "cluster_id_1": 1 if predicted_cluster == 1 else 0,
    "cluster_id_2": 1 if predicted_cluster == 2 else 0,
    "city_encoded": mapped_city_encoding
}

    X_reg = pd.DataFrame([regression_features])

    # EXECUTE REGRESSION ENGINE
    predicted_log_price = regressor_model.predict(X_reg)[0]
    final_predicted_price = float(np.exp(predicted_log_price))
    
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
            - Size: {input_data["sqft_living"]} sqft living area ({input_data["sqft_lot"]} sqft lot)
            - Layout: {input_data["bedrooms"]} beds, {input_data["bathrooms"]} baths, {input_data["floors"]} floors
            - Context: Built in {input_data["yr_built"]}, Condition Rating: {input_data["condition"]}/5, View Quality: {input_data["view"]}/4

            Provide a tight 3-sentence summary:
            1. Analyze the core driver of this value (e.g., size vs vintage balance).
            2. Evaluate if this valuation looks realistic and justified from a market perspective.
            3. Synthesize the final investment trust outlook.
            """

            response = ai_client.chat.send(
                model="google/gemma-4-26b-a4b-it:free",
                messages=[
                    {"role": "system", "content": sys_instruction},
                    {"role": "user", "content": prompt_payload},
                ],
            )
            llm_interpretation = response.choices[0].message.content
        except Exception as api_err:
            llm_interpretation = (
                f"Live interpretation generation suspended: {str(api_err)}"
            )

    return {
        "classification": {
            "cluster_id": predicted_cluster,
            "portfolio_category": assigned_category,
        },
        "regression": {
            "predicted_price_raw": final_predicted_price,
            "predicted_price_formatted": f"${final_predicted_price:,.2f}",
        },
        "llm_interpretation": llm_interpretation,
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=9999)
