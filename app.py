import sys
import os
from typing import Optional
from datetime import datetime
from contextlib import asynccontextmanager
import certifi
from dotenv import load_dotenv
import pymongo
import pandas as pd
from pathlib import Path

from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from uvicorn import run as app_run
from fastapi.responses import Response, JSONResponse, FileResponse
from starlette.responses import RedirectResponse
from fastapi.templating import Jinja2Templates

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.pipeline.training_pipeline import TrainingPipeline
from networksecurity.utils.main_utils.utils import load_object
from networksecurity.constants.training_pipeline import DATA_INGESTION_COLLECTION_NAME
from networksecurity.constants.training_pipeline import DATA_INGESTION_DATABASE_NAME
from networksecurity.utils.ml_utils.model.estimator import NetworkModel

# Load environment variables
load_dotenv()

# MongoDB Configuration
ca = certifi.where()
mongo_db_uri = os.getenv("MONGODB_URI")
if not mongo_db_uri:
    raise ValueError("MONGODB_URI not found in environment variables")

client = pymongo.MongoClient(mongo_db_uri, tlsCAFile=ca)
database = client[DATA_INGESTION_DATABASE_NAME]
collection = database[DATA_INGESTION_COLLECTION_NAME]

# Model paths configuration
MODEL_DIR = os.getenv("MODEL_DIR", "final_model")
OUTPUT_DIR = "prediction_output"
Path(OUTPUT_DIR).mkdir(exist_ok=True)

# Global model cache
model_cache = {"preprocessor": None, "model": None}


def load_models():
    """Load models with caching to avoid repeated I/O"""
    try:
        if model_cache["preprocessor"] is None:
            model_cache["preprocessor"] = load_object(f"{MODEL_DIR}/preprocessor.pkl")
            logging.info("Preprocessor loaded successfully")
        
        if model_cache["model"] is None:
            model_cache["model"] = load_object(f"{MODEL_DIR}/model.pkl")
            logging.info("Model loaded successfully")
        
        return model_cache["preprocessor"], model_cache["model"]
    except Exception as e:
        logging.error(f"Error loading models: {str(e)}")
        raise HTTPException(status_code=500, detail="Models not found. Please train the model first.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan event handler for startup and shutdown events
    """
    # Startup: Preload models
    try:
        logging.info("Application starting up...")
        if os.path.exists(f"{MODEL_DIR}/preprocessor.pkl") and os.path.exists(f"{MODEL_DIR}/model.pkl"):
            load_models()
            logging.info("Models preloaded successfully")
        else:
            logging.warning("Models not found. Please train the model before making predictions.")
    except Exception as e:
        logging.warning(f"Could not preload models: {str(e)}")
    
    yield  # Application runs here
    
    # Shutdown: Cleanup
    logging.info("Application shutting down...")
    client.close()
    logging.info("MongoDB connection closed")


# FastAPI App with lifespan
app = FastAPI(
    title="Network Security ML API",
    description="API for training and prediction of network security threats",
    version="1.0.0",
    lifespan=lifespan
)


# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Templates and static files
templates = Jinja2Templates(directory='./templates')

# Mount static files for serving output CSV
try:
    app.mount("/prediction_output", StaticFiles(directory="prediction_output"), name="prediction_output")
except Exception:
    pass  # Directory might not exist yet


@app.get("/", tags=["Root"])
async def index():
    """Serve main dashboard page"""
    return FileResponse('templates/index.html')


@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint"""
    try:
        # Check MongoDB connection
        client.admin.command('ping')
        
        # Check if models exist
        preprocessor_exists = os.path.exists(f"{MODEL_DIR}/preprocessor.pkl")
        model_exists = os.path.exists(f"{MODEL_DIR}/model.pkl")
        
        return {
            "status": "healthy",
            "mongodb": "connected",
            "models_available": preprocessor_exists and model_exists
        }
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "error": str(e)}
        )


@app.get("/train", tags=["Training"])
async def train_route():
    """Trigger model training pipeline"""
    try:
        logging.info("Training pipeline started")
        train_pipeline = TrainingPipeline()
        train_pipeline.run_pipeline()
        
        # Clear model cache to force reload of new models
        model_cache["preprocessor"] = None
        model_cache["model"] = None
        
        logging.info("Training pipeline completed successfully")
        return {"message": "Training completed successfully"}
    except Exception as e:
        logging.error(f"Training failed: {str(e)}")
        raise NetworkSecurityException(e, sys)


@app.post("/predict", tags=["Prediction"])
async def predict_route(request: Request, file: UploadFile = File(...)):
    """
    Make predictions on uploaded CSV file and return HTML table
    
    - **file**: CSV file containing features for prediction
    """
    try:
        # Validate file type
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Only CSV files are supported")
        
        # Read CSV with size limit (e.g., 50MB)
        contents = await file.read()
        if len(contents) > 50 * 1024 * 1024:  # 50MB limit
            raise HTTPException(status_code=400, detail="File size exceeds 50MB limit")
        
        # Parse CSV
        from io import StringIO
        df = pd.read_csv(StringIO(contents.decode('utf-8')))
        
        # Validate data
        if df.empty:
            raise HTTPException(status_code=400, detail="CSV file is empty")
        
        logging.info(f"Received file with {len(df)} rows")
        
        # Load models
        preprocessor, model = load_models()
        network_model = NetworkModel(preprocessor=preprocessor, model=model)
        
        # Make predictions
        y_pred = network_model.predict(df)
        df['predicted_column'] = y_pred
        
        # Convert -1 to 0 if needed (for user-friendly display)
        # Uncomment the line below if you want to convert -1 to 0
        # df['predicted_column'] = df['predicted_column'].replace(-1, 0)
        
        # Save output
        output_path = f"{OUTPUT_DIR}/output.csv"
        df.to_csv(output_path, index=False)
        logging.info(f"Predictions saved to {output_path}")
        
        # Return HTML table
        table_html = df.to_html(classes='table table-striped', index=False)
        return templates.TemplateResponse(
            "table.html", 
            {
                "request": request, 
                "table": table_html, 
                "rows": len(df),
                "now": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        )
        
    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="CSV file is empty or invalid")
    except pd.errors.ParserError:
        raise HTTPException(status_code=400, detail="Invalid CSV format")
    except Exception as e:
        logging.error(f"Prediction failed: {str(e)}")
        raise NetworkSecurityException(e, sys)


@app.post("/predict/json", tags=["Prediction"])
async def predict_json_route(file: UploadFile = File(...)):
    """
    Make predictions and return JSON response
    
    - **file**: CSV file containing features for prediction
    """
    try:
        # Validate file
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Only CSV files are supported")
        
        # Read and parse CSV
        contents = await file.read()
        if len(contents) > 50 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File size exceeds 50MB limit")
        
        from io import StringIO
        df = pd.read_csv(StringIO(contents.decode('utf-8')))
        
        if df.empty:
            raise HTTPException(status_code=400, detail="CSV file is empty")
        
        # Load models and predict
        preprocessor, model = load_models()
        network_model = NetworkModel(preprocessor=preprocessor, model=model)
        y_pred = network_model.predict(df)
        
        # Prepare response
        df['predicted_column'] = y_pred
        
        # Convert -1 to 0 if needed
        # Uncomment the line below if you want to convert -1 to 0
        # df['predicted_column'] = df['predicted_column'].replace(-1, 0)
        
        return {
            "predictions": df['predicted_column'].tolist(),
            "total_records": len(df),
            "summary": {
                "positive_predictions": int((df['predicted_column'] == 1).sum()),
                "negative_predictions": int((df['predicted_column'] == -1).sum() + (df['predicted_column'] == 0).sum())
            }
        }
        
    except Exception as e:
        logging.error(f"Prediction failed: {str(e)}")
        raise NetworkSecurityException(e, sys)


if __name__ == "__main__":
    app_run(app, host="0.0.0.0", port=8000)