from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import numpy as np
import cv2
import easyocr
import re
import os
import io
import asyncpg
from typing import Optional
import uuid
from datetime import datetime
import logging
from contextlib import asynccontextmanager
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database configuration
DATABASE_URL = f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"


# Model classes (same as in your notebook)
class MultiModalClassifier(nn.Module):
    def __init__(self, num_classes=4, ocr_feature_dim=100):
        super(MultiModalClassifier, self).__init__()

        self.backbone = models.efficientnet_b0(pretrained=True)
        self.backbone.classifier = nn.Identity()

        for param in list(self.backbone.parameters())[:-10]:
            param.requires_grad = False

        self.ocr_processor = nn.Sequential(
            nn.Linear(ocr_feature_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
        )

        cnn_feature_dim = 1280
        combined_dim = cnn_feature_dim + 32

        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, image, ocr_features):
        img_features = self.backbone(image)
        ocr_features = self.ocr_processor(ocr_features)
        combined = torch.cat([img_features, ocr_features], dim=1)
        output = self.classifier(combined)
        return output


class OCRProcessor:
    def __init__(self):
        self.reader = easyocr.Reader(["ru", "en"], gpu=torch.cuda.is_available())
        self.vectorizer = None
        self.is_fitted = False

    def extract_text(self, image_array):
        try:
            results = self.reader.readtext(image_array, detail=0)
            text = " ".join(results).lower()
            text = re.sub(r"[^\w\s©]", " ", text)
            text = re.sub(r"\s+", " ", text).strip()
            return text
        except:
            return ""

    def extract_features(self, texts):
        if not self.is_fitted:
            # Load pre-fitted vectorizer or create dummy features
            try:
                self.vectorizer = joblib.load("tfidf_vectorizer.pkl")
                self.is_fitted = True
            except:
                # Create dummy features if vectorizer not found
                logger.warning("TF-IDF vectorizer not found, using dummy features")
                return np.zeros((len(texts), 100))

        try:
            features = self.vectorizer.transform(texts).toarray()
            return features
        except:
            return np.zeros((len(texts), 100))


# Global variables
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None
ocr_processor = None
class_names = ["Cover", "Title Page", "Back Title Page", "Colophon"]

# Database connection pool
db_pool = None


async def init_db():
    """Initialize database connection pool"""
    global db_pool
    try:
        db_pool = await asyncpg.create_pool(DATABASE_URL)
        logger.info("Database pool created successfully")

        # Create table if not exists
        async with db_pool.acquire() as conn:
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS image_classifications (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    image_data BYTEA NOT NULL,
                    model_class INTEGER NOT NULL,
                    user_class INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )
        logger.info("Database table created/verified")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise


async def close_db():
    """Close database connection pool"""
    global db_pool
    if db_pool:
        await db_pool.close()
        logger.info("Database pool closed")


def load_model():
    """Load the trained model"""
    global model, ocr_processor
    try:
        model = MultiModalClassifier(num_classes=4, ocr_feature_dim=100).to(device)
        model.load_state_dict(
            torch.load("book_page_classifier.pth", map_location=device)
        )
        model.eval()

        ocr_processor = OCRProcessor()
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Model loading failed: {e}")
        raise


def preprocess_image(image_bytes):
    """Preprocess image for model prediction"""
    try:
        # Convert bytes to PIL Image
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Apply transformations
        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        image_tensor = transform(image).unsqueeze(0).to(device)

        # Convert to numpy array for OCR
        image_array = np.array(Image.open(io.BytesIO(image_bytes)))

        return image_tensor, image_array
    except Exception as e:
        logger.error(f"Image preprocessing failed: {e}")
        raise


async def predict_image_class(image_bytes):
    """Predict image class using the model"""
    try:
        # Preprocess image
        image_tensor, image_array = preprocess_image(image_bytes)

        # Extract OCR features
        text = ocr_processor.extract_text(image_array)
        ocr_features = ocr_processor.extract_features([text])
        ocr_tensor = torch.FloatTensor(ocr_features).to(device)

        # Make prediction
        with torch.no_grad():
            outputs = model(image_tensor, ocr_tensor)
            _, predicted = outputs.max(1)
            predicted_class = predicted.item()

        return predicted_class
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    # Startup
    logger.info("Starting up...")
    await init_db()
    load_model()
    yield
    # Shutdown
    logger.info("Shutting down...")
    await close_db()


# FastAPI app
app = FastAPI(
    title="Book Page Classification API",
    description="API for classifying book pages into Cover, Title Page, Back Title Page, and Colophon",
    version="1.0.0",
    lifespan=lifespan,
)
app.mount("/static", StaticFiles(directory="static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/predict", response_model=dict)
async def predict_page_class(file: UploadFile = File(...)):
    """
    Predict the class of a book page image

    Returns:
    - predicted_class: The predicted class index
    - class_name: The name of the predicted class
    - image_id: UUID of the stored image
    """
    try:
        # Validate file type
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")

        # Read image data
        image_data = await file.read()

        # Predict class
        predicted_class = await predict_image_class(image_data)

        # Store in database
        async with db_pool.acquire() as conn:
            image_id = await conn.fetchval(
                "INSERT INTO image_classifications (image_data, model_class) VALUES ($1, $2) RETURNING id",
                image_data,
                predicted_class,
            )

        return {
            "image_id": str(image_id),
            "predicted_class": predicted_class,
            "class_name": class_names[predicted_class],
            "message": "Image classified successfully",
        }

    except Exception as e:
        logger.error(f"Prediction endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/correct/{image_id}")
async def correct_classification(image_id: str, user_class: int = Form(...)):
    """
    Correct the classification of an image

    Args:
    - image_id: UUID of the image to correct
    - user_class: The correct class provided by the user (0-3)
    """
    try:
        # Validate class
        if user_class not in range(4):
            raise HTTPException(
                status_code=400, detail="user_class must be between 0 and 3"
            )

        # Update database
        async with db_pool.acquire() as conn:
            result = await conn.execute(
                "UPDATE image_classifications SET user_class = $1, updated_at = CURRENT_TIMESTAMP WHERE id = $2",
                user_class,
                uuid.UUID(image_id),
            )

            if result == "UPDATE 0":
                raise HTTPException(status_code=404, detail="Image not found")

        return {
            "image_id": image_id,
            "user_class": user_class,
            "class_name": class_names[user_class],
            "message": "Classification corrected successfully",
        }

    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid image_id format")
    except Exception as e:
        logger.error(f"Correction endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/classification/{image_id}")
async def get_classification(image_id: str):
    """
    Get classification information for a specific image
    """
    try:
        async with db_pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT id, model_class, user_class, created_at, updated_at FROM image_classifications WHERE id = $1",
                uuid.UUID(image_id),
            )

            if not row:
                raise HTTPException(status_code=404, detail="Image not found")

        return {
            "image_id": str(row["id"]),
            "model_class": row["model_class"],
            "model_class_name": class_names[row["model_class"]],
            "user_class": row["user_class"],
            "user_class_name": (
                class_names[row["user_class"]]
                if row["user_class"] is not None
                else None
            ),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid image_id format")
    except Exception as e:
        logger.error(f"Get classification endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/images")
async def get_all_images():
    async with db_pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT id, model_class, user_class FROM image_classifications"
        )
        return {
            "images": [
                {
                    "image_id": str(row["id"]),
                    "model_class": row["model_class"],
                    "model_class_name": class_names[row["model_class"]],
                    "user_class": row["user_class"],
                    "user_class_name": (
                        class_names[row["user_class"]]
                        if row["user_class"] is not None
                        else None
                    ),
                }
                for row in rows
            ]
        }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "device": str(device)}


@app.get("/stats")
async def get_stats():
    """Get classification statistics"""
    try:
        async with db_pool.acquire() as conn:
            stats = await conn.fetch(
                """
                SELECT 
                    model_class,
                    COUNT(*) as count,
                    COUNT(user_class) as corrected_count
                FROM image_classifications 
                GROUP BY model_class
                ORDER BY model_class
            """
            )

        result = []
        for row in stats:
            result.append(
                {
                    "class": row["model_class"],
                    "class_name": class_names[row["model_class"]],
                    "total_predictions": row["count"],
                    "user_corrections": row["corrected_count"],
                }
            )

        return {"statistics": result}

    except Exception as e:
        logger.error(f"Stats endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
