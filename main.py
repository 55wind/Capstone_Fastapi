# main.py (FastAPI)
from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
import tensorflow as tf
import io

app = FastAPI()

# ✅ 로컬에서 .keras 모델 로드
model = tf.keras.models.load_model("garbage_classification_test_model.keras")

@app.post("/predict")
async def predict(file: UploadFile = Form(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = image.resize((224, 224))
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)

        predictions = model.predict(image_array)
        class_id = int(np.argmax(predictions))
        confidence = float(np.max(predictions))

        return JSONResponse(content={
            "category": str(class_id),  # 결과는 Spring에서 string으로 받아야 안전
            "guide": confidence
        })

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)