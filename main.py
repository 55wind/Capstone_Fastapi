# main.py (FastAPI)
from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
import tensorflow as tf
import io

app = FastAPI()

classes = [
    '종이(paper)', '유리(glass)', '캔(can)', '배터리(battery)', '플라스틱(plastic)',
    '의류(clothes)', '일반쓰레기(trash)', '음식물 쓰레기(food organic)',
    '비닐(vinyl)', '스티로폼(styrofoam)'
]

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
        category = classes[class_id]

        return JSONResponse(content={
            "category": category,
            "guide": f"{confidence:.2f}"
        })

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)