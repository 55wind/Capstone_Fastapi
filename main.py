from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, HTMLResponse
from PIL import Image
import numpy as np
import tensorflow as tf
import io
import json

app = FastAPI()

# 모델 로드
model = tf.keras.models.load_model("garbage_classification_test_model.keras")

# 클래스 인덱스 로드 (index → class name 매핑용)
with open("class_indices.json", "r") as f:
    class_indices = json.load(f)
# 인덱스를 기준으로 정렬된 클래스 이름 리스트 생성
classes = [None] * len(class_indices)
for label, index in class_indices.items():
    classes[index] = label

# 최근 예측 결과 저장용 전역 변수
last_result = {
    "category": "없음",
    "guide": "없음"
}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    global last_result
    try:
        # 이미지 로드
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = image.resize((224, 224))

        # 밝기 검사
        img_mean = np.array(image).mean()
        if img_mean < 30 or img_mean > 230:
            return JSONResponse(
                content={"error": "이미지가 너무 어둡거나 밝습니다. 다시 촬영해주세요."},
                status_code=400
            )

        # 전처리
        image_array = np.array(image).astype('float32') / 255.0
        image_array = np.expand_dims(image_array, axis=0)

        # 예측
        predictions = model.predict(image_array)
        class_id = int(np.argmax(predictions))
        confidence = float(np.max(predictions))
        category = classes[class_id]

        # 결과 저장
        last_result["category"] = category
        last_result["guide"] = f"{confidence:.2f}"

        return JSONResponse(content={
            "category": category,
            "confidence": f"{confidence:.2f}"
        })

    except Exception as e:
        return JSONResponse(content={"error": f"예측 중 오류 발생: {str(e)}"}, status_code=500)

@app.get("/", response_class=HTMLResponse)
async def root():
    return f"""
    <html>
        <head><title>최근 예측 결과</title></head>
        <body>
            <h2>🗂 최근 분류 결과</h2>
            <p><strong>분류:</strong> {last_result["category"]}</p>
            <p><strong>정확도:</strong> {last_result["guide"]}</p>
        </body>
    </html>
    """
