from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse
from utils.preprocess import preprocess_image
import boto3

app = FastAPI()

# ✅ SageMaker 설정
SAGEMAKER_ENDPOINT = "garbage-endpoint"  # 사용자가 생성한 엔드포인트 이름
AWS_REGION = "ap-southeast-2"            # 시드니 리전
runtime = boto3.client("sagemaker-runtime", region_name=AWS_REGION)

@app.post("/predict")
async def predict(file: UploadFile, name: str = Form(...)):
    try:
        print(f"📥 요청: 사용자={name}, 파일={file.filename}")
        raw_bytes = await file.read()

        # ✅ 이미지 전처리
        payload = preprocess_image(raw_bytes)

        # ✅ SageMaker 호출
        response = runtime.invoke_endpoint(
            EndpointName=SAGEMAKER_ENDPOINT,
            ContentType="application/json",
            Body=str(payload).encode("utf-8")
        )

        result = response['Body'].read().decode("utf-8")
        print("✅ 예측 결과:", result)

        return JSONResponse(content={
            "nickname": name,
            "result": result
        })
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)