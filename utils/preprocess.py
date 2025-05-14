from PIL import Image
import numpy as np
import io

def preprocess_image(file) -> list:
    image = Image.open(io.BytesIO(file)).convert("RGB")
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    img_array = img_array.astype(np.float32)
    img_array = np.expand_dims(img_array, axis=0)  # (1, 224, 224, 3)
    return img_array.tolist()