import os
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import shutil

app = FastAPI()

# Load Models
disease_model = YOLO('./dis_model.pt')


def predict_is_abnormal(img_path, conf_threshold=0.25):
    results = disease_model.predict(source=img_path, conf=conf_threshold)
    for result in results:
        classes = result.boxes.cls if result.boxes else []
        for cls in classes:
            if int(cls) == 1:
                return False  # Class 1 detected
    return True  # No class 1 found

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    file_location = f"temp_{file.filename}"
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        if predict_is_abnormal(file_location):
            result = "Abnormal Foot"
        else:
            result = "Normal Foot"
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        os.remove(file_location)

    return {"result": result}
