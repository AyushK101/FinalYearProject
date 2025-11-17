from fastapi import FastAPI
import os
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from keras.models import load_model
from keras.preprocessing.image import img_to_array

model = load_model("model/deepfake_model.h5")

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_EXT = {"png", "jpg", "jpeg", "mp4", "avi", "mov"}


def allowed(filename: str):
    return filename.split(".")[-1].lower() in ALLOWED_EXT


def predict_image_file(path: str):
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Failed to read image at path: {path}")
    img = cv2.resize(img, (224, 224))
    img = img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    if model is None:
        raise FileNotFoundError(f"model load failed.")
    pred = model.predict(img)[0][0]
    return "Fake" if pred > 0.5 else "Real"


def predict_video_file(path: str):
    cap = cv2.VideoCapture(path)
    total, fake_frames = 0, 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        total += 1
        img = cv2.resize(frame, (224, 224))
        img = img_to_array(img) / 255.0
        img = np.expand_dims(img, axis=0)

        if model is None:
            raise FileNotFoundError(f"model load failed.")
        pred = model.predict(img)[0][0]
        if pred > 0.5:
            fake_frames += 1

    cap.release()
    percent = (fake_frames / total) * 100
    return percent


def live_generator():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img = cv2.resize(frame, (224, 224))
        img = img_to_array(img) / 255.0
        img = np.expand_dims(img, axis=0)

        if model is None:
            raise ValueError(f"model is undefined")
        pred = model.predict(img)[0][0]
        label = "Fake" if pred > 0.5 else "Real"

        cv2.putText(frame, label, (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 0, 255), 2)

        _, encoded = cv2.imencode(".jpg", frame)
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" +
               encoded.tobytes() + b"\r\n")

    cap.release()


app = FastAPI()


@app.post("/image")
async def analyze_image(file: UploadFile = File(...)):
    if file.filename is None or not allowed(file.filename):
        raise HTTPException(400, "Invalid video type")

    filepath = f"{UPLOAD_FOLDER}/{file.filename}"
    with open(filepath, "wb") as f:
        f.write(await file.read())

    result = predict_image_file(filepath)
    return {"filename": file.filename, "classification": result}


@app.post("/video")
async def analyze_video(file: UploadFile = File(...)):
    if file.filename is None or not allowed(file.filename):
        raise HTTPException(400, "Invalid video type")

    filepath = f"{UPLOAD_FOLDER}/{file.filename}"
    with open(filepath, "wb") as f:
        f.write(await file.read())

    percent_fake = predict_video_file(filepath)

    return {
        "filename": file.filename,
        "fake_percentage": round(percent_fake, 2)
    }


@app.get("/live")
async def live_video_stream():
    return StreamingResponse(live_generator(),
                             media_type="multipart/x-mixed-replace; boundary=frame")
