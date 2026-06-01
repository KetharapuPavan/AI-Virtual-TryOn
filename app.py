from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
import cv2
import mediapipe as mp
import numpy as np
import os

app = FastAPI(title="AI Virtual Try-On")

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

@app.get("/")
def home():
    return {"message": "AI Virtual Try-On API Running"}

@app.post("/tryon")
async def tryon(file: UploadFile = File(...)):

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)

    with open(file_path, "wb") as f:
        f.write(await file.read())

    image = cv2.imread(file_path)

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = pose.process(rgb)

    if results.pose_landmarks:

        mp.solutions.drawing_utils.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

    output_path = os.path.join(
        OUTPUT_FOLDER,
        "processed_" + file.filename
    )

    cv2.imwrite(output_path, image)

    return FileResponse(output_path)
