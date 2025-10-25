from django.shortcuts import render
from django.conf import settings
import os
import numpy as np
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import TimeDistributed, LSTM, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2

MODEL_PATH = os.path.join(settings.BASE_DIR, 'best_video_model.h5')

def build_model():
    cnn_base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(128,128,3))
    cnn_base.trainable = False

    model = Sequential([
        TimeDistributed(cnn_base, input_shape=(40,128,128,3)),
        TimeDistributed(GlobalAveragePooling2D()),
        LSTM(128, return_sequences=False, dropout=0.5),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    return model

model = build_model()
model.load_weights(MODEL_PATH)

def preprocess_video(video_path, n_frames=40, img_size=(128,128)):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(total // n_frames, 1)
    frames = []

    for f in range(0, total, step):
        if len(frames) >= n_frames:
            break
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, img_size)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame / 255.0)

    cap.release()
    while len(frames) < n_frames and len(frames) > 0:
        frames.append(frames[-1])

    return np.array(frames)

def index(request):
    context = {}
    if request.method == 'POST' and 'video' in request.FILES:
        video_file = request.FILES['video']
        video_path = os.path.join(settings.MEDIA_ROOT, 'temp_video.mp4')

        with open(video_path, 'wb+') as f:
            for chunk in video_file.chunks():
                f.write(chunk)

        frames = preprocess_video(video_path)
        preds = model.predict(np.expand_dims(frames, axis=0))[0][0]

        context['result'] = "🚨 Shoplifting Detected" if preds > 0.5 else "✅ Normal Activity"

        os.remove(video_path)

    return render(request, 'detection/index.html', context)
