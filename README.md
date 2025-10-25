# 🛍️ Shoplifting Detection from Surveillance Videos

This project focuses on detecting **shoplifting incidents** in retail shop videos using deep learning.  
Two different models were implemented and compared:

1. **CNN + LSTM model (from scratch)**
2. **MobileNetV2 + LSTM model (transfer learning)**

---

## 🎥 Dataset

- **Name:** `Shop DataSet`
- **Classes:**
  - `shop lifters` → theft activity (label = 1)
  - `non shop lifters` → normal activity (label = 0)
- **Total videos:** 855  
  - 324 shop lifters  
  - 531 non-shop lifters  

Each video is processed into **40 frames** to capture temporal motion features.

---

## 🧩 Model 1: From Scratch (CNN + LSTM)

**Input shape:** `(40, 64, 64, 3)`  
**Architecture:**
- TimeDistributed Conv2D blocks (32 → 64 → 128 filters)
- Batch Normalization + MaxPooling
- Flatten → LSTM(128)
- Dense layers with Dropout (0.4)
- Output: Sigmoid

**Optimizer:** Adam (lr = 5e-5)  
**Loss:** Binary Crossentropy  
**Epochs:** 10  
**Batch size:** 2  

**Results:**
| Metric | Value |
|--------|--------|
| Training Accuracy | 99.8% |
| Validation Accuracy | 100% |
| Validation Loss | 0.04 |

> ⚠️ Very high accuracy due to full in-memory training — potential overfitting or data overlap.

---

## 🧠 Model 2: Pretrained (MobileNetV2 + LSTM)

**Input shape:** `(40, 128, 128, 3)`  
**Base model:** MobileNetV2 (ImageNet weights, frozen)  
**Architecture:**
- `TimeDistributed(MobileNetV2)`
- `GlobalAveragePooling2D`
- `LSTM(128, dropout=0.5)`
- `Dense(64, relu)` → `Dropout(0.5)` → `Dense(1, sigmoid)`

**Training setup:**
- Data loaded via generator (`video_generator`)
- Real-time augmentation using Albumentations
- Saved best checkpoint: `best_video_model.h5`

**Results (after 50 epochs):**
| Metric | Value |
|--------|--------|
| Training Accuracy | ~90.8% |
| Validation Accuracy | ~84.4% |
| Validation Loss | ~0.36 |

✅ Much better generalization and robustness than the scratch model.

---

## 🌐 Deployment with Django

The best-performing **MobileNetV2 + LSTM** model was **successfully deployed using Django**.  
A simple and intuitive **web interface** was built, allowing users to:

- Upload surveillance videos directly from the browser.  
- Automatically process and predict whether a shoplifting incident occurred.  
- Display the result instantly on the webpage.  

The Django app integrates the trained deep learning model and runs real-time inference,  
making the system practical for real-world retail monitoring.
