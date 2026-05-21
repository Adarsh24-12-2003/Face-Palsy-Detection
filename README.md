---
title: Stroke Detection
emoji: 🧠
colorFrom: blue
colorTo: red
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: false
---

# Face Palsy Detection API

FastAPI server for face palsy detection with YOLOv8 and heatmap visualization.

## Endpoints

- `GET /` - Health check
- `POST /predict` - Upload image for prediction

## Response Format

```json
{
  "predictions": [
    {"class": "SlightPalsy_Eyes", "confidence": 0.85}
  ],
  "severity": "Mild",
  "gradcam_image": "base64_encoded_heatmap"
}
```
