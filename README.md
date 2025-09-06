# Face Palsy Detection API

YOLOv8-based face palsy detection model with 6 classes:
- Normal_Eyes, Normal_Mouth
- SlightPalsy_Eyes, SlightPalsy_Mouth  
- StrongPalsy_Eyes, StrongPalsy_Mouth

## Performance
- mAP@0.5: 94.1%
- mAP@0.5-0.95: 91.5%

## API Usage
```bash
curl -X POST -F "image=@image.jpg" http://your-app.onrender.com/predict
```

## Deploy to Render
1. Push to GitHub
2. Connect to Render
3. Deploy as Web Service