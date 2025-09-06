from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import torch
from PIL import Image, ImageDraw, ImageFont
import io
import numpy as np
try:
    import cv2
except ImportError:
    cv2 = None
import base64

app = Flask(__name__)
CORS(app)

# Load model with ultralytics
from ultralytics import YOLO
import os
os.environ['TORCH_SERIALIZATION_SAFE_GLOBALS'] = 'True'
model = YOLO('runs/detect/train11/weights/best.pt')

class_names = ['Normal_Eyes', 'Normal_Mouth', 'SlightPalsy_Eyes', 'SlightPalsy_Mouth', 'StrongPalsy_Eyes', 'StrongPalsy_Mouth']

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return jsonify({
            'error': 'GET method not supported',
            'message': 'Use POST method with image file',
            'example': 'curl -X POST -F "image=@photo.jpg" https://stroke-detection-loa4.onrender.com/predict'
        }), 405
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        if not file.content_type.startswith('image/'):
            return jsonify({'error': 'File is not an image'}), 400

        image = Image.open(file.stream).convert("RGB")
        
        # Preprocess for better camera image detection
        img_array = np.array(image)
        # Normalize lighting
        img_array = cv2.convertScaleAbs(img_array, alpha=1.2, beta=10)
        image = Image.fromarray(img_array)
        
        results = model(image, conf=0.25)
        
        # Create annotated image (not true heatmap)
        img_array = np.array(image)
        annotated_img = img_array.copy()
        
        predictions = []
        for r in results:
            boxes = r.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].int().tolist()
                    conf = float(box.conf)
                    cls = int(box.cls)
                    class_name = model.names[cls]
                    
                    # Color based on severity
                    if 'Normal' in class_name:
                        color = (0, 255, 0)  # Green
                    elif 'Slight' in class_name:
                        color = (255, 165, 0)  # Orange
                    else:  # Strong
                        color = (255, 0, 0)  # Red
                    
                    # Draw rectangle and label
                    if cv2 is not None:
                        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)
                        label = f"{class_name}: {conf:.2f}"
                        cv2.putText(annotated_img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    predictions.append({
                        'class': class_name,
                        'confidence': conf,
                        'bbox': [x1, y1, x2, y2]
                    })
        
        # Convert to base64
        if cv2 is not None:
            _, buffer = cv2.imencode('.jpg', cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR))
            img_base64 = base64.b64encode(buffer).decode('utf-8')
        else:
            img_base64 = ""
        
        return jsonify({
            'predictions': predictions,
            'annotated_image': img_base64
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/')
def home():
    return jsonify({
        'message': 'Face Palsy Detection API',
        'endpoints': {
            'predict': 'POST /predict (upload image)',
            'health': 'GET /health'
        },
        'usage': 'curl -X POST -F "image=@image.jpg" /predict'
    })

@app.route('/health')
def health():
    return jsonify({'status': 'healthy'})

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'error': '404 Not Found',
        'message': 'Endpoint not found',
        'available_endpoints': {
            'home': 'GET /',
            'health': 'GET /health',
            'predict': 'POST /predict (with image file)'
        }
    }), 404

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    print(f"Starting Flask app on port {port}")
    print("Available routes:")
    for rule in app.url_map.iter_rules():
        print(f"  {rule.rule} [{', '.join(rule.methods)}]")
    app.run(host='0.0.0.0', port=port, debug=True)