#!/usr/bin/env python3
"""
FastAPI Face Palsy Detection API with clinical heatmap
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from ultralytics import YOLO
import torch
import numpy as np
import cv2
from sklearn.preprocessing import StandardScaler
import joblib
import os
import tempfile
import uvicorn
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import base64
from io import BytesIO

app = FastAPI(title="Face Palsy Detection API", version="1.0.0")

class FrozenYOLOFeatureExtractor:
    def __init__(self, model_path='best.pt'):
        import torch
        original_load = torch.load
        torch.load = lambda *args, **kwargs: original_load(*args, **{**kwargs, 'weights_only': False})
        try:
            self.model = YOLO(model_path)
        finally:
            torch.load = original_load
        # Don't set to eval mode - keep trainable for Grad-CAM
        self.gradients = None
        self.activations = None
        
        # Register hooks for Grad-CAM
        target_layer = self._get_target_layer()
        if target_layer:
            target_layer.register_forward_hook(self._save_activation)
            target_layer.register_full_backward_hook(self._save_gradient)
    
    def _get_target_layer(self):
        """Get the last convolutional layer for Grad-CAM"""
        for module in reversed(list(self.model.model.model.modules())):
            if isinstance(module, torch.nn.Conv2d):
                return module
        return None
    
    def _save_activation(self, module, input, output):
        self.activations = output.detach().clone()
    
    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach().clone()
    
    def generate_gradcam(self, image_path):
        """Generate activation heatmap from YOLO detections"""
        img = cv2.imread(image_path)
        h, w = img.shape[:2]
        
        # Get predictions
        with torch.no_grad():
            results = self.model.predict(image_path, verbose=False, imgsz=640)
        
        # Create heatmap from detection confidence and locations
        heatmap = np.zeros((h, w), dtype=np.float32)
        
        if len(results[0].boxes) > 0:
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                
                # Convert to int
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Create smooth radial gradient for each detection
                center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                radius = max(x2 - x1, y2 - y1) * 1.5
                
                y_coords, x_coords = np.ogrid[:h, :w]
                dist = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
                
                # Gaussian-like falloff
                intensity = conf * np.exp(-dist**2 / (2 * (radius/2)**2))
                heatmap = np.maximum(heatmap, intensity)
            
            # Smooth for medical imaging look
            if heatmap.max() > 0:
                heatmap = cv2.GaussianBlur(heatmap, (101, 101), 0)
                heatmap = heatmap / heatmap.max()
        
        return heatmap
    
    def extract_features(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            return None
        
        # Ensure eval mode for inference
        self.model.model.eval()
        for param in self.model.model.parameters():
            param.requires_grad = False
        
        with torch.no_grad():
            results = self.model.predict(image_path, verbose=False)
        features = []
        if len(results[0].boxes) > 0:
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxyn[0].cpu().numpy()
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                box_area = (x2 - x1) * (y2 - y1)
                aspect_ratio = (x2 - x1) / (y2 - y1) if (y2 - y1) > 0 else 0
                features.extend([x1, y1, x2, y2, conf, cls_id, box_area, aspect_ratio])
        target_size = 48
        if len(features) < target_size:
            features.extend([0.0] * (target_size - len(features)))
        else:
            features = features[:target_size]
        return np.array(features)

class FrozenFeaturePipeline:
    def __init__(self):
        self.extractor = None
        self.mlp_model = None
        self.xgb_model = None
        self.scaler = None
    
    def load_models(self):
        self.extractor = FrozenYOLOFeatureExtractor()
        import numpy as np
        import warnings
        warnings.filterwarnings('ignore', category=UserWarning)
        
        if not hasattr(np.random, '_pickle'):
            np.random._pickle = type('_pickle', (), {})()
        
        try:
            import sklearn
            self.mlp_model = joblib.load('weighted_mlp_frozen.pkl')
            self.xgb_model = joblib.load('weighted_xgb_frozen.pkl')
            self.scaler = joblib.load('frozen_feature_scaler.pkl')
        except Exception as e:
            print(f'Joblib load failed: {e}, trying pickle...')
            import pickle
            with open('weighted_mlp_frozen.pkl', 'rb') as f:
                self.mlp_model = pickle.load(f)
            with open('weighted_xgb_frozen.pkl', 'rb') as f:
                self.xgb_model = pickle.load(f)
            with open('frozen_feature_scaler.pkl', 'rb') as f:
                self.scaler = pickle.load(f)
    
    def predict(self, image_path):
        # Preprocess image first
        processed_path = _preprocess_phone_image(image_path)
        
        with torch.no_grad():
            results = self.extractor.model.predict(processed_path, verbose=False)
        features = []
        detections = []
        if len(results[0].boxes) > 0:
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxyn[0].cpu().numpy()
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                box_area = (x2 - x1) * (y2 - y1)
                aspect_ratio = (x2 - x1) / (y2 - y1) if (y2 - y1) > 0 else 0
                features.extend([x1, y1, x2, y2, conf, cls_id, box_area, aspect_ratio])
                detections.append({
                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                    'confidence': conf,
                    'class_id': cls_id
                })
        target_size = 48
        if len(features) < target_size:
            features.extend([0.0] * (target_size - len(features)))
        else:
            features = features[:target_size]
        features = np.array(features).reshape(1, -1)
        features_scaled = self.scaler.transform(features)
        mlp_proba = self.mlp_model.predict_proba(features_scaled)[0][1]
        xgb_proba = self.xgb_model.predict_proba(features)[0][1]
        ensemble_proba = 0.6 * xgb_proba + 0.4 * mlp_proba
        
        # Clean up preprocessed file
        if processed_path != image_path and os.path.exists(processed_path):
            try:
                os.remove(processed_path)
            except:
                pass
        
        return {
            'mlp_score': mlp_proba,
            'xgb_score': xgb_proba,
            'ensemble_score': ensemble_proba,
            'prediction': 'palsy' if ensemble_proba > 0.5 else 'normal',
            'confidence': ensemble_proba if ensemble_proba > 0.5 else 1 - ensemble_proba,
            'detections': detections
        }

pipeline = None

def get_pipeline():
    global pipeline
    if pipeline is None:
        pipeline = FrozenFeaturePipeline()
        pipeline.load_models()
    return pipeline

def _preprocess_phone_image(image_path):
    """Preprocess phone camera images for better detection"""
    img = cv2.imread(image_path)
    if img is None:
        return image_path
    
    # Normalize lighting and enhance contrast
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # CLAHE for adaptive histogram equalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    
    enhanced = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    
    # Denoise while preserving edges
    denoised = cv2.fastNlMeansDenoisingColored(enhanced, None, 10, 10, 7, 21)
    
    # Sharpen for better feature detection
    kernel = np.array([[-1,-1,-1], [-1, 9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(denoised, -1, kernel)
    
    # Blend original and sharpened
    result = cv2.addWeighted(denoised, 0.7, sharpened, 0.3, 0)
    
    # Save preprocessed image
    preprocessed_path = image_path.replace('.', '_preprocessed.')
    cv2.imwrite(preprocessed_path, result)
    
    return preprocessed_path

def _create_clinical_heatmap(image_path, result):
    """Create medical-grade heatmap with clear patient visibility"""
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    
    # Generate heatmap
    pipeline_obj = get_pipeline()
    heatmap = pipeline_obj.extractor.generate_gradcam(image_path)
    
    # Medical-grade figure with better contrast
    fig = plt.figure(figsize=(10, 8), facecolor='#001a4d')
    ax = fig.add_subplot(111)
    ax.set_facecolor('#001a4d')
    
    # Show patient image with higher opacity for visibility
    ax.imshow(img_rgb, alpha=0.85, interpolation='lanczos')
    
    # Medical thermal colormap
    colors = ['#001a4d', '#0033cc', '#0066ff', '#00ccff', '#00ff99', 
              '#ffff00', '#ffcc00', '#ff6600', '#ff0000']
    cmap = LinearSegmentedColormap.from_list('thermal', colors, N=256)
    
    im = None
    if heatmap is not None and heatmap.max() > 0:
        # Lower heatmap opacity so patient is clearly visible
        im = ax.imshow(heatmap, cmap=cmap, alpha=0.5, vmin=0, vmax=1, 
                       interpolation='bicubic', aspect='auto')
    
    # Professional medical axes
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
    
    # Clear pixel coordinate scales
    tick_y = np.linspace(0, h, 9, dtype=int)
    tick_x = np.linspace(0, w, 9, dtype=int)
    
    ax.set_yticks(tick_y)
    ax.set_xticks(tick_x)
    ax.set_yticklabels(tick_y, fontsize=10, color='white', fontfamily='monospace', weight='bold')
    ax.set_xticklabels(tick_x, fontsize=10, color='white', fontfamily='monospace', weight='bold')
    ax.tick_params(colors='white', width=1.5, length=6)
    
    ax.set_xlabel('Pixel X', fontsize=11, color='white', weight='bold', labelpad=8)
    ax.set_ylabel('Pixel Y', fontsize=11, color='white', weight='bold', labelpad=8)
    
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    
    # Professional colorbar
    if im is not None:
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Activation Intensity', color='white', fontsize=11, weight='bold')
        cbar.ax.yaxis.set_tick_params(color='white', labelcolor='white', labelsize=9)
        cbar.outline.set_edgecolor('white')
        cbar.outline.set_linewidth(1.5)
    
    plt.tight_layout()
    
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=200, facecolor='#001a4d', 
                edgecolor='none', bbox_inches='tight')
    buffer.seek(0)
    heatmap_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close(fig)
    
    return heatmap_b64

def _create_interactive_html(heatmap_b64, result):
    """Generate interactive HTML with zoom functionality"""
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Medical Heatmap Viewer</title>
        <style>
            body {{
                margin: 0;
                padding: 20px;
                background: #001a4d;
                font-family: 'Arial', sans-serif;
                color: white;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
            }}
            .header {{
                text-align: center;
                margin-bottom: 20px;
            }}
            .viewer {{
                position: relative;
                overflow: auto;
                border: 2px solid white;
                background: #001a4d;
                max-height: 80vh;
            }}
            #heatmap {{
                display: block;
                cursor: zoom-in;
                transition: transform 0.2s;
                transform-origin: center;
            }}
            #heatmap.zoomed {{
                cursor: zoom-out;
            }}
            .controls {{
                margin-top: 15px;
                text-align: center;
            }}
            button {{
                background: #0066ff;
                color: white;
                border: none;
                padding: 10px 20px;
                margin: 5px;
                cursor: pointer;
                font-size: 14px;
                border-radius: 5px;
                font-weight: bold;
            }}
            button:hover {{
                background: #0088ff;
            }}
            .info {{
                margin-top: 15px;
                padding: 15px;
                background: rgba(0, 102, 255, 0.2);
                border-radius: 5px;
                border: 1px solid #0066ff;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🏥 Medical Thermal Heatmap Analysis</h1>
                <p>Prediction: <strong>{result['prediction'].upper()}</strong> | Confidence: <strong>{result['confidence']:.2%}</strong></p>
            </div>
            <div class="viewer" id="viewer">
                <img id="heatmap" src="data:image/png;base64,{heatmap_b64}" alt="Heatmap">
            </div>
            <div class="controls">
                <button onclick="zoomIn()">🔍 Zoom In</button>
                <button onclick="zoomOut()">🔍 Zoom Out</button>
                <button onclick="resetZoom()">↺ Reset</button>
                <button onclick="toggleFullscreen()">⛶ Fullscreen</button>
            </div>
            <div class="info">
                <strong>Instructions:</strong> Click image to zoom | Use buttons for precise control | Scroll to pan when zoomed
            </div>
        </div>
        <script>
            let scale = 1;
            const img = document.getElementById('heatmap');
            const viewer = document.getElementById('viewer');
            
            img.addEventListener('click', function() {{
                if (scale === 1) {{
                    scale = 2;
                }} else {{
                    scale = 1;
                }}
                img.style.transform = `scale(${{scale}})`;
                img.classList.toggle('zoomed', scale > 1);
            }});
            
            function zoomIn() {{
                scale = Math.min(scale + 0.5, 5);
                img.style.transform = `scale(${{scale}})`;
                img.classList.add('zoomed');
            }}
            
            function zoomOut() {{
                scale = Math.max(scale - 0.5, 1);
                img.style.transform = `scale(${{scale}})`;
                if (scale === 1) img.classList.remove('zoomed');
            }}
            
            function resetZoom() {{
                scale = 1;
                img.style.transform = 'scale(1)';
                img.classList.remove('zoomed');
                viewer.scrollTop = 0;
                viewer.scrollLeft = 0;
            }}
            
            function toggleFullscreen() {{
                if (!document.fullscreenElement) {{
                    viewer.requestFullscreen();
                }} else {{
                    document.exitFullscreen();
                }}
            }}
        </script>
    </body>
    </html>
    """
    return html

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    try:
        ext = '.jpg'
        if image.filename:
            ext = '.' + image.filename.split('.')[-1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp_file:
            content = await image.read()
            temp_file.write(content)
            temp_path = temp_file.name
        try:
            pipeline_obj = get_pipeline()
            result = pipeline_obj.predict(temp_path)
            heatmap_b64 = _create_clinical_heatmap(temp_path, result)
            interactive_html = _create_interactive_html(heatmap_b64, result)
            return {
                'prediction': result['prediction'],
                'confidence': float(result['confidence']),
                'mlp_score': float(result['mlp_score']),
                'xgb_score': float(result['xgb_score']),
                'ensemble_score': float(result['ensemble_score']),
                'heatmap': heatmap_b64,
                'interactive_viewer': interactive_html,
                'model_type': 'frozen_yolov8n_ensemble'
            }
        except Exception as e:
            print(f'Prediction error: {str(e)}')
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f'Prediction failed: {str(e)}')
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    except Exception as e:
        print(f'Request error: {str(e)}')
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    try:
        pipeline_status = get_pipeline()
        return {'status': 'healthy', 'model': 'frozen_ensemble', 'loaded': True}
    except Exception as e:
        return {'status': 'error', 'error': str(e), 'loaded': False}

@app.get("/")
async def home():
    return {
        'message': 'Face Palsy Detection API',
        'model': 'Frozen YOLOv8n Ensemble',
        'endpoint': '/predict (POST with image file)',
        'docs': '/docs'
    }

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 7860))
    uvicorn.run(app, host='0.0.0.0', port=port)
