#!/usr/bin/env python3
"""
Optimized Facial Palsy Detection Training with Explainable AI
Pretrained YOLO + MediaPipe + Heatmaps + XGBoost+MLP Ensemble + XAI
Root Dataset + Augmented Dataset
"""

import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb
import mediapipe as mp
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# Import YOLO
from ultralytics import YOLO

# Import Explainable AI
from explainable_ai import ExplainableAI

print("🚀 Starting Optimized Facial Palsy Detection Training with XAI...")

class OptimizedFeatureExtractor:
    """Optimized feature extractor with pretrained YOLO + MediaPipe + Heatmaps"""
    
    def __init__(self):
        # Load standard YOLO model for face detection (not your palsy detection model)
        print("📷 Loading standard YOLO face detection model...")
        self.yolo_model = YOLO('yolov8n.pt')  # Use standard YOLO for face detection
        
        # Initialize MediaPipe for facial landmarks
        print("👤 Initializing MediaPipe face mesh...")
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # Define landmark indices for key facial features
        self.landmark_indices = {
            'left_eye': [33, 7, 163, 173, 157, 158, 159, 160, 161, 246],
            'right_eye': [362, 382, 381, 380, 374, 373, 390, 398, 384, 385],
            'left_eyebrow': [70, 63, 105, 66, 107, 55, 65, 52, 53, 46],
            'right_eyebrow': [296, 334, 293, 300, 276, 283, 282, 295, 285, 336],
            'nose': [1, 2, 5, 4, 6, 19, 20, 94, 125, 141],
            'mouth': [13, 14, 78, 80, 81, 82, 87, 88, 95, 308],
            'mouth_corners': [61, 291],
            'face_outline': [10, 338, 297, 332, 284, 251, 389, 356, 454, 323]
        }
    
    def detect_face_with_yolo(self, image):
        """Detect face using standard YOLO for person detection"""
        try:
            # Run YOLO detection for person detection
            results = self.yolo_model(image, verbose=False)
            
            # Get person detections (class 0 is person in COCO)
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Filter for person class (0) with reasonable confidence
                        if box.cls == 0 and box.conf > 0.2:  # Low threshold for person detection
                            # Get bounding box coordinates
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            
                            # Add padding to ensure full face is captured
                            padding = 30
                            x1 = max(0, int(x1 - padding))
                            y1 = max(0, int(y1 - padding))
                            x2 = int(x2 + padding)
                            y2 = int(y2 + padding)
                            
                            return x1, y1, x2, y2
            
            return None
        except Exception as e:
            print(f"❌ YOLO detection error: {e}")
            return None
    
    def extract_features_with_heatmap(self, image_path):
        """Extract features and generate heatmap using frozen YOLO backbone"""
        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                return None, None, None
            
            # Use frozen YOLO backbone for face detection
            face_bbox = self.detect_face_with_yolo(image)
            if face_bbox is None:
                print(f"   ⚠️ No face detected by YOLO in {os.path.basename(image_path)}")
                return None, None, None
            
            x1, y1, x2, y2 = face_bbox
            print(f"   📷 YOLO detected face in {os.path.basename(image_path)}: ({x1},{y1}) to ({x2},{y2})")
            
            # Crop face region using YOLO detection
            face_region = image[y1:y2, x1:x2]
            
            # Convert to RGB for MediaPipe
            face_rgb = cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe on YOLO-cropped face
            results = self.face_mesh.process(face_rgb)
            
            if not results.multi_face_landmarks:
                print(f"   ⚠️ MediaPipe failed on YOLO-cropped face {os.path.basename(image_path)}")
                return None, None, None
            
            landmarks = results.multi_face_landmarks[0]
            
            # Extract features from YOLO-cropped face
            features = self._calculate_asymmetry_features(landmarks, face_region.shape)
            
            # Generate heatmap for YOLO-cropped face
            heatmap = self._generate_asymmetry_heatmap(landmarks, face_region.shape)
            
            return features, face_bbox, heatmap
            
        except Exception as e:
            print(f"   ❌ Failed to extract features from {image_path}: {e}")
            return None, None, None
    
    def _calculate_asymmetry_features(self, landmarks, face_shape):
        """Calculate optimized asymmetry features"""
        try:
            h, w = face_shape[:2]
            points = []
            
            # Convert normalized landmarks to pixel coordinates
            for landmark in landmarks.landmark:
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                points.append([x, y])
            
            points = np.array(points)
            
            # Calculate 10 optimized features
            features = []
            
            # 1. Eye horizontal asymmetry
            left_eye_center = np.mean(points[self.landmark_indices['left_eye']], axis=0)
            right_eye_center = np.mean(points[self.landmark_indices['right_eye']], axis=0)
            eye_distance = np.linalg.norm(left_eye_center - right_eye_center)
            eye_asymmetry = abs(left_eye_center[0] - right_eye_center[0]) / eye_distance if eye_distance > 0 else 0
            features.append(eye_asymmetry)
            
            # 2. Eyebrow vertical asymmetry
            left_eyebrow_center = np.mean(points[self.landmark_indices['left_eyebrow']], axis=0)
            right_eyebrow_center = np.mean(points[self.landmark_indices['right_eyebrow']], axis=0)
            eyebrow_asymmetry = abs(left_eyebrow_center[1] - right_eyebrow_center[1]) / h
            features.append(eyebrow_asymmetry)
            
            # 3. Mouth corner asymmetry
            left_mouth_corner = points[self.landmark_indices['mouth_corners'][0]]
            right_mouth_corner = points[self.landmark_indices['mouth_corners'][1]]
            mouth_asymmetry = abs(left_mouth_corner[1] - right_mouth_corner[1]) / h
            features.append(mouth_asymmetry)
            
            # 4. Nose-mouth ratio
            nose_tip = points[self.landmark_indices['nose'][0]]
            mouth_center = np.mean(points[self.landmark_indices['mouth']], axis=0)
            nose_mouth_dist = np.linalg.norm(nose_tip - mouth_center)
            features.append(nose_mouth_dist)
            
            # 5. Eye-nose-mouth ratio
            eye_center = (left_eye_center + right_eye_center) / 2
            eye_nose_dist = np.linalg.norm(eye_center - nose_tip)
            ratio = eye_nose_dist / nose_mouth_dist if nose_mouth_dist > 0 else 0
            features.append(ratio)
            
            # 6. Landmark standard deviation
            landmark_std = np.std(points)
            features.append(landmark_std)
            
            # 7. Landmark range
            landmark_range = np.max(points) - np.min(points)
            features.append(landmark_range)
            
            # 8. Landmark mean
            landmark_mean = np.mean(points)
            features.append(landmark_mean)
            
            # 9. Eye closure asymmetry
            left_eye_height = np.mean(points[159:163]) - np.mean(points[145:149])
            right_eye_height = np.mean(points[386:390]) - np.mean(points[374:378])
            eye_closure_asym = abs(left_eye_height - right_eye_height)
            features.append(eye_closure_asym)
            
            # 10. Total asymmetry score
            total_asymmetry = eye_asymmetry + mouth_asymmetry
            features.append(total_asymmetry)
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            print(f"❌ Feature calculation error: {e}")
            return None
    
    def _generate_asymmetry_heatmap(self, landmarks, face_shape):
        """Generate asymmetry heatmap visualization"""
        try:
            h, w = face_shape[:2]
            
            # Create blank heatmap
            heatmap = np.zeros((h, w), dtype=np.float32)
            
            # Convert landmarks to pixel coordinates
            points = []
            for landmark in landmarks.landmark:
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                points.append([x, y])
            
            points = np.array(points)
            
            # Calculate asymmetry for each landmark
            for i, point in enumerate(points):
                # Mirror point across vertical center
                mirror_x = w - point[0]
                
                # Find closest landmark to mirror position
                distances = np.sqrt((points[:, 0] - mirror_x)**2 + (points[:, 1] - point[1])**2)
                closest_idx = np.argmin(distances)
                
                if distances[closest_idx] < w * 0.1:  # Only if mirror point exists
                    mirror_point = points[closest_idx]
                    
                    # Calculate asymmetry at this point
                    asymmetry = np.linalg.norm(point - mirror_point) / w
                    
                    # Add to heatmap with Gaussian blur
                    cv2.circle(heatmap, (int(point[0]), int(point[1])), 5, asymmetry, -1)
            
            # Apply Gaussian blur for smooth heatmap
            heatmap = cv2.GaussianBlur(heatmap, (15, 15), 0)
            
            # Normalize to 0-255
            heatmap = np.clip(heatmap * 255, 0, 255).astype(np.uint8)
            
            # Apply colormap
            heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            
            return heatmap_colored
            
        except Exception as e:
            print(f"❌ Heatmap generation error: {e}")
            return None

class OptimizedEnsembleTrainer:
    """Optimized ensemble trainer for Root + Aug datasets"""
    
    def __init__(self):
        self.feature_extractor = OptimizedFeatureExtractor()
        self.scaler = StandardScaler()
        self.mlp_model = None
        self.xgb_model = None
        self.ensemble_weights = None
        
    def load_frames_dataset(self, frames_dir):
        """Load frames dataset (affected + unaffected + aug + train + valid + test)"""
        print(f"📁 Loading frames dataset from {frames_dir}")
        
        image_files = []
        labels = []
        subjects = []
        
        # Load affected (palsy) from frames
        self._load_frames_class(frames_dir, "affected", image_files, labels, subjects, 1)
        
        # Load unaffected (normal) from frames
        self._load_frames_class(frames_dir, "unaffected", image_files, labels, subjects, 0)
        
        # Load from aug directory
        aug_dir = "../aug"
        if os.path.exists(aug_dir):
            print(f"   Also loading from {aug_dir}")
            self._load_augmented_dataset(aug_dir, image_files, labels, subjects)
        
        # Load from train directory
        train_dir = "../train"
        if os.path.exists(train_dir):
            print(f"   Also loading from {train_dir}")
            self._load_yolo_dataset(train_dir, image_files, labels, subjects)
        
        # Load from valid directory
        valid_dir = "../valid"
        if os.path.exists(valid_dir):
            print(f"   Also loading from {valid_dir}")
            self._load_yolo_dataset(valid_dir, image_files, labels, subjects)
        
        # Load from test directory
        test_dir = "../test"
        if os.path.exists(test_dir):
            print(f"   Also loading from {test_dir}")
            self._load_yolo_dataset(test_dir, image_files, labels, subjects)
        
        print(f"📊 Dataset loaded: {len(image_files)} images")
        print(f"   Palsy cases: {sum(labels)}")
        print(f"   Normal cases: {len(labels) - sum(labels)}")
        print(f"   Unique subjects: {len(set(subjects))}")
        
        # If still no data, use a sample from frames/affected/1
        if len(image_files) == 0:
            print("   🚨 No data found, using sample from frames/affected/1")
            self._load_sample_data(image_files, labels, subjects)
        
        return image_files, np.array(labels), np.array(subjects)
    
    def _load_sample_data(self, image_files, labels, subjects):
        """Load sample data from frames/affected/1 for testing"""
        # Try different path formats
        sample_dirs = [
            os.path.join("../frames", "affected", "1"),
            os.path.join("C:", "Users", "adars", "Downloads", "face-palsy.v2i.yolov8", "frames", "affected", "1"),
            os.path.join("frames", "affected", "1"),
        ]
        
        for sample_dir in sample_dirs:
            if os.path.exists(sample_dir):
                print(f"   Loading sample data from {sample_dir}")
                
                count = 0
                for img_file in os.listdir(sample_dir):
                    if img_file.lower().endswith(('.jpg', '.jpeg', '.png')) and count < 50:  # Limit to 50 images
                        img_path = os.path.join(sample_dir, img_file)
                        image_files.append(img_path)
                        labels.append(1)  # Palsy
                        subjects.append(f"sample_1_{count}")
                        count += 1
                
                print(f"   ✅ Loaded {count} sample palsy images")
                break
        
        # Also load some normal samples
        normal_dirs = [
            os.path.join("../frames", "unaffected", "1"),
            os.path.join("C:", "Users", "adars", "Downloads", "face-palsy.v2i.yolov8", "frames", "unaffected", "1"),
            os.path.join("frames", "unaffected", "1"),
        ]
        
        for normal_dir in normal_dirs:
            if os.path.exists(normal_dir):
                print(f"   Loading sample data from {normal_dir}")
                
                count = 0
                for img_file in os.listdir(normal_dir):
                    if img_file.lower().endswith(('.jpg', '.jpeg', '.png')) and count < 50:  # Limit to 50 images
                        img_path = os.path.join(normal_dir, img_file)
                        image_files.append(img_path)
                        labels.append(0)  # Normal
                        subjects.append(f"sample_normal_1_{count}")
                        count += 1
                
                print(f"   ✅ Loaded {count} sample normal images")
                break
    
    def _load_frames_class(self, frames_dir, class_name, image_files, labels, subjects, class_label):
        """Load specific class from frames directory"""
        class_dir = os.path.join(frames_dir, class_name)
        if not os.path.exists(class_dir):
            print(f"   ⚠️ {class_name} directory not found: {class_dir}")
            return
        
        for subject_dir in os.listdir(class_dir):
            subject_path = os.path.join(class_dir, subject_dir)
            if os.path.isdir(subject_path):
                for img_file in os.listdir(subject_path):
                    if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        img_path = os.path.join(subject_path, img_file)
                        image_files.append(img_path)
                        labels.append(class_label)
                        subjects.append(f"frames_{class_name}_{subject_dir}")
        
        print(f"   ✅ Loaded {len([s for s in subjects if s.startswith(f'frames_{class_name}_')])} {class_name} images")
    
    def _load_yolo_dataset(self, yolo_dir, image_files, labels, subjects):
        """Load YOLO format dataset (images + labels)"""
        images_dir = os.path.join(yolo_dir, "images")
        labels_dir = os.path.join(yolo_dir, "labels")
        
        if not os.path.exists(images_dir):
            print(f"   ⚠️ Images directory not found: {images_dir}")
            return
        
        # Load images and match with labels
        for img_file in os.listdir(images_dir):
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(images_dir, img_file)
                
                # Try to find corresponding label file
                label_file = os.path.splitext(img_file)[0] + '.txt'
                label_path = os.path.join(labels_dir, label_file)
                
                # Default to palsy (1) for YOLO datasets
                class_label = 1
                subject_id = os.path.splitext(img_file)[0]
                
                image_files.append(img_path)
                labels.append(class_label)
                subjects.append(f"yolo_{subject_id}")
        
        print(f"   ✅ Loaded {len([s for s in subjects if s.startswith('yolo_')])} YOLO images")
    
    def _load_dataset_folder(self, data_dir, image_files, labels, subjects, dataset_type):
        """Load images from a dataset folder"""
        for class_name in ["affected", "unaffected"]:
            class_path = os.path.join(data_dir, class_name)
            if not os.path.exists(class_path):
                continue
                
            class_label = 1 if class_name == "affected" else 0
            
            for subject_dir in os.listdir(class_path):
                subject_path = os.path.join(class_path, subject_dir)
                if os.path.isdir(subject_path):
                    for img_file in os.listdir(subject_path):
                        if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                            img_path = os.path.join(subject_path, img_file)
                            image_files.append(img_path)
                            labels.append(class_label)
                            subjects.append(f"{dataset_type}_{subject_dir}")
    
    def extract_all_features(self, image_files):
        """Extract features from all images"""
        print("🔍 Extracting features from all images...")
        
        features = []
        valid_indices = []
        heatmaps = []
        
        for i, img_path in enumerate(image_files):
            if i % 100 == 0:
                print(f"   Processing {i}/{len(image_files)}...")
            
            feature_vector, bbox, heatmap = self.feature_extractor.extract_features_with_heatmap(img_path)
            
            if feature_vector is not None:
                features.append(feature_vector)
                valid_indices.append(i)
                heatmaps.append(heatmap)
            else:
                print(f"   ❌ Failed to extract features from {img_path}")
        
        features = np.array(features)
        print(f"✅ Successfully extracted features from {len(features)}/{len(image_files)} images")
        
        return features, valid_indices, heatmaps
    
    def train_optimized_ensemble(self, X_train, y_train, X_val, y_val):
        """Train optimized MLP + XGBoost ensemble"""
        print("🤖 Training optimized MLP + XGBoost ensemble...")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Train optimized MLP
        print("   Training optimized MLP...")
        self.mlp_model = MLPClassifier(
            hidden_layer_sizes=(64, 32),  # Smaller for generalization
            activation='relu',
            solver='adam',
            alpha=0.01,  # Higher regularization
            batch_size=16,
            learning_rate_init=0.001,
            max_iter=500,  # Fewer iterations
            random_state=42,
            early_stopping=True,
            validation_fraction=0.2
        )
        self.mlp_model.fit(X_train_scaled, y_train)
        
        # Train optimized XGBoost
        print("   Training optimized XGBoost...")
        self.xgb_model = xgb.XGBClassifier(
            n_estimators=100,  # Fewer trees
            max_depth=4,  # Shallower trees
            learning_rate=0.05,  # Lower learning rate
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='logloss',
            reg_alpha=0.5,  # Higher regularization
            reg_lambda=2.0
        )
        self.xgb_model.fit(X_train_scaled, y_train)
        
        # Get validation predictions
        mlp_proba = self.mlp_model.predict_proba(X_val_scaled)[:, 1]
        xgb_proba = self.xgb_model.predict_proba(X_val_scaled)[:, 1]
        
        # Optimize ensemble weights
        print("   Optimizing ensemble weights...")
        best_score = 0
        best_weights = [0.6, 0.4]  # Favor XGBoost
        
        for mlp_weight in np.arange(0.3, 0.7, 0.05):
            xgb_weight = 1.0 - mlp_weight
            ensemble_proba = mlp_weight * mlp_proba + xgb_weight * xgb_proba
            
            # Calculate AUC score
            auc_score = roc_auc_score(y_val, ensemble_proba)
            
            if auc_score > best_score:
                best_score = auc_score
                best_weights = [mlp_weight, xgb_weight]
        
        self.ensemble_weights = best_weights
        print(f"✅ Best ensemble weights: MLP={best_weights[0]:.2f}, XGBoost={best_weights[1]:.2f}")
        print(f"✅ Validation AUC: {best_score:.4f}")
        
        return best_score
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate ensemble performance"""
        print("📊 Evaluating model performance...")
        
        X_test_scaled = self.scaler.transform(X_test)
        
        # Get individual predictions
        mlp_proba = self.mlp_model.predict_proba(X_test_scaled)[:, 1]
        xgb_proba = self.xgb_model.predict_proba(X_test_scaled)[:, 1]
        
        # Ensemble prediction
        ensemble_proba = (self.ensemble_weights[0] * mlp_proba + 
                         self.ensemble_weights[1] * xgb_proba)
        ensemble_pred = (ensemble_proba > 0.5).astype(int)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, ensemble_pred)
        precision = precision_score(y_test, ensemble_pred)
        recall = recall_score(y_test, ensemble_pred)
        f1 = f1_score(y_test, ensemble_pred)
        auc = roc_auc_score(y_test, ensemble_proba)
        
        print(f"📈 Final Results:")
        print(f"   Accuracy: {accuracy:.4f}")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall: {recall:.4f}")
        print(f"   F1-Score: {f1:.4f}")
        print(f"   AUC: {auc:.4f}")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc
        }
    
    def save_models(self, output_dir):
        """Save trained models"""
        os.makedirs(output_dir, exist_ok=True)
        
        joblib.dump(self.scaler, os.path.join(output_dir, 'optimized_scaler.pkl'))
        joblib.dump(self.mlp_model, os.path.join(output_dir, 'optimized_mlp.pkl'))
        joblib.dump(self.xgb_model, os.path.join(output_dir, 'optimized_xgb.pkl'))
        joblib.dump(self.ensemble_weights, os.path.join(output_dir, 'optimized_ensemble_weights.pkl'))
        
        print(f"💾 Models saved to {output_dir}")
    
    def initialize_xai(self, X_background):
        """Initialize Explainable AI system"""
        print("🧠 Initializing Explainable AI system...")
        
        self.xai = ExplainableAI(
            self.feature_extractor,
            self.scaler,
            self.mlp_model,
            self.xgb_model,
            self.ensemble_weights
        )
        
        # Initialize explainers
        self.xai.initialize_explainers(X_background)
        
        print("✅ Explainable AI system ready")
    
    def generate_explanations(self, sample_images, save_dir="./xai_explanations"):
        """Generate explanations for sample images"""
        print("🔍 Generating explainable AI analysis...")
        
        os.makedirs(save_dir, exist_ok=True)
        
        for i, img_path in enumerate(sample_images[:10]):  # Explain first 10 images
            print(f"   Explaining {i+1}/10: {os.path.basename(img_path)}")
            
            explanations = self.xai.explain_prediction(img_path, save_dir)
            
            if explanations:
                # Generate summary report
                report_path = os.path.join(save_dir, f"explanation_{i+1}_summary.txt")
                self.xai.generate_summary_report(explanations, report_path)
        
        print(f"✅ Explanations generated and saved to {save_dir}")

def main():
    """Main training function"""
    print("🚀 Starting Optimized Facial Palsy Detection Training")
    
    # Initialize trainer
    trainer = OptimizedEnsembleTrainer()
    
    # Load combined dataset (frames affected + frames unaffected + aug + train + valid + test)
    frames_dir = "../frames"  # Use frames directory directly
    image_files, labels, subjects = trainer.load_frames_dataset(frames_dir)
    
    # Extract features with heatmaps
    features, valid_indices, heatmaps = trainer.extract_all_features(image_files)
    labels = labels[valid_indices]
    subjects = subjects[valid_indices]
    
    # Subject-wise cross-validation
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    
    fold_results = []
    for fold, (train_idx, val_idx) in enumerate(sgkf.split(features, labels, subjects)):
        print(f"\n🔄 Fold {fold + 1}/5")
        
        X_train, X_val = features[train_idx], features[val_idx]
        y_train, y_val = labels[train_idx], labels[val_idx]
        
        # Train ensemble
        auc_score = trainer.train_optimized_ensemble(X_train, y_train, X_val, y_val)
        
        # Evaluate
        metrics = trainer.evaluate_model(X_val, y_val)
        fold_results.append(metrics)
    
    # Print overall results
    print("\n🎯 Overall Results:")
    avg_metrics = {
        'accuracy': np.mean([r['accuracy'] for r in fold_results]),
        'precision': np.mean([r['precision'] for r in fold_results]),
        'recall': np.mean([r['recall'] for r in fold_results]),
        'f1': np.mean([r['f1'] for r in fold_results]),
        'auc': np.mean([r['auc'] for r in fold_results])
    }
    
    for metric, value in avg_metrics.items():
        print(f"   {metric.capitalize()}: {value:.4f} ± {np.std([r[metric] for r in fold_results]):.4f}")
    
    # Train final model on all data
    print("\n🎓 Training final model on all data...")
    trainer.train_optimized_ensemble(features, labels, features, labels)
    
    # Initialize Explainable AI
    trainer.initialize_xai(features)
    
    # Generate explanations for sample images
    sample_images = image_files[:20]  # Explain first 20 images
    trainer.generate_explanations(sample_images)
    
    # Save models
    trainer.save_models("./optimized_trained_models")
    
    print("🎉 Optimized training with XAI completed successfully!")

if __name__ == "__main__":
    main()
