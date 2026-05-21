#!/usr/bin/env python3
"""
Face Palsy Detection Training Script
EfficientNet-B3 Transfer Learning Approach
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Try to import EfficientNet
try:
    from efficientnet_pytorch import EfficientNet
except ImportError:
    print("Installing efficientnet_pytorch...")
    os.system("pip install efficientnet_pytorch")
    from efficientnet_pytorch import EfficientNet

class PalsyDataset(Dataset):
    """Custom Dataset for Face Palsy Detection"""
    
    def __init__(self, data_path, split='train'):
        self.data_path = Path(data_path)
        self.split = split
        
        # Load metadata
        metadata_path = self.data_path / 'preprocessed_metadata.json'
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Filter images by split
        self.images = [img for img in metadata['images'] if img['split'] == split]
        
        # Create label mapping
        self.label_map = {'normal': 0, 'palsy': 1}
        
        print(f"Loaded {len(self.images)} {split} samples")
        print(f"Class distribution: {self.get_class_distribution()}")
    
    def get_class_distribution(self):
        """Get class distribution"""
        counts = {'normal': 0, 'palsy': 0}
        for img in self.images:
            counts[img['class']] += 1
        return counts
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_info = self.images[idx]
        
        # Load preprocessed tensor
        img_path = self.data_path / img_info['split'] / img_info['class'] / img_info['filename']
        img_array = np.load(img_path)
        
        # Convert to tensor and fix dimensions (HWC -> CHW)
        if img_array.ndim == 3:
            img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float()
        else:
            img_tensor = torch.from_numpy(img_array).float().unsqueeze(0)
        
        # Get label
        label = self.label_map[img_info['class']]
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return img_tensor, label_tensor

class LabelSmoothingCrossEntropy(nn.Module):
    """Label Smoothing Cross Entropy Loss"""
    
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
        
    def forward(self, pred, target):
        pred = pred.log_softmax(dim=-1)
        true_dist = torch.zeros_like(pred)
        true_dist.fill_(self.smoothing / (pred.size(1) - 1))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=-1))

class PalsyTrainer:
    """Training class for Face Palsy Detection"""
    
    def __init__(self, data_path="c:\\Users\\adars\\Downloads\\face-palsy.v2i.yolov8\\preprocessed_dataset"):
        self.data_path = Path(data_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Create datasets
        self.train_dataset = PalsyDataset(self.data_path, 'train')
        self.val_dataset = PalsyDataset(self.data_path, 'val')
        self.test_dataset = PalsyDataset(self.data_path, 'test')
        
        # Create data loaders
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=32, shuffle=True, num_workers=0
        )
        self.val_loader = DataLoader(
            self.val_dataset, batch_size=32, shuffle=False, num_workers=0
        )
        self.test_loader = DataLoader(
            self.test_dataset, batch_size=32, shuffle=False, num_workers=0
        )
        
        # Initialize model
        self.model = self.create_model()
        self.criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=0.01)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=50, eta_min=1e-6
        )
        
        # Training history
        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'lr': []
        }
        
        # Early stopping
        self.best_val_acc = 0.0
        self.patience = 10
        self.patience_counter = 0
        
        # Create output directory
        self.output_dir = Path("training_results")
        self.output_dir.mkdir(exist_ok=True)
    
    def create_model(self):
        """Create EfficientNet-B3 model"""
        model = EfficientNet.from_pretrained('efficientnet-b3', num_classes=2)
        
        # Phase 1: Freeze backbone, only train classifier
        for param in model.parameters():
            param.requires_grad = False
        
        # Unfreeze only the final classifier layer
        for param in model._fc.parameters():
            param.requires_grad = True
        
        model = model.to(self.device)
        return model
    
    def unfreeze_top_layers(self):
        """Unfreeze top layers for fine-tuning"""
        print("Unfreezing top layers for fine-tuning...")
        
        # Unfreeze the last few blocks
        for param in model._blocks[-3:].parameters():
            param.requires_grad = True
        
        # Unfreeze the classifier
        for param in model._fc.parameters():
            param.requires_grad = True
        
        # Reduce learning rate for fine-tuning
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = 1e-5
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1} [Train]')
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            # Update progress bar
            acc = 100. * correct / total
            pbar.set_postfix({
                'Loss': f'{running_loss/(batch_idx+1):.4f}',
                'Acc': f'{acc:.2f}%'
            })
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        self.history['train_loss'].append(epoch_loss)
        self.history['train_acc'].append(epoch_acc)
        
        return epoch_loss, epoch_acc
    
    def validate_epoch(self, epoch):
        """Validate for one epoch"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Epoch {epoch+1} [Val]')
            for batch_idx, (data, target) in enumerate(pbar):
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                loss = self.criterion(output, target)
                
                running_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
                
                # Update progress bar
                acc = 100. * correct / total
                pbar.set_postfix({
                    'Loss': f'{running_loss/(batch_idx+1):.4f}',
                    'Acc': f'{acc:.2f}%'
                })
        
        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = 100. * correct / total
        
        self.history['val_loss'].append(epoch_loss)
        self.history['val_acc'].append(epoch_acc)
        self.history['lr'].append(self.optimizer.param_groups[0]['lr'])
        
        return epoch_loss, epoch_acc
    
    def save_checkpoint(self, epoch, val_acc, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_acc': val_acc,
            'history': self.history
        }
        
        # Save latest checkpoint
        torch.save(checkpoint, self.output_dir / 'latest_checkpoint.pth')
        
        # Save best model
        if is_best:
            torch.save(checkpoint, self.output_dir / 'best_model.pth')
            print(f"🎉 New best model saved with validation accuracy: {val_acc:.2f}%")
    
    def plot_training_history(self):
        """Plot training history"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss plot
        axes[0, 0].plot(self.history['train_loss'], label='Train Loss')
        axes[0, 0].plot(self.history['val_loss'], label='Val Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy plot
        axes[0, 1].plot(self.history['train_acc'], label='Train Acc')
        axes[0, 1].plot(self.history['val_acc'], label='Val Acc')
        axes[0, 1].set_title('Training and Validation Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Learning rate plot
        axes[1, 0].plot(self.history['lr'])
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].grid(True)
        
        # Remove empty subplot
        axes[1, 1].remove()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'training_history.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def evaluate_model(self, loader, split_name='Test'):
        """Evaluate model on dataset"""
        self.model.eval()
        all_preds = []
        all_targets = []
        all_probs = []
        
        with torch.no_grad():
            for data, target in tqdm(loader, desc=f'Evaluating {split_name}'):
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                probs = torch.softmax(output, dim=1)[:, 1]  # Probability of positive class
                _, predicted = output.max(1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        # Calculate metrics
        accuracy = 100. * np.mean(np.array(all_preds) == np.array(all_targets))
        
        # Classification report
        class_names = ['Normal', 'Palsy']
        report = classification_report(all_targets, all_preds, target_names=class_names, output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(all_targets, all_preds)
        
        # AUC-ROC
        auc_score = roc_auc_score(all_targets, all_probs)
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title(f'{split_name} Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(self.output_dir / f'{split_name.lower()}_confusion_matrix.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm,
            'auc_score': auc_score,
            'predictions': all_preds,
            'targets': all_targets,
            'probabilities': all_probs
        }
    
    def train(self, epochs=50, fine_tune_epoch=20):
        """Main training loop"""
        print("🚀 Starting Face Palsy Detection Training")
        print(f"📊 Dataset: {len(self.train_dataset)} train, {len(self.val_dataset)} val, {len(self.test_dataset)} test")
        print(f"🎯 Target: {epochs} epochs, fine-tuning from epoch {fine_tune_epoch}")
        
        for epoch in range(epochs):
            print(f"\n{'='*50}")
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"{'='*50}")
            
            # Fine-tuning phase
            if epoch == fine_tune_epoch:
                self.unfreeze_top_layers()
            
            # Train
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_acc = self.validate_epoch(epoch)
            
            # Learning rate scheduling
            self.scheduler.step()
            
            # Print epoch summary
            print(f"\nEpoch {epoch+1} Summary:")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            print(f"  Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Save checkpoint
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            self.save_checkpoint(epoch, val_acc, is_best)
            
            # Early stopping
            if self.patience_counter >= self.patience:
                print(f"\n⏰ Early stopping triggered after {epoch+1} epochs")
                print(f"Best validation accuracy: {self.best_val_acc:.2f}%")
                break
        
        # Plot training history
        self.plot_training_history()
        
        # Load best model for final evaluation
        print("\n🔍 Loading best model for final evaluation...")
        checkpoint = torch.load(self.output_dir / 'best_model.pth')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Final evaluation
        print("\n📊 Final Evaluation on Test Set:")
        test_results = self.evaluate_model(self.test_loader, 'Test')
        
        print(f"Test Accuracy: {test_results['accuracy']:.2f}%")
        print(f"AUC-ROC Score: {test_results['auc_score']:.4f}")
        print("\nClassification Report:")
        for class_name, metrics in test_results['classification_report'].items():
            if isinstance(metrics, dict):
                print(f"  {class_name:10s}: P={metrics['precision']:.3f}, R={metrics['recall']:.3f}, F1={metrics['f1-score']:.3f}")
        
        # Save final results
        results = {
            'best_val_acc': self.best_val_acc,
            'test_results': {
                'accuracy': test_results['accuracy'],
                'auc_score': test_results['auc_score'],
                'classification_report': test_results['classification_report'],
                'confusion_matrix': test_results['confusion_matrix'].tolist()
            },
            'training_history': self.history,
            'model_info': {
                'architecture': 'EfficientNet-B3',
                'pretrained': True,
                'num_classes': 2,
                'input_size': (224, 224, 3)
            }
        }
        
        with open(self.output_dir / 'final_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✅ Training completed successfully!")
        print(f"📁 Results saved to: {self.output_dir}")
        print(f"🏆 Best validation accuracy: {self.best_val_acc:.2f}%")
        print(f"🎯 Test accuracy: {test_results['accuracy']:.2f}%")
        
        return results

if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create trainer
    trainer = PalsyTrainer()
    
    # Start training
    results = trainer.train(
        epochs=50,           # Total training epochs
        fine_tune_epoch=20    # Start fine-tuning at epoch 20
    )
