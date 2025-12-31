"""
LSTM Model Module
PyTorch implementation based on paper specifications.
"""
import torch
import torch.nn as nn
import logging
import os

logger = logging.getLogger(__name__)

class LSTMModel(nn.Module):
    """
    LSTM Classifier for Stock Prediction.
    Architecture:
    - LSTM Layer (Stackable)
    - Fully Connected Output Layer (to 3 classes)
    """
    def __init__(self, input_size=5, hidden_size=8, num_layers=2, num_classes=3, dropout=0.0):
        super(LSTMModel, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # Output layer
        # Paper implies simple dense layer on top of last time step
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        # x shape: (Batch, Window, Features)
        
        # LSTM output: (Batch, Window, Hidden)
        # Hidden/Cell states: (Layers, Batch, Hidden)
        out, _ = self.lstm(x)
        
        # Take the output of the last time step
        # Shape: (Batch, Hidden)
        last_out = out[:, -1, :]
        
        # Classification head
        logits = self.fc(last_out)
        
        return logits

class ModelWrapper:
    """Wrapper for handling Training/Saving/Loading of the PyTorch model."""
    
    def __init__(self, config):
        self.config = config.ml
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')
            
        self.model = LSTMModel(
            input_size=self.config.input_size,
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_layers,
            num_classes=self.config.num_classes,
            dropout=self.config.dropout
        ).to(self.device)
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        
        logger.info(f"Initialized LSTM Model on {self.device}: H={self.config.hidden_size}, L={self.config.num_layers}")

    def train_one_epoch(self, train_loader, progress_callback=None):
        """Train for one epoch with optional progress callback."""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        num_batches = len(train_loader)
        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(X_batch)
            loss = self.criterion(outputs, y_batch)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()

            # Report progress if callback provided
            if progress_callback and batch_idx % 10 == 0:  # Update every 10 batches
                current_loss = total_loss / (batch_idx + 1)
                current_acc = correct / total if total > 0 else 0
                progress_callback(batch_idx + 1, num_batches, current_loss, current_acc)

        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total
        return avg_loss, accuracy

    def evaluate(self, loader, progress_callback=None):
        """Evaluate the model with optional progress callback."""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        num_batches = len(loader)
        with torch.no_grad():
            for batch_idx, (X_batch, y_batch) in enumerate(loader):
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)

                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()

                # Report progress if callback provided
                if progress_callback and batch_idx % 5 == 0:  # Update every 5 batches
                    current_loss = total_loss / (batch_idx + 1)
                    current_acc = correct / total if total > 0 else 0
                    progress_callback(batch_idx + 1, num_batches, current_loss, current_acc)

        return total_loss / len(loader), correct / total

    def predict(self, X_tensor):
        """Make predictions."""
        self.model.eval()
        with torch.no_grad():
            X_tensor = X_tensor.to(self.device)
            outputs = self.model(X_tensor)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
        return predicted.cpu().numpy(), probs.cpu().numpy()

    def save_checkpoint(self, path, epoch, metrics):
        """Save full checkpoint including optimizer state."""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'metrics': metrics
        }, path)
        
    def load_checkpoint(self, path):
        """Load checkpoint and return epoch/metrics."""
        if not os.path.exists(path):
            logger.warning(f"Checkpoint not found: {path}")
            return None, {}
            
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
        epoch = checkpoint.get('epoch', 0)
        metrics = checkpoint.get('metrics', {})
        
        logger.info(f"Loaded checkpoint from epoch {epoch}")
        return epoch, metrics
