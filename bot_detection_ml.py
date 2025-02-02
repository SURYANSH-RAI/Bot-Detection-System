import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
import joblib
import json
import logging

class TextEncoder:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.max_length = 128

    def encode(self, texts):
        """Encode text using BERT embeddings"""
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        with torch.no_grad():
            outputs = self.model(**encodings)
            embeddings = outputs.last_hidden_state[:, 0, :].numpy()
        
        return embeddings

class BotDetectionModel(nn.Module):
    def __init__(self, input_dim):
        super(BotDetectionModel, self).__init__()
        
        # Neural network architecture
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)

class BotDetectionTrainer:
    def __init__(self, config_path=None):
        # Load configuration if provided
        self.config = {
            'random_seed': 42,
            'test_size': 0.2,
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 10,
            'early_stopping_patience': 3
        } if config_path is None else json.load(open(config_path))

        # Initialize components
        self.text_encoder = TextEncoder()
        self.scaler = StandardScaler()
        self.model = None
        
        # Set random seeds
        np.random.seed(self.config['random_seed'])
        torch.manual_seed(self.config['random_seed'])

    def extract_features(self, df):
        """Extract features from raw data"""
        features = {}
        
        # Text features
        if 'content' in df.columns:
            text_embeddings = self.text_encoder.encode(df['content'].tolist())
            for i in range(text_embeddings.shape[1]):
                features[f'text_embedding_{i}'] = text_embeddings[:, i]

        # Behavioral features
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            
            # Calculate posting frequency
            time_diffs = df.groupby('user_id')['timestamp'].diff().dt.total_seconds()
            features['avg_time_between_posts'] = time_diffs.groupby(df['user_id']).mean()
            features['std_time_between_posts'] = time_diffs.groupby(df['user_id']).std()

        # Engagement features
        for col in ['likes', 'shares', 'comments']:
            if col in df.columns:
                features[f'avg_{col}'] = df.groupby('user_id')[col].mean()
                features[f'std_{col}'] = df.groupby('user_id')[col].std()

        # Convert features to DataFrame
        feature_df = pd.DataFrame(features)
        return feature_df.fillna(0)

    def prepare_data(self, data_path):
        """Prepare data for training"""
        # Load and preprocess data
        df = pd.read_csv(data_path)
        
        # Extract features
        X = self.extract_features(df)
        y = df['is_bot'].astype(int)  # Target variable
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.config['test_size'],
            random_state=self.config['random_seed']
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test

    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Train the model"""
        input_dim = X_train.shape[1]
        self.model = BotDetectionModel(input_dim)
        
        # Loss function and optimizer
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config['learning_rate']
        )
        
        # Convert data to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train.values).reshape(-1, 1)
        
        if X_val is not None:
            X_val_tensor = torch.FloatTensor(X_val)
            y_val_tensor = torch.FloatTensor(y_val.values).reshape(-1, 1)
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config['epochs']):
            self.model.train()
            
            # Training step
            optimizer.zero_grad()
            outputs = self.model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()
            
            # Validation step
            if X_val is not None:
                self.model.eval()
                with torch.no_grad():
                    val_outputs = self.model(X_val_tensor)
                    val_loss = criterion(val_outputs, y_val_tensor)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= self.config['early_stopping_patience']:
                    print(f"Early stopping at epoch {epoch}")
                    break
                
                print(f"Epoch {epoch}: Train Loss = {loss.item():.4f}, Val Loss = {val_loss.item():.4f}")
            else:
                print(f"Epoch {epoch}: Train Loss = {loss.item():.4f}")

    def evaluate(self, X_test, y_test):
        """Evaluate the model"""
        self.model.eval()
        
        # Make predictions
        X_test_tensor = torch.FloatTensor(X_test)
        with torch.no_grad():
            y_pred = self.model(X_test_tensor).numpy()
        
        # Convert predictions to binary
        y_pred_binary = (y_pred > 0.5).astype(int)
        
        # Calculate metrics
        report = classification_report(y_test, y_pred_binary)
        conf_matrix = confusion_matrix(y_test, y_pred_binary)
        
        return {
            'classification_report': report,
            'confusion_matrix': conf_matrix,
            'predictions': y_pred
        }

    def save_model(self, path):
        """Save the trained model and preprocessing components"""
        model_state = {
            'model_state_dict': self.model.state_dict(),
            'scaler': self.scaler,
            'config': self.config
        }
        torch.save(model_state, path)

    def load_model(self, path):
        """Load a trained model"""
        model_state = torch.load(path, weights_only=False)
        
        # Load configuration
        self.config = model_state['config']
        
        # Initialize model
        input_dim = next(iter(model_state['model_state_dict'].values())).shape[1]
        self.model = BotDetectionModel(input_dim)
        self.model.load_state_dict(model_state['model_state_dict'])
        
        # Load scaler
        self.scaler = model_state['scaler']

    def predict(self, data):
        """Make predictions on new data"""
        # Extract features
        features = self.extract_features(data)
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Make predictions
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(torch.FloatTensor(features_scaled)).numpy()
        
        return predictions

# Example usage
if __name__ == "__main__":
    # Initialize trainer
    trainer = BotDetectionTrainer()
    
    # Prepare data
    X_train, X_test, y_train, y_test = trainer.prepare_data('bot_detection_data.csv')
    
    # Train model
    trainer.train(X_train, y_train, X_test, y_test)
    
    # Evaluate model
    results = trainer.evaluate(X_test, y_test)
    print("\nEvaluation Results:")
    print(results['classification_report'])
    
    # Save model
    trainer.save_model('bot_detection_model.pt')