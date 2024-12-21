import tensorflow as tf
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

@dataclass
class TransformerConfig:
    d_model: int = 512
    num_heads: int = 8
    num_layers: int = 6
    d_ff: int = 2048
    dropout_rate: float = 0.1
    max_seq_length: int = 1000
    vocab_size: int = 10000

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.d_model = config.d_model
        
        assert self.d_model % self.num_heads == 0
        
        self.depth = self.d_model // self.num_heads
        
        self.query_dense = tf.keras.layers.Dense(config.d_model)
        self.key_dense = tf.keras.layers.Dense(config.d_model)
        self.value_dense = tf.keras.layers.Dense(config.d_model)
        
        self.dense = tf.keras.layers.Dense(config.d_model)
        
    def split_heads(self, inputs, batch_size):
        inputs = tf.reshape(
            inputs, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(inputs, perm=[0, 2, 1, 3])
        
    def call(self, inputs):
        query, key, value, mask = inputs['query'], inputs['key'], inputs['value'], inputs['mask']
        batch_size = tf.shape(query)[0]
        
        # Linear layers
        query = self.query_dense(query)
        key = self.key_dense(key)
        value = self.value_dense(value)
        
        # Split heads
        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)
        
        # Scaled dot-product attention
        scaled_attention = tf.matmul(query, key, transpose_b=True)
        scaled_attention = scaled_attention / tf.math.sqrt(tf.cast(self.depth, tf.float32))
        
        if mask is not None:
            scaled_attention += (mask * -1e9)
        
        attention_weights = tf.nn.softmax(scaled_attention, axis=-1)
        output = tf.matmul(attention_weights, value)
        
        # Reshape and concatenate heads
        output = tf.transpose(output, perm=[0, 2, 1, 3])
        output = tf.reshape(output, (batch_size, -1, self.d_model))
        
        return self.dense(output)

class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(config.d_ff, activation='relu'),
            tf.keras.layers.Dense(config.d_model)
        ])
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(config.dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(config.dropout_rate)
        
    def call(self, inputs, training, mask=None):
        attn_output = self.attention({
            'query': inputs,
            'key': inputs,
            'value': inputs,
            'mask': mask
        })
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.d_model = config.d_model
        self.max_seq_length = config.max_seq_length
        
        # Create positional encoding matrix
        position = np.arange(self.max_seq_length)[:, np.newaxis]
        div_term = np.exp(np.arange(0, self.d_model, 2) * -(np.log(10000.0) / self.d_model))
        
        pe = np.zeros((self.max_seq_length, self.d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        
        self.pe = tf.constant(pe, dtype=tf.float32)[tf.newaxis, ...]
        
    def call(self, inputs):
        return inputs + self.pe[:, :tf.shape(inputs)[1], :]

class MarketTransformer(tf.keras.Model):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        
        self.embedding = tf.keras.layers.Dense(config.d_model)
        self.pos_encoding = PositionalEncoding(config)
        
        self.transformer_blocks = [
            TransformerBlock(config) for _ in range(config.num_layers)
        ]
        
        self.dropout = tf.keras.layers.Dropout(config.dropout_rate)
        self.final_layer = tf.keras.layers.Dense(1)
        
    def call(self, inputs, training=True):
        # inputs shape: (batch_size, seq_length, features)
        x = self.embedding(inputs)
        x = self.pos_encoding(x)
        x = self.dropout(x, training=training)
        
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, training=training)
            
        # Global average pooling
        x = tf.reduce_mean(x, axis=1)
        
        return self.final_layer(x)

class PricePredictor:
    def __init__(self, config: Optional[TransformerConfig] = None):
        if config is None:
            config = TransformerConfig()
        self.config = config
        self.model = MarketTransformer(config)
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=tf.keras.optimizers.schedules.CosineDecayRestarts(
                initial_learning_rate=1e-4,
                first_decay_steps=1000
            )
        )
        
    def prepare_data(self, 
                    price_data: np.ndarray,
                    window_size: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """Prépare les données pour l'entraînement avec fenêtre glissante."""
        X, y = [], []
        for i in range(len(price_data) - window_size):
            X.append(price_data[i:(i + window_size)])
            y.append(price_data[i + window_size])
        return np.array(X), np.array(y)
    
    @tf.function
    def train_step(self, x, y):
        with tf.GradientTape() as tape:
            predictions = self.model(x, training=True)
            loss = tf.keras.losses.mean_squared_error(y, predictions)
            
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        return loss
    
    def train(self, 
              train_data: np.ndarray,
              validation_data: np.ndarray,
              epochs: int = 100,
              batch_size: int = 32):
        """Entraîne le modèle sur les données historiques."""
        X_train, y_train = self.prepare_data(train_data)
        X_val, y_val = self.prepare_data(validation_data)
        
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))\
            .shuffle(10000)\
            .batch(batch_size)
        
        val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))\
            .batch(batch_size)
        
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training
            train_losses = []
            for x_batch, y_batch in train_dataset:
                loss = self.train_step(x_batch, y_batch)
                train_losses.append(loss)
            
            # Validation
            val_losses = []
            for x_batch, y_batch in val_dataset:
                predictions = self.model(x_batch, training=False)
                val_loss = tf.keras.losses.mean_squared_error(y_batch, predictions)
                val_losses.append(val_loss)
            
            avg_train_loss = tf.reduce_mean(train_losses)
            avg_val_loss = tf.reduce_mean(val_losses)
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
            
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"Train Loss: {avg_train_loss:.4f}")
            print(f"Val Loss: {avg_val_loss:.4f}")
    
    def predict(self, market_data: np.ndarray) -> np.ndarray:
        """Prédit les prix futurs."""
        X = self.prepare_data(market_data)[0]
        return self.model(X, training=False).numpy()
    
    def get_confidence_interval(self, 
                              predictions: np.ndarray,
                              confidence_level: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
        """Calcule l'intervalle de confiance des prédictions."""
        z_score = 1.96  # Pour 95% de confiance
        std = np.std(predictions, axis=0)
        margin_of_error = z_score * (std / np.sqrt(len(predictions)))
        
        lower_bound = predictions - margin_of_error
        upper_bound = predictions + margin_of_error
        
        return lower_bound, upper_bound

class EnsemblePredictor:
    def __init__(self, n_models: int = 5):
        self.models = [PricePredictor() for _ in range(n_models)]
        
    def train(self, train_data: np.ndarray, validation_data: np.ndarray):
        """Entraîne l'ensemble des modèles."""
        for i, model in enumerate(self.models):
            print(f"Training model {i+1}/{len(self.models)}")
            model.train(train_data, validation_data)
    
    def predict(self, market_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Fait des prédictions avec l'ensemble des modèles."""
        predictions = []
        for model in self.models:
            pred = model.predict(market_data)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)
        
        return mean_pred, std_pred, predictions
