import tensorflow as tf
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import tensorflow_probability as tfp

@dataclass
class AttentionLSTMConfig:
    input_dim: int = 128
    lstm_units: List[int] = (256, 512, 256)
    attention_heads: int = 8
    dropout_rate: float = 0.2
    learning_rate: float = 1e-4
    batch_size: int = 32
    sequence_length: int = 100
    prediction_length: int = 20
    num_samples: int = 1000
    quantile_levels: List[float] = (0.1, 0.5, 0.9)

class TemporalAttention(tf.keras.layers.Layer):
    def __init__(self, units: int):
        super().__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)
        
    def call(self, query, values):
        # query: [batch_size, query_dim]
        # values: [batch_size, seq_len, value_dim]
        
        # Expand query dims
        query_with_time = tf.expand_dims(query, 1)
        
        # Score function
        score = self.V(tf.nn.tanh(
            self.W1(query_with_time) + self.W2(values)
        ))
        
        # Attention weights
        attention_weights = tf.nn.softmax(score, axis=1)
        
        # Context vector
        context = attention_weights * values
        context = tf.reduce_sum(context, axis=1)
        
        return context, attention_weights

class MultiHeadTemporalAttention(tf.keras.layers.Layer):
    def __init__(self, config: AttentionLSTMConfig):
        super().__init__()
        self.num_heads = config.attention_heads
        self.head_dim = config.lstm_units[-1] // config.attention_heads
        
        self.heads = [TemporalAttention(self.head_dim) for _ in range(self.num_heads)]
        self.linear = tf.keras.layers.Dense(config.lstm_units[-1])
        
    def call(self, query, values):
        head_outputs = []
        attention_weights = []
        
        for head in self.heads:
            context, weights = head(query, values)
            head_outputs.append(context)
            attention_weights.append(weights)
        
        multi_head = tf.concat(head_outputs, axis=-1)
        final_context = self.linear(multi_head)
        
        return final_context, attention_weights

class ProbabilisticLayer(tf.keras.layers.Layer):
    def __init__(self, config: AttentionLSTMConfig):
        super().__init__()
        self.quantile_levels = config.quantile_levels
        self.num_samples = config.num_samples
        
        # Paramètres de la distribution
        self.loc_layer = tf.keras.layers.Dense(1)
        self.scale_layer = tf.keras.layers.Dense(1, activation='softplus')
        self.df_layer = tf.keras.layers.Dense(1, activation='softplus')
        
    def call(self, inputs):
        # Student's t distribution parameters
        loc = self.loc_layer(inputs)
        scale = self.scale_layer(inputs)
        df = self.df_layer(inputs) + 2.0  # Ensure df > 2 for finite variance
        
        # Création de la distribution
        distribution = tfp.distributions.StudentT(
            df=df,
            loc=loc,
            scale=scale
        )
        
        # Échantillonnage et quantiles
        samples = distribution.sample(self.num_samples)
        quantiles = [
            distribution.quantile(q) for q in self.quantile_levels
        ]
        
        return {
            'distribution': distribution,
            'samples': samples,
            'quantiles': quantiles,
            'parameters': {
                'loc': loc,
                'scale': scale,
                'df': df
            }
        }

class ResidualGatedBlock(tf.keras.layers.Layer):
    def __init__(self, units: int, dropout_rate: float):
        super().__init__()
        self.gate = tf.keras.layers.Dense(units, activation='sigmoid')
        self.activation = tf.keras.layers.Dense(units, activation='tanh')
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.layer_norm = tf.keras.layers.LayerNormalization()
        
    def call(self, inputs, training=False):
        gate_values = self.gate(inputs)
        activated = self.activation(inputs)
        gated = gate_values * activated
        dropped = self.dropout(gated, training=training)
        residual = inputs + dropped
        normalized = self.layer_norm(residual)
        return normalized

class DeepAttentionLSTM(tf.keras.Model):
    def __init__(self, config: AttentionLSTMConfig):
        super().__init__()
        self.config = config
        
        # Couches d'embedding
        self.input_projection = tf.keras.layers.Dense(config.input_dim)
        
        # LSTM multi-couches avec residual connections
        self.lstm_layers = []
        self.residual_blocks = []
        
        for units in config.lstm_units:
            lstm = tf.keras.layers.LSTM(
                units,
                return_sequences=True,
                return_state=True
            )
            residual = ResidualGatedBlock(
                units,
                config.dropout_rate
            )
            self.lstm_layers.append(lstm)
            self.residual_blocks.append(residual)
        
        # Attention multi-tête
        self.attention = MultiHeadTemporalAttention(config)
        
        # Couche probabiliste
        self.probabilistic_output = ProbabilisticLayer(config)
        
        # Métriques de performance
        self.loss_tracker = tf.keras.metrics.Mean(name='loss')
        self.quantile_loss_tracker = tf.keras.metrics.Mean(name='quantile_loss')
        self.calibration_error_tracker = tf.keras.metrics.Mean(name='calibration_error')
    
    def call(self, inputs, training=False):
        # Projection initiale
        x = self.input_projection(inputs)
        
        # LSTM + Residual processing
        states = []
        for lstm, residual in zip(self.lstm_layers, self.residual_blocks):
            lstm_out, state_h, state_c = lstm(x)
            x = residual(lstm_out, training=training)
            states.extend([state_h, state_c])
        
        # Attention sur la séquence
        context, attention_weights = self.attention(states[-2], x)
        
        # Prédiction probabiliste
        outputs = self.probabilistic_output(context)
        
        return outputs
    
    def compute_loss(self, y_true, distribution_params):
        """Calcul de la loss avec multiple composantes."""
        # Negative log likelihood
        distribution = distribution_params['distribution']
        nll_loss = -tf.reduce_mean(distribution.log_prob(y_true))
        
        # Quantile loss
        quantile_losses = []
        for q, pred in zip(self.config.quantile_levels, distribution_params['quantiles']):
            error = y_true - pred
            quantile_loss = tf.reduce_mean(
                tf.maximum(q * error, (q - 1) * error)
            )
            quantile_losses.append(quantile_loss)
        
        # Calibration error
        samples = distribution_params['samples']
        calibration_error = self._compute_calibration_error(y_true, samples)
        
        # Loss totale
        total_loss = (
            nll_loss +
            tf.reduce_mean(quantile_losses) +
            0.1 * calibration_error
        )
        
        # Mise à jour des métriques
        self.loss_tracker.update_state(total_loss)
        self.quantile_loss_tracker.update_state(tf.reduce_mean(quantile_losses))
        self.calibration_error_tracker.update_state(calibration_error)
        
        return total_loss
    
    def _compute_calibration_error(self, y_true, samples):
        """Calcul de l'erreur de calibration."""
        # Calcul des rangs empiriques
        ranks = tf.reduce_mean(
            tf.cast(samples <= y_true, tf.float32),
            axis=0
        )
        
        # Calcul de l'erreur de calibration
        sorted_ranks = tf.sort(ranks)
        n = tf.cast(tf.shape(ranks)[0], tf.float32)
        expected_ranks = tf.range(1, n + 1) / n
        
        return tf.reduce_mean(tf.abs(sorted_ranks - expected_ranks))
    
    @tf.function
    def train_step(self, data):
        x, y = data
        
        with tf.GradientTape() as tape:
            predictions = self(x, training=True)
            loss = self.compute_loss(y, predictions)
        
        # Calcul et application des gradients
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        return {
            'loss': self.loss_tracker.result(),
            'quantile_loss': self.quantile_loss_tracker.result(),
            'calibration_error': self.calibration_error_tracker.result()
        }
    
    @tf.function
    def test_step(self, data):
        x, y = data
        predictions = self(x, training=False)
        loss = self.compute_loss(y, predictions)
        
        return {
            'loss': self.loss_tracker.result(),
            'quantile_loss': self.quantile_loss_tracker.result(),
            'calibration_error': self.calibration_error_tracker.result()
        }
    
    def predict_distribution(self, x):
        """Prédiction avec intervalles de confiance et distribution complète."""
        predictions = self(x, training=False)
        
        return {
            'mean': predictions['parameters']['loc'],
            'std': predictions['parameters']['scale'],
            'quantiles': predictions['quantiles'],
            'samples': predictions['samples'],
            'distribution': predictions['distribution']
        }
    
    def get_attention_weights(self, x):
        """Récupère les poids d'attention pour l'interprétabilité."""
        # Forward pass jusqu'à l'attention
        projected = self.input_projection(x)
        
        lstm_out = projected
        for lstm, residual in zip(self.lstm_layers, self.residual_blocks):
            lstm_out, state_h, state_c = lstm(lstm_out)
            lstm_out = residual(lstm_out)
        
        _, attention_weights = self.attention(state_h, lstm_out)
        
        return attention_weights
