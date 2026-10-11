"""
Core AI Model Base Architectures

Domain-agnostic base model architectures that get specialized through
domain-specific training data. Each model type can be reused across
every domain stack (cybersecurity, finance, manufacturing, agriculture, etc.).

These 10 foundational architectures cover the full spectrum of AI tasks
required by the platform's microservices.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple



# Abstract Base

class BaseModel(ABC, nn.Module):
    """Abstract base for all platform models."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config

    @abstractmethod
    def forward(self, *args, **kwargs):
        pass

    def get_num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def get_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# 1. ANOMALY_DETECTOR — Unsupervised Learning
#    Architectures: Autoencoder, Isolation Forest, One-Class SVM
#    Purpose: Detect deviations from normal behavior


class AnomalyDetectorBase(BaseModel):
    """
    Autoencoder-based anomaly detection.

    Learns to reconstruct normal inputs. High reconstruction error
    indicates anomalous behavior. Used by:
      - IoT/OT Anomaly Detection
      - AI-Driven Threat Detection
      - Ransomware Behavior Detection
      - Data Exfiltration Predictor
      - Insider Threat Synthesizer
    """

    def __init__(self, input_dim: int, latent_dim: int = 32, config: Dict[str, Any] = None):
        super().__init__(config or {})
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed

    def get_anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruction error as anomaly score."""
        reconstructed = self.forward(x)
        return torch.mean((x - reconstructed) ** 2, dim=1)



# 2. BEHAVIOR_ANALYZER — Sequential Pattern Recognition
#    Architectures: LSTM, GRU, Transformer (Time Series)
#    Purpose: Analyze sequences and temporal patterns

class BehaviorAnalyzerBase(BaseModel):
    """
    LSTM-based sequence analysis for temporal behavior patterns.

    Processes ordered event sequences to detect behavioral anomalies
    or predict next actions. Used by:
      - Insider Threat Synthesizer
      - Ransomware Behavior Detection
      - IoT/OT Anomaly Detection
      - Data Exfiltration Predictor
      - ADHO Deception Orchestrator
      - Malware Sandbox Analysis
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_classes: int = None,
        bidirectional: bool = True,
        config: Dict[str, Any] = None,
    ):
        super().__init__(config or {})
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=0.2 if num_layers > 1 else 0.0,
        )

        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim

        if num_classes:
            # Classification mode
            self.head = nn.Sequential(
                nn.Linear(lstm_output_dim, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, num_classes),
            )
        else:
            # Prediction mode (next-step)
            self.head = nn.Linear(lstm_output_dim, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len, input_dim)
        lstm_out, (h_n, c_n) = self.lstm(x)
        # Use last timestep output
        last_output = lstm_out[:, -1, :]
        return self.head(last_output)


# 3. THREAT_CLASSIFIER — Supervised Learning
#    Architectures: CNN, BERT, Random Forest, XGBoost
#    Purpose: Categorize and label threats/events


class ThreatClassifierBase(BaseModel):
    """
    Multi-class classification with MLP.

    Categorizes events into predefined threat categories using
    labeled training data. Used by:
      - AI-Driven Threat Detection
      - FUSER Multi-Source Correlator
      - Phishing & Malicious Link Detection
      - Malware Sandbox Analysis
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dims: list = None,
        dropout: float = 0.3,
        config: Dict[str, Any] = None,
    ):
        super().__init__(config or {})

        if hidden_dims is None:
            hidden_dims = [256, 128]

        layers = []
        prev_dim = input_dim

        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.ReLU(),
                nn.BatchNorm1d(h_dim),
                nn.Dropout(dropout),
            ])
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, num_classes))
        self.classifier = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


# 4. FORECASTER — Time Series Prediction
#    Architectures: LSTM, Prophet, ARIMA, Transformer
#    Purpose: Predict future states and events

class ForecasterBase(BaseModel):
    """
    Transformer-based time series forecasting.

    Predicts future states from historical sequences. Used by:
      - Ransomware Behavior Detection
      - IoT/OT Anomaly Detection
      - Attack Path Predictor
      - Data Exfiltration Predictor
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        forecast_horizon: int = 1,
        config: Dict[str, Any] = None,
    ):
        super().__init__(config or {})

        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, 512, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(d_model, input_dim * forecast_horizon)
        self.forecast_horizon = forecast_horizon
        self.input_dim = input_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len, input_dim)
        seq_len = x.size(1)
        embedded = self.embedding(x) + self.pos_encoding[:, :seq_len, :]
        transformed = self.transformer(embedded)
        forecast = self.fc(transformed[:, -1, :])
        return forecast.view(-1, self.forecast_horizon, self.input_dim)


# 5. CORRELATOR — Graph Neural Networks
#    Architectures: GNN, Graph Attention Networks
#    Purpose: Find relationships between entities/events

class CorrelatorBase(BaseModel):
    """
    Graph-based entity and event correlation.

    Models relationships between entities (hosts, users, services)
    as a graph and identifies suspicious connection patterns. Used by:
      - FUSER Multi-Source Correlator
      - Attack Path Predictor
    """

    def __init__(
        self,
        node_features: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        output_dim: int = 1,
        config: Dict[str, Any] = None,
    ):
        super().__init__(config or {})

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(node_features, hidden_dim))

        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))

        self.output = nn.Linear(hidden_dim, output_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self, x: torch.Tensor, adjacency: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: Node features (num_nodes, node_features)
            adjacency: Adjacency matrix (num_nodes, num_nodes)
        """
        for layer in self.layers:
            # Message passing: aggregate neighbor features
            x = torch.matmul(adjacency, x)
            x = layer(x)
            x = torch.relu(x)
            x = self.norm(x)

        return self.output(x)


# 6. CONTENT_ANALYZER — NLP / Computer Vision
#    Architectures: BERT, GPT, ResNet, YOLO
#    Purpose: Analyze text, images, code, URLs

class ContentAnalyzerBase(BaseModel):
    """
    Transformer-based text and content analysis.

    Processes textual content (emails, URLs, logs, code) for
    classification and feature extraction. Used by:
      - Phishing & Malicious Link Detection
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        num_classes: int = 2,
        max_seq_len: int = 512,
        config: Dict[str, Any] = None,
    ):
        super().__init__(config or {})

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Embedding(max_seq_len, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len) — token IDs
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)

        embedded = self.embedding(x) + self.pos_embedding(positions)
        transformed = self.transformer(embedded)
        pooled = transformed.mean(dim=1)  # Mean pooling
        return self.classifier(pooled)


# 7. RL_AGENT — Reinforcement Learning
#    Architectures: DQN, PPO, A3C
#    Purpose: Automated response and adaptive behavior

class RLAgentBase(BaseModel):
    """
    Deep Q-Network for reinforcement learning.

    Learns optimal response actions through environment interaction.
    Used by:
      - ADHO Deception & Honeypot Orchestrator
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        config: Dict[str, Any] = None,
    ):
        super().__init__(config or {})

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

        # Target network for stable training
        self.target_network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
        self.update_target()

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Return Q-values for all actions."""
        return self.network(state)

    def update_target(self):
        """Copy weights to target network."""
        self.target_network.load_state_dict(self.network.state_dict())

    def select_action(self, state: torch.Tensor, epsilon: float = 0.1) -> int:
        """Epsilon-greedy action selection."""
        if torch.rand(1).item() < epsilon:
            return torch.randint(0, self.network[-1].out_features, (1,)).item()
        with torch.no_grad():
            q_values = self.forward(state)
            return q_values.argmax(dim=-1).item()


# 8. GENERATOR — Generative Models
#    Architectures: GAN, VAE, Diffusion Models
#    Purpose: Generate synthetic data, simulate scenarios

class GeneratorBase(BaseModel):
    """
    GAN generator for synthetic data generation.

    Generates realistic synthetic samples for data augmentation,
    scenario simulation, and adversarial testing. Used by:
      - ADHO Deception Orchestrator (generate decoy data)
      - Insider Threat Synthesizer (augment rare attack data)
    """

    def __init__(
        self,
        latent_dim: int = 100,
        output_dim: int = 128,
        config: Dict[str, Any] = None,
    ):
        super().__init__(config or {})
        self.latent_dim = latent_dim

        self.generator = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
            nn.Tanh(),
        )

        self.discriminator = nn.Sequential(
            nn.Linear(output_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Generate samples from noise."""
        return self.generator(z)

    def discriminate(self, x: torch.Tensor) -> torch.Tensor:
        """Classify real vs generated."""
        return self.discriminator(x)

    def sample(self, n_samples: int, device: torch.device = None) -> torch.Tensor:
        """Generate n random samples."""
        device = device or next(self.parameters()).device
        z = torch.randn(n_samples, self.latent_dim, device=device)
        return self.forward(z)


# 9. RECOMMENDER — Collaborative Filtering
#    Architectures: Matrix Factorization, Neural CF
#    Purpose: Suggest actions, configurations, policies

class RecommenderBase(BaseModel):
    """
    Neural collaborative filtering for action/policy recommendations.

    Recommends optimal security configurations, response actions,
    or segmentation policies. Used by:
      - Attack Path Predictor & Micro-Segmentation Planner
    """

    def __init__(
        self,
        num_entities: int,
        num_actions: int,
        embed_dim: int = 64,
        config: Dict[str, Any] = None,
    ):
        super().__init__(config or {})

        self.entity_embedding = nn.Embedding(num_entities, embed_dim)
        self.action_embedding = nn.Embedding(num_actions, embed_dim)

        self.fc = nn.Sequential(
            nn.Linear(embed_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(
        self, entity_ids: torch.Tensor, action_ids: torch.Tensor
    ) -> torch.Tensor:
        entity_embed = self.entity_embedding(entity_ids)
        action_embed = self.action_embedding(action_ids)
        concat = torch.cat([entity_embed, action_embed], dim=1)
        return self.fc(concat).squeeze(-1)


# 10. EXPLAINER — Interpretability
#     Architectures: LIME, SHAP, Attention Mechanisms
#     Purpose: Explain model decisions for transparency

class ExplainerBase(BaseModel):
    """
    Attention-based model explainability.

    Provides interpretable attention weights over input features,
    enabling analysts to understand why a model made a specific
    decision. Used by:
      - FUSER Multi-Source Correlator
      - Malware Sandbox Analysis
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_heads: int = 4,
        config: Dict[str, Any] = None,
    ):
        super().__init__(config or {})

        self.fc_in = nn.Linear(input_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(
            hidden_dim, num_heads=num_heads, batch_first=True
        )
        self.fc_out = nn.Linear(hidden_dim, 1)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            output: Prediction scores
            attention_weights: Feature importance weights for explanation
        """
        transformed = self.fc_in(x)

        if transformed.dim() == 2:
            transformed = transformed.unsqueeze(1)

        attended, attention_weights = self.attention(
            transformed, transformed, transformed
        )
        output = self.fc_out(attended.mean(dim=1))
        return output, attention_weights


# Model Registry — Maps model type names to classes

MODEL_REGISTRY = {
    "ANOMALY_DETECTOR": AnomalyDetectorBase,
    "BEHAVIOR_ANALYZER": BehaviorAnalyzerBase,
    "THREAT_CLASSIFIER": ThreatClassifierBase,
    "FORECASTER": ForecasterBase,
    "CORRELATOR": CorrelatorBase,
    "CONTENT_ANALYZER": ContentAnalyzerBase,
    "RL_AGENT": RLAgentBase,
    "GENERATOR": GeneratorBase,
    "RECOMMENDER": RecommenderBase,
    "EXPLAINER": ExplainerBase,
}


def get_model_class(model_type: str) -> type:
    """Get model class by type name."""
    if model_type not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model type: {model_type}. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[model_type]
