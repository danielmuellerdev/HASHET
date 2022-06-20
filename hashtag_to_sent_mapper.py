from numpy import identity
import torch
import pytorch_lightning as pl
import tensorflow as tf
import torch.nn as nn

pl.seed_everything(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Hashtag2SentMapper(pl.LightningModule):
    """Maps from hashtag space to sentence space. """
    
    def __init__(
        self, in_features: int, out_features: int,
        hidden_layer1_size: int = 350, hidden_layer2_size: int = 250, learning_rate: float = 3e-5
    ):
        super().__init__()
        self.learning_rate = learning_rate
        self._model = nn.Sequential(
          nn.Linear(in_features, hidden_layer1_size),
          nn.ReLU(),
          nn.Linear(hidden_layer1_size, hidden_layer2_size),
          nn.ReLU(),
          nn.Linear(hidden_layer2_size, out_features)
        )

        self._cosine_distance_loss = nn.CosineEmbeddingLoss()
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def forward(self, targets):
        predicted_sent_emb = self._model(targets)
        return predicted_sent_emb
    
    def training_step(self, targets, sent_emb):
        predicted_sent_emb = self(targets)
        loss = self._cosine_distance_loss(sent_emb, predicted_sent_emb)

        self.log('train_loss', loss)
        self.log('cosine_distance', self._cosine_distance(sent_emb, predicted_sent_emb))

    def test_step(self, targets, sent_emb):
        predicted_sent_emb = self(targets)
        loss = self._cosine_distance_loss(sent_emb, predicted_sent_emb)

        self.log('test_loss', loss)
        self.log('cosine_distance', self._cosine_distance(sent_emb, predicted_sent_emb))

    @staticmethod
    def _cosine_distance(y_true, y_pred):
        return tf.compat.v1.losses.cosine_distance(tf.nn.l2_normalize(y_pred, 0), tf.nn.l2_normalize(y_true, 0), dim=0)
    