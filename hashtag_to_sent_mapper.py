import torch
import pytorch_lightning as pl
import tensorflow as tf
import torch.nn as nn
from torch import Tensor


class Hashtag2SentMapper(pl.LightningModule):
    """ Maps from hashtag space to sentence space. """
    
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
        self._cosine_similarity = nn.CosineSimilarity()
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def forward(self, x):
        y_hat = self._model(x)
        return y_hat

    def _calc_cosine_distance_loss(self, y: Tensor, y_hat: Tensor) -> Tensor:
        return self._cosine_distance_loss(
            y, 
            y_hat, 
            target=torch.tensor([1] * y.shape[0])
        )

    def training_step(self, batch, _):
        y_hat = self(batch['x'])
        loss = self._calc_cosine_distance_loss(batch['y'], y_hat)

        self.log('train_loss', loss)
        self.log('train_cosine_similarity', self._cosine_similarity(batch['y'], y_hat).mean())

        return loss

    def validation_step(self, batch, _):
        y_hat = self(batch['x'])
        loss = self._calc_cosine_distance_loss(batch['y'], y_hat)

        self.log('val_loss', loss)
        self.log('val_cosine_distance', self._cosine_distance(batch['y'], y_hat))
        self.log('val_cosine_similarity', self._cosine_similarity(batch['y'], y_hat).mean())

    def test_step(self, batch, _):
        y_hat = self(batch['x'])
        loss = self._calc_cosine_distance_loss(batch['y'], y_hat)
        
        self.log('test_loss', loss)
        self.log('test_cosine_distance', self._cosine_distance(batch['y'], y_hat))

    @staticmethod
    def _cosine_distance(y_true, y_pred):
        return tf.compat.v1.losses.cosine_distance(tf.nn.l2_normalize(y_pred, 0), tf.nn.l2_normalize(y_true, 0), dim=0).numpy()
    