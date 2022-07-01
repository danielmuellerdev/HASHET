from typing import List

from transformers import BertTokenizer, BertModel
import torch

class SentenceEmbeddingModel:
    OUTPUT_DIM = 768

    def __init__(self):
        self._tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self._model = BertModel.from_pretrained('bert-base-uncased')

    def _generate_embedding(self, sentence: str):
        input_ids = torch.tensor(self._tokenizer.encode(sentence)).unsqueeze(0)  # Batch size 1
        outputs = self._model(input_ids)
        last_hidden_states = outputs[0].squeeze()  # The last hidden-state is the first element of the output tuple
        last_hidden_state_at_cls_token = last_hidden_states[0]
        
        return last_hidden_state_at_cls_token.detach()

    def generate_embeddings(self, sentences: List[str]) -> torch.tensor:
        sents_embs = [self._generate_embedding(sent) for sent in sentences]
        return torch.stack(tuple(sents_embs))
