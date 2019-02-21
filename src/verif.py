from time import time
from typing import Dict, List

import torch

from src.dataset import Post, VerifInstance, load_verif_instances


class Verif:
    class Hyperparameters:
        def __init__(self,
                     max_sentence_length: int,
                     batch_size: int,
                     num_epochs: float,
                     learning_rate: float):
            self.max_sentence_length = max_sentence_length
            self.batch_size = batch_size
            self.num_epochs = num_epochs
            self.learning_rate = learning_rate

    class Dataset(torch.utils.data.Dataset):
        def __init__(self,
                     instances: List[VerifInstance],
                     posts: Dict[str, Post],
                     post_embeddings: Dict[str, torch.Tensor],
                     hparams: 'Verif.Hyperparameters',
                     device: torch.device):
            self._dataset: List[Dict[str, torch.Tensor]] = []
            for instance in instances:
                post = posts[instance.post_id]
                self._dataset.append({
                    'post_id': post.id,
                    'emb': post_embeddings[post.id],
                    'label': torch.tensor(instance.label.value, device=device),
                })

        def __len__(self) -> int:
            return len(self._dataset)

        def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
            return self._dataset[index]

    def __init__(self,
                 posts: Dict[str, Post],
                 post_embeddings: Dict[str, torch.Tensor],
                 hparams: 'Verif.Hyperparameters',
                 device: torch.device):
        self._posts = posts
        self._post_embeddings = post_embeddings
        self._hparams = hparams
        self._device = device

        self._load_data()

    def _load_data(self):
        print()
        print('Loading Verification data...')
        time_before = time()
        train, dev, test = load_verif_instances()
        print('  Number of instances: train={:d}, dev={:d}, test={:d}'
              .format(len(train), len(dev), len(test) if test else 0))

        self._train_data = self.Dataset(
            train, self._posts, self._post_embeddings, self._hparams,
            self._device)
        self._dev_data = self.Dataset(
            dev, self._posts, self._post_embeddings, self._hparams,
            self._device)
        self._test_data = self.Dataset(
            test, self._posts, self._post_embeddings, self._hparams,
            self._device)
        time_after = time()
        print('  Took {:.2f}s.'.format(time_after - time_before))
