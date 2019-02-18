from collections.__init__ import OrderedDict
from itertools import chain
from time import time
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import Post, SdqcInstance, load_sdcq_instances

EVAL_DEV_EVERY_N_EPOCH = 5


class Sdqc:
    class Hyperparameters:
        def __init__(self,
                     max_sentence_length: int,
                     batch_size: int,
                     num_epochs: int,
                     learning_rate: float,
                     num_input_dimensions: int,
                     conv_num_layers: int,
                     conv_kernel_sizes: List[int],
                     conv_num_channels: int,
                     dense_num_layers: int,
                     dense_num_hidden: int,
                     dense_dropout: float):
            self.max_sentence_length = max_sentence_length
            self.batch_size = batch_size
            self.num_epochs = num_epochs
            self.learning_rate = learning_rate
            self.num_input_dimensions = num_input_dimensions
            self.conv_num_layers = conv_num_layers
            self.conv_kernel_sizes = conv_kernel_sizes
            self.conv_num_channels = conv_num_channels
            self.dense_num_layers = dense_num_layers
            self.dense_num_hidden = dense_num_hidden
            self.dense_dropout = dense_dropout

    def __init__(self,
                 posts: Dict[str, Post],
                 post_embeddings: Dict[str, torch.Tensor],
                 hparams: 'Sdqc.Hyperparameters',
                 device: torch.device):
        self._posts = posts
        self._post_embeddings = post_embeddings
        self._hparams = hparams
        self._device = device
        self.model = None

        self._load_data()

    class Dataset(torch.utils.data.Dataset):
        def __init__(self,
                     instances: List[SdqcInstance],
                     posts: Dict[str, Post],
                     post_embeddings: Dict[str, torch.Tensor],
                     hparams: 'Sdqc.Hyperparameters',
                     device: torch.device):
            self._dataset: List[Dict[str, torch.Tensor]] = []
            for instance in instances:
                post = posts[instance.post_id]

                self._dataset.append({
                    'post_id': post.id,
                    'embedding': post_embeddings[post.id],
                    'label': torch.tensor(instance.label.value, device=device),
                })

        def __len__(self) -> int:
            return len(self._dataset)

        def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
            return self._dataset[index]

    def _load_data(self):
        print()
        print('Loading SDQC data...')
        time_before = time()
        train, dev, test = load_sdcq_instances()
        print('  Number of instances: train={:d}, dev={:d}, test={:d}'.format(
            len(train), len(dev), len(test) if test else 0))

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

    class Model(nn.Module):
        def __init__(self, hparams: 'Sdqc.Hyperparameters'):
            super().__init__()
            self._hparams = hparams

            self._conv_layers = nn.ModuleList()
            for i in range(self._hparams.conv_num_layers):
                layer = nn.ModuleDict()
                for size in self._hparams.conv_kernel_sizes:
                    conv = nn.Conv1d(
                        in_channels=(self._hparams.num_input_dimensions
                                     if i == 0 else
                                     len(self._hparams.conv_kernel_sizes)
                                     * self._hparams.conv_num_channels),
                        out_channels=self._hparams.conv_num_channels,
                        kernel_size=size)
                    batch_norm = nn.BatchNorm1d(
                        num_features=self._hparams.conv_num_channels)
                    layer['kernel_size{:d}'.format(size)] = nn.ModuleDict(
                        {'conv': conv, 'batch_norm': batch_norm})
                self._conv_layers.append(layer)

            self._dense_layers = nn.ModuleList()
            for i in range(self._hparams.dense_num_layers):
                self._dense_layers.append(nn.Linear(
                    in_features=((len(self._hparams.conv_kernel_sizes)
                                  * self._hparams.conv_num_channels)
                                 if i == 0 else self._hparams.dense_num_hidden),
                    out_features=self._hparams.dense_num_hidden
                ))

            self._linear = nn.Linear(
                in_features=self._hparams.dense_num_hidden,
                out_features=len(SdqcInstance.Label))

            # num_total_params = 0
            # for i, (n, w) in enumerate(self.named_parameters()):
            #     if w.requires_grad:
            #         print(i, n, w.shape, w.numel())
            #         num_total_params += w.numel()
            # print('Num Total Parameters: {}'.format(num_total_params))

        def forward(self, embedding):
            x = embedding

            for layer in self._conv_layers:
                h = []
                for size in self._hparams.conv_kernel_sizes:
                    conv_batch_norm = layer['kernel_size{:d}'.format(size)]
                    conv = conv_batch_norm['conv']
                    batch_norm = conv_batch_norm['batch_norm']

                    h.append(batch_norm(F.relu(conv(
                        F.pad(x, [(size - 1) // 2, size // 2])))))
                x = torch.cat(h, dim=1)

            x = F.avg_pool1d(x, kernel_size=self._hparams.max_sentence_length)
            x.squeeze_(dim=2)

            for dense in self._dense_layers:
                x = F.dropout(F.relu(dense(x)),
                              p=self._hparams.dense_dropout,
                              training=self.training)

            logits = self._linear(x)

            return logits

    def train(self) -> Dict[str, str]:
        print('Training SDQC model...')
        self.model = self.Model(self._hparams).to(self._device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            self.model.parameters(), lr=self._hparams.learning_rate)

        train_loader = DataLoader(
            self._train_data, batch_size=self._hparams.batch_size, shuffle=True)

        for epoch_no in range(1, self._hparams.num_epochs + 1):
            losses, labels, predictions = [], [], []

            with tqdm(total=(len(train_loader)), unit='batch',
                      desc='Epoch: {:{}d}/{:d}'.format(
                          epoch_no, len(str(self._hparams.num_epochs)),
                          self._hparams.num_epochs)) \
                    as progress_bar:
                for batch_no, batch in enumerate(train_loader):
                    optimizer.zero_grad()

                    self.model.train()
                    batch_logits = self.model(batch['embedding'])
                    with torch.no_grad():
                        batch_prediction = torch.argmax(batch_logits, dim=1)

                    loss = criterion(batch_logits, batch['label'])
                    loss.backward()
                    optimizer.step()

                    losses.append(loss.item())
                    labels.append(batch['label'].data.cpu().numpy())
                    predictions.append(batch_prediction.data.cpu().numpy())

                    progress_bar.set_postfix({
                        'loss': '{:.2e}'.format(loss.item()),
                    })
                    progress_bar.update(1)

            # print(list(self.model.elmo._scalar_mixes[0].scalar_parameters))

            labels = np.concatenate(labels)
            predictions = np.concatenate(predictions)

            epoch_loss = np.mean(losses)
            epoch_acc = accuracy_score(labels, predictions)
            epoch_f1 = f1_score(labels, predictions, average='macro')
            print('  Loss={:.2e}  Accuracy={:.2%}  F1-score={:.2%}'
                  .format(epoch_loss, epoch_acc, epoch_f1))

            if (epoch_no == self._hparams.num_epochs
                    or not epoch_no % EVAL_DEV_EVERY_N_EPOCH):
                dev_acc, dev_f1, _ = self.eval(self._dev_data)
                print('  Validation:    Accuracy={:.2%}  F1-score={:.2%}'
                      .format(dev_acc, dev_f1))

        test_acc, test_f1, test_results = self.eval(self._test_data)
        print('Test:            Accuracy={:.2%}  F1-score={:.2%}'
              .format(test_acc, test_f1))

        return test_results

    def eval(self, data: 'Sdqc.Dataset') \
            -> (float, float, Dict[str, str]):
        post_ids, labels, predictions = [], [], []

        with torch.no_grad():
            data_loader = DataLoader(data, batch_size=self._hparams.batch_size)
            for batch_no, batch in enumerate(data_loader):
                self.model.eval()
                batch_logits = self.model(batch['embedding'])
                batch_prediction = torch.argmax(batch_logits, dim=1)

                post_ids.append(batch['post_id'])
                labels.append(batch['label'].data.cpu().numpy())
                predictions.append(batch_prediction.data.cpu().numpy())

        labels = np.concatenate(labels)
        predictions = np.concatenate(predictions)

        acc = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions, average='macro')

        results = OrderedDict()
        for post_id, prediction in zip(chain.from_iterable(post_ids),
                                       predictions):
            results[post_id] = SdqcInstance.Label(prediction).name

        return acc, f1, results
