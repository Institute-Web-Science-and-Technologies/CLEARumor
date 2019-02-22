# Copyright 2019 Lukas Schmelzeisen
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from time import time
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import Post, SdqcInstance, load_sdcq_instances
from src.util import DatasetHelper, ScalingMode

EVAL_DEV_EVERY_N_EPOCH = 5


class Sdqc:
    class Hyperparameters:
        def __init__(self,
                     max_sentence_length: int,
                     batch_size: int,
                     num_epochs: int,
                     learning_rate: float,
                     class_weights: List[float],
                     input_num_emb_dims: int,
                     input_num_aux_dims: int,
                     input_aux_scaling_features: List[int],
                     input_aux_scaling_mode: ScalingMode,
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
            self.class_weights = class_weights
            self.input_num_emb_dims = input_num_emb_dims
            self.input_num_aux_dims = input_num_aux_dims
            self.input_aux_scaling_features = input_aux_scaling_features
            self.input_aux_scaling_mode = input_aux_scaling_mode
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

    class Dataset(DatasetHelper):
        def __init__(self,
                     instances: Iterable[SdqcInstance],
                     posts: Dict[str, Post],
                     post_embeddings: Dict[str, torch.Tensor],
                     hparams: 'Sdqc.Hyperparameters',
                     device: torch.device):
            super().__init__(post_embeddings)

            for instance in instances:
                post = posts[instance.post_id]
                post_embedding = post_embeddings[post.id]

                post_platform, post_author, post_similarity_to_source = \
                    self.calc_shared_features(post)

                post_type = [post.has_source_depth,
                             post.has_reply_depth,
                             post.has_nested_depth]

                post_features = (np.concatenate((post_platform,
                                                 post_author,
                                                 [post_similarity_to_source],
                                                 post_type))
                                 .astype(np.float32))

                self._dataset.append({
                    'post_id': post.id,
                    'emb': post_embedding,
                    'features': torch.from_numpy(post_features).to(device),
                    'label': (torch.tensor(instance.label.value, device=device)
                              if instance.label else 0),
                })

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

        def filter_func(post_id: str) -> bool:
            return self._posts[post_id].platform == Post.Platform.twitter

        for index in self._hparams.input_aux_scaling_features:
            min, max, mean, std = \
                self._train_data.calc_stats_for_aux_feature(index, filter_func)

            if self._hparams.input_aux_scaling_mode == ScalingMode.none:
                pass
            elif self._hparams.input_aux_scaling_mode == ScalingMode.min_max:
                args = [index, min, max, filter_func]
                self._train_data.min_max_scale_aux_feature(*args)
                self._dev_data.min_max_scale_aux_feature(*args)
                self._test_data.min_max_scale_aux_feature(*args)
            elif self._hparams.input_aux_scaling_mode == ScalingMode.standard:
                args = [index, mean, std, filter_func]
                self._train_data.standard_scale_aux_feature(*args)
                self._dev_data.standard_scale_aux_feature(*args)
                self._test_data.standard_scale_aux_feature(*args)
            else:
                raise ValueError('Unimplemented enum variant.')

        time_after = time()
        print('  Took {:.2f}s.'.format(time_after - time_before))

    class Model(nn.Module):
        def __init__(self, hparams: 'Sdqc.Hyperparameters'):
            super().__init__()
            self._hparams = hparams

            emb_num_output_dims = self._hparams.input_num_emb_dims

            # -- convolutional layers ------------------------------------------
            conv_num_input_dims = emb_num_output_dims
            conv_num_output_dims = (len(self._hparams.conv_kernel_sizes)
                                    * self._hparams.conv_num_channels)
            self._conv_layers = nn.ModuleList()
            for i in range(self._hparams.conv_num_layers):
                layer = nn.ModuleDict()
                for size in self._hparams.conv_kernel_sizes:
                    conv = nn.Conv1d(
                        in_channels=(conv_num_input_dims
                                     if i == 0 else conv_num_output_dims),
                        out_channels=self._hparams.conv_num_channels,
                        kernel_size=size)
                    batch_norm = nn.BatchNorm1d(
                        num_features=self._hparams.conv_num_channels)
                    layer['kernel_size{:d}'.format(size)] = nn.ModuleDict(
                        {'conv': conv, 'batch_norm': batch_norm})
                self._conv_layers.append(layer)

            # -- dense layers --------------------------------------------------
            if self._hparams.conv_num_layers:
                dense_num_input_dims = \
                    conv_num_output_dims + self._hparams.input_num_aux_dims
            else:
                dense_num_input_dims = \
                    emb_num_output_dims + self._hparams.input_num_aux_dims
            dense_num_output_dims = self._hparams.dense_num_hidden
            self._dense_layers = nn.ModuleList()
            for i in range(self._hparams.dense_num_layers):
                self._dense_layers.append(nn.Linear(
                    in_features=(dense_num_input_dims
                                 if i == 0 else dense_num_output_dims),
                    out_features=self._hparams.dense_num_hidden))

            # -- linear layer --------------------------------------------------
            if self._hparams.dense_num_layers:
                linear_num_input_dims = dense_num_output_dims
            elif self._hparams.conv_num_layers:
                linear_num_input_dims = \
                    conv_num_output_dims + self._hparams.input_num_aux_dims
            else:
                linear_num_input_dims = \
                    emb_num_output_dims + self._hparams.input_num_aux_dims
            self._linear = nn.Linear(
                in_features=linear_num_input_dims,
                out_features=len(SdqcInstance.Label))

            # num_total_params = 0
            # for i, (n, w) in enumerate(self.named_parameters()):
            #     if w.requires_grad:
            #         print(i, n, w.shape, w.numel())
            #         num_total_params += w.numel()
            # print('Num Total Parameters: {}'.format(num_total_params))

        def forward(self, emb, aux):
            x = emb

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

            if self._hparams.input_num_aux_dims:
                x = torch.cat((x, aux), dim=1)

            for dense in self._dense_layers:
                x = F.dropout(F.relu(dense(x)),
                              p=self._hparams.dense_dropout,
                              training=self.training)

            logits = self._linear(x)

            return logits

    def train(self) -> None:
        print('Training SDQC model...')
        self.model = self.Model(self._hparams).to(self._device)

        criterion = nn.CrossEntropyLoss(
            weight=torch.tensor(self._hparams.class_weights,
                                dtype=torch.float32,
                                device=self._device))
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
                    batch_logits = self.model(batch['emb'], batch['features'])
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
                dev_acc, dev_f1 = self.eval(self._dev_data)
                print('  Validation:    Accuracy={:.2%}  F1-score={:.2%}'
                      .format(dev_acc, dev_f1))

        test_acc, test_f1 = self.eval(self._test_data)
        print('Test:            Accuracy={:.2%}  F1-score={:.2%}'
              .format(test_acc, test_f1))

    def eval(self, dataset: 'Sdqc.Dataset') -> (float, float):
        labels, predictions = [], []

        with torch.no_grad():
            data_loader = DataLoader(dataset,
                                     batch_size=self._hparams.batch_size)
            for batch in data_loader:
                self.model.eval()
                batch_logits = self.model(batch['emb'], batch['features'])
                batch_prediction = torch.argmax(batch_logits, dim=1)

                labels.append(batch['label'].data.cpu().numpy())
                predictions.append(batch_prediction.data.cpu().numpy())

        labels = np.concatenate(labels)
        predictions = np.concatenate(predictions)

        acc = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions, average='macro')

        return acc, f1

    def predict(self, post_ids: Iterable[str]) \
            -> Dict[str, Tuple[SdqcInstance.Label,
                               Dict[SdqcInstance.Label, float]]]:
        instances = [SdqcInstance(post_id, None) for post_id in post_ids]
        dataset = self.Dataset(instances, self._posts, self._post_embeddings,
                               self._hparams, self._device)

        results = {}
        with torch.no_grad():
            data_loader = DataLoader(dataset,
                                     batch_size=self._hparams.batch_size)
            for batch in data_loader:
                self.model.eval()
                batch_logits = self.model(batch['emb'], batch['features'])
                batch_probs = F.softmax(batch_logits, dim=1)
                batch_prediction = torch.argmax(batch_logits, dim=1)

                for post_id, prediction, probs in zip(
                        batch['post_id'], batch_prediction, batch_probs):
                    results[post_id] = \
                        (SdqcInstance.Label(prediction.item()),
                         dict(zip(SdqcInstance.Label, probs.tolist())))

        return results
