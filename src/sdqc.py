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

from itertools import chain
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import Post, SdqcInstance, load_sdcq_instances
from src.util import DatasetHelper, ScalingMode, \
    generate_folds_for_k_fold_cross_validation_helper

EVAL_DEV_EVERY_N_EPOCH = 20


class Sdqc:
    class Hyperparameters:
        def __init__(self,
                     max_sentence_length: int,
                     batch_size: int,
                     num_epochs: int,
                     learning_rate: float,
                     weight_decay: float,
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
            self.weight_decay = weight_decay
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

    def _load_datasets(self,
                       train_instances: Iterable[SdqcInstance],
                       dev_instances: Optional[Iterable[SdqcInstance]],
                       test_instances: Optional[Iterable[SdqcInstance]]) \
            -> ('Sdqc.Dataset',
                Optional['Sdqc.Dataset'],
                Optional['Sdqc.Dataset']):
        print('Number of instances: train={:d}, dev={:d}, test={:d}'
              .format(len(train_instances),
                      len(dev_instances or []),
                      len(test_instances or [])))

        train_dataset = self.Dataset(
            train_instances, self._posts, self._post_embeddings, self._hparams,
            self._device)

        dev_dataset = None
        if dev_instances:
            dev_dataset = self.Dataset(
                dev_instances, self._posts, self._post_embeddings,
                self._hparams, self._device)

        test_dataset = None
        if test_instances:
            test_dataset = self.Dataset(
                test_instances, self._posts, self._post_embeddings,
                self._hparams, self._device)

        def filter_func(post_id: str) -> bool:
            return self._posts[post_id].platform == Post.Platform.twitter

        for index in self._hparams.input_aux_scaling_features:
            min, max, mean, std = \
                train_dataset.calc_stats_for_aux_feature(index, filter_func)

            for dataset in (train_dataset, dev_dataset, test_dataset):
                if not dataset:
                    continue

                if self._hparams.input_aux_scaling_mode == ScalingMode.none:
                    pass
                elif (self._hparams.input_aux_scaling_mode
                      == ScalingMode.min_max):
                    dataset.min_max_scale_aux_feature(
                        index, min, max, filter_func)
                elif (self._hparams.input_aux_scaling_mode
                      == ScalingMode.standard):
                    dataset.standard_scale_aux_feature(
                        index, mean, std, filter_func)
                else:
                    raise ValueError('Unimplemented enum variant.')

        return train_dataset, dev_dataset, test_dataset

    def load_organizer_split(self) -> ('Sdqc.Dataset',
                                       'Sdqc.Dataset',
                                       Optional['Sdqc.Dataset']):
        train_instances, dev_instances, test_instances = load_sdcq_instances()
        return self._load_datasets(
            train_instances, dev_instances, test_instances)

    def generate_folds_for_k_fold_cross_validation(self, num_folds: int) \
            -> List[List[SdqcInstance]]:
        train_instances, dev_instances, test_instances = load_sdcq_instances()
        return generate_folds_for_k_fold_cross_validation_helper(
            num_folds, self._posts, train_instances, dev_instances,
            test_instances)

    def arrange_folds_for_k_fold_cross_validation(
            self, folds: List[List[SdqcInstance]], index: int) \
            -> ('Sdqc.Dataset', 'Sdqc.Dataset'):
        train_instances = list(chain.from_iterable(
            fold for i, fold in enumerate(folds) if i != index))
        test_instances = folds[index]

        train_dataset, _, test_dataset = \
            self._load_datasets(train_instances, None, test_instances)
        return train_dataset, test_dataset

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

    def train(self,
              train_dataset: 'Sdqc.Dataset',
              dev_dataset: Optional['Sdqc.Dataset'] = None,
              print_progress: bool = False) -> 'Sdqc.Model':
        model = self.Model(self._hparams).to(self._device)

        criterion = nn.CrossEntropyLoss(
            weight=torch.tensor(self._hparams.class_weights,
                                dtype=torch.float32,
                                device=self._device))
        optimizer = optim.Adam(model.parameters(),
                               lr=self._hparams.learning_rate,
                               weight_decay=self._hparams.weight_decay)

        train_loader = DataLoader(
            train_dataset, batch_size=self._hparams.batch_size, shuffle=True)

        for epoch_no in range(1, self._hparams.num_epochs + 1):
            losses, labels, predictions = [], [], []

            progress_bar = None
            if print_progress:
                progress_bar = tqdm(total=(len(train_loader)),
                                    unit='batch',
                                    desc='Epoch: {:{}d}/{:d}'.format(
                                        epoch_no,
                                        len(str(self._hparams.num_epochs)),
                                        self._hparams.num_epochs))

            for batch_no, batch in enumerate(train_loader):
                optimizer.zero_grad()

                model.train()
                batch_logits = model(batch['emb'], batch['features'])
                with torch.no_grad():
                    batch_prediction = torch.argmax(batch_logits, dim=1)

                loss = criterion(batch_logits, batch['label'])
                loss.backward()
                optimizer.step()

                losses.append(loss.item())
                labels.append(batch['label'].data.cpu().numpy())
                predictions.append(batch_prediction.data.cpu().numpy())

                if progress_bar:
                    progress_bar.set_postfix({
                        'loss': '{:.2e}'.format(loss.item()),
                    })
                    progress_bar.update(1)

            if progress_bar:
                progress_bar.close()

                labels = np.concatenate(labels)
                predictions = np.concatenate(predictions)

                epoch_loss = np.mean(losses)
                epoch_acc = accuracy_score(labels, predictions)
                epoch_f1 = f1_score(labels, predictions, average='macro')

                print('  Loss={:.2e}  Accuracy={:.2%}  F1-score={:.2%}'
                      .format(epoch_loss, epoch_acc, epoch_f1))

            if print_progress and dev_dataset and \
                    (epoch_no == self._hparams.num_epochs
                     or not epoch_no % EVAL_DEV_EVERY_N_EPOCH):
                dev_acc, dev_f1 = self.eval(model, dev_dataset)
                print('  Validation:    Accuracy={:.2%}  F1-score={:.2%}'
                      .format(dev_acc, dev_f1))

        return model

    def eval(self, model: 'Sdqc.Model', dataset: 'Sdqc.Dataset') \
            -> (float, float):
        labels, predictions = [], []

        with torch.no_grad():
            data_loader = DataLoader(
                dataset, batch_size=self._hparams.batch_size)
            for batch in data_loader:
                model.eval()
                batch_logits = model(batch['emb'], batch['features'])
                batch_prediction = torch.argmax(batch_logits, dim=1)

                labels.append(batch['label'].data.cpu().numpy())
                predictions.append(batch_prediction.data.cpu().numpy())

        labels = np.concatenate(labels)
        predictions = np.concatenate(predictions)

        acc = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions, average='macro')

        return acc, f1

    def predict(self, model: 'Sdqc.Model', post_ids: Iterable[str]) \
            -> Dict[str, Tuple[SdqcInstance.Label,
                               Dict[SdqcInstance.Label, float]]]:
        instances = [SdqcInstance(post_id) for post_id in post_ids]
        dataset = self.Dataset(instances, self._posts, self._post_embeddings,
                               self._hparams, self._device)

        results = {}
        with torch.no_grad():
            data_loader = DataLoader(dataset,
                                     batch_size=self._hparams.batch_size)
            for batch in data_loader:
                model.eval()
                batch_logits = model(batch['emb'], batch['features'])
                batch_probs = F.softmax(batch_logits, dim=1)
                batch_prediction = torch.argmax(batch_logits, dim=1)

                for post_id, prediction, probs in zip(
                        batch['post_id'], batch_prediction, batch_probs):
                    results[post_id] = \
                        (SdqcInstance.Label(prediction.item()),
                         dict(zip(SdqcInstance.Label, probs.tolist())))

        return results
