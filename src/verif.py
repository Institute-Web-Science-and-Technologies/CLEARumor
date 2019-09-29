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

from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import Post, SdqcInstance, VerifInstance
from src.util import DatasetHelper, ScalingMode, \
    rmse_score

EVAL_DEV_EVERY_N_EPOCH = 20


class Verif:
    class Hyperparameters:
        def __init__(self,
                     batch_size: int,
                     num_epochs: int,
                     learning_rate: float,
                     weight_decay: float,
                     class_weights: List[float],
                     input_num_dims: int,
                     input_scaling_features: List[int],
                     input_scaling_mode: ScalingMode,
                     dense_num_layers: int,
                     dense_num_hidden: int,
                     dense_dropout: float):
            self.batch_size = batch_size
            self.num_epochs = num_epochs
            self.learning_rate = learning_rate
            self.weight_decay = weight_decay
            self.class_weights = class_weights
            self.input_num_dims = input_num_dims
            self.input_scaling_features = input_scaling_features
            self.input_scaling_mode = input_scaling_mode
            self.dense_num_layers = dense_num_layers
            self.dense_num_hidden = dense_num_hidden
            self.dense_dropout = dense_dropout

    def __init__(self,
                 posts: Dict[str, Post],
                 post_embeddings: Dict[str, torch.Tensor],
                 hparams: 'Verif.Hyperparameters',
                 device: torch.device):
        self._posts = posts
        self._post_embeddings = post_embeddings
        self._hparams = hparams
        self._device = device

    class Dataset(DatasetHelper):
        def __init__(self,
                     instances: Iterable[VerifInstance],
                     posts: Dict[str, Post],
                     post_embeddings: Dict[str, torch.Tensor],
                     sdqc_estimates: Dict[str, Tuple[SdqcInstance.Label,
                                                     Dict[SdqcInstance.Label,
                                                          float]]],
                     hparams: 'Verif.Hyperparameters',
                     device: torch.device):
            super().__init__(post_embeddings)

            for instance in instances:
                source_post = posts[instance.post_id]

                post_features = self.calc_features(source_post, posts, sdqc_estimates)

                self._dataset.append({
                    'post_id': source_post.id,
                    'features': torch.from_numpy(post_features).to(device),
                    'label': (torch.tensor(instance.label.value, device=device)
                              if instance.label else 0),
                })

        @classmethod
        def calc_features(cls, source_post: Post, posts: List[Post],post_embeddings: Dict[str, torch.tensor], sdqc_estimates:  Dict[str, Tuple[SdqcInstance.Label,
                                                     Dict[SdqcInstance.Label,
                                                          float]]]):
            post_platform, post_author, post_similarity_to_source = \
                cls.calc_shared_features(source_post,post_embeddings)
            post_has_media = [source_post.has_media,
                              not source_post.has_media]
            post_upvote_ratio = 0.5
            if source_post.upvote_ratio:
                post_upvote_ratio = source_post.upvote_ratio
            num_childs = 0
            depths = {'reply': 0, 'nested': 0}
            predictions = {label: 0 for label in SdqcInstance.Label}
            estimates = {label: 0 for label in SdqcInstance.Label}
            for post in posts.values():
                if post.source_id == source_post.id:
                    num_childs += 1

                    if post.has_reply_depth:
                        depths['reply'] += 1
                    elif post.has_nested_depth:
                        depths['nested'] += 1

                    predictions[sdqc_estimates[post.id][0]] += 1
                    for label, prob in sdqc_estimates[post.id][1].items():
                        estimates[label] += prob
            depths = {depth: num / num_childs
                      for depth, num in depths.items()}
            predictions = {label: num / num_childs
                           for label, num in predictions.items()}
            estimates = {label: prob / num_childs
                         for label, prob in estimates.items()}
            depths = [depths['reply'],
                      depths['nested']]
            predictions = [predictions[SdqcInstance.Label.support],
                           predictions[SdqcInstance.Label.deny],
                           predictions[SdqcInstance.Label.query]]
            estimates = [estimates[SdqcInstance.Label.support],
                         estimates[SdqcInstance.Label.deny],
                         estimates[SdqcInstance.Label.query]]
            return (np.concatenate((post_platform,
                                             post_author,
                                             [post_similarity_to_source],
                                             post_has_media,
                                             [post_upvote_ratio],
                                             depths,
                                             # predictions,
                                             estimates
                                    ))
                             .astype(np.float32))


    def build_datasets(self,
                       train_instances: Iterable[VerifInstance],
                       dev_instances: Optional[Iterable[VerifInstance]],
                       test_instances: Optional[Iterable[VerifInstance]],
                       sdqc_estimates: Dict[str, Tuple[SdqcInstance.Label,
                                                       Dict[SdqcInstance.Label,
                                                            float]]]) \
            -> ('Verif.Dataset',
                Optional['Verif.Dataset'],
                Optional['Verif.Dataset']):
        print('Number of instances: train={:d}, dev={:d}, test={:d}'
              .format(len(train_instances),
                      len(dev_instances or []),
                      len(test_instances or [])))

        train_dataset = self.Dataset(
            train_instances, self._posts, self._post_embeddings,
            sdqc_estimates, self._hparams, self._device)

        dev_dataset = None
        if dev_instances:
            dev_dataset = self.Dataset(
                dev_instances, self._posts, self._post_embeddings,
                sdqc_estimates, self._hparams, self._device)

        test_dataset = None
        if test_instances:
            test_dataset = self.Dataset(
                test_instances, self._posts, self._post_embeddings,
                sdqc_estimates, self._hparams, self._device)

        def filter_func(post_id: str) -> bool:
            return self._posts[post_id].platform == Post.Platform.twitter

        for index in self._hparams.input_scaling_features:
            min, max, mean, std = \
                train_dataset.calc_stats_for_aux_feature(index, filter_func)

            for dataset in (train_dataset, dev_dataset, test_dataset):
                if not dataset:
                    continue

                if self._hparams.input_scaling_mode == ScalingMode.none:
                    pass
                elif self._hparams.input_scaling_mode == ScalingMode.min_max:
                    dataset.min_max_scale_aux_feature(
                        index, min, max, filter_func)
                elif self._hparams.input_scaling_mode == ScalingMode.standard:
                    dataset.standard_scale_aux_feature(
                        index, mean, std, filter_func)
                else:
                    raise ValueError('Unimplemented enum variant.')

        return train_dataset, dev_dataset, test_dataset

    class Model(nn.Module):
        def __init__(self, hparams: 'Verif.Hyperparameters'):
            super().__init__()
            self._hparams = hparams

            # -- dense layers --------------------------------------------------
            dense_num_input_dims = self._hparams.input_num_dims
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
            else:
                linear_num_input_dims = self._hparams.input_num_dims
            self._linear = nn.Linear(
                in_features=linear_num_input_dims,
                out_features=len(VerifInstance.Label))

        def forward(self, features):
            x = features

            for dense in self._dense_layers:
                x = F.dropout(F.relu(dense(x)),
                              p=self._hparams.dense_dropout,
                              training=self.training)

            logits = self._linear(x)

            return logits

    def train(self,
              train_dataset: 'Verif.Dataset',
              dev_dataset: Optional['Verif.Dataset'] = None,
              print_progress: bool = False) -> 'Verif.Model':
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
            losses, labels, predictions, prediction_probs = [], [], [], []

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
                batch_logits = model(batch['features'])
                with torch.no_grad():
                    batch_prediction_prob, batch_prediction = \
                        F.softmax(batch_logits, dim=1).max(dim=1)

                loss = criterion(batch_logits, batch['label'])
                loss.backward()
                optimizer.step()

                losses.append(loss.item())
                labels.append(batch['label'].data.cpu().numpy())
                predictions.append(batch_prediction.data.cpu().numpy())
                prediction_probs.append(
                    batch_prediction_prob.data.cpu().numpy())

                if progress_bar:
                    progress_bar.set_postfix({
                        'loss': '{:.2e}'.format(loss.item()),
                    })
                    progress_bar.update(1)

            if progress_bar:
                progress_bar.close()

                labels = np.concatenate(labels)
                predictions = np.concatenate(predictions)
                prediction_probs = np.concatenate(prediction_probs)

                confidences = np.maximum(0.5, prediction_probs)
                confidences[
                    predictions == VerifInstance.Label.unverified.value] = 0

                epoch_loss = np.mean(losses)
                epoch_acc = accuracy_score(labels, predictions)
                epoch_f1 = f1_score(labels, predictions, average='macro')
                epoch_rmse = rmse_score(labels, predictions, confidences)

                print('  Loss={:.2e}  Accuracy={:.2%}  F1-score={:.2%}  '
                      'RMSE={:.4f}'.format(epoch_loss, epoch_acc, epoch_f1,
                                           epoch_rmse))

            if print_progress and dev_dataset and \
                    (epoch_no == self._hparams.num_epochs
                     or not epoch_no % EVAL_DEV_EVERY_N_EPOCH):
                dev_acc, dev_f1, dev_rmse, _ = self.eval(model, dev_dataset)
                print('  Validation:    Accuracy={:.2%}  F1-score={:.2%}  '
                      'RMSE={:.4f}'.format(dev_acc, dev_f1, dev_rmse))

        return model

    def eval(self, model: 'Verif.Model', dataset: 'Verif.Dataset') \
            -> (float, float, float, Dict[str, Dict[str, float]]):
        labels, predictions, prediction_probs = [], [], []

        with torch.no_grad():
            data_loader = DataLoader(
                dataset, batch_size=self._hparams.batch_size)
            for batch in data_loader:
                model.eval()
                batch_logits = model(batch['features'])
                batch_prediction_probs, batch_predictions = \
                    F.softmax(batch_logits, dim=1).max(dim=1)

                labels.append(batch['label'].data.cpu().numpy())
                predictions.append(batch_predictions.data.cpu().numpy())
                prediction_probs.append(
                    batch_prediction_probs.data.cpu().numpy())

        labels = np.concatenate(labels)
        predictions = np.concatenate(predictions)
        prediction_probs = np.concatenate(prediction_probs)

        confidences = np.maximum(0.5, prediction_probs)
        confidences[predictions == VerifInstance.Label.unverified.value] = 0

        acc = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions, average='macro')
        rmse = rmse_score(labels, predictions, confidences)
        report = classification_report(
            labels, predictions, output_dict=True,
            labels=range(len(VerifInstance.Label)),
            target_names=[label.name for label in VerifInstance.Label])

        return acc, f1, rmse, report

    def predict(self,
                model: 'Verif.Model',
                post_ids: Iterable[str],
                sdqc_estimates: Dict[str, Tuple[SdqcInstance.Label,
                                                Dict[SdqcInstance.Label,
                                                     float]]]) \
            -> Dict[str, Tuple[VerifInstance.Label, float]]:
        instances = [VerifInstance(post_id) for post_id in post_ids]
        dataset = self.Dataset(
            instances, self._posts, self._post_embeddings, sdqc_estimates,
            self._hparams, self._device)

        results = {}
        with torch.no_grad():
            data_loader = DataLoader(dataset,
                                     batch_size=self._hparams.batch_size)
            for batch in data_loader:
                model.eval()
                batch_logits = model(batch['features'])
                batch_prediction_probs, batch_predictions = \
                    F.softmax(batch_logits, dim=1).max(dim=1)

                batch_predictions = batch_predictions.data.cpu().numpy()
                batch_prediction_probs = \
                    batch_prediction_probs.data.cpu().numpy()

                batch_confidences = np.maximum(0.5, batch_prediction_probs)
                batch_confidences[batch_predictions
                                  == VerifInstance.Label.unverified.value] = 0
                batch_predictions[batch_predictions
                                  == VerifInstance.Label.unverified.value] = \
                    VerifInstance.Label.true.value

                for post_id, prediction, confidence in zip(
                        batch['post_id'], batch_predictions, batch_confidences):
                    results[post_id] = (VerifInstance.Label(prediction.item()),
                                        confidence.item())

        return results
