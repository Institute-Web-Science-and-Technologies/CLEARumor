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

import json
from collections import OrderedDict, defaultdict
from enum import Enum
from itertools import chain
from math import sqrt
from pathlib import Path
from pprint import pprint
from random import shuffle
from sys import maxsize
from time import time
from typing import Callable, Dict, Iterable, List, Optional, Set, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from allennlp.modules.elmo import Elmo, batch_to_ids
from torch.utils.data import Dataset

from src.dataset import ELMO_OPTIONS_FILE, ELMO_WEIGHTS_FILE, Post, \
    SdqcInstance, VerifInstance


class ScalingMode(Enum):
    none = 0
    min_max = 1
    standard = 2


class DatasetHelper(Dataset):
    def __init__(self, post_embeddings: Dict[str, torch.tensor]):
        super().__init__()
        self._dataset = []
        self._post_embeddings = post_embeddings

    def calc_shared_features(self, post: Post) \
            -> (np.ndarray, np.ndarray, np.ndarray):
        post_platform = [post.platform == Post.Platform.twitter,
                         post.platform == Post.Platform.reddit]

        post_author = [0, 0, 0, 0, 0]
        if post.platform == Post.Platform.twitter:
            post_author = [post.user_verified,
                           not post.user_verified,
                           post.followers_count,
                           post.friends_count,
                           post.followers_count / (post.friends_count + 1e-8)]

        post_similarity_to_source = np.array(1)
        if not post.has_source_depth:
            post_emb_mean = self._post_embeddings[post.id].mean(dim=1)
            source_emb_mean = self._post_embeddings[post.source_id].mean(dim=1)
            post_similarity_to_source = F.cosine_similarity(
                post_emb_mean, source_emb_mean, dim=0).cpu().numpy()

        return post_platform, post_author, post_similarity_to_source

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return self._dataset[index]

    def calc_stats_for_aux_feature(self,
                                   index: int,
                                   filter_func: Optional[
                                       Callable[[str], bool]] = None) \
            -> (float, float, float, float):
        if not filter_func:
            def filter_func(_post_id: str) -> bool:
                return True

        feature_values = np.array([post['features'][index].item()
                                   for post in self._dataset
                                   if filter_func(post['post_id'])])
        return (feature_values.min(),
                feature_values.max(),
                feature_values.mean(),
                feature_values.std())

    def min_max_scale_aux_feature(self,
                                  index: int,
                                  min: float,
                                  max: float,
                                  filter_func: Optional[
                                      Callable[[str], bool]] = None) \
            -> None:
        if not filter_func:
            def filter_func(_post_id: str) -> bool:
                return True

        for post in self._dataset:
            if filter_func(post['post_id']):
                value = post['features'][index]
                post['features'][index] = (value - min) / (max - min)

    def standard_scale_aux_feature(self,
                                   index: int,
                                   mean: float,
                                   std: float,
                                   filter_func: Optional[
                                       Callable[[str], bool]] = None) \
            -> None:
        if not filter_func:
            def filter_func(_post_id: str) -> bool:
                return True

        for post in self._dataset:
            if filter_func(post['post_id']):
                value = post['features'][index]
                post['features'][index] = (value - mean) / std


def calculate_post_elmo_embeddings(posts: Dict[str, Post],
                                   max_sentence_length: int,
                                   batch_size: int,
                                   scalar_mix_parameters: List[float],
                                   device: torch.device) \
        -> Dict[str, torch.Tensor]:
    """Calculate ELMo embeddings of all posts in the dataset.

    Calculating these embeddings one time before training the actual models
    allows for extremely fast training later. The downsides are that we can't
    propagate gradients through the embeddings, but fine-tuning these would
    probably lead to be overfitting, since our dataset is very small.
    Additionally, we also can't learn the scalar_mix_parameters, but since
    training is so much faster, adjusting these by hand should be sufficient.

    Since we are going to load the entire dataset into GPU memory later anyways,
    we keep the embeddings in GPU memory here already.

    Args:
        posts: A dictionary mapping post IDs to their respective posts. Load
            this with `src.dataset.load_posts()`.
        max_sentence_length: Number of tokens after which sentences will be
            truncated.
        batch_size: Batch size for calculating the ELMo embeddings.
        scalar_mix_parameters: Parameters for mixing the different ELMo layers.
            See the paper for details on this.
        device: Device to execute on.

    Returns:
        A dictionary mapping post IDs to their respective ELMo embedding in a
        PyTorch tensor. Each tensor will have shape
        `(num_elmo_dimensions, max_sentence_length)`.
    """

    print('Calculating post embeddings...')
    time_before = time()

    elmo = Elmo(ELMO_OPTIONS_FILE,
                ELMO_WEIGHTS_FILE,
                num_output_representations=1,
                dropout=0,
                requires_grad=False,
                do_layer_norm=False,
                scalar_mix_parameters=scalar_mix_parameters).to(device)
    elmo.eval()

    post_embeddings = {}
    batch_ids = []
    # Add a dummy sentence with max_sentence_length to each batch to enforce
    # that each batch of embeddings has the same shape. `batch_to_id()` and
    # `elmo()` take care of zero padding shorter sentences for us.
    batch_texts = [['' for _ in range(max_sentence_length)]]
    for i, post in enumerate(posts.values()):
        batch_ids.append(post.id)
        batch_texts.append(post.text[:max_sentence_length])

        if not i % batch_size or i == len(posts) - 1:
            batch_character_ids = batch_to_ids(batch_texts).to(device)
            batch_texts = [['' for _ in range(max_sentence_length)]]

            # - [0] to select first output representation (there is only one
            #   because of `num_output_representations=1` at `elmo` creation.
            # - [1:] to ignore dummy sentence added at the start.
            batch_embeddings = \
                elmo(batch_character_ids)['elmo_representations'][0][1:]
            batch_embeddings = batch_embeddings.split(split_size=1, dim=0)
            del batch_character_ids  # Free up memory sooner.

            for post_id, post_embedding in zip(batch_ids, batch_embeddings):
                post_embedding.squeeze_(dim=0)
                post_embedding.transpose_(0, 1)
                post_embeddings[post_id] = post_embedding
            batch_ids = []

    time_after = time()
    print('  Took {:.2f}s.'.format(time_after - time_before))

    return post_embeddings


def generate_folds_for_k_fold_cross_validation(posts: Dict[str, Post],
                                               num_folds: int) \
        -> List[Set[str]]:
    posts_per_discriminator = defaultdict(set)
    for post in posts.values():
        if post.platform == Post.Platform.twitter:
            discriminator = post.topic
        elif post.platform == Post.Platform.reddit:
            discriminator = post.source_id
        else:
            raise ValueError('Unimplemented enum variant.')
        posts_per_discriminator[discriminator].add(post.id)
    posts_per_discriminator = list(posts_per_discriminator.values())
    shuffle(posts_per_discriminator)

    folds = [set() for _ in range(num_folds)]
    for post_ids in posts_per_discriminator:
        # Find fold with fewest elements
        index = None
        num_elements = maxsize
        for i, fold in enumerate(folds):
            if num_elements > len(fold):
                num_elements = len(fold)
                index = i

        # Add post to that fold
        folds[index].update(post_ids)

    return folds


def arrange_folds_for_k_fold_cross_validation(folds: List[Set[str]],
                                              index: int) \
        -> (Set[str], Set[str]):
    train_post_ids = set(chain.from_iterable(
        fold for i, fold in enumerate(folds) if i != index))
    test_post_ids = folds[index]
    return train_post_ids, test_post_ids


def filter_instances(train_post_ids: Set[str],
                     test_post_ids: Set[str],
                     instances: Iterable[Union[SdqcInstance, VerifInstance]]) \
        -> (List[Union[SdqcInstance, VerifInstance]],
            List[Union[SdqcInstance, VerifInstance]]):
    train_instances = [i for i in instances if i.post_id in train_post_ids]
    test_instances = [i for i in instances if i.post_id in test_post_ids]

    shuffle(train_instances)
    shuffle(test_instances)

    return train_instances, test_instances


def rmse_score(labels, predictions, confidences):
    rmse = 0
    for label, prediction, confidence in \
            zip(labels, predictions, confidences):
        if label == prediction and \
                (label == VerifInstance.Label.true.value
                 or label == VerifInstance.Label.false.value):
            rmse += (1 - confidence) ** 2
        elif label == VerifInstance.Label.unverified.value:
            rmse += confidence ** 2
        else:
            rmse += 1
    rmse = sqrt(rmse / len(labels))
    return rmse


def display_results(sdqc_accs: Iterable[float],
                    sdqc_f1s: Iterable[float],
                    sdqc_reports: Iterable[Dict[str, Dict[str, float]]],
                    verif_accs: Iterable[float],
                    verif_f1s: Iterable[float],
                    verif_rmses: Iterable[float],
                    verif_reports: Iterable[Dict[str, Dict[str, float]]]):
    def display_report(reports: Iterable[Dict[str, Dict[str, float]]]):
        report_lists = defaultdict(lambda: defaultdict(list))
        for report in reports:
            for outer_key, inner_report in report.items():
                if outer_key == 'accuracy':
                    report_lists[outer_key]['accuracy'].append(inner_report)
                else:
                    for inner_key, value in inner_report.items():
                        report_lists[outer_key][inner_key].append(value)

        report_stats = {}
        for outer_key, inner_report in report_lists.items():
            report_stats[outer_key] = {}
            for inner_key, values in inner_report.items():
                report_stats[outer_key][inner_key] = '{:.1%}±{:.1%}'.format(
                    np.mean(values), np.std(values))

        pprint(report_stats)

    sdqc_acc = (np.mean(sdqc_accs), np.std(sdqc_accs))
    sdqc_f1 = (np.mean(sdqc_f1s), np.std(sdqc_f1s))
    print('Task A: SDQC')
    print('  Accuracy: {:.1%}±{:.1%}'
          '  F1-Score: {:.1%}±{:.1%}'
          .format(sdqc_acc[0], sdqc_acc[1],
                  sdqc_f1[0], sdqc_f1[1]))
    display_report(sdqc_reports)

    verif_acc = (np.mean(verif_accs), np.std(verif_accs))
    verif_f1 = (np.mean(verif_f1s), np.std(verif_f1s))
    verif_rmse = (np.mean(verif_rmses), np.std(verif_rmses))
    print('Task B: Verification')
    print('  Accuracy: {:.1%}±{:.1%}'
          '  F1-Score: {:.1%}±{:.1%}'
          '  RMSE: {:.3f}±{:.3f}'
          .format(verif_acc[0], verif_acc[1],
                  verif_f1[0], verif_f1[1],
                  verif_rmse[0], verif_rmse[1]))
    display_report(verif_reports)


def write_answers_json(
        path: Path,
        sdqc_instances: List[SdqcInstance],
        verif_instances: List[SdqcInstance],
        sdqc_estimates: Dict[str, Tuple[SdqcInstance.Label,
                                        Dict[SdqcInstance.Label, float]]],
        verif_estimates: Dict[str, Tuple[VerifInstance.Label, float]]):
    sdqc_answers = OrderedDict()
    for instance in sdqc_instances:
        answer = sdqc_estimates[instance.post_id]
        sdqc_answers[instance.post_id] = answer[0].name

    verif_answers = OrderedDict()
    for instance in verif_instances:
        answer = verif_estimates[instance.post_id]
        verif_answers[instance.post_id] = (answer[0].name, answer[1])

    answers = OrderedDict()
    answers['subtaskaenglish'] = sdqc_answers
    answers['subtaskbenglish'] = verif_answers

    with path.open('w', encoding='UTF-8') as fout:
        json.dump(answers, fout, indent=2)
