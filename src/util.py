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

from enum import Enum
from math import sqrt
from time import time
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from allennlp.modules.elmo import Elmo, batch_to_ids
from torch.utils.data import Dataset

from src.dataset import ELMO_OPTIONS_FILE, ELMO_WEIGHTS_FILE, Post, \
    VerifInstance


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

        post_author = [0, 1, 0, 0, 1]
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
