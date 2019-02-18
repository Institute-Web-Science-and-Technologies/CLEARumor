from time import time
from typing import Dict, List

import torch
from allennlp.modules.elmo import Elmo, batch_to_ids

from src.dataset import ELMO_OPTIONS_FILE, ELMO_WEIGHTS_FILE, Post


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
