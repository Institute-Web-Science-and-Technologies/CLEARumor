from typing import List

import torch
from allennlp.modules.elmo import batch_to_ids
from torch.nn import functional as F

ELMO_INPUT_CHARACTERS_PER_TOKEN = batch_to_ids([['']]).shape[2]
ELMO_INPUT_DTYPE = batch_to_ids([['']]).dtype


def sentence_to_tensor(sentence: List[str],
                       max_sentence_length: int) -> torch.Tensor:
    if not sentence:
        return torch.zeros(
            (max_sentence_length, ELMO_INPUT_CHARACTERS_PER_TOKEN),
            dtype=ELMO_INPUT_DTYPE)

    return F.pad(
        batch_to_ids([(sentence[:max_sentence_length])]).squeeze(dim=0),
        [0, 0, 0, max(0, max_sentence_length - len(sentence))])
