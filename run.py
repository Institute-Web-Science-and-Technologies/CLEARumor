#!/usr/bin/env python3
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
from warnings import filterwarnings

import torch
from sklearn.exceptions import UndefinedMetricWarning

from src.dataset import check_for_required_external_data_files, load_posts
from src.sdqc import Sdqc
from src.util import calculate_post_elmo_embeddings
from src.verif import Verif

time_before = time()

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.fastest = True

filterwarnings('ignore', category=UndefinedMetricWarning)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

check_for_required_external_data_files()

posts = load_posts()
post_embeddings = calculate_post_elmo_embeddings(
    posts,
    max_sentence_length=32,
    batch_size=512,
    scalar_mix_parameters=[0, 0, 0],
    device=device)
num_emb_dims = next(iter(post_embeddings.values())).shape[0]

sdqc_hparams = Sdqc.Hyperparameters(
    max_sentence_length=32,
    batch_size=512,
    num_epochs=50,
    learning_rate=1e-3,
    class_weights=[1, 1, 1, 0.2],
    input_num_emb_dims=num_emb_dims,
    input_num_aux_dims=11,
    input_aux_scaling_features=[7, 8, 9],
    input_aux_scaling_mode=Sdqc.Hyperparameters.ScalingMode.min_max,
    conv_num_layers=1,
    conv_kernel_sizes=[2, 3],
    conv_num_channels=64,
    dense_num_layers=3,
    dense_num_hidden=128,
    dense_dropout=0.5)
sdqc = Sdqc(posts, post_embeddings, sdqc_hparams, device)
sdqc.train()
sdqc_estimates = sdqc.predict(posts.keys())

verif_hparams = Verif.Hyperparameters(
    max_sentence_length=32, batch_size=256, num_epochs=10, learning_rate=0.01)
verif = Verif(posts, post_embeddings, verif_hparams, device)

time_after = time()
print('Program ran for {:.2f}s in total'.format(time_after - time_before))
