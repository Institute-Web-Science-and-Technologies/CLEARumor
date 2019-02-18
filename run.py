#!/usr/bin/env python3
from warnings import filterwarnings

import torch
from sklearn.exceptions import UndefinedMetricWarning

from src.dataset import check_for_required_external_data_files, load_posts
from src.sdqc import Sdqc
from src.util import calculate_post_elmo_embeddings
from src.verif import Verif

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
num_embedding_dimensions = next(iter(post_embeddings.values())).shape[0]

sdqc_hparams = Sdqc.Hyperparameters(
    max_sentence_length=32,
    batch_size=512,
    num_epochs=100,
    learning_rate=0.001,
    num_input_dimensions=num_embedding_dimensions,
    conv_num_layers=2,
    conv_kernel_sizes=[2, 3],
    conv_num_channels=64,
    dense_num_layers=3,
    dense_num_hidden=128,
    dense_dropout=0.5)
sdqc = Sdqc(posts, post_embeddings, sdqc_hparams, device)
sdqc_results = sdqc.train()

verif_hparams = Verif.Hyperparameters(
    max_sentence_length=32, batch_size=256, num_epochs=10, learning_rate=0.01)
verif = Verif(posts, post_embeddings, verif_hparams, device)
