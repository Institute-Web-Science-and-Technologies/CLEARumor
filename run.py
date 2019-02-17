#!/usr/bin/env python3
from warnings import filterwarnings

import torch
from sklearn.exceptions import UndefinedMetricWarning

from src.dataset import check_for_required_external_data_files, load_posts
from src.sdqc import Sdqc
from src.verif import Verif

torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.fastest = True

filterwarnings('ignore', category=UndefinedMetricWarning)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

check_for_required_external_data_files()

posts = load_posts()

sdqc_hparams = Sdqc.Hyperparameters(
    max_sentence_length=32, batch_size=256, num_epochs=10, learning_rate=0.01)
sdqc = Sdqc(posts, sdqc_hparams, device)
sdqc_results = sdqc.train()

verif_hparams = Verif.Hyperparameters(
    max_sentence_length=32, batch_size=256, num_epochs=10, learning_rate=0.01)
verif = Verif(posts, verif_hparams, device)
