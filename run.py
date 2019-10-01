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

from datetime import datetime
from itertools import chain
from pathlib import Path
from time import time
from warnings import filterwarnings

import numpy as np
import torch
from sklearn.exceptions import UndefinedMetricWarning

from src.dataset import check_for_required_external_data_files, load_posts, \
    load_sdcq_instances, load_verif_instances
from src.sdqc import Sdqc
from src.util import ScalingMode, arrange_folds_for_k_fold_cross_validation, \
    calculate_post_elmo_embeddings, display_results, filter_instances, \
    generate_folds_for_k_fold_cross_validation, write_answers_json
from src.verif import Verif

time_before = time()

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.fastest = True

filterwarnings('ignore', category=UndefinedMetricWarning)

NUM_ORGA_REPETITIONS = 10
NUM_CV_REPETITIONS = 1
NUM_CV_FOLDS = 10

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
    num_epochs=100,
    learning_rate=1e-3,
    weight_decay=1e-2,
    class_weights=[1, 1, 1, 0.2],
    input_num_emb_dims=num_emb_dims,
    input_num_aux_dims=11,
    input_aux_scaling_features=[4, 5, 6],
    input_aux_scaling_mode=ScalingMode.min_max,
    conv_num_layers=1,
    conv_kernel_sizes=[2, 3],
    conv_num_channels=64,
    dense_num_layers=3,
    dense_num_hidden=128,
    dense_dropout=0.5)
sdqc = Sdqc(posts, post_embeddings, sdqc_hparams, device)

verif_hparams = Verif.Hyperparameters(
    batch_size=128,
    num_epochs=5000,
    learning_rate=1e-3,
    weight_decay=1e-2,
    class_weights=[1, 1, 0.3],
    input_num_dims=16,
    input_scaling_features=[4, 5, 6],
    input_scaling_mode=ScalingMode.min_max,
    dense_num_layers=2,
    dense_num_hidden=512,
    dense_dropout=0.25)
verif = Verif(posts, post_embeddings, verif_hparams, device)

sdqc_train_instances, sdqc_dev_instances, sdqc_test_instances = \
    load_sdcq_instances()
sdqc_all_instances = list(chain(
    sdqc_train_instances, sdqc_dev_instances, sdqc_test_instances))
verif_train_instances, verif_dev_instances, verif_test_instances = \
    load_verif_instances()
verif_all_instances = list(chain(
    verif_train_instances, verif_dev_instances, verif_test_instances))

sdqc_times, verif_times = [], []
sdqc_dev_accs, sdqc_dev_f1s, sdqc_dev_reports = [], [], []
sdqc_test_accs, sdqc_test_f1s, sdqc_test_reports = [], [], []
sdqc_cv_accs, sdqc_cv_f1s, sdqc_cv_reports = [], [], []
verif_dev_accs, verif_dev_f1s, verif_dev_rmses, verif_dev_reports = \
    [], [], [], []
verif_test_accs, verif_test_f1s, verif_test_rmses, verif_test_reports = \
    [], [], [], []
verif_cv_accs, verif_cv_f1s, verif_cv_rmses, verif_cv_reports = [], [], [], []

answers_dir = Path('answers') / (datetime.utcnow().isoformat() + 'Z')
answers_dir.mkdir(parents=True, exist_ok=False)

print()
print('-- Organizer Split ----------------------------------------------------')

for repetition_no in range(NUM_ORGA_REPETITIONS):
    print()
    print('## Repetition {}/{}'.format(repetition_no + 1, NUM_ORGA_REPETITIONS))

    print('Task A: SDQC')
    t1 = time()
    sdqc_train_dataset, sdqc_dev_dataset, sdqc_test_dataset = \
        sdqc.build_datasets(sdqc_train_instances,
                            sdqc_dev_instances,
                            sdqc_test_instances)
    sdqc_model = sdqc.train(sdqc_train_dataset,
                            sdqc_dev_dataset,
                            print_progress=False)
    t2 = time()
    sdqc_times.append(t2 - t1)

    sdqc_estimates = sdqc.predict(sdqc_model, posts.keys())
    if sdqc_dev_dataset:
        acc, f1, report = sdqc.eval(sdqc_model, sdqc_dev_dataset)
        print('Validation:  Accuracy={:.1%}  F1-score={:.1%}'.format(acc, f1))
        sdqc_dev_accs.append(acc)
        sdqc_dev_f1s.append(f1)
        sdqc_dev_reports.append(report)
    if sdqc_test_dataset:
        acc, f1, report = sdqc.eval(sdqc_model, sdqc_test_dataset)
        print('Test:        Accuracy={:.1%}  F1-score={:.1%}'.format(acc, f1))
        sdqc_test_accs.append(acc)
        sdqc_test_f1s.append(f1)
        sdqc_test_reports.append(report)

    # model_path = 'data/sdqc_model_{}.pth'.format(repetition_no)
    # torch.save(sdqc_model.state_dict(),model_path)

    print('Task B: Verification')
    t1 = time()
    verif_train_dataset, verif_dev_dataset, verif_test_dataset = \
        verif.build_datasets(verif_train_instances,
                             verif_dev_instances,
                             verif_test_instances,
                             sdqc_estimates)
    verif_model = verif.train(verif_train_dataset,
                              verif_dev_dataset,
                              print_progress=False)
    t2 = time()
    verif_times.append(t2 - t1)

    verif_estimates = verif.predict(
        verif_model,
        [post.id for post in posts.values() if post.has_source_depth],
        sdqc_estimates)
    if verif_dev_dataset:
        acc, f1, rmse, report = verif.eval(verif_model, verif_dev_dataset)
        print('Validation:  Accuracy={:.1%}  F1-score={:.1%}  RMSE={:.3f}'
              .format(acc, f1, rmse))
        verif_dev_accs.append(acc)
        verif_dev_f1s.append(f1)
        verif_dev_rmses.append(rmse)
        verif_dev_reports.append(report)
    if verif_test_dataset:
        acc, f1, rmse, report = verif.eval(verif_model, verif_test_dataset)
        print('Test:        Accuracy={:.1%}  F1-score={:.1%}  RMSE={:.3f}'
              .format(acc, f1, rmse))
        verif_test_accs.append(acc)
        verif_test_f1s.append(f1)
        verif_test_rmses.append(rmse)
        verif_test_reports.append(report)

        # model_path = 'data/verif_model_{}.pth'.format(repetition_no)
        # torch.save(verif_model.state_dict(),model_path)

    write_answers_json(
        answers_dir / 'answers.organizers_rep{}_train.json'.format(
            repetition_no),
        sdqc_train_instances, verif_train_instances,
        sdqc_estimates, verif_estimates)
    if sdqc_dev_instances and verif_dev_instances:
        write_answers_json(
            answers_dir / 'answers.organizers_rep{}_dev.json'.format(
                repetition_no),
            sdqc_dev_instances, verif_dev_instances,
            sdqc_estimates, verif_estimates)
    if sdqc_test_instances and verif_test_instances:
        write_answers_json(
            answers_dir / 'answers.organizers_rep{}_test.json'.format(
                repetition_no),
            sdqc_test_instances, verif_test_instances,
            sdqc_estimates, verif_estimates)

print()
print('-- k-fold Cross Validation --------------------------------------------')

for repetition_no in range(NUM_CV_REPETITIONS):
    print()
    print('## Repetition {}/{}'.format(repetition_no + 1, NUM_CV_REPETITIONS))

    folds = generate_folds_for_k_fold_cross_validation(posts, NUM_CV_FOLDS)
    for i in range(NUM_CV_FOLDS):
        print()
        print('# Cross Validation {}/{}'.format(i + 1, NUM_CV_FOLDS))

        train_post_ids, test_post_ids = \
            arrange_folds_for_k_fold_cross_validation(folds, i)

        print('Task A: SDQC')
        sdqc_train_instances, sdqc_test_instances = \
            filter_instances(train_post_ids, test_post_ids, sdqc_all_instances)
        sdqc_train_dataset, _, sdqc_test_dataset = \
            sdqc.build_datasets(sdqc_train_instances, None, sdqc_test_instances)
        sdqc_model = sdqc.train(sdqc_train_dataset, print_progress=False)
        sdqc_estimates = sdqc.predict(sdqc_model, posts.keys())
        acc, f1, report = sdqc.eval(sdqc_model, sdqc_test_dataset)
        print('Test:        Accuracy={:.1%}  F1-score={:.1%}'
              .format(acc, f1))
        sdqc_cv_accs.append(acc)
        sdqc_cv_f1s.append(f1)
        sdqc_cv_reports.append(report)

        print('Task B: Verification')
        verif_train_instances, verif_test_instances = \
            filter_instances(train_post_ids, test_post_ids, verif_all_instances)
        verif_train_dataset, _, verif_test_dataset = \
            verif.build_datasets(verif_train_instances, None,
                                 verif_test_instances,
                                 sdqc_estimates)
        verif_model = verif.train(verif_train_dataset, print_progress=False)
        verif_estimates = verif.predict(
            verif_model,
            [post.id for post in posts.values() if post.has_source_depth],
            sdqc_estimates)
        acc, f1, rmse, report = verif.eval(verif_model, verif_test_dataset)
        print('Test:        Accuracy={:.1%}  F1-score={:.1%}  RMSE={:.3f}'
              .format(acc, f1, rmse))
        verif_cv_accs.append(acc)
        verif_cv_f1s.append(f1)
        verif_cv_rmses.append(rmse)
        verif_cv_reports.append(report)

        write_answers_json(
            answers_dir / 'answers.kfold_rep{}_cv{}_train.json'.format(
                repetition_no, i),
            sdqc_train_instances, verif_train_instances,
            sdqc_estimates, verif_estimates)
        write_answers_json(
            answers_dir / 'answers.kfold_rep{}_cv{}_test.json'.format(
                repetition_no, i),
            sdqc_test_instances, verif_test_instances,
            sdqc_estimates, verif_estimates)

print()
print('-- Results ------------------------------------------------------------')

print()
print('# Validation')
display_results(
    sdqc_dev_accs, sdqc_dev_f1s, sdqc_dev_reports,
    verif_dev_accs, verif_dev_f1s, verif_dev_rmses, verif_dev_reports)

print()
print('# Test')
display_results(
    sdqc_test_accs, sdqc_test_f1s, sdqc_test_reports,
    verif_test_accs, verif_test_f1s, verif_test_rmses, verif_test_reports)

print()
print('# Cross-Validation')
display_results(
    sdqc_cv_accs, sdqc_cv_f1s, sdqc_cv_reports,
    verif_cv_accs, verif_cv_f1s, verif_cv_rmses, verif_cv_reports)

print()
print('# Runtime')
print('SDQC:  {:.2}±{:.2}s'.format(np.mean(sdqc_times), np.std(sdqc_times)))
print('Verif: {:.2}±{:.2}s'.format(np.mean(verif_times), np.std(verif_times)))

print()
time_after = time()
print('Program ran for {:.2f}s in total'.format(time_after - time_before))
