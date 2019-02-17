from collections.__init__ import OrderedDict
from itertools import chain
from time import time
from typing import Dict, List

import numpy as np
import torch
from allennlp.modules import Elmo
from sklearn.metrics import accuracy_score, f1_score
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import ELMO_OPTIONS_FILE, ELMO_WEIGHTS_FILE, Post, \
    SdqcInstance, load_sdcq_instances
from src.util import sentence_to_tensor

EVAL_DEV_EVERY_N_EPOCH = 5


class Sdqc:
    class Hyperparameters:
        def __init__(self,
                     max_sentence_length: int,
                     batch_size: int,
                     num_epochs: int,
                     learning_rate: float):
            self.max_sentence_length = max_sentence_length
            self.batch_size = batch_size
            self.num_epochs = num_epochs
            self.learning_rate = learning_rate

    def __init__(self,
                 posts: Dict[str, Post],
                 hparams: 'Sdqc.Hyperparameters',
                 device: torch.device):
        self._posts = posts
        self._hparams = hparams
        self._device = device
        self.model = None

        self._load_data()

    class Dataset(torch.utils.data.Dataset):
        def __init__(self,
                     instances: List[SdqcInstance],
                     posts: Dict[str, Post],
                     hparams: 'Sdqc.Hyperparameters',
                     device: torch.device):
            self._dataset: List[Dict[str, torch.Tensor]] = []
            for instance in instances:
                post = posts[instance.post_id]
                self._dataset.append({
                    'post_id': post.id,
                    'text': sentence_to_tensor(
                        post.text, hparams.max_sentence_length).to(device),
                    'label': torch.tensor(instance.label.value, device=device),
                })

        def __len__(self) -> int:
            return len(self._dataset)

        def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
            return self._dataset[index]

    def _load_data(self):
        print()
        print('Loading SDQC data...')
        time_before = time()
        train, dev, test = load_sdcq_instances()
        print('  Number of instances: train={:d}, dev={:d}, test={:d}'.format(
            len(train), len(dev), len(test) if test else 0))

        self._train_data = self.Dataset(
            train, self._posts, self._hparams, self._device)
        self._dev_data = self.Dataset(
            dev, self._posts, self._hparams, self._device)
        self._test_data = self.Dataset(
            test, self._posts, self._hparams, self._device)
        time_after = time()
        print('  Took {:.2f}s.'.format(time_after - time_before))

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.elmo = Elmo(
                ELMO_OPTIONS_FILE, ELMO_WEIGHTS_FILE,
                num_output_representations=1,
                dropout=0.5, requires_grad=False, do_layer_norm=False,
                scalar_mix_parameters=[1 / 3, 1 / 3, 1 / 3])
            self.linear = nn.Sequential(
                nn.Linear(256, 100),
                nn.ReLU(),
                nn.Linear(100, 4),
            )

        def forward(self, text):
            # print('-- text')
            # print(text)
            # print(text.shape)

            elmo = self.elmo(text)
            # print('-- elmo representations')
            # print(elmo['elmo_representations'][0])
            # print(elmo['elmo_representations'][0].shape)
            # print('-- sequence length')
            # print(elmo['mask'])
            # print(elmo['mask'].shape)

            mean = elmo['elmo_representations'][0].mean(dim=1)
            # print('-- mean')
            # print(mean)
            # print(mean.shape)

            logits = self.linear(mean)
            # print('-- logits')
            # print(logits)
            # print(logits.shape)
            return logits

    def train(self) -> Dict[str, str]:
        print('Training SDQC model...')
        self.model = self.Model().to(self._device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            self.model.parameters(), lr=self._hparams.learning_rate)

        train_loader = DataLoader(
            self._train_data, batch_size=self._hparams.batch_size, shuffle=True)

        for epoch_no in range(1, self._hparams.num_epochs + 1):
            losses, labels, predictions = [], [], []

            with tqdm(total=(len(train_loader)), unit='batch',
                      desc='Epoch: {:{}d}/{:d}'.format(
                          epoch_no, len(str(self._hparams.num_epochs)),
                          self._hparams.num_epochs)) \
                    as progress_bar:
                for batch_no, batch in enumerate(train_loader):
                    optimizer.zero_grad()

                    self.model.train()
                    batch_logits = self.model(batch['text'])
                    with torch.no_grad():
                        batch_prediction = torch.argmax(batch_logits, dim=1)

                    loss = criterion(batch_logits, batch['label'])
                    loss.backward()
                    optimizer.step()

                    losses.append(loss.item())
                    labels.append(batch['label'].data.cpu().numpy())
                    predictions.append(batch_prediction.data.cpu().numpy())

                    progress_bar.set_postfix({
                        'loss': '{:.2e}'.format(loss.item()),
                    })
                    progress_bar.update(1)

            # print(list(self.model.elmo._scalar_mixes[0].scalar_parameters))

            labels = np.concatenate(labels)
            predictions = np.concatenate(predictions)

            epoch_loss = np.mean(losses)
            epoch_acc = accuracy_score(labels, predictions)
            epoch_f1 = f1_score(labels, predictions, average='macro')
            print('  Loss={:.2e}  Accuracy={:.2%}  F1-score={:.2%}'
                  .format(epoch_loss, epoch_acc, epoch_f1))

            if (epoch_no == self._hparams.num_epochs
                    or not epoch_no % EVAL_DEV_EVERY_N_EPOCH):
                dev_acc, dev_f1, _ = self.eval(self._dev_data)
                print('  Validation:    Accuracy={:.2%}  F1-score={:.2%}'
                      .format(dev_acc, dev_f1))

        test_acc, test_f1, test_results = self.eval(self._test_data)
        print('Test:            Accuracy={:.2%}  F1-score={:.2%}'
              .format(test_acc, test_f1))

        return test_results

    def eval(self, data: 'Sdqc.Dataset') \
            -> (float, float, Dict[str, str]):
        post_ids, labels, predictions = [], [], []

        with torch.no_grad():
            data_loader = DataLoader(data, batch_size=self._hparams.batch_size)
            for batch_no, batch in enumerate(data_loader):
                self.model.eval()
                batch_logits = self.model(batch['text'])
                batch_prediction = torch.argmax(batch_logits, dim=1)

                post_ids.append(batch['post_id'])
                labels.append(batch['label'].data.cpu().numpy())
                predictions.append(batch_prediction.data.cpu().numpy())

        labels = np.concatenate(labels)
        predictions = np.concatenate(predictions)

        acc = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions, average='macro')

        results = OrderedDict()
        for post_id, prediction in zip(chain.from_iterable(post_ids),
                                       predictions):
            results[post_id] = SdqcInstance.Label(prediction).name

        return acc, f1, results
