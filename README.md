# WeST @ RumorEval 2019

## Data

* Place the ELMo weights and options files
  [`elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5`](https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5)
  and [`elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json`](https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json)
  in the `data/external` subdirectory.
  They can be obtained from the [ELMo website](https://allennlp.org/elmo).
* Place the training, test data, and evaluation files
  `rumoureval-2019-training-data.zip`, `rumoureval-2019-test-data.zip`, and
  `home_scorer_macro.py` in the `data/external/` subdirectory.
  They can be obtained from the [RumorEval 2019 competetion
  page](https://competitions.codalab.org/competitions/19938).
* If you also want to evaluate on the final evaluation data from RumorEval,
  place the `final-eval-key.json` file in the `data/external` subdirectory.
  So far, this file has only been shared privately with all participants in the
  competition. It is under an embargo for a couple of months, but will be
  released publicly later.
