# CLEARumor: ConvoLving ELMo Against Rumors @&nbsp;RumorEval 2019

## Dependencies

Python 3.6+ is required.

* See the [PyTorch Getting Started](https://pytorch.org/get-started) page for
  how to install it.
* Install [AllenNLP](https://allennlp.org/tutorials):
  ```sh
  pip3 install allennlp
  ```
* Install the [tokenizer for Twitter and Reddit posts](https://github.com/erikavaris/tokenizer):
  ```sh
  pip3 install git+https://github.com/erikavaris/tokenizer.git
  ```

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

## Contacts

If you have any questions regarding the code or the employed machine learning
architectures, please, don't hesitate to contact the authors or report an issue.

* Lukas Schmelzeisen, [lukas@uni-koblenz.de](mailto:lukas@uni-koblenz.de)
* Ipek Baris, [ibaris@uni-koblenz.de](mailto:ibaris@uni-koblenz.de)
* [Institute for Web Science and Technoloogies](https://west.uni-koblenz.de),
  University of Koblenz-Landau

## License

All code is license under the [Apache License Version 2.0](https://www.apache.org/licenses/LICENSE-2.0).
