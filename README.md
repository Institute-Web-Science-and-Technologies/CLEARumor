# CLEARumor: ConvoLving ELMo Against Rumors @&nbsp;RumorEval 2019

This repository contains the code used in the following paper:

[Ipek Baris, Lukas Schmelzeisen, and Steffen Staab (2019). “CLEARumor at
SemEval-2019 Task 7: ConvoLving ELMo Against Rumors”. In: SemEval@NAACL-HLT.
Association for Computational Linguistics,
pp. 1105–1109.](paper.pdf)
[[ACL Anthology]](https://www.aclweb.org/anthology/papers/S/S19/S19-2193/)
[[arXiv]](https://arxiv.org/abs/1904.03084)
[[Poster]](poster.pdf)

If you use this in your work, please cite as:

```text
@inproceedings{DBLP:conf/semeval/BarisSS19,
  author    = {Ipek Baris and
               Lukas Schmelzeisen and
               Steffen Staab},
  title     = {{CLEAR}umor at {S}em{E}val-2019 Task 7: {C}onvo{L}ving {ELM}o
               {A}gainst {R}umors},
  booktitle = {SemEval@NAACL-HLT},
  pages     = {1105--1109},
  publisher = {Association for Computational Linguistics},
  year      = {2019},
  url       = {https://www.aclweb.org/anthology/S19-2193},
}
```

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

* Lukas Schmelzeisen, [lukas@uni-koblenz.de](mailto:lukas@uni-koblenz.de), [@lschmelzeisen](https://twitter.com/lschmelzeisen)
* Ipek Baris, [ibaris@uni-koblenz.de](mailto:ibaris@uni-koblenz.de), [@ipekbrs](https://twitter.com/ipekbrs)
* [Institute for Web Science and Technoloogies](https://west.uni-koblenz.de),
  University of Koblenz-Landau

## License

All code is license under the [Apache License Version 2.0](https://www.apache.org/licenses/LICENSE-2.0).
