## KorSpacing , pytorch 기반 한국어 띄어쓰기 라이브러리
- 240625, wygo
- torch 기반 띄어쓰기 라이브러리
---------------
python package for korean spacing model by torch.

This project is developed by referring to [KoSpacing](https://github.com/haven-jeon/KoSpacing).

[![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](http://www.gnu.org/licenses/gpl-3.0)


#### Introduction

Word spacing is one of the important parts of the preprocessing of Korean text analysis. Accurate spacing greatly affects the accuracy of subsequent text analysis. `PyKoSpacing` has fairly accurate automatic word spacing performance,especially good for online text originated from SNS or SMS.

For example.

"아버지가방에들어가신다." can be spaced both of below.


1. "아버지가 방에 들어가신다." means  "My father enters the room."
1. "아버지 가방에 들어가신다." means  "My father goes into the bag."

Common sense, the first is the right answer.

`PyKoSpacing` is based on Deep Learning model trained from large corpus(more than 100 million NEWS articles from [Chan-Yub Park](https://github.com/mrchypark)). 


#### Performance

| Test Set  | Accuracy | 
|---|---|
| Sejong(colloquial style) Corpus(1M) | 97.1% |
| OOOO(literary style)  Corpus(3M)   | 94.3% |

- Accuracy = # correctly spaced characters/# characters in the test data.
  - Might be increased performance if normalize compound words. 


#### Install

##### PyPI Install
Pre-requisite:
```bash
proper installation of python3
proper installation of pip

pip install torch

To install from GitHub, use

    pip install git+https://github.com/airobotlab/korspacing


#### Example 

```python
>>> from korspacing import KorSpacing
>>> spacing = KorSpacing()
>>> result = spacing('아버지가방에들어가신다')
>>> print(result)

>>> # Apply rules dictionary
>>> from korspacing import KorSpacing
>>> rules = {
    '아버지 가방에': '아버지가 방에', 
    '아 버지가방': '아버지가 방',     
}
>>> spacing = KorSpacing(rules=rules)
>>> result = spacing('아버지가방에들어가신다')
>>> print(result)
```

Setting rules with json file. (Add rules to './resources/rules.json')


Run on command line(thanks [lqez](https://github.com/lqez)). 

```bash
$ cat test_in.txt
김형호영화시장분석가는'1987'의네이버영화정보네티즌10점평에서언급된단어들을지난해12월27일부터올해1월10일까지통계프로그램R과KoNLP패키지로텍스트마이닝하여분석했다.
아버지가방에들어가신다.
$ python -m korspacing.run_file test_in.txt
김형호 영화시장 분석가는 '1987'의 네이버 영화 정보 네티즌 10점 평에서 언급된 단어들을 지난해 12월 27일부터 올해 1월 10일까지 통계 프로그램 R과 KoNLP 패키지로 텍스트마이닝하여 분석했다.
아버지가 방에 들어가신다.
```

#### Model Architecture

![](kospacing_arch.png)


#### For Training

- Training code uses an architecture that is more advanced than PyKoSpacing, but also contains the learning logic of PyKoSpacing.
  - https://github.com/haven-jeon/Train_KoSpacing

#### Citation

```markdowns
@misc{airobotlab2018,
author = {airobotlab},
title = {korspacing: Automatic Korean word spacing by torch},
publisher = {GitHub},
journal = {GitHub repository},
howpublished = {\url{https://github.com/airobotlab/korspacing}}
```

### Star History

[![Star History Chart](https://api.star-history.com/svg?repos=airobotlab/korspacing&type=Date)](https://star-history.com/#airobotlab/korspacing&Date)

