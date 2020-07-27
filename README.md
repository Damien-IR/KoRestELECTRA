# KoRestELECTRA, ELECTRA based on Korean restaurant reviews
맛집 리뷰, 위키피디아, 뉴스 등의 corpus 들을 모아 학습시킨 ELECTRA 모델입니다.

구글의 [ELECTRA](https://github.com/google-research/electra) 와 monologg 님의 [KoELECTRA](https://github.com/monologg/KoELECTRA)를 주로 참고하여 만든 모델이지만,<br>
이 모델은 사용 목적을 **"맛집 리뷰"들과 관련된 태스크에만 적용하는 것으로 한정하고서 만든 모델입니다.**<br>

## How to use
```python
from transformers import ElectraTokenizer, ElectraModel

tokenizer = ElectraTokenizer.from_pretrained("damien-ir/ko-rest-electra-discriminator")
model = ElectraModel.from_pretrained("damien-ir/ko-rest-electra-discriminator")
```

## About Model
일반적인 리뷰나 댓글들을 보면, 굉장히 많은 오탈자나 줄임말 등이 존재하고, 심지어는 띄어쓰기를 아예 쓰지 않은 문장 또한 존재합니다.<br>
이를 모두 처리할 수 있게끔, **대부분 의도적으로 전처리를 하지 않은 corpus로 만든 모델입니다.**<br>
단, 한글과 영어 처리에 집중하기 위해 다수의 모델을 참고하여 한글과 영어, 일부 특수문자 외의 문자들은 corpus에서 제거하였습니다.

또한, 별도의 토크나이저나 형태소 분석기를 사용하지 않았기 때문에, 즉시 transformers 라이브러리를 이용하여 태스크에 적용 가능합니다.<br>
단, 모델의 제작 의도 자체가 맛집 리뷰들을 분석하기 위함이므로,<br>
뉴스, 논문, 특허 등과 같은 다소 **잘 정제된 데이터셋을 사용하는 태스크에 사용할 모델을 찾으신다면, [KoELECTRA](https://github.com/monologg/KoELECTRA) 모델을 사용하시는 것을 권장드립니다.**<br>
**잘 정제되지 않은 문장들의 태스크에 적용할 모델을 찾으신다면, [KcBERT](https://github.com/Beomi/KcBERT) 모델을 사용하시는 것을 권장드립니다.**

KoRestELECTRA 모델 학습에는 24GB, 6200만 줄 가량의 corpus를 사용하였고, 맛집 리뷰들의 용량은 이 중 17.3GB 입니다.<br>
모델의 설정은 vocab 크기를 제외하고, 구글의 [electra-base 모델 설정](https://s3.amazonaws.com/models.huggingface.co/bert/google/electra-base-discriminator/config.json)과 동일합니다.<br>
[vocab](https://cdn.huggingface.co/damien-ir/ko-rest-electra-discriminator/vocab.txt)의 경우 [monologg](https://github.com/monologg)님의 [KoELECTRA](https://github.com/monologg/KoELECTRA) 모델을 많이 참고하였습니다.

corpus에 사용한 리뷰들은 다양한 사이트에서 다양한 맛집 포스팅들, 댓글, 리뷰들을 모은 것이고, 일부는 긴 내용의 포스팅을 포함하고 있는 경우 또한 많습니다.<br>
또한 각 문장을 맥락 없이 줄 나눔을 하거나, 중간에 이미지를 넣은 후 다른 이야기를 하는 경우도 많습니다.<br>
이를 kss 등을 사용하여 줄을 분리하고 모델을 만들어 보았으나, 눈으로 보기에도 corpus 품질이 좋지 않았고, 개인적으로 가지고 있는 맛집 관련 벤치마크나 nsmc에서도 벤치마크 결과가 좋지 않아 폐기하게 되었습니다.<br>
kss의 경우 위키피디아나 뉴스 등과 같은 잘 정제된 문어체에는 잘 작동하지만, 제가 필요로 하는 맛집 리뷰 관련 태스크에서는 구어체를 기반으로 하고 있기 때문에 발생하는 문제로 보입니다.

또한 맛집 리뷰만을 단독으로 사용하였을 경우 모델의 성능이 NSMC를 제외한 다른 태스크에서 처참한 성적이 나오기 때문에,<br>
뉴스나 위키피디아 등의 잘 정제된 corpus 를 일부 섞어주니 문어체/구어체, 정제/비정제 데이터셋을 가리지 않고 성능이 향상되었고,<br>
이를 기반으로 조합한 다양한 모델 중 가장 준수한 성적을 보이는 모델을 업로드 하였습니다.

## About Tokenizer
위에서 말씀드린 것처럼, 이 모델은 오직 맛집 리뷰 관련 태스크에 적용하기 위해 학습시킨 모델입니다.<br>
맛집 리뷰 관련 문장에서 토크나이징을 하는 경우,<br>
기존의 모델들이 띄어쓰기를 적절히 쓰지 않은 문장, 혹은 신조어 등에서 적절히 토크나이징을 하지 못하는 것을 개선하였습니다.

```python
>>> from transformers import ElectraTokenizer
>>> tokenizer = ElectraTokenizer.from_pretrained("damien-ir/ko-rest-electra-discriminator")
>>> koelectra_tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-base-discriminator")
>>> koelectra_v2_tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-base-v2-discriminator")

>>> from konlpy.tag import Mecab
>>> mecab = Mecab()

>>> mecab.morphs("가성비넘좋구분위기도최고에용ㅎ")
['가성', '비', '넘', '좋', '구', '분위기', '도', '최고', '에', '용', 'ㅎ']
>>> tokenizer.tokenize("[CLS] 가성비넘좋구분위기도최고에용ㅎ [SEP]")
['[CLS]', '가성비', '##넘', '##좋', '##구', '##분위기', '##도', '##최고', '##에용', '##ㅎ', '[SEP]']
>>> koelectra_tokenizer.tokenize("[CLS] 가성비넘좋구분위기도최고에용ㅎ [SEP]")
['[CLS]', '가', '##성', '##비', '##넘', '##좋', '##구', '##분', '##위기', '##도', '##최', '##고에', '##용', '##ㅎ', '[SEP]']
>>> koelectra_v2_tokenizer.tokenize("[CLS] 가성비넘좋구분위기도최고에용ㅎ [SEP]")
['[CLS]', '[UNK]', '[SEP]']
```

## Benchmark Result
| Model | NSMC | Naver NER | PAWS | KorNLI | KorSTS | Question Pair | KorQuad |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| reviews + wiki | 89.85 | 85.31 | 76.85 | 78.70 | 80.34 | 94.72  | 64.09 / 87.98 |
| reviews + wiki + news + etc<br>(Current Model) | 90.38 | TBC | TBC | TBC | TBC | TBC | TBC |

## Acknowledgement
TensorFlow Research Cloud(TFRC) 의 지원을 받아 Cloud TPU로 모델을 학습하였습니다.<br>
