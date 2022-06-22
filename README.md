# ViT with ONCF Framework

### 보조 정보를 효과적으로 사용하기 위한 새로운 방법 : ONCF 프레임워크를 사용한 ViT
### A New Methology for Effective Use of Auxiliary Information : ViT with ONCF Framework
[오주영](https://github.com/Jy0923), [권유진](https://github.com/rnjsdb72), [최민석](https://github.com/ChoiMinS), [장서윤](https://github.com/sy00n)

본 연구에서는 보조정보를 효과적으로 활용하기 위한 시도로 보조 분류기를 도입하여 새로운 추천 시스템을 제안하고자 한다. 이를 위해 다음과 같은 접근 방법을 설계하였다.

1. Multi Embedding Layer 구조로 설계한다.
2. ONCF에 기반한 외적 방법론을 적용하고 합성곱 신경망 구조 대신 ViT 구조를 적용한다.
3. 보조 분류기를 사용하여 잠재 벡터가 보조 정보를 내포하도록 한다.

* 모델 구조
<image src = "https://github.com/Jy0923/ViT-with-ONCF-Framework/blob/main/figure/figure1.png" width="600" height="500">
* 실험 결과
<image src = "https://github.com/Jy0923/ViT-with-ONCF-Framework/blob/main/figure/figure2.png" width="300" height="150>

<br>
## Contents
1. [Environment Settings](#environment-settings)
2. [Training](#training)


## Environment Settings
* Clone the repository
```
git clone https://github.com/Jy0923/ViT-with-ONCF-Framework.git
cd ViT-with-ONCF-Framework
```
* Setup python environment
```
conda create -n vitoncf python=3.10 -y
conda activate vitoncf
pip install -r requirements.txt
```


## Training
* [Setup](#environment-settings) your environment
  
### Data Preprocessing

* Download the [MovieLens 1M Dataset](https://grouplens.org/datasets/movielens/)
* Run data.ipynb to preprocess the data

### Training Vit with ONCF model
* Change the config in the train_config.json file, Run
```
python train.py train_config.json
```
