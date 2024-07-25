# Noise2Noise Denoiser
이 리포지토리는 논문 [Speech Denoising Without Clean Training Data: A Noise2Noise Approach](https://arxiv.org/abs/2104.03838)에서 제공된 코드를 실제로 활용하여 wav 음성의 잡음을 제거하기 위해 만들어졌습니다.
## 요구 사항
- Python <= 3.10
## 필요 패키지 설치
```sh
pip install -r requirements.txt
```
혹은
```sh
pip3 install -r requirements.txt
```
## train.py
모델을 학습하는 코드입니다. 이때, 반드시 노이즈 음성만 들어 있는 디렉토리를 지정해야 합니다.
## denoise.py
학습한 모델을 기반으로 음성을 denoise하는 코드입니다. 이때, ```model_weights_path```를 반드시 모델 파일의 경로로 지정해야 합니다.
```python
model_weights_path = "Weights/dc20_model_30.pth"
```
