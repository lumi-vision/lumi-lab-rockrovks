# LUMI - 광과민성 발작 위험 영상 부분 보정 딥러닝 모델

**LUMI**는 영상 내 광과민성 발작(Photosensitive Epilepsy, PSE) 유발 위험 요소를 국제 표준 및 의학적 근거에 기반하여 탐지하고 보정하는 딥러닝 모델을 구축하기 위한 프로젝트입니다.

---

## 1. 프로젝트 폴더 구조

```text
lumi-lab-rockrovks/
├── data/
│   └── PSE_triggering_dataset/          # 테스트 및 라벨링 대상 원본 동영상 데이터셋
├── ImageDiscriminator/                  # 영상 알고리즘 개발을 위한 초기 단계
│   ├── image_discriminator_1.ipynb
│   └── image_discriminator_2_adstripe.ipynb
├── VideoDiscriminator/                  # 동영상 5대 위험 트리거 판별 및 데이터셋 빌더
│   ├── vdcoreops_v1.py                  # [핵심 엔진] 물리 휘도·시야각 변환 및 5대 위험 구간 판별 모듈
│   ├── runvd_v1.py                      # [단일 실행] 개별 동영상 1개 빠른 판별 실행 스크립트
│   ├── dsbuilder_v1.py                  # [배치 실행] 다수 동영상 자동 판별 및 딥러닝용 CSV 라벨 생성기
│   ├── video_discriminator_1.ipynb
│   └── video_discriminator_2.ipynb
├── requirements.txt                     # 가상환경 패키지 의존성 목록 (PyTorch CUDA 12.8 포함)
└── README.md
```
<sub>* `.ipynb` 파일들은 초기 개발 단계 과정에서의 체크포인트입니다.</sub>

---

## 2. 실행 환경 및 가상환경(venv) 세팅

본 프로젝트는 **Windows 환경**, **Python 3.12** 및 **CUDA 12.8 지원 GPU 환경(PyTorch)**에서 실시되었습니다.

### 2.1 가상환경 생성 및 활성화

저장소 루트 폴더(`lumi-lab-rockrovks`)에서 터미널을 열고 아래 명령어를 순서대로 실행합니다.

```powershell
# 1. Python 가상환경(venv) 생성
python -m venv venv

# 2. 가상환경 활성화
.\venv\Scripts\Activate.ps1
```

### 2.2 필수 패키지 설치

`requirements.txt` 상단에 PyTorch CUDA 12.8 인덱스 URL(`--extra-index-url`)이 포함되어 있으므로, 아래 명령어 한 줄로 GPU 버전 PyTorch와 필수 라이브러리(`opencv-python-headless`, `pandas`, `numpy` 등)가 일괄 설치됩니다. (CUDA 환경은 개인 버전에 맞게 설치하세요.)

```powershell
# 3. pip 업그레이드 및 패키지 일괄 설치
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 2.3 설치 및 GPU 연동 확인

```powershell
python -c "import torch, cv2, pandas; print('PyTorch:', torch.__version__, '| CUDA 사용 가능:', torch.cuda.is_available())"
```

---

## 3. 실행 방법

```powershell
# 1. 단일 동영상 판별 실행
python VideoDiscriminator/runvd_v1.py

# (선택) 다른 영상 경로 또는 기기 조건 지정 실행
python VideoDiscriminator/runvd_v1.py --video "data/PSE_triggering_dataset/1.mp4" --device laptop --display_area 15

# 2. 데이터셋 폴더 내 다수 영상 일괄 판별 및 CSV 라벨 생성
python VideoDiscriminator/dsbuilder_v1.py --dataset_dir "data/PSE_triggering_dataset" --output_dir "DatasetBuilder"
```
