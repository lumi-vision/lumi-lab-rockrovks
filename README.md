# LUMI - 광과민성 발작 위험 영상 부분 보정 딥러닝 모델

**LUMI**는 영상 내 광과민성 발작(Photosensitive Epilepsy, PSE) 유발 위험 요소를 국제 표준 및 의학적 근거에 기반하여 탐지하고 보정하는 딥러닝 모델을 구축하기 위한 프로젝트입니다.

---

## 1. 프로젝트 폴더 구조

```text
lumi-lab-rockrovks/
├── data/
│   └── PSE_triggering_dataset/              # 원본 동영상 데이터셋
├── ImageDiscriminator/                      # 이미지 기반 알고리즘 개발을 위한 초기 단계
│   ├── image_discriminator_1.ipynb
│   └── image_discriminator_2_adstripe.ipynb
├── VideoDiscriminator/                      # 동영상 5대 위험 트리거 판별 알고리즘 초기 단계
│   ├── vdcoreops_v1.py                 
│   ├── runvd_v1.py                       
│   └── dsbuilder_v1.py                    
├── Fullscreen/                              # 전체 화면(Fullscreen) 및 부분 감광(Partial) 밝기 보정 · 폐루프 검증 모듈
│   ├── A1/                                  # [A1] 물리 휘도 안전 상한(18 cd/m²) 기반 전체 화면 & 부분 감광 보정 (1112)
│   │   ├── vdcorefs/                        # [핵심 엔진] 7대 위험 구간 및 4대 연속 위험도 지표 분석 패키지
│   │   ├── vdcoreps/                        # [마스크 엔진] 픽셀 단위 위험 영역 국소화 및 시공간 소프트 마스크 생성 패키지
│   │   ├── fullscreen_coreops_1112.py       # [보정 엔진] A1 전체 화면 균일 선형 감쇠(base/hold/safe 통합) 모듈
│   │   ├── run_fullscreen_1112.py           # [단일 실행] A1 전체 화면 보정 영상(.mp4) 및 4단 그래프(.png) 생성
│   │   ├── partial_coreops_1112.py          # [보정 엔진] A1 시공간 부분 감광(수식 A · 잔여 구간 0 보장) 모듈
│   │   ├── run_partial_1112.py              # [단일 실행] A1 부분 감광 보정 영상(.mp4) 및 4단 그래프(.png) 생성
│   │   ├── runvd.py                         # [단일 실행] 개별 동영상 7대 트리거·4대 연속 지표 진단 및 CSV/그래프 출력
│   │   └── dsbuilder.py                     # [배치 실행] 다수 동영상 자동 진단 및 4대 연속 지표 포함 CSV 생성기
│   └── B2/                                  # [B2] 7대 트리거·4대 연속 지표 기반 위험도 비례 & 적색 분리 보정 (1112)
│       ├── vdcorefs/                        # [핵심 엔진] 7대 위험 구간 및 4대 연속 위험도 지표 분석 패키지
│       ├── fullscreen_coreops_1112.py       # [보정 엔진] risk_calibrated 비례 + 적색 분리(base/hold/safe 통합) 모듈
│       ├── run_fullscreen_1112.py           # [단일 실행] B2 모드별(base/hold/safe/all) 보정 영상 및 5단 그래프 생성
│       ├── runvd.py                         # [단일 실행] 개별 동영상 7대 트리거·4대 연속 지표 진단 및 CSV/그래프 출력
│       └── dsbuilder.py                     # [배치 실행] 다수 동영상 자동 진단 및 4대 연속 지표 포함 CSV 생성기
├── requirements.txt                         # 가상환경 패키지 의존성 목록 (PyTorch CUDA 12.8 포함)
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
