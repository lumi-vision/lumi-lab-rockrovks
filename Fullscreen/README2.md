# LUMI Fullscreen & Partial — 밝기 보정 및 폐루프 검증 모듈 (`A1` · `B2`)

`Fullscreen/` 폴더는 광과민성 발작(Photosensitive Epilepsy, PSE) 유발 위험 구간이 탐지된 동영상에 대해 **선형 물리 휘도(및 적색 채널) 공간에서 밝기를 감쇠 보정**하고, 원본 해상도로 보정 영상을 인코딩하면서 **동시에 위험 지표를 재검증(Closed-loop Verification)**하는 독립 실행형 모듈(`A1`, `B2`, 모두 `1112` 통합 버전)을 포함합니다.

* **`A1/`**: 물리 휘도 안전 상한(`18.0 cd/m²`) 기반 **전체 화면 균일 보정(`run_fullscreen_1112.py`)** 및 **시공간 부분 감광 보정(`run_partial_1112.py`)** 통합 폴더 (`vdcorefs/` + `vdcoreps/` 내장)
* **`B2/`**: `risk_calibrated` 위험도 비례 감쇠 + 포화 적색(`R`) 채널 분리 감쇠 기반 **전체 화면 보정(`run_fullscreen_1112.py`)** 폴더 (`vdcorefs/` 내장)

> 각 폴더(`A1/`, `B2/`)는 자체 탐지·마스크 패키지를 내부에 포함하고 있어 외부 상위 폴더나 별도 `PYTHONPATH` 설정 없이 어느 위치에서든 바로 실행할 수 있습니다.

---

## 1. `A1 (Fullscreen)` vs `A1 (Partial)` vs `B2 (Fullscreen)` 핵심 비교

| 구분 | `A1/` 전체 화면 (`run_fullscreen_1112.py`) | `A1/` 부분 감광 (`run_partial_1112.py`) | `B2/` 전체 화면 (`run_fullscreen_1112.py`) |
| :--- | :--- | :--- | :--- |
| **탐지 · 검증 엔진** | `vdcorefs/` (7대 트리거 + 4대 연속 지표) | Pass 1: `vdcoreps/` (픽셀 국소화 + 7대/4대 동일)<br>Pass 2 검증: `vdcorefs/` | `vdcorefs/` (7대 트리거 + 4대 연속 지표) |
| **시청 환경 기본값** | `laptop` (`15.6"`, `56cm`) / `desktop` / `tv` | `laptop` (`15.6"`, `56cm`) / `desktop` / `tv` | `laptop` (`15.6"`, `56cm`) / `desktop` / `tv` |
| **분석 · 저장 해상도** | 분석 `max_width = 640` / **원본 해상도 저장** | 분석 `max_width = 640` / **원본 해상도 저장** | 분석 `max_width = 640` / **원본 해상도 저장** |
| **공간 적용 범위** | 프레임 전체 픽셀 균일 감쇠 | 위험 코어 영역(`W_t = 1`) 집중 감쇠 + 외곽 구조 유도 소프트 페더링 (안전 배경 `W_t = 0` 보존) | 프레임 전체 픽셀 균일 감쇠 (`G, B` vs `R` 채널 분리) |
| **밝기 감쇠(`alpha_t`)** | **방식 A**: 프레임 최대 휘도 `<= 18.0 cd/m²` | **방식 A (마스크 내부 기준)**: 코어 위험 영역 내 최대 휘도 `<= 18.0 cd/m²` + 잔여 위험 `0` 폐루프 보강 | **방식 B (`risk_calibrated` 비례)**: `50.0 -> 50%`, `100.0 -> 9%` 선형 보간 |
| **적색(`R`) 처리** | **선택 1**: 포화 적색 상위 24% 초과분 `<= 0.058`이 되도록 전 채널 `alpha_t` 하향 | **선택 1 (마스크 내부)**: 위험 영역 내 적색 초과분 `<= 0.058`이 되도록 해당 영역 `alpha_t` 하향 | **선택 2 (채널 분리)**: `G, B`는 `alpha_t` 유지, `R`에만 `beta_R,t` 추가 적용 (`hybrid` / `abs`) |
| **출력 파일 접미사** | `_fullscreen_dimmed.mp4`<br>`_fullscreen_graph.png` | `_partial_dimmed.mp4`<br>`_partial_graph.png` | `_fullscreen_dimmed.mp4`<br>`_fullscreen_graph.png` |
| **비교 그래프 구성** | **4단 (`Layout A`)**<br>휘도 · 전/후 4종 지표 · 전체 배율(`alpha_t`) | **4단 (`Layout A`)**<br>휘도 · 전/후 4종 지표 · 화면 평균 배율 및 감광 면적 비율(%) | **5단 (`Layout 1`)**<br>휘도 · 전/후 4종 지표 · `G/B` 배율 · `R` 추가/최종 배율 |

---

## 2. 폴더별 파일 및 패키지 구성

### 2.1 `A1/` 및 `B2/` 수록 파일 목록

| 경로 | 포함 폴더 | 역할 및 설명 |
| :--- | :---: | :--- |
| `vdcorefs/` *(8개 모듈)* | `A1/`, `B2/` | **전역 판별·검증 엔진**: IEC 61966-2-1 광도·색도 변환, 시계열 전이/연장 상태 기계, 2D FFT 결합 정적 공간 패턴 탐지, 7대 라벨 및 4대 연속 위험도 산출 (`python -m vdcorefs <영상>` 지원) |
| `vdcoreps/` *(7개 모듈)* | `A1/` | **시공간 부분 마스크 엔진**: `vdcorefs`와 100% 동일한 7대/4대 판정을 유지하면서 점멸·적색·정적 패턴 유발 픽셀 코어를 국소화하고 구조 유도(Guided) 보더리스 소프트 가중치 마스크 `W_t(y, x)` 생성 (`python -m vdcoreps <영상>` 지원) |
| `fullscreen_coreops_1112.py` | `A1/`, `B2/` | **전체 화면 보정 엔진**: `A1`(물리 휘도·적색 상한 균일 LUT) / `B2`(`risk_calibrated` 비례 + 적색 분리 3채널 LUT) 스케줄러 및 원본 해상도 스트리밍 인코딩·재검증 |
| `run_fullscreen_1112.py` | `A1/`, `B2/` | **전체 화면 통합 실행 CLI**: `--mode base\|hold\|safe\|all` 지원 (Pass 1 결과 공유) 및 `.mp4` + 4단(`A1`)/5단(`B2`) 비교 그래프 저장 |
| `partial_coreops_1112.py` | `A1/` | **부분 감광 보정 엔진**: 수식 A `c_lin' = [1 - W_t * (1 - alpha_t)] * c_lin` 적용 및 잔여 위험 구간 발생 시 자동 보더리스 폐루프 보강(`enforce_zero_residual=True`) |
| `run_partial_1112.py` | `A1/` | **부분 감광 통합 실행 CLI**: `--mode base\|hold\|safe\|all` 지원 (Pass 1 시공간 마스크 공유) 및 `.mp4` + 4단 비교 그래프 저장 |
| `runvd.py` | `A1/`, `B2/` | **단일 영상 상세 진단**: 7대 라벨 간트 차트·4대 위험도·1초 전이 수 3단 그래프(`.png`) 및 `<stem>_intervals.csv`, `<stem>_frame_metrics.csv` 출력 |
| `dsbuilder.py` | `A1/`, `B2/` | **배치 라벨링 빌더**: 폴더 전체 자연 정렬 분석 후 `pse_video_summary.csv`, `pse_interval_details.csv` 생성 |

### 2.2 공통 7대 위험 라벨 · 4대 연속 지표 · 스케줄링 모드(`--mode`)

* **7대 구조화 위험 라벨**: `휘도 플래시`, `교집합`(휘도 플래시 ∩ 빠른 장면 전환), `빠른 장면 전환`, `적색 플래시`, `휘도 연장`, `적색 연장`, `정적 공간 패턴` (`안전` 포함)
* **4대 연속 위험도 지표**: `risk_calibrated` `[0~100]` (경계 `50.0`), `risk_ppr_pct` `[0~100%]`, `risk_nomura` `[0~1]` (경계 `0.50`), `risk_vfm_weibull` `[0~100]`
* **`--mode` 선택** (`run_fullscreen_1112.py` · `run_partial_1112.py` 공통):
  * `base` *(기본값)*: 확정 7대 위험 구간 보정 + 대칭 `0.20s` 코사인 Fade-in/out (`_dimmed.mp4`, `_graph.png`)
  * `hold`: 확정 7대 위험 구간 보정 + `0.20s` 절반 완화 → `0.50s` Half-Hold 유지 → `0.20s` 완전 해제 (`_hold_dimmed.mp4`, `_hold_graph.png`)
  * `safe`: `(확정 위험 구간) ∪ (4대 연속 지표 >= 50% 합집합)` 선제 보정 + Half-Hold 해제 (`_safe_dimmed.mp4`, `_safe_graph.png`)
  * `all`: Pass 1 분석 1회를 공유하여 `base`, `hold`, `safe` 3종을 한 번에 생성

---

## 3. 바로 실행하기 (`A1` — 전체 화면 & 부분 감광)

저장소 루트(`lumi-lab-rockrovks`)에서 바로 실행하거나, 각 스크립트 상단의 `VIDEO_PATH` / `OUTPUT_DIR`(`"경로를 입력하세요!"`) 변수를 직접 수정해 실행할 수 있습니다.

```powershell
# 1. A1 전체 화면(Fullscreen) 보정 실행 (base | hold | safe | all)
python Fullscreen/A1/run_fullscreen_1112.py --video "data/PSE_triggering_dataset/4.mp4" --output-dir "Output/A1/fullscreen" --mode base --no-show

# 2. A1 부분 감광(Partial) 보정 실행 (base | hold | safe | all)
python Fullscreen/A1/run_partial_1112.py --video "data/PSE_triggering_dataset/4.mp4" --output-dir "Output/A1/partial" --mode base --no-show

# (선택) 부분 감광 시간창 반경(기본 0.16초) 및 공간 페더링 폭(기본 0.40도) 지정 실행
python Fullscreen/A1/run_partial_1112.py --video "data/PSE_triggering_dataset/4.mp4" --output-dir "Output/A1/partial" --mode safe --temporal-window-sec 0.16 --feather-deg 0.40 --no-show

# 3. 단일 영상 7대 트리거·4대 연속 지표 진단 (CSV + 3단 그래프)
python Fullscreen/A1/runvd.py "data/PSE_triggering_dataset/1.mp4" --out-dir "Output/A1_vd"

# 4. 데이터셋 폴더 일괄 라벨링 CSV 생성
python Fullscreen/A1/dsbuilder.py --input-dir "data/PSE_triggering_dataset" --output-dir "Output/A1_dataset"
```

---

## 4. 바로 실행하기 (`B2` — 위험도 비례 & 적색 채널 분리 보정)

* `--red-mode`: `hybrid` *(기본값, `red_ratio' <= 0.72` 이탈과 적색 초과분 `<= 0.058` 중 완화된 상한 선택)* 또는 `abs` *(`<= 0.058` 엄격 적용)*
* `--alpha-at-50` / `--alpha-at-100`: `risk_calibrated = 50.0` 및 `100.0`에서의 선형 밝기 배율 지정 (기본값 `0.50`, `0.09`)

```powershell
# 1. B2 단일 모드 보정 실행 (base | hold | safe)
python Fullscreen/B2/run_fullscreen_1112.py --video "data/PSE_triggering_dataset/4.mp4" --output-dir "Output/B2" --mode safe --no-show

# 2. B2 전체 모드(base + hold + safe) 일괄 비교 실행
python Fullscreen/B2/run_fullscreen_1112.py --video "data/PSE_triggering_dataset/4.mp4" --output-dir "Output/B2" --mode all --red-mode hybrid --no-show

# 3. 단일 영상 진단 및 데이터셋 일괄 라벨링
python Fullscreen/B2/runvd.py "data/PSE_triggering_dataset/1.mp4" --out-dir "Output/B2_vd"
python Fullscreen/B2/dsbuilder.py --input-dir "data/PSE_triggering_dataset" --output-dir "Output/B2_dataset"
```
