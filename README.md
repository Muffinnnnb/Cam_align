# Cam_align

Windows 로컬 Python에서 실행하는 카메라·점군 보정 및 색상 융합 실험 도구입니다.

- RealSense Depth → 우측 IR/Color: 평면 기반 RT 계산, 픽셀별 깊이 조회, `rs.align()` 비교와 PLY 저장.
- NSL-3110CA LiDAR → 별도 Color 카메라: 자동 보드 평면 검출, RANSAC + LM 보정, RGB 점군 저장.
- Qt 화면은 `.ui` 파일을 직접 읽으므로 Qt Designer에서 수정할 수 있습니다.

현재는 **개발·검증용 기준 버전**입니다. NSL 보드 오탐·미검출과 색상 정합 오차가 남아 있으며, RT의 절대 정확도는 독립적으로 검증되지 않았습니다. 목록 색상이나 학습 잔차가 작다는 이유만으로 보정 성공을 판정하면 안 됩니다.

## 설치와 실행

Windows / Python 3.12에서 확인했습니다. 저장소 루트에서 실행합니다.

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe nsl_color_gui.py
```

RealSense Depth 정렬 창은 다음과 같이 실행합니다.

```powershell
.venv\Scripts\python.exe calibration_gui.py
```

`@run_nsl.bat`와 `@run.bat`도 사용할 수 있습니다. 프로젝트의 `.venv`가 있으면 우선 사용하고, 없으면 로컬 `python`을 사용합니다. 단순 16-bit Depth 확인은 `python depth_viewer.py <depth_raw.png>`로 실행합니다.

오프라인 분석에는 카메라 연결이 필요하지 않습니다. 연결 장비의 공장 보정값을 읽는 기능에는 RealSense 장치가 필요합니다. `opencv-python-headless` 대신 화면 출력이 가능한 `opencv-python`을 설치하세요.

## 입력 데이터

촬영 원본과 기존 RT는 저장소에 포함하지 않습니다. 처음 실행하면 입력 폴더를 지정하고 해당 데이터로 RT를 계산하거나 호환되는 RT를 불러와야 합니다.

### RealSense Depth → 대상 영상

화면에서 Depth 폴더와 대상 IR/Color 폴더를 선택합니다. 각 폴더의 `Intrinsic.txt`, `Distortion.txt`를 사용하며, Depth에는 16-bit 원본과 깊이 단위 정보가 필요합니다. 표시용 컬러 Depth 이미지는 거리 원본이 아닙니다. 저장소의 `cal/`, `cal_260901/` 내부 파라미터는 기존 장비의 값이므로 다른 장비·해상도에 그대로 적용하지 마세요.

### NSL → Color

`Capture Root`에 촬영 프로그램이 만든 폴더를 지정합니다. 현재 로더는 `capture_manifest_*.json`의 `capture_timestamp`, `files`를 통해 다음 파일을 대응시킵니다.

- `NSL_pointcloud/`의 binary `.pcd`.
- `RealSense_color/`의 `.png`.
- WF 복원 모드에서는 `NSL_distance_raw_u16_*.png` 원본 거리맵도 필요합니다.

Color 내부 파라미터 폴더에는 `Intrinsic.txt`, `Distortion.txt`가 있어야 합니다. 보드는 한 칸 30 mm, 판 전체 250 × 280 mm를 가정합니다.

기본 입력 위치는 저장소의 `captures/`이며, GUI에서 선택한 경로는 로컬 설정에 저장됩니다. 명령행 기본 경로는 `CAM_ALIGN_CAPTURE_ROOT` 환경변수로도 지정할 수 있습니다. 예:

```powershell
python fuse_color_pointcloud.py --capture-root D:\captures\session --color-cal D:\cal\c --rt output\calibration\nsl_to_color_rt.json --all
```

PCD 원본 좌표와 WF 복원 좌표는 서로 다른 기하 모델이므로 RT를 혼용하지 않습니다. WF 복원은 제조사 명목 렌즈표를 사용하는 비교 모델이지 개별 장비의 실측 내부 보정값이 아닙니다. 렌즈표 출처와 제약은 [cal/nsl/SOURCE.txt](cal/nsl/SOURCE.txt)에 정리했습니다.

## 파일 구성과 저장 위치

| 위치 | 역할 |
| --- | --- |
| `calibration_gui.py`, `calibration_viewer.ui` | RealSense 정렬 화면 |
| `nsl_color_gui.py`, `nsl_color_viewer.ui` | NSL–Color 보정·융합 화면 |
| `calibrate_d_to_r.py`, `realsense_compare.py` | Depth 보정·정렬·SDK 비교 |
| `nsl_color_calibration.py`, `nsl_plane_fit.py`, `nsl_geometry.py` | NSL 평면 검출·RT·광선 모델 |
| `fuse_color_pointcloud.py` | 입력 대응, 색상 투영, 명령행 융합 |
| `tests/unit/` | 합성 데이터·격리 설정을 쓰는 회귀 테스트 |
| `tests/integration/` | 특정 로컬 촬영 자료가 필요한 GUI 검사 |
| `tools/legacy_validate_nsl_color.py` | 과거 방식 실험 보관; 현재 v6 합격 검사로 사용하지 않음 |
| `관련_논문/논문_색인.md` | 참고 자료 색인 |
| `output/` | 생성물 전용, Git 제외 |
| `휴지통/` | 사용자가 검토·복원할 로컬 보관 자료, Git 제외 |

기본 출력은 `output/calibration/`(RT), `pointclouds/`(PLY), `images/`(PNG), `comparison/`(정렬 비교), `fusion/`(명령행 융합), `validation/`(검사 결과), `logs/`(로그)로 나뉩니다. 폴더는 실행 중 필요할 때 생성됩니다.

RGB PLY에는 투영 위치의 원본 Color 값을 저장합니다. CloudCompare에서 intensity나 거리 스칼라 대신 **Colors/RGB** 표시를 선택해야 합니다. 색이 저장됐다는 사실은 물리적 대응점이 정확하다는 뜻은 아닙니다.

## 검증 범위와 남은 한계

NSL 화면의 하늘색은 **평면 후보 있음**, 빨강은 **평면쌍 없음**, 회색은 **미검사**입니다. RT 학습 사용/제외 이력과 잔차는 별도 표시합니다. 주황색 표시가 없어졌다는 것은 판정 표시 정책의 변경이며, 탐지 성공률 100%를 뜻하지 않습니다.

기존 76장 육안 점검에서는 보드 위치 후보 61장(부분 검출 포함), 신체 오탐 5장, 미검출 10장이 확인됐습니다. 이는 해당 촬영 묶음에 대한 관찰이지 일반화된 성능 보증이 아닙니다. 이번 저장소 정리는 검출·보정 알고리즘의 성능 개선 작업이 아닙니다.

평면 잔차와 시간 구간 교차검증은 내부 일관성 검사입니다. 독립 기준점 재투영 오차, 장비별 거리·광선 오차, 취득 시각 차이 검증이 추가로 필요합니다. RANSAC 임계값을 센서 정확도나 허용 RGB 픽셀 오차로 해석하지 마세요.

촬영 자료 없이 실행 가능한 회귀 검사:

```powershell
python -m unittest discover -s tests/unit -p "test_*.py" -v
```

단위 테스트는 실제 RT 정확도가 아니라 계산·입출력·상태 분리 등의 동작을 검사합니다. GitHub Actions도 이 범위만 검사합니다.

실제 Qt 작업 스레드·RGB 저장·깊이 클릭 검사는 선택적으로 실행합니다. `260922` 촬영 76장과 기존 `output/calibration/nsl_to_color_wf_range_auto_v5_rt.json`이 필요하며, 결과는 `output/validation/gui_smoke/`에만 저장합니다.

```powershell
$env:CAM_ALIGN_CAPTURE_ROOT = 'D:\captures\260922'
python tests/integration/nsl_gui_smoke.py
```

이 검사는 해당 촬영 묶음에 맞춘 회귀 검사이며, 새 데이터의 성공률을 측정하는 도구가 아닙니다. 과거 검증 도구는 현재 코드와 가정이 달라 별도 보관했습니다.

## 버전 관리

기존 `Muffinnnnb/Cam_align` 저장소의 `main` 이력을 이어갑니다. 코드·UI·테스트·문서는 추적하며, 원본 촬영·점군·개인 설정·로그·새 RT/검사 결과는 제외합니다. 이미 과거 커밋에 들어간 파일은 `.gitignore`로 과거 이력에서 지워지지 않습니다.

```powershell
git status
git add <변경한-코드나-문서>
git commit -m "변경 내용"
git push origin main
```

커밋 전 테스트와 변경 파일 목록을 확인하세요. 촬영 자료나 제조사 파일을 추가로 공개할 때는 개인정보·배포 권한을 별도로 확인해야 합니다.
