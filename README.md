# iPad용 구독형 전기요금 최적제어 웹앱

Google OR-Tools 기반 연구용 시뮬레이터입니다. Streamlit Community Cloud에 배포하면 iPad Safari에서 별도 설치 없이 사용할 수 있습니다.

## 포함 파일

- `streamlit_app.py`: iPad용 화면
- `subscription_energy_optimizer.py`: OR-Tools 최적화 엔진
- `구독형_전기요금_최적제어_입력자료.xlsx`: 내장 4인 가구 예시
- `requirements.txt`: 서버 설치 패키지

## 배포

1. GitHub에서 새 저장소를 만듭니다.
2. 이 폴더의 파일을 저장소에 모두 업로드합니다.
3. Streamlit Community Cloud에 로그인하고 GitHub 저장소를 연결합니다.
4. 실행 파일로 `streamlit_app.py`를 선택해 배포합니다.
5. 생성된 `*.streamlit.app` 주소를 iPad Safari에서 엽니다.

## 개인정보 주의

공개 배포 앱에는 실제 고객의 AMI, 주소, 생활패턴 또는 기타 개인정보를 업로드하지 마십시오. 실제 고객자료를 사용하려면 사내 보안 서버나 접근이 통제된 비공개 환경으로 이전해야 합니다.
