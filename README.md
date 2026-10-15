# Lovedict (다음 단어 예측 연애 시뮬레이션)

사용자가 문장을 보내면 하루(Llama-3.1-8B)가 `You are so ___`의 다음 단어 후보 3개와 확률을 떠올리고,
그중 하나를 확률에 따라 말합니다. 단어의 호감도 점수(gpt-6-luna, 실패 시 내장 사전)가 호감도에 더해집니다.
100 도달 = 해피엔딩, 0 도달 = 배드엔딩.

## 실행
    pip install -r requirements.txt
    # api_keys.txt는 server.py 옆에 이미 들어 있음 (그대로 사용, 수정 금지)
    python server.py          # http://localhost:5000

## 배포 (웹 링크 1개)
Render 등에서 Web Service 생성: Build `pip install -r requirements.txt`, Start `gunicorn server:app`.
api_keys.txt는 Secret File(경로 KEYS_FILE)로 올리세요. GitHub에는 절대 올리지 마세요(.gitignore 처리됨).
