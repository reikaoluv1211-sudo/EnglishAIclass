"""Lovedict - next-word prediction dating sim (backend).

키는 api_keys.txt에서만 읽습니다(프론트로 절대 전달하지 않음).
- Llama-3.1-8B-Instruct:novita (Hugging Face): 다음 단어 후보 + 확률
- gpt-6-luna (OpenAI): 후보 단어의 호감도 점수 매기기 (실패 시 다른 모델로 바꾸지 않고 오류 안내)
"""
import json
import math
import os
import random
import re
from pathlib import Path

import requests
from flask import Flask, jsonify, request, send_from_directory

BASE = Path(__file__).parent
KEYS_FILE = Path(os.environ.get("KEYS_FILE", BASE / "api_keys.txt"))


def load_config():
    """api_keys.txt의 KEY=VALUE 줄을 읽는다 (파일은 수정하지 않음)."""
    cfg = {}
    if KEYS_FILE.exists():
        for line in KEYS_FILE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                cfg[k.strip()] = v.strip()
    return cfg


CFG = load_config()
KEYS = {"hf": CFG.get("HF_TOKEN"), "openai": CFG.get("OPENAI_API_KEY")}
LLAMA = CFG.get("HF_MODEL", "meta-llama/Llama-3.1-8B-Instruct:novita")
LUNA = CFG.get("OPENAI_MODEL", "gpt-6-luna")
HF_URL = CFG.get("HF_BASE_URL", "https://router.huggingface.co/v1") + "/chat/completions"
OPENAI_URL = CFG.get("OPENAI_BASE_URL", "https://api.openai.com/v1") + "/chat/completions"
PREFIX = "You are so"

app = Flask(__name__, static_folder=str(BASE / "static"), static_url_path="/static")


# ---------- next-word candidates (Llama) ----------
def next_word_candidates(user_text):
    """'You are so ___' 다음에 올 단어 상위 3개와 확률."""
    system = (
        "You are Haru, a charming university student on a date with the user. "
        "React to the user's message with exactly one short sentence that starts "
        f"with '{PREFIX}' and then one adjective about the user. Nothing else."
    )
    r = requests.post(
        HF_URL,
        headers={"Authorization": f"Bearer {KEYS['hf']}"},
        json={
            "model": LLAMA,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_text},
            ],
            "max_tokens": 12,
            "temperature": 0.7,
            "logprobs": True,
            "top_logprobs": 10,
        },
        timeout=40,
    )
    r.raise_for_status()
    choice = r.json()["choices"][0]
    tokens = choice["logprobs"]["content"]

    # PREFIX 바로 다음 토큰의 위치 찾기
    acc, target = "", None
    for i, t in enumerate(tokens):
        acc += t["token"]
        if re.sub(r"[^a-z]", "", acc.lower()).endswith("youareso"):
            target = i + 1
            break
    if target is None or target >= len(tokens):
        raise ValueError("prefix not found in model output")

    words = {}
    for alt in tokens[target]["top_logprobs"]:
        w = re.sub(r"[^A-Za-z]", "", alt["token"]).lower()
        if len(w) >= 3:
            words[w] = words.get(w, 0) + math.exp(alt["logprob"])
    top = sorted(words.items(), key=lambda kv: -kv[1])[:3]
    total = sum(p for _, p in top) or 1
    return [{"word": w, "prob": p / total} for w, p in top]


# ---------- affection scoring ----------
def score_words(words):
    """각 단어를 -10(최악)~+10(최고) 호감도로 변환 (gpt-6-luna).
    모델이 동작하지 않으면 다른 모델로 바꾸지 않고 오류를 그대로 올린다."""
    r = requests.post(
        OPENAI_URL,
        headers={"Authorization": f"Bearer {KEYS['openai']}"},
        json={
            "model": LUNA,
            "messages": [
                {"role": "system", "content": (
                    "Rate how much each adjective, said by a date to someone, "
                    "would raise (+) or lower (-) that person's romantic mood, "
                    "from -10 to 10. Reply ONLY with a JSON object mapping "
                    "each word to an integer.")},
                {"role": "user", "content": json.dumps(words)},
            ],
        },
        timeout=30,
    )
    r.raise_for_status()
    text = r.json()["choices"][0]["message"]["content"]
    data = json.loads(re.search(r"\{.*\}", text, re.S).group(0))
    return {w: max(-10, min(10, int(data.get(w, 0)))) for w in words}


# ---------- routes ----------
@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.post("/api/chat")
def chat():
    body = request.get_json(silent=True) or {}
    text = str(body.get("message", "")).strip()[:300]
    affection = int(body.get("affection", 30))
    if not text:
        return jsonify(error="메시지를 입력해 주세요."), 400
    if not KEYS["hf"] or not KEYS["openai"]:
        return jsonify(error="서버에서 api_keys.txt의 키를 찾지 못했어요. 파일 위치를 확인해 주세요."), 500
    try:
        cands = next_word_candidates(text)
    except Exception as e:  # noqa: BLE001
        app.logger.error("llama failed: %s", e)
        return jsonify(error="Llama 모델이 응답하지 않아요. 다른 모델로 바꾸지 않고 멈춥니다. 키/모델 설정을 확인해 주세요."), 502

    try:
        scores = score_words([c["word"] for c in cands])
    except Exception as e:  # noqa: BLE001
        app.logger.error("gpt-6-luna failed: %s", e)
        return jsonify(error="gpt-6-luna 모델이 응답하지 않아요. 다른 모델로 바꾸지 않고 멈춥니다. 키/모델 설정을 확인해 주세요."), 502
    for c in cands:
        c["score"] = scores[c["word"]]

    # 확률에 비례해 하루가 실제로 말하는 단어를 하나 뽑는다
    chosen = random.choices(cands, weights=[c["prob"] for c in cands])[0]
    delta = chosen["score"]
    return jsonify(
        candidates=cands,
        chosen=chosen["word"],
        reply=f"{PREFIX} {chosen['word']}.",
        delta=delta,
        affection=max(0, min(100, affection + delta)),
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
