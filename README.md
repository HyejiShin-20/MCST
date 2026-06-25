# Emotion Daily Culture Recommender MVP

감정·일상 기록을 분석해 문화 콘텐츠를 추천하는 백엔드 중심 MVP입니다. 현재 데모 범위는 **음악과 도서**입니다. 영화 샘플 데이터는 보존하지만 데모 추천에서는 숨깁니다.

## 현재 구현 범위

- 텍스트 일기 저장, 감정/상황/관심사 분석, 장기 기억 프로필 저장
- Google Vision OCR: 이미지/손글씨/그림 업로드 → 후보 텍스트 생성
- OpenAI STT: 음성 업로드 → 후보 텍스트 생성
- 이미지 캡셔닝: OCR 텍스트와 별도로 장면/분위기/맥락 설명 생성
- 링크 입력: 웹페이지 본문을 읽어 후보 텍스트 생성
- 후보 텍스트 사용자 확인/수정 후 감정 분석과 추천 실행
- 음악/도서 추천 결과와 추천 이유 제공
- Melon/Bugs/Kyobo raw 데이터 기반 import, 요약/키워드/감정 태그 자동화
- Chroma 기반 후보 검색, 로컬 HuggingFace 임베딩 기본 사용
- 원문 일기, 분석 결과, 추천 로그, 장기 기억, 미디어 변환 결과 저장
- 기억 초기화와 사용자 데이터 삭제 API

## 데이터 정책

- `data/raw/`의 원천 파일은 수정하지 않습니다.
- 음악 가사는 import 중에만 읽고, DB에는 원문 가사를 저장하지 않습니다.
- 음악 DB에는 가사 요약, 키워드, 감정 태그, 상황/주제 태그만 저장합니다.
- 정제 결과는 `data/processed/culture_import_music_book.jsonl`에 저장됩니다.

## 환경 준비

PowerShell 기준입니다.

```powershell
conda create -n emotion-culture-mvp python=3.11 -y
conda activate emotion-culture-mvp
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

이미 환경을 만들었다면:

```powershell
conda activate emotion-culture-mvp
python -m pip install -r requirements.txt
```

## `.env` 주요 설정

루트의 `.env`에 아래 공개 설정이 있어야 합니다. API 키 값은 직접 넣으세요.

```dotenv
APP_DB_PATH=data/app.sqlite3
DEFAULT_CONTENT_TYPES=music,book
HIDE_SAMPLE_CONTENT_IN_DEMO=true
RETRIEVAL_BACKEND=chroma
EMBEDDING_PROVIDER=local
HF_EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

OCR_PROVIDER=google
GOOGLE_CLOUD_PROJECT=mcst-500412
GOOGLE_VISION_FEATURE=DOCUMENT_TEXT_DETECTION
GOOGLE_VISION_LANGUAGE_HINTS=ko,en

STT_PROVIDER=openai
OPENAI_TRANSCRIPTION_MODEL=gpt-4o-mini-transcribe
OPENAI_TRANSCRIPTION_LANGUAGE=ko
OPENAI_ANALYSIS_ENABLED=true
OPENAI_EMOTION_MODEL=gpt-5.4-mini
OPENAI_EMOTION_FALLBACK_MODEL=gpt-5.4-nano
OPENAI_VISION_MODEL=gpt-5.4-mini
OPENAI_IMAGE_CAPTION_ENABLED=true

LINK_TIMEOUT_SECONDS=10
LINK_MAX_BYTES=2097152
LINK_MAX_TEXT_CHARS=6000
```

Google OCR ADC 설정 예시:

```powershell
gcloud config set project mcst-500412
gcloud services enable vision.googleapis.com --project mcst-500412
gcloud auth application-default set-quota-project mcst-500412
```

## 실제 데이터 import

raw 파일을 읽어 음악 300개, 도서 300개를 정제하고 DB에 넣습니다.

```powershell
python scripts/import_real_culture_data.py --music-limit 300 --book-limit 300
```

자동 태깅을 실행합니다.

```powershell
python tagger/tag_worker.py --content-type music --batch-size 300 --limit 300
python tagger/tag_worker.py --content-type book --batch-size 300 --limit 300
```

더 작게 테스트하려면:

```powershell
python scripts/import_real_culture_data.py --music-limit 20 --book-limit 20 --dry-run
```

## 서버 실행

```powershell
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --log-level info
```

상태 확인:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/api/health
```

프론트는 정적 HTML입니다.

```text
E:\web\frontend\index.html
```

브라우저에서 위 파일을 열면 기본 API 주소는 `http://127.0.0.1:8000`입니다. 기존 디자인은 유지했고, 업로드 UI 안에 변환 후보 텍스트 확인 영역만 추가했습니다.

## API 빠른 테스트

텍스트 일기 생성:

```powershell
$diary = Invoke-RestMethod http://127.0.0.1:8000/api/diary `
  -Method Post `
  -ContentType 'application/json; charset=utf-8' `
  -Body (@{
    user_id = 1
    input_type = 'text'
    text = '오늘은 일이 계속 꼬여서 지쳤지만 조용히 위로받고 싶다.'
  } | ConvertTo-Json)
```

추천 요청:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/api/recommend `
  -Method Post `
  -ContentType 'application/json; charset=utf-8' `
  -Body (@{
    entry_id = $diary.entry_id
    content_types = @('music', 'book')
    per_type = 4
  } | ConvertTo-Json)
```

미디어 업로드는 프론트에서 파일을 선택하거나 음성을 녹음하면 `/api/media/transcribe`로 전송됩니다. 서버는 후보 텍스트를 반환하고, 사용자가 수정한 텍스트로 `/api/diary/from-transcription`을 호출합니다.

링크 입력은 `/api/link/transcribe`로 전송됩니다. 웹페이지 본문을 후보 텍스트로 추출한 뒤 같은 확인/수정 흐름을 거쳐 일기와 추천으로 연결됩니다.

## 품질 검증 명령

Python 컴파일:

```powershell
python -m compileall app tagger scripts tests
```

HTTP 기반 데모 검증:

```powershell
python scripts/demo_quality_check.py --port 8010
```

검증 결과는 `data/exports/demo_quality_check.json`에 저장됩니다.

## 이번 세션 검증 결과

- 실제 데이터 import/태깅 대상: 음악 315개, 도서 315개, 총 630개
- raw source: Melon 월간 Top50, Melon 가사, Bugs 연도별 가요, Kyobo 연도별 베스트셀러
- raw 가사 DB 저장 여부: 저장하지 않음
- 자동 태깅: 음악 315/315 성공, 도서 315/315 성공
- 평균 태깅 신뢰도: 음악 0.7602, 도서 0.7125
- recommendable music/book 데이터: 630개
- HTTP 데모 품질 검증: 4/4 통과
- 단위 테스트: `python -m unittest discover -s tests` 통과
- 추천 응답: `music`, `book`만 채움, `movie`는 0개
- 샘플 데이터 누출: 0건
- 검색 backend: Chroma 확인
- 임베딩 backend: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- Google OCR 실제 호출: 성공, 후보 텍스트 반환, 임시 파일 삭제 확인
- OpenAI STT 실제 호출: 성공, 한국어 텍스트 반환, 임시 파일 삭제 확인
- OpenAI 감정 분석 보강: 네트워크 권한에서 실제 호출 성공, `llm_augmented=true` 확인

## 참고

- 첫 Chroma 추천 호출은 컬렉션 동기화 때문에 느릴 수 있습니다.
- 네트워크가 제한된 환경에서는 OpenAI 분석 보강이 실패해도 로컬 분석으로 자동 폴백합니다.
- Node.js가 PATH에 없으면 프론트 JS 문법 검사는 생략될 수 있습니다. 현재 프론트 연결은 HTTP e2e와 실제 API 응답 기준으로 검증했습니다.

## 기록/저장소/취향 분석

이번 MVP는 데모용 로컬 사용자 전환을 지원합니다. 비밀번호 인증은 없고, 프론트의 `USER` 입력에 이름을 넣고 `LOAD`를 누르면 해당 사용자 기준으로 기록과 저장소가 분리됩니다.

- `/api/users`: 사용자 생성/전환
- `/api/activity`: 일기 원문, 변환 텍스트, 추천 로그, 저장 항목 조회
- `/api/repository`: 저장한 추천 항목 조회
- `/api/repository/save`: 추천 카드를 나의 저장소에 저장
- `/api/taste-analysis`: 저장한 항목만 기준으로 취향 요약, 감정 패턴, 콘텐츠 분야 비율, 다음 추천 방향 생성

프론트 연결:

- `frontend/index.html`: 마지막 변환 텍스트/추천 결과를 브라우저 로컬 저장소에 복원하고, 추천 카드의 `SAVE TO REPOSITORY`로 DB 저장
- `frontend/archive.html`: DB의 일기, 미디어 변환 결과, 추천 로그를 타임라인으로 표시
- `frontend/repository.html`: 저장한 음악/도서 추천 항목 표시
- `frontend/discovery.html`: 저장한 항목만 기반으로 취향 분석 표시

추가 검증 명령:

```powershell
python scripts/frontend_static_check.py
python scripts/http_persistence_check.py
python scripts/http_multimodal_check.py
python -m unittest discover -s tests
```
