"""LLM 콘텐츠 재태깅 러너.

목적: 키워드 규칙으로 잘못/대충 붙은 기존 태그를 LLM(gpt-5.4-mini)으로 전량 재태깅.

모드:
  sync         : 소량을 실시간 호출로 태깅(품질 확인용). 기본 5개.
  batch-build  : OpenAI Batch API 요청 파일(JSONL) 생성. (전량 재태깅이 기본)
  batch-submit : 요청 파일 업로드 + 배치 생성 → batch id 출력.
  batch-status : 배치 진행상태 확인.
  batch-apply  : 완료된 배치 결과를 DB에 반영(또는 --output-file 로컬 파일 사용).

비용 절감: Batch API 는 자동 -50%. 공통 시스템 프롬프트는 프롬프트 캐싱으로 추가 절감.

사전 준비: 환경변수 OPENAI_API_KEY 설정. (config.py 가 .env 도 읽음)

예시:
  python tagger/llm_tag_worker.py sync --limit 5 --dry-run     # 품질 미리보기(저장 안 함)
  python tagger/llm_tag_worker.py sync --limit 5               # 5개만 실제 저장
  python tagger/llm_tag_worker.py batch-build --out data/exports/llm_tag_requests.jsonl
  python tagger/llm_tag_worker.py batch-submit --in data/exports/llm_tag_requests.jsonl
  python tagger/llm_tag_worker.py batch-status --batch-id batch_xxx
  python tagger/llm_tag_worker.py batch-apply --batch-id batch_xxx
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config
from app.database import Database
from tagger.llm_tagger import (
    build_request_messages,
    build_tag_result,
    extract_output_text,
    parse_llm_tags,
)

# gpt-5.x 계열이 추론(reasoning) 모델이면 추론 토큰이 출력 예산을 먹으므로 여유를 둔다.
# 너무 작으면 보이는 JSON 전에 잘려 빈 출력이 날 수 있다. sync 소량 테스트로 먼저 확인할 것.
ENDPOINT = "/v1/responses"
MAX_OUTPUT_TOKENS = 512


def _client():
    if not config.OPENAI_API_KEY:
        raise SystemExit("OPENAI_API_KEY 가 설정되어 있지 않습니다. (.env 또는 환경변수)")
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise SystemExit("openai 패키지가 필요합니다: pip install openai") from exc
    return OpenAI(api_key=config.OPENAI_API_KEY)


def _targets(database: Database, content_type: str, limit: int, retag: bool) -> list[dict[str, Any]]:
    targets = database.list_tagging_targets(
        content_type=content_type,
        limit=limit if limit > 0 else 100000,
        retag=retag,
    )
    return targets


def _apply_one(database: Database, content: dict[str, Any], text: str) -> float:
    llm_tags = parse_llm_tags(text)
    tag_result, embedding_hash, model_name = build_tag_result(content, llm_tags)
    database.save_content_tagging_result(
        content_id=str(content["content_id"]),
        tag_result=tag_result,
        embedding_hash=embedding_hash,
        embedding_model_name=model_name,
    )
    return float(tag_result["final"]["tag_confidence"])


# ---------------------------------------------------------------- sync (small test)
def cmd_sync(args: argparse.Namespace) -> None:
    database = Database()
    database.initialize()
    client = _client()
    targets = _targets(database, args.content_type, args.limit, retag=args.retag)[: args.limit]
    print(f"[sync] 대상 {len(targets)}개, 모델={config.OPENAI_EMOTION_MODEL}, dry_run={args.dry_run}")
    ok = 0
    for content in targets:
        cid = str(content["content_id"])
        try:
            response = client.responses.create(
                model=config.OPENAI_EMOTION_MODEL,
                input=build_request_messages(content),
                max_output_tokens=MAX_OUTPUT_TOKENS,
            )
            text = str(getattr(response, "output_text", "") or "")
            tags = parse_llm_tags(text)
            if args.dry_run:
                print(f"  - {cid} {content.get('title','')}: {json.dumps(tags, ensure_ascii=False)}")
            else:
                conf = _apply_one(database, content, text)
                print(f"  - {cid} {content.get('title','')}: {json.dumps(tags, ensure_ascii=False)} (conf={conf})")
            ok += 1
        except Exception as exc:
            print(f"  ! {cid} 실패: {exc}")
    print(f"[sync] 완료: {ok}/{len(targets)}")


# ---------------------------------------------------------------- batch-build
def cmd_batch_build(args: argparse.Namespace) -> None:
    database = Database()
    database.initialize()
    targets = _targets(database, args.content_type, args.limit, retag=args.retag)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with out_path.open("w", encoding="utf-8") as handle:
        for content in targets:
            line = {
                "custom_id": str(content["content_id"]),
                "method": "POST",
                "url": ENDPOINT,
                "body": {
                    "model": config.OPENAI_EMOTION_MODEL,
                    "input": build_request_messages(content),
                    "max_output_tokens": MAX_OUTPUT_TOKENS,
                },
            }
            handle.write(json.dumps(line, ensure_ascii=False) + "\n")
            written += 1
    print(f"[batch-build] {written}개 요청 작성 → {out_path}")
    print(f"  모델={config.OPENAI_EMOTION_MODEL}, endpoint={ENDPOINT}, retag={args.retag}")
    print(f"  다음: python tagger/llm_tag_worker.py batch-submit --in {out_path}")


# ---------------------------------------------------------------- batch-submit
def cmd_batch_submit(args: argparse.Namespace) -> None:
    client = _client()
    in_path = Path(args.in_path)
    if not in_path.exists():
        raise SystemExit(f"요청 파일이 없습니다: {in_path}")
    uploaded = client.files.create(file=in_path.open("rb"), purpose="batch")
    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint=ENDPOINT,
        completion_window="24h",
        metadata={"purpose": "content_llm_retag"},
    )
    print(f"[batch-submit] batch_id={batch.id} status={batch.status}")
    print(f"  다음: python tagger/llm_tag_worker.py batch-status --batch-id {batch.id}")


# ---------------------------------------------------------------- batch-status
def cmd_batch_status(args: argparse.Namespace) -> None:
    client = _client()
    batch = client.batches.retrieve(args.batch_id)
    counts = getattr(batch, "request_counts", None)
    print(f"[batch-status] id={batch.id} status={batch.status}")
    if counts:
        print(f"  완료={counts.completed} 실패={counts.failed} 전체={counts.total}")
    if batch.status == "completed":
        print(f"  output_file_id={batch.output_file_id}")
        print(f"  다음: python tagger/llm_tag_worker.py batch-apply --batch-id {batch.id}")
    elif batch.status in {"failed", "expired", "cancelled"}:
        print(f"  error_file_id={getattr(batch, 'error_file_id', None)}")


# ---------------------------------------------------------------- batch-apply
def cmd_batch_apply(args: argparse.Namespace) -> None:
    database = Database()
    database.initialize()

    lines: list[str]
    if args.output_file:
        lines = Path(args.output_file).read_text(encoding="utf-8").splitlines()
    else:
        client = _client()
        batch = client.batches.retrieve(args.batch_id)
        if batch.status != "completed":
            raise SystemExit(f"배치가 아직 완료되지 않았습니다: status={batch.status}")
        content_bytes = client.files.content(batch.output_file_id).read()
        lines = content_bytes.decode("utf-8").splitlines()

    ok = 0
    fail = 0
    for raw_line in lines:
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        record = json.loads(raw_line)
        cid = record.get("custom_id")
        try:
            body = (record.get("response") or {}).get("body") or {}
            text = extract_output_text(body)
            content = database.get_content(str(cid))
            if not content:
                raise ValueError("content not found")
            conf = _apply_one(database, content, text)
            ok += 1
            if ok <= 10:
                print(f"  - {cid} {content.get('title','')} (conf={conf})")
        except Exception as exc:
            fail += 1
            print(f"  ! {cid} 실패: {exc}")
    print(f"[batch-apply] 반영 {ok}개, 실패 {fail}개")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LLM content re-tagging runner")
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--content-type", choices=["movie", "music", "book", "all"], default="all")
    common.add_argument(
        "--no-retag",
        dest="retag",
        action="store_false",
        help="기본은 전량 재태깅. 지정 시 pending/실패 항목만.",
    )
    common.set_defaults(retag=True)

    p_sync = sub.add_parser("sync", parents=[common], help="소량 실시간 태깅(품질 확인)")
    p_sync.add_argument("--limit", type=int, default=5, help="태깅할 개수")
    p_sync.add_argument("--dry-run", action="store_true", help="저장하지 않고 결과만 출력")
    p_sync.set_defaults(func=cmd_sync)

    p_build = sub.add_parser("batch-build", parents=[common], help="배치 요청 JSONL 생성")
    p_build.add_argument("--limit", type=int, default=0, help="0=전체")
    p_build.add_argument("--out", default="data/exports/llm_tag_requests.jsonl")
    p_build.set_defaults(func=cmd_batch_build)

    p_submit = sub.add_parser("batch-submit", help="배치 업로드+생성")
    p_submit.add_argument("--in", dest="in_path", default="data/exports/llm_tag_requests.jsonl")
    p_submit.set_defaults(func=cmd_batch_submit)

    p_status = sub.add_parser("batch-status", help="배치 상태 확인")
    p_status.add_argument("--batch-id", required=True)
    p_status.set_defaults(func=cmd_batch_status)

    p_apply = sub.add_parser("batch-apply", help="배치 결과 DB 반영")
    p_apply.add_argument("--batch-id", help="완료된 배치 id")
    p_apply.add_argument("--output-file", help="로컬 결과 JSONL 경로(배치 id 대신 사용)")
    p_apply.set_defaults(func=cmd_batch_apply)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
