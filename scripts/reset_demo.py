"""테스트로 더러워진 demo 계정과 그 데이터를 완전히 삭제한다.

공모전 배포 전 1회 실행:
    python scripts/reset_demo.py

- users 테이블에 password_hash / is_guest 컬럼이 없으면 먼저 마이그레이션한다.
- username='demo' 계정의 일기·분석·추천 로그·저장·프로필·변환 기록을 모두 지우고
  계정 자체도 삭제한다.
- 체험(게스트) 계정은 보장만 하고 그대로 둔다.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.database import Database  # noqa: E402


def main() -> None:
    database = Database()
    # password_hash / is_guest 컬럼 마이그레이션 + 게스트 계정 보장
    database.initialize()

    demo = database.get_user_by_username("demo")
    if not demo:
        print("demo 계정이 이미 없습니다. 정리할 내용이 없습니다.")
    else:
        user_id = int(demo["user_id"])
        deleted = database.delete_user(user_id)
        print(
            f"demo 계정(user_id={user_id})과 모든 데이터를 삭제했습니다."
            if deleted
            else f"demo 계정(user_id={user_id}) 데이터는 삭제했지만 계정 행 삭제에 실패했습니다."
        )

    guest = database.ensure_guest_user()
    print(f"체험(게스트) 계정 확인: user_id={guest['user_id']}, username={guest['username']}")
    print(f"DB 경로: {database.db_path}")


if __name__ == "__main__":
    main()
