"""비밀번호 해시/검증 + 비밀번호 정책 유틸 (외부 의존성 없이 표준 라이브러리만 사용)."""
from __future__ import annotations

import hashlib
import hmac
import secrets

# pbkdf2-sha256 파라미터
_ALGORITHM = "pbkdf2_sha256"
_ITERATIONS = 200_000
_SALT_BYTES = 16

MIN_PASSWORD_LENGTH = 8


def hash_password(password: str, *, iterations: int = _ITERATIONS) -> str:
    """비밀번호를 'pbkdf2_sha256$iterations$salt_hex$hash_hex' 형태로 저장 가능한 문자열로 반환."""
    salt = secrets.token_bytes(_SALT_BYTES)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return f"{_ALGORITHM}${iterations}${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    """저장된 해시 문자열과 입력 비밀번호를 상수 시간 비교로 검증."""
    if not stored:
        return False
    try:
        algorithm, iteration_str, salt_hex, hash_hex = stored.split("$")
    except ValueError:
        return False
    if algorithm != _ALGORITHM:
        return False
    try:
        iterations = int(iteration_str)
        salt = bytes.fromhex(salt_hex)
        expected = bytes.fromhex(hash_hex)
    except (ValueError, TypeError):
        return False
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return hmac.compare_digest(digest, expected)


# 너무 흔하거나 단순한 비밀번호 차단 목록
_COMMON_PASSWORDS = {
    "password", "12345678", "123456789", "1234567890", "qwerty", "qwerty123",
    "111111", "123123", "abc12345", "password1", "password123", "iloveyou",
    "admin", "admin123", "welcome", "letmein", "1q2w3e4r", "qwer1234",
    "asdf1234", "zxcv1234", "00000000", "11111111",
}


def validate_password_strength(password: str) -> str | None:
    """비밀번호가 정책을 통과하면 None, 아니면 한국어 오류 메시지를 반환한다.

    정책:
    - 최소 8자
    - 공백만으로 구성 불가
    - 영문/숫자/기호 중 2종류 이상 포함
    - 동일 문자 반복(예: aaaaaaaa) 불가
    - 흔한 비밀번호 목록 차단
    """
    if password is None:
        return "비밀번호를 입력해 주세요."
    if len(password) < MIN_PASSWORD_LENGTH:
        return f"비밀번호는 최소 {MIN_PASSWORD_LENGTH}자 이상이어야 합니다."
    if " " in password:
        return "비밀번호에 공백을 사용할 수 없습니다."
    if password.lower() in _COMMON_PASSWORDS:
        return "너무 흔한 비밀번호입니다. 다른 비밀번호를 사용해 주세요."
    if len(set(password)) == 1:
        return "같은 문자만 반복된 비밀번호는 사용할 수 없습니다."

    has_letter = any(character.isalpha() for character in password)
    has_digit = any(character.isdigit() for character in password)
    has_symbol = any((not character.isalnum()) for character in password)
    classes = sum([has_letter, has_digit, has_symbol])
    if classes < 2:
        return "비밀번호는 영문, 숫자, 기호 중 2가지 이상을 조합해야 합니다."
    return None
