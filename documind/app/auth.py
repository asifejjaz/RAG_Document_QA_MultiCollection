"""Password hashing + signed session cookies."""
import bcrypt
from fastapi import Request, HTTPException
from itsdangerous import URLSafeSerializer, BadSignature
from app import config, db

_ser = URLSafeSerializer(config.SESSION_SECRET, salt="session")
COOKIE = "dm_session"


def hash_pw(pw: str) -> str:
    return bcrypt.hashpw(pw.encode("utf-8")[:72], bcrypt.gensalt()).decode("utf-8")


def verify_pw(pw: str, pw_hash: str) -> bool:
    try:
        return bcrypt.checkpw(pw.encode("utf-8")[:72], pw_hash.encode("utf-8"))
    except Exception:
        return False


def make_cookie(user_id: int) -> str:
    return _ser.dumps({"uid": user_id})


def read_cookie(token: str) -> int | None:
    try:
        return _ser.loads(token).get("uid")
    except BadSignature:
        return None


def current_user(request: Request):
    tok = request.cookies.get(COOKIE)
    if not tok:
        return None
    uid = read_cookie(tok)
    if uid is None:
        return None
    return db.get_user(uid)


def require_user(request: Request):
    u = current_user(request)
    if not u:
        raise HTTPException(status_code=401, detail="Login required")
    return u
