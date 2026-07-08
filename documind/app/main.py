import shutil
import uuid
from pathlib import Path
from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException, Depends
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from app import config, db, auth
from app.rag import ingest, chat
from app import reports

db.init()
app = FastAPI(title=config.PRODUCT_NAME)
BASE = Path(__file__).resolve().parent
app.mount("/static", StaticFiles(directory=str(BASE / "static")), name="static")
tpl = Jinja2Templates(directory=str(BASE / "templates"))
tpl.env.cache = None  # workaround: Jinja LRUCache key breaks under Python 3.14
CTX = {"product": config.PRODUCT_NAME, "tagline": config.TAGLINE}


# ---------- pages ----------
@app.get("/", response_class=HTMLResponse)
def landing(request: Request):
    return tpl.TemplateResponse(request, "landing.html", {**CTX, "user": auth.current_user(request)})


@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request):
    return tpl.TemplateResponse(request, "auth.html", {**CTX, "mode": "login", "error": None})


@app.get("/signup", response_class=HTMLResponse)
def signup_page(request: Request):
    return tpl.TemplateResponse(request, "auth.html", {**CTX, "mode": "signup", "error": None})


@app.get("/app", response_class=HTMLResponse)
def app_page(request: Request):
    user = auth.current_user(request)
    if not user:
        return RedirectResponse("/login", status_code=302)
    return tpl.TemplateResponse(request, "app.html", {**CTX, "email": user["email"]})


# ---------- auth ----------
@app.post("/login")
def login(request: Request, email: str = Form(...), password: str = Form(...)):
    u = db.get_user_by_email(email.strip().lower())
    if not u or not auth.verify_pw(password, u["pw_hash"]):
        return tpl.TemplateResponse(request, "auth.html", {**CTX, "mode": "login", "error": "Invalid email or password."})
    resp = RedirectResponse("/app", status_code=302)
    resp.set_cookie(auth.COOKIE, auth.make_cookie(u["id"]), httponly=True, samesite="lax", max_age=60 * 60 * 24 * 14)
    return resp


@app.post("/signup")
def signup(request: Request, email: str = Form(...), password: str = Form(...)):
    email = email.strip().lower()
    if len(password) < 6:
        return tpl.TemplateResponse(request, "auth.html", {**CTX, "mode": "signup", "error": "Password must be at least 6 characters."})
    if db.get_user_by_email(email):
        return tpl.TemplateResponse(request, "auth.html", {**CTX, "mode": "signup", "error": "That email is already registered."})
    uid = db.create_user(email, auth.hash_pw(password))
    resp = RedirectResponse("/app", status_code=302)
    resp.set_cookie(auth.COOKIE, auth.make_cookie(uid), httponly=True, samesite="lax", max_age=60 * 60 * 24 * 14)
    return resp


@app.get("/logout")
def logout():
    resp = RedirectResponse("/", status_code=302)
    resp.delete_cookie(auth.COOKIE)
    return resp


# ---------- API ----------
@app.post("/api/ingest")
async def api_ingest(request: Request, files: list[UploadFile] = File(...)):
    user = auth.require_user(request)
    results = []
    for f in files:
        dest = config.UPLOADS / f"{uuid.uuid4()}_{Path(f.filename).name}"
        with dest.open("wb") as out:
            shutil.copyfileobj(f.file, out)
        size_mb = dest.stat().st_size / 1e6
        if size_mb > config.MAX_UPLOAD_MB:
            dest.unlink(missing_ok=True)
            results.append({"filename": f.filename, "error": f"exceeds {config.MAX_UPLOAD_MB}MB"})
            continue
        try:
            r = ingest.ingest_file(user["id"], dest, f.filename)
            if r["doc_id"]:
                db.add_document(user["id"], r["doc_id"], f.filename, r["chunks"])
                db.log_usage(user["id"], "ingest", embed_t=r["embed_tokens"])
                results.append({"filename": f.filename, "chunks": r["chunks"]})
            else:
                results.append({"filename": f.filename, "error": "no extractable text"})
        except Exception as e:
            results.append({"filename": f.filename, "error": str(e)[:120]})
        finally:
            dest.unlink(missing_ok=True)
    return {"results": results, "documents": db.list_documents(user["id"])}


@app.get("/api/docs")
def api_docs(request: Request):
    user = auth.require_user(request)
    return {"documents": db.list_documents(user["id"])}


@app.post("/api/chat")
async def api_chat(request: Request):
    user = auth.require_user(request)
    payload = await request.json()
    question = (payload.get("question") or "").strip()
    doc_id = payload.get("doc_id") or None
    if not question:
        raise HTTPException(400, "empty question")
    if len(question) > 1000:
        raise HTTPException(400, "question too long")
    res = chat.answer(user["id"], question, doc_id=doc_id)
    u = res["usage"]
    db.log_usage(user["id"], "chat", embed_t=u["embed_tokens"], prompt_t=u["prompt_tokens"], output_t=u["output_tokens"])
    return res


@app.get("/api/usage")
def api_usage(request: Request):
    user = auth.require_user(request)
    return db.usage_stats(user["id"])


@app.post("/api/report")
async def api_report(request: Request):
    user = auth.require_user(request)
    payload = await request.json()
    title = (payload.get("title") or "DocuMind Report").strip()[:120]
    body = (payload.get("body") or "").strip()
    sources = payload.get("sources") or []
    if not body:
        raise HTTPException(400, "nothing to report")
    pdf = reports.build_pdf(title, body, sources, author_email=user["email"])
    db.log_usage(user["id"], "report")
    fname = "".join(ch if ch.isalnum() or ch in "-_ " else "" for ch in title).strip().replace(" ", "_") or "report"
    return Response(pdf, media_type="application/pdf",
                    headers={"content-disposition": f'attachment; filename="{fname}.pdf"'})


@app.get("/healthz")
def health():
    return {"ok": True, "product": config.PRODUCT_NAME}
