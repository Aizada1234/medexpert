from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from google import genai
import os
import traceback
import sqlite3
import hashlib
import secrets
from datetime import datetime
from typing import Optional


# =========================
# Gemini AI
# =========================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY не найден в переменных окружения")

client = genai.Client(api_key=GEMINI_API_KEY)

GEMINI_MODEL = "gemini-2.0-flash"


# =========================
# FastAPI app
# =========================
app = FastAPI(
    title="MedExpert Backend",
    version="1.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DB_PATH = "medexpert.db"


# =========================
# Database helpers
# =========================
def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def init_db():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        first_name TEXT NOT NULL,
        last_name TEXT NOT NULL,
        phone TEXT NOT NULL UNIQUE,
        password_hash TEXT NOT NULL,
        created_at TEXT NOT NULL
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS sessions (
        token TEXT PRIMARY KEY,
        user_id INTEGER NOT NULL,
        created_at TEXT NOT NULL,
        FOREIGN KEY(user_id) REFERENCES users(id)
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS diagnosis_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        diagnosis TEXT NOT NULL,
        recommendation TEXT NOT NULL,
        symptoms TEXT NOT NULL,
        risk_level TEXT NOT NULL,
        created_at TEXT NOT NULL,
        FOREIGN KEY(user_id) REFERENCES users(id)
    )
    """)

    conn.commit()
    conn.close()


@app.on_event("startup")
def on_startup():
    init_db()


def get_current_user_id(authorization: Optional[str]) -> int:
    if not authorization:
        raise HTTPException(status_code=401, detail="Не передан Authorization token")

    token = authorization.replace("Bearer ", "").strip()

    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT user_id FROM sessions WHERE token = ?", (token,))
    row = cursor.fetchone()
    conn.close()

    if row is None:
        raise HTTPException(status_code=401, detail="Неверный или устаревший token")

    return int(row["user_id"])


def fallback_medical_reply(user_message: str) -> str:
    return (
        "ИИ-сервис временно недоступен из-за ограничения бесплатной квоты Gemini API.\n\n"
        "Предварительная оценка: по вашему сообщению можно провести только общую ориентировочную оценку. "
        "Для более точной предварительной рекомендации нужно указать симптомы, их длительность, температуру, "
        "наличие боли, слабости, тошноты, кашля, головокружения и хронических заболеваний.\n\n"
        "Возможные причины могут быть разными и зависят от конкретных симптомов. Это может быть временное "
        "функциональное состояние, инфекционный процесс, переутомление, воспаление или другое нарушение.\n\n"
        "Что можно сделать сейчас: наблюдайте за состоянием, измерьте температуру, пейте достаточно воды, "
        "избегайте самолечения сильными препаратами и при возможности обратитесь к врачу.\n\n"
        "Срочно обратитесь за медицинской помощью, если есть сильная боль, одышка, потеря сознания, "
        "кровотечение, высокая температура, выраженная слабость, боль в груди или резкое ухудшение состояния."
    )


def make_ai_response(reply_text: str, success: bool = True):
    reply_text = reply_text.strip()

    return {
        "reply": reply_text,
        "message": reply_text,
        "text": reply_text,
        "success": success
    }


# =========================
# DTO models
# =========================
class RegisterRequest(BaseModel):
    firstName: str
    lastName: str
    phone: str
    password: str


class LoginRequest(BaseModel):
    phone: str
    password: str


class AuthResponse(BaseModel):
    userId: int
    token: str
    firstName: str
    lastName: str
    phone: str
    success: bool = True


class ProfileResponse(BaseModel):
    userId: int
    firstName: str
    lastName: str
    phone: str
    success: bool = True


class UpdateProfileRequest(BaseModel):
    firstName: str
    lastName: str
    newPassword: Optional[str] = None


class DiagnosisHistoryRequest(BaseModel):
    diagnosis: str
    recommendation: str
    symptoms: str
    riskLevel: str
    createdAt: Optional[str] = None


class DiagnosisHistoryResponse(BaseModel):
    id: int
    diagnosis: str
    recommendation: str
    symptoms: str
    riskLevel: str
    createdAt: str


class AiChatRequest(BaseModel):
    message: str
    symptoms: list[str] = Field(default_factory=list)
    metrics: dict[str, str] = Field(default_factory=dict)
    mode: str = "general"


class AiChatResponse(BaseModel):
    reply: str
    message: str
    text: str
    success: bool = True


# =========================
# Base endpoints
# =========================
@app.get("/")
async def root():
    return {
        "status": "ok",
        "message": "MedExpert backend is running",
        "database": "SQLite connected"
    }

@app.get("/debug-key")
async def debug_key():
    key = os.getenv("GEMINI_API_KEY", "")

    return {
        "length": len(key),
        "prefix": key[:8] if key else None,
        "suffix": key[-4:] if key else None
    }
@app.get("/db/test")
async def db_test():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) AS count FROM users")
    users_count = cursor.fetchone()["count"]

    cursor.execute("SELECT COUNT(*) AS count FROM diagnosis_history")
    history_count = cursor.fetchone()["count"]

    conn.close()

    return {
        "status": "ok",
        "users": users_count,
        "history": history_count
    }


# =========================
# Auth endpoints
# =========================
@app.post("/auth/register", response_model=AuthResponse)
async def register(request: RegisterRequest):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT id FROM users WHERE phone = ?", (request.phone,))
    if cursor.fetchone() is not None:
        conn.close()
        raise HTTPException(status_code=400, detail="Пользователь уже существует")

    created_at = datetime.now().isoformat()

    cursor.execute(
        """
        INSERT INTO users (first_name, last_name, phone, password_hash, created_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            request.firstName,
            request.lastName,
            request.phone,
            hash_password(request.password),
            created_at
        )
    )

    conn.commit()

    user_id = cursor.lastrowid
    token = secrets.token_hex(32)

    cursor.execute(
        "INSERT INTO sessions (token, user_id, created_at) VALUES (?, ?, ?)",
        (token, user_id, created_at)
    )

    conn.commit()
    conn.close()

    return AuthResponse(
        userId=user_id,
        token=token,
        firstName=request.firstName,
        lastName=request.lastName,
        phone=request.phone
    )


@app.post("/auth/login", response_model=AuthResponse)
async def login(request: LoginRequest):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT id, first_name, last_name, phone
        FROM users
        WHERE phone = ? AND password_hash = ?
        """,
        (request.phone, hash_password(request.password))
    )

    user = cursor.fetchone()

    if user is None:
        conn.close()
        raise HTTPException(status_code=401, detail="Неверный телефон или пароль")

    token = secrets.token_hex(32)
    created_at = datetime.now().isoformat()

    cursor.execute(
        "INSERT INTO sessions (token, user_id, created_at) VALUES (?, ?, ?)",
        (token, user["id"], created_at)
    )

    conn.commit()
    conn.close()

    return AuthResponse(
        userId=user["id"],
        token=token,
        firstName=user["first_name"],
        lastName=user["last_name"],
        phone=user["phone"]
    )


@app.post("/auth/logout")
async def logout(authorization: Optional[str] = Header(None)):
    token = authorization.replace("Bearer ", "").strip() if authorization else ""

    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM sessions WHERE token = ?", (token,))
    conn.commit()
    conn.close()

    return {
        "success": True,
        "message": "Выход выполнен"
    }


# =========================
# Profile endpoints
# =========================
@app.get("/profile", response_model=ProfileResponse)
async def get_profile(authorization: Optional[str] = Header(None)):
    user_id = get_current_user_id(authorization)

    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, first_name, last_name, phone FROM users WHERE id = ?",
        (user_id,)
    )
    user = cursor.fetchone()
    conn.close()

    if user is None:
        raise HTTPException(status_code=404, detail="Пользователь не найден")

    return ProfileResponse(
        userId=user["id"],
        firstName=user["first_name"],
        lastName=user["last_name"],
        phone=user["phone"]
    )


@app.put("/profile", response_model=ProfileResponse)
async def update_profile(
    request: UpdateProfileRequest,
    authorization: Optional[str] = Header(None)
):
    user_id = get_current_user_id(authorization)

    first_name = request.firstName.strip()
    last_name = request.lastName.strip()

    if not first_name or not last_name:
        raise HTTPException(status_code=400, detail="Имя и фамилия не должны быть пустыми")

    conn = get_connection()
    cursor = conn.cursor()

    if request.newPassword and request.newPassword.strip():
        cursor.execute(
            """
            UPDATE users
            SET first_name = ?, last_name = ?, password_hash = ?
            WHERE id = ?
            """,
            (first_name, last_name, hash_password(request.newPassword.strip()), user_id)
        )
    else:
        cursor.execute(
            """
            UPDATE users
            SET first_name = ?, last_name = ?
            WHERE id = ?
            """,
            (first_name, last_name, user_id)
        )

    conn.commit()

    cursor.execute(
        "SELECT id, first_name, last_name, phone FROM users WHERE id = ?",
        (user_id,)
    )
    user = cursor.fetchone()
    conn.close()

    if user is None:
        raise HTTPException(status_code=404, detail="Пользователь не найден")

    return ProfileResponse(
        userId=user["id"],
        firstName=user["first_name"],
        lastName=user["last_name"],
        phone=user["phone"]
    )


# =========================
# History endpoints
# =========================
@app.post("/history", response_model=DiagnosisHistoryResponse)
async def save_history(
    request: DiagnosisHistoryRequest,
    authorization: Optional[str] = Header(None)
):
    user_id = get_current_user_id(authorization)
    created_at = request.createdAt or datetime.now().isoformat()

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO diagnosis_history
        (user_id, diagnosis, recommendation, symptoms, risk_level, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            user_id,
            request.diagnosis,
            request.recommendation,
            request.symptoms,
            request.riskLevel,
            created_at
        )
    )

    conn.commit()
    history_id = cursor.lastrowid
    conn.close()

    return DiagnosisHistoryResponse(
        id=history_id,
        diagnosis=request.diagnosis,
        recommendation=request.recommendation,
        symptoms=request.symptoms,
        riskLevel=request.riskLevel,
        createdAt=created_at
    )


@app.get("/history", response_model=list[DiagnosisHistoryResponse])
async def get_history(authorization: Optional[str] = Header(None)):
    user_id = get_current_user_id(authorization)

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT id, diagnosis, recommendation, symptoms, risk_level, created_at
        FROM diagnosis_history
        WHERE user_id = ?
        ORDER BY created_at DESC
        """,
        (user_id,)
    )

    rows = cursor.fetchall()
    conn.close()

    return [
        DiagnosisHistoryResponse(
            id=row["id"],
            diagnosis=row["diagnosis"],
            recommendation=row["recommendation"],
            symptoms=row["symptoms"],
            riskLevel=row["risk_level"],
            createdAt=row["created_at"]
        )
        for row in rows
    ]




@app.delete("/history/{history_id}")
async def delete_history_item(
    history_id: int,
    authorization: Optional[str] = Header(None)
):
    user_id = get_current_user_id(authorization)

    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "DELETE FROM diagnosis_history WHERE id = ? AND user_id = ?",
        (history_id, user_id)
    )
    conn.commit()
    deleted_count = cursor.rowcount
    conn.close()

    if deleted_count == 0:
        raise HTTPException(status_code=404, detail="Запись истории не найдена")

    return {
        "success": True,
        "message": "Запись истории удалена"
    }


@app.delete("/history")
async def clear_history(authorization: Optional[str] = Header(None)):
    user_id = get_current_user_id(authorization)

    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM diagnosis_history WHERE user_id = ?", (user_id,))
    conn.commit()
    conn.close()

    return {
        "success": True,
        "message": "История очищена"
    }


# =========================
# Gemini endpoints
# =========================
@app.get("/test-gemini")
async def test_gemini():
    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents="Ответь по-русски одним предложением: привет"
        )

        reply_text = response.text or "Не удалось получить ответ Gemini."

        return JSONResponse(
            content=make_ai_response(reply_text, success=True),
            media_type="application/json; charset=utf-8"
        )

    except Exception as e:
        traceback.print_exc()

        return JSONResponse(
            content={
                "success": False,
                "raw_error": str(e)
            }
        )
@app.get("/models")
async def list_models():
    try:
        models = client.models.list()

        result = []

        for model in models:
            result.append(model.name)

        return {"models": result}

    except Exception as e:
        return {"error": str(e)}
        
@app.post("/ai/chat", response_model=AiChatResponse)
async def ai_chat(request: AiChatRequest):
    symptoms_text = ", ".join(request.symptoms) if request.symptoms else "не указаны"

    metrics_text = ", ".join(
        [f"{key}: {value}" for key, value in request.metrics.items()]
    ) if request.metrics else "не указаны"

    prompt = f"""
Ты медицинский AI-ассистент учебного мобильного приложения MedExpert.

Правила:
Не ставь окончательный диагноз.
Отвечай только на русском языке.
Пиши, что это предварительная оценка.
Рекомендуй обратиться к врачу.
Если симптомы опасные, советуй срочно обратиться за медицинской помощью.
Не используй markdown-разметку, звездочки, решетки, тройные тире.
Пиши обычным простым текстом.

Режим: {request.mode}

Сообщение пользователя:
{request.message}

Симптомы:
{symptoms_text}

Показатели:
{metrics_text}

Сформируй ответ по разделам:
1. Предварительная оценка
2. Возможные причины
3. Что можно сделать сейчас
4. Когда срочно нужен врач
"""

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt
        )

        reply_text = response.text or "Не удалось получить ответ Gemini."

        return AiChatResponse(
            **make_ai_response(reply_text, success=True)
        )

    except Exception as e:
        traceback.print_exc()

        error_text = str(e)

        if "429" in error_text or "RESOURCE_EXHAUSTED" in error_text or "quota" in error_text.lower():
            reply_text = fallback_medical_reply(request.message)

            return AiChatResponse(
                **make_ai_response(reply_text, success=True)
            )

        reply_text = (
            "Произошла временная ошибка при обращении к ИИ-сервису. "
            "Проверьте подключение backend, API key и модель Gemini."
        )

        return AiChatResponse(
            **make_ai_response(reply_text, success=False)
        )
