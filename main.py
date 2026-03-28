#!/usr/bin/env python3
"""
CubeMeta Backend Solver API — v2.0
====================================
FastAPI sunucusu; küp yüz verisi alarak Kociemba algoritmasıyla çözüm üretir
ve Groq LLaMA modeli aracılığıyla AI ipuçları sağlar.

Kurulum:
    pip install fastapi uvicorn kociemba groq python-dotenv

Çalıştırma:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import logging
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import kociemba
from dotenv import load_dotenv

# Groq — opsiyonel; anahtar yoksa AI ipucu devre dışı
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cubemeta")

app = FastAPI(title="CubeMeta Solver API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Modeller ─────────────────────────────────────────────────────────────────

class SolveRequest(BaseModel):
    faces: List[List[str]]          # 6 yüz × N kare (3×3 için 9 kare)
    grid_size: Optional[int] = 9    # Yüz başına kare sayısı

class SolveResponse(BaseModel):
    solution: List[str]
    move_count: int
    error: Optional[str] = None

class AiHintRequest(BaseModel):
    moves: List[str]                # Çözüm hamlelerinin listesi
    language: Optional[str] = "tr" # "tr" veya "en"
    groq_api_key: Optional[str] = None  # Uygulamadan gelen anahtar

class AiHintResponse(BaseModel):
    hint: str
    error: Optional[str] = None

# ── Renk → Kociemba dönüşümü ─────────────────────────────────────────────────

def faces_to_kociemba_string(faces: List[List[str]]) -> str:
    """
    6 yüzlük renk tablosunu kociemba formatına çevirir.
    Kociemba: 54 karakterli string — U(9) R(9) F(9) D(9) L(9) B(9)
    Uygulama sırası: [0]=Ön [1]=Arka [2]=Sol [3]=Sağ [4]=Üst [5]=Alt
    """
    front, back, left, right, top, bottom = faces

    # Merkez renkten yüz harfi eşleşmesi
    color_to_face = {
        top[4]:    'U',
        right[4]:  'R',
        front[4]:  'F',
        bottom[4]: 'D',
        left[4]:   'L',
        back[4]:   'B',
    }

    if len(color_to_face) != 6:
        raise ValueError("Yinelenen merkez rengi — geçersiz küp durumu")

    kociemba_order = [top, right, front, bottom, left, back]
    result = ""
    for face in kociemba_order:
        for sticker in face:
            if sticker not in color_to_face:
                raise ValueError(f"Bilinmeyen renk kodu: {sticker}")
            result += color_to_face[sticker]

    if len(result) != 54:
        raise ValueError(f"Geçersiz küp dizisi uzunluğu: {len(result)}")

    return result

# ── Groq AI ───────────────────────────────────────────────────────────────────

def get_groq_client(api_key: Optional[str]) -> Optional[object]:
    """Groq istemcisi oluştur — önce parametredeki, sonra env anahtarını kullan."""
    if not GROQ_AVAILABLE:
        return None
    key = api_key or os.getenv("GROQ_API_KEY", "")
    if not key:
        return None
    try:
        return Groq(api_key=key)
    except Exception:
        return None

def generate_ai_hint(moves: List[str], language: str = "tr", api_key: Optional[str] = None) -> str:
    """Groq LLaMA ile çözüm hakkında ipucu üret."""
    client = get_groq_client(api_key)
    if not client:
        return "Groq API anahtarı eksik. Lütfen Ayarlar > Yapay Zeka bölümünden girin."

    move_str = " → ".join(moves[:15])  # İlk 15 hamle
    more = f" ... ve {len(moves) - 15} hamle daha" if len(moves) > 15 else ""

    if language == "tr":
        prompt = f"""Bir Rubik Küp çözüm algoritması uzmanısın. Aşağıdaki çözüm adımlarını analiz et ve kullanıcıya Türkçe, kısa (3-4 cümle) ve anlaşılır bir ipucu ver. 
Hamleler: {move_str}{more}
Toplam hamle sayısı: {len(moves)}
Yanıt yalnızca Türkçe olmalı ve pratik tavsiyeler içermeli."""
    else:
        prompt = f"""You are a Rubik's Cube solving expert. Analyze the solution steps and provide a short (3-4 sentence) helpful hint.
Moves: {move_str}{more}
Total moves: {len(moves)}
Reply in English with practical advice."""

    try:
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=256,
            temperature=0.7,
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Groq hatası: {e}")
        return "Yapay zeka ipucu şu an kullanılamıyor."

# ── Rotalar ───────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "status": "CubeMeta Solver API çalışıyor",
        "version": "2.0.0",
        "groq_available": GROQ_AVAILABLE and bool(os.getenv("GROQ_API_KEY"))
    }

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/api/v1/solve", response_model=SolveResponse)
def solve(request: SolveRequest):
    logger.info(f"Çözüm isteği: {len(request.faces)} yüz, izgara={request.grid_size}")

    if len(request.faces) != 6:
        raise HTTPException(status_code=400, detail="Tam olarak 6 yüz gereklidir")

    grid_size = request.grid_size or 9

    for i, face in enumerate(request.faces):
        if len(face) != grid_size:
            raise HTTPException(
                status_code=400,
                detail=f"Yüz {i}: {grid_size} kare gerekli, {len(face)} geldi"
            )

    # Standart olmayan ızgara boyutu — merkez kare 3×3 alt kümesi al
    working_faces = request.faces
    if grid_size != 9:
        try:
            working_faces = [extract_center_9(face, grid_size) for face in request.faces]
            logger.info(f"Izgara {grid_size}→9'a indirgendt")
        except Exception as e:
            return SolveResponse(solution=[], move_count=0, error=f"Izgara dönüşümü başarısız: {e}")

    try:
        cube_string = faces_to_kociemba_string(working_faces)
        logger.info(f"Kociemba girdisi: {cube_string}")
        solution_str = kociemba.solve(cube_string)
        moves = solution_str.strip().split()
        logger.info(f"Çözüm: {moves} ({len(moves)} hamle)")
        return SolveResponse(solution=moves, move_count=len(moves))
    except ValueError as e:
        logger.error(f"Doğrulama hatası: {e}")
        return SolveResponse(solution=[], move_count=0, error=str(e))
    except Exception as e:
        logger.error(f"Çözücü hatası: {e}")
        raise HTTPException(status_code=500, detail=f"Çözüm başarısız: {str(e)}")

@app.post("/api/v1/ai-hint", response_model=AiHintResponse)
def ai_hint(request: AiHintRequest):
    """Groq LLaMA ile çözüm ipucu üret."""
    logger.info(f"AI ipucu isteği: {len(request.moves)} hamle, dil={request.language}")
    hint = generate_ai_hint(request.moves, request.language, request.groq_api_key)
    return AiHintResponse(hint=hint)

# ── Yardımcı ──────────────────────────────────────────────────────────────────

def extract_center_9(face: List[str], grid_size: int) -> List[str]:
    """
    Büyük bir ızgaradan merkez 3×3 bloğu çıkar.
    Örnek: 4×3=12 kareden merkez 3×3 → 9 kare
    """
    rows = _approx_rows(grid_size)
    cols = grid_size // rows
    matrix = [face[r * cols:(r + 1) * cols] for r in range(rows)]
    # Merkezi bul
    start_r = (rows - 3) // 2
    start_c = (cols - 3) // 2
    result = []
    for r in range(start_r, start_r + 3):
        for c in range(start_c, start_c + 3):
            result.append(matrix[r][c])
    return result

def _approx_rows(size: int) -> int:
    import math
    return max(2, round(math.sqrt(size)))

# ── Giriş noktası ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
