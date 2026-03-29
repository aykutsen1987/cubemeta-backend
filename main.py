#!/usr/bin/env python3
"""
CubeMeta Backend Solver API — v2.2
====================================
v2.2 değişiklikleri:
  - /api/v1/solve endpoint'ine detaylı debug logging eklendi
  - Gelen faces verisi loglanıyor (hangi renkler geliyor?)
  - "cubestring is invalid" hatasında daha açıklayıcı mesaj
  - --reload FLAG KALDIRILDI (Render prod için gerekli değil, restart döngüsüne neden oluyor)
"""

import os
import logging
import math
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import kociemba
from dotenv import load_dotenv

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cubemeta")

app = FastAPI(title="CubeMeta Solver API", version="2.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Modeller ─────────────────────────────────────────────────────────────────

class SolveRequest(BaseModel):
    faces: List[List[str]]
    grid_size: Optional[int] = 9

class SolveResponse(BaseModel):
    solution: List[str]
    move_count: int
    error: Optional[str] = None
    debug: Optional[str] = None   # debug bilgisi

class AiHintRequest(BaseModel):
    moves: List[str]
    language: Optional[str] = "tr"

class AiHintResponse(BaseModel):
    hint: str
    error: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    version: str
    groq_available: bool

# ── Kociemba dönüşümü ────────────────────────────────────────────────────────

def faces_to_kociemba_string(faces: List[List[str]]) -> str:
    """
    Uygulama sırası: [0]=Ön [1]=Arka [2]=Sol [3]=Sağ [4]=Üst [5]=Alt
    Kociemba sırası: U R F D L B
    """
    front, back, left, right, top, bottom = faces

    centers = {
        "top(U)":    top[4],
        "right(R)":  right[4],
        "front(F)":  front[4],
        "bottom(D)": bottom[4],
        "left(L)":   left[4],
        "back(B)":   back[4],
    }
    logger.info(f"Merkez renkler: {centers}")

    try:
        color_to_face = {
            top[4]:    'U',
            right[4]:  'R',
            front[4]:  'F',
            bottom[4]: 'D',
            left[4]:   'L',
            back[4]:   'B',
        }
    except IndexError:
        raise ValueError("Yüz verileri eksik — merkez kareler bulunamadı.")

    if len(color_to_face) != 6:
        raise ValueError(
            f"Merkez renk çakışması! 6 farklı renk gerekli. "
            f"Mevcut merkezler: {centers}"
        )

    kociemba_order = [top, right, front, bottom, left, back]
    result = ""
    for face_name, face in zip(["top","right","front","bottom","left","back"], kociemba_order):
        for sticker in face:
            if sticker not in color_to_face:
                raise ValueError(f"Bilinmeyen renk kodu: '{sticker}' ({face_name} yüzünde)")
            result += color_to_face[sticker]

    return result

# ── Groq ─────────────────────────────────────────────────────────────────────

def get_groq_client():
    if not GROQ_AVAILABLE:
        return None
    key = os.getenv("GROQ_API_KEY", "")
    if not key:
        return None
    try:
        return Groq(api_key=key)
    except Exception:
        return None

def generate_ai_hint(moves: List[str], language: str = "tr") -> str:
    client = get_groq_client()
    if not client:
        return "Groq API anahtarı tanımlı değil."
    move_str = " → ".join(moves[:15])
    more = f" ... ve {len(moves)-15} hamle daha" if len(moves) > 15 else ""
    if language == "tr":
        prompt = f"Rubik Küp uzmanısın. Kısa Türkçe ipucu ver: {move_str}{more}"
    else:
        prompt = f"Rubik's Cube expert. Short hint: {move_str}{more}"
    try:
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=256, temperature=0.7,
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        return f"AI ipucu hatası: {e}"

# ── Rotalar ───────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "status": "CubeMeta Solver API çalışıyor",
        "version": "2.2.0",
        "groq_available": GROQ_AVAILABLE and bool(os.getenv("GROQ_API_KEY")),
        "endpoints": ["/api/v1/health", "/api/v1/solve", "/api/v1/ai-hint"]
    }

@app.get("/api/v1/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        version="2.2.0",
        groq_available=GROQ_AVAILABLE and bool(os.getenv("GROQ_API_KEY"))
    )

@app.post("/api/v1/solve", response_model=SolveResponse)
def solve(request: SolveRequest):
    if len(request.faces) != 6:
        return SolveResponse(solution=[], move_count=0,
                             error=f"6 yüz gerekli, {len(request.faces)} geldi")

    # ── Debug: gelen renkleri logla ──────────────────────────────────────────
    face_names = ["Ön", "Arka", "Sol", "Sağ", "Üst", "Alt"]
    color_summary = {}
    total_colors = {}
    for i, face in enumerate(request.faces):
        logger.info(f"Yüz[{i}] {face_names[i]}: {face}")
        for c in face:
            total_colors[c] = total_colors.get(c, 0) + 1
        color_summary[face_names[i]] = f"merkez={face[4] if len(face)>4 else '?'}"

    logger.info(f"Renk dağılımı: {total_colors}")
    logger.info(f"Merkez özeti: {color_summary}")

    # ── Renk dağılımı kontrolü ───────────────────────────────────────────────
    white_count = total_colors.get("W", 0)
    if white_count > 9:
        return SolveResponse(
            solution=[], move_count=0,
            error=f"Renk tespiti hatası: Beyaz (W) rengi {white_count} kez sayıldı (max 9). "
                  f"Kamera taraması yeniden yapılmalı.",
            debug=f"Toplam renk dağılımı: {total_colors}"
        )

    working_faces = request.faces
    if (request.grid_size or 9) != 9:
        try:
            working_faces = [extract_center_9(f, request.grid_size) for f in request.faces]
        except Exception as e:
            return SolveResponse(solution=[], move_count=0, error=f"Izgara dönüşüm hatası: {e}")

    for i, face in enumerate(working_faces):
        if len(face) != 9:
            return SolveResponse(solution=[], move_count=0,
                                 error=f"Yüz {i} ({face_names[i]}): 9 kare gerekli, {len(face)} var")

    try:
        cube_string = faces_to_kociemba_string(working_faces)
        logger.info(f"Kociemba string ({len(cube_string)} karakter): {cube_string}")
        solution_str = kociemba.solve(cube_string)
        moves = solution_str.strip().split()
        logger.info(f"Çözüm: {len(moves)} hamle → {moves}")
        return SolveResponse(solution=moves, move_count=len(moves))
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Çözücü hatası: {error_msg}")
        if "invalid" in error_msg.lower():
            return SolveResponse(
                solution=[], move_count=0,
                error=f"Geçersiz küp durumu: Renk dağılımı fiziksel olarak imkansız. "
                      f"Her renkten tam 9 kare olmalı. Mevcut dağılım: {total_colors}",
                debug=f"Kociemba hatası: {error_msg}"
            )
        return SolveResponse(solution=[], move_count=0, error=error_msg)

@app.post("/api/v1/ai-hint", response_model=AiHintResponse)
def ai_hint(request: AiHintRequest):
    return AiHintResponse(hint=generate_ai_hint(request.moves, request.language))

# ── Yardımcılar ───────────────────────────────────────────────────────────────

def extract_center_9(face: List[str], grid_size: int) -> List[str]:
    rows = _approx_rows(grid_size)
    cols = grid_size // rows
    matrix = [face[r*cols:(r+1)*cols] for r in range(rows)]
    sr = (rows-3)//2; sc = (cols-3)//2
    return [matrix[r][c] for r in range(sr, sr+3) for c in range(sc, sc+3)]

def _approx_rows(size: int) -> int:
    return max(2, round(math.sqrt(size)))

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
