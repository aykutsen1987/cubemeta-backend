#!/usr/bin/env python3
"""
CubeMeta Backend Solver API — v2.0
====================================
FastAPI sunucusu; küp yüz verisi alarak Kociemba algoritmasıyla çözüm üretir
ve Groq LLaMA modeli aracılığıyla AI ipuçları sağlar.
"""

import os
import logging
import math  # _approx_rows için eklendi
from typing import List, Optional

from fastapi import FastAPI, HTTPException
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
    language: Optional[str] = "tr"  # "tr" veya "en"

class AiHintResponse(BaseModel):
    hint: str
    error: Optional[str] = None

# ── Renk → Kociemba dönüşümü ─────────────────────────────────────────────────

def faces_to_kociemba_string(faces: List[List[str]]) -> str:
    """
    6 yüzlük renk tablosunu kociemba formatına çevirir.
    Kociemba sırası: U(Up) R(Right) F(Front) D(Down) L(Left) B(Back)
    Senin uygulamanın gönderdiği sıra: [0]=Ön [1]=Arka [2]=Sol [3]=Sağ [4]=Üst [5]=Alt
    """
    front, back, left, right, top, bottom = faces

    # Merkez renkten yüz harfi eşleşmesi
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
        raise ValueError("Yüz verileri eksik veya merkez kareler bulunamadı.")

    if len(color_to_face) != 6:
        raise ValueError("Yinelenen merkez rengi — geçersiz küp durumu")

    # Kociemba algoritmasının beklediği özel yüz sırası (U R F D L B)
    kociemba_order = [top, right, front, bottom, left, back]
    result = ""
    for face in kociemba_order:
        for sticker in face:
            if sticker not in color_to_face:
                raise ValueError(f"Bilinmeyen renk kodu: {sticker}")
            result += color_to_face[sticker]

    return result

# ── Groq AI ───────────────────────────────────────────────────────────────────

def get_groq_client() -> Optional[object]:
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
        return "Groq API anahtarı sunucuda tanımlı değil. Render Environment Variables'a GROQ_API_KEY ekleyin."

    move_str = " → ".join(moves[:15])
    more = f" ... ve {len(moves) - 15} hamle daha" if len(moves) > 15 else ""

    if language == "tr":
        prompt = f"Rubik Küp çözüm uzmanısın. Şu hamleleri analiz et ve kısa (3-4 cümle) Türkçe ipucu ver: {move_str}{more}"
    else:
        prompt = f"You are a Rubik's Cube expert. Analyze these moves and give a short hint: {move_str}{more}"

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

@app.post("/api/v1/solve", response_model=SolveResponse)
def solve(request: SolveRequest):
    if len(request.faces) != 6:
        raise HTTPException(status_code=400, detail="Tam olarak 6 yüz gereklidir")

    grid_size = request.grid_size or 9

    # Izgara boyutu kontrolü ve gerekirse 3x3'e (9 kare) indirgeme
    working_faces = request.faces
    if grid_size != 9:
        try:
            working_faces = [extract_center_9(face, grid_size) for face in request.faces]
        except Exception as e:
            return SolveResponse(solution=[], move_count=0, error=f"Izgara dönüşümü hatası: {e}")

    try:
        cube_string = faces_to_kociemba_string(working_faces)
        solution_str = kociemba.solve(cube_string)
        moves = solution_str.strip().split()
        return SolveResponse(solution=moves, move_count=len(moves))
    except Exception as e:
        logger.error(f"Çözücü hatası: {e}")
        return SolveResponse(solution=[], move_count=0, error=str(e))

@app.post("/api/v1/ai-hint", response_model=AiHintResponse)
def ai_hint(request: AiHintRequest):
    hint = generate_ai_hint(request.moves, request.language)
    return AiHintResponse(hint=hint)

# ── Yardımcılar ───────────────────────────────────────────────────────────────

def extract_center_9(face: List[str], grid_size: int) -> List[str]:
    rows = _approx_rows(grid_size)
    cols = grid_size // rows
    matrix = [face[r * cols:(r + 1) * cols] for r in range(rows)]
    start_r = (rows - 3) // 2
    start_c = (cols - 3) // 2
    result = []
    for r in range(start_r, start_r + 3):
        for c in range(start_c, start_c + 3):
            result.append(matrix[r][c])
    return result

def _approx_rows(size: int) -> int:
    return max(2, round(math.sqrt(size)))

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
