#!/usr/bin/env python3
"""
CubeMeta Backend Solver API — v2.3
  - Renk dağılımı "fiziksel olarak imkansız" ise açık hata mesajı
  - Backend artık her zaman gönderilen veriyi işler (validation Android tarafında)
"""

import os, logging, math
from typing import List, Optional
from fastapi import FastAPI
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

app = FastAPI(title="CubeMeta Solver API", version="2.3.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class SolveRequest(BaseModel):
    faces: List[List[str]]
    grid_size: Optional[int] = 9

class SolveResponse(BaseModel):
    solution: List[str]
    move_count: int
    error: Optional[str] = None

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

FACE_NAMES = ["Ön", "Arka", "Sol", "Sağ", "Üst", "Alt"]

def faces_to_kociemba_string(faces):
    front, back, left, right, top, bottom = faces
    centers = {"top": top[4], "right": right[4], "front": front[4],
                "bottom": bottom[4], "left": left[4], "back": back[4]}
    logger.info(f"Merkezler: {centers}")

    color_to_face = {}
    for color, letter in [(top[4],'U'),(right[4],'R'),(front[4],'F'),
                           (bottom[4],'D'),(left[4],'L'),(back[4],'B')]:
        if color in color_to_face:
            raise ValueError(
                f"Merkez renk çakışması: '{color}' birden fazla yüzde merkez. "
                f"Merkezler: {centers}"
            )
        color_to_face[color] = letter

    result = ""
    for face in [top, right, front, bottom, left, back]:
        for sticker in face:
            if sticker not in color_to_face:
                raise ValueError(f"Bilinmeyen renk: '{sticker}'")
            result += color_to_face[sticker]
    logger.info(f"Kociemba: {result}")
    return result

def get_groq_client():
    if not GROQ_AVAILABLE: return None
    key = os.getenv("GROQ_API_KEY", "")
    if not key: return None
    try: return Groq(api_key=key)
    except: return None

def generate_ai_hint(moves, language="tr"):
    client = get_groq_client()
    if not client:
        return "Groq API anahtarı tanımlı değil."
    move_str = " → ".join(moves[:15])
    more = f" ... ve {len(moves)-15} hamle daha" if len(moves) > 15 else ""
    prompt = (f"Rubik Küp uzmanısın. Kısa Türkçe ipucu ver: {move_str}{more}"
              if language == "tr"
              else f"Rubik's Cube expert. Short hint: {move_str}{more}")
    try:
        r = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role":"user","content":prompt}],
            max_tokens=256, temperature=0.7)
        return r.choices[0].message.content.strip()
    except Exception as e:
        return f"AI ipucu hatası: {e}"

@app.get("/")
def root():
    return {"status": "ok", "version": "2.3.0"}

@app.get("/api/v1/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok", version="2.3.0",
                          groq_available=GROQ_AVAILABLE and bool(os.getenv("GROQ_API_KEY")))

@app.post("/api/v1/solve", response_model=SolveResponse)
def solve(request: SolveRequest):
    if len(request.faces) != 6:
        return SolveResponse(solution=[], move_count=0,
                             error=f"6 yüz gerekli, {len(request.faces)} geldi")

    # Renk dağılımı logu
    total = {}
    for i, face in enumerate(request.faces):
        logger.info(f"Yüz[{i}] {FACE_NAMES[i]}: {face}")
        for c in face: total[c] = total.get(c, 0) + 1
    logger.info(f"Dağılım: {total}")

    working_faces = request.faces
    if (request.grid_size or 9) != 9:
        try:
            working_faces = [extract_center_9(f, request.grid_size) for f in request.faces]
        except Exception as e:
            return SolveResponse(solution=[], move_count=0, error=f"Izgara hatası: {e}")

    try:
        cube_string = faces_to_kociemba_string(working_faces)
        solution_str = kociemba.solve(cube_string)
        moves = solution_str.strip().split()
        logger.info(f"Çözüm: {len(moves)} hamle")
        return SolveResponse(solution=moves, move_count=len(moves))
    except ValueError as e:
        # Merkez çakışması veya bilinmeyen renk — kullanıcıya göster
        logger.error(f"Veri hatası: {e}")
        return SolveResponse(solution=[], move_count=0, error=str(e))
    except Exception as e:
        err = str(e)
        logger.error(f"Kociemba hatası: {err}")
        # Kociemba "invalid" dediğinde dağılımı ekle
        dist_str = ", ".join(f"{k}:{v}" for k, v in sorted(total.items()))
        return SolveResponse(
            solution=[], move_count=0,
            error=(
                f"Küp fiziksel olarak geçersiz — renk dağılımı: {dist_str}. "
                f"Her renkten tam 9 kare olmalı. "
                f"Preview ekranında renkleri düzeltin ve tekrar deneyin."
            )
        )

@app.post("/api/v1/ai-hint", response_model=AiHintResponse)
def ai_hint(request: AiHintRequest):
    return AiHintResponse(hint=generate_ai_hint(request.moves, request.language))

def extract_center_9(face, grid_size):
    rows = max(2, round(math.sqrt(grid_size)))
    cols = grid_size // rows
    matrix = [face[r*cols:(r+1)*cols] for r in range(rows)]
    sr, sc = (rows-3)//2, (cols-3)//2
    return [matrix[r][c] for r in range(sr, sr+3) for c in range(sc, sc+3)]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
