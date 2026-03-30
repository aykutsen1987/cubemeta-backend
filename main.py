#!/usr/bin/env python3
"""
CubeMeta Backend Solver API — v2.4
  - Groq yerine Anthropic Claude API ile AI ipucu
  - Kociemba yüz sırası düzeltmesi: U R F D L B
  - Renk dağılımı "fiziksel olarak imkansız" ise açık hata mesajı
"""

import os, logging, math
from typing import List, Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import kociemba
from dotenv import load_dotenv

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cubemeta")

app = FastAPI(title="CubeMeta Solver API", version="2.4.0")
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
    ai_available: bool

FACE_NAMES = ["Ön", "Arka", "Sol", "Sağ", "Üst", "Alt"]

# Android toApiFormat() gönderim sırası:
#   index 0 = FRONT, 1 = BACK, 2 = LEFT, 3 = RIGHT, 4 = TOP, 5 = BOTTOM
# Kociemba string sırası: U R F D L B
#   U = faces[4] (TOP), R = faces[3] (RIGHT), F = faces[0] (FRONT)
#   D = faces[5] (BOTTOM), L = faces[2] (LEFT), B = faces[1] (BACK)

def faces_to_kociemba_string(faces):
    front  = faces[0]
    back   = faces[1]
    left   = faces[2]
    right  = faces[3]
    top    = faces[4]
    bottom = faces[5]

    centers = {
        "U": top[4], "R": right[4], "F": front[4],
        "D": bottom[4], "L": left[4], "B": back[4]
    }
    logger.info(f"Merkezler: {centers}")

    color_to_face = {}
    for face_letter, face_data in [("U", top), ("R", right), ("F", front),
                                     ("D", bottom), ("L", left), ("B", back)]:
        center_color = face_data[4]
        if center_color in color_to_face:
            raise ValueError(
                f"Merkez renk cakismasi: '{center_color}' birden fazla yuzde merkez. "
                f"Merkezler: {centers}"
            )
        color_to_face[center_color] = face_letter

    # Kociemba string: U R F D L B sirasıyla her yuzun 9 karesi
    result = ""
    for face_data in [top, right, front, bottom, left, back]:
        for sticker in face_data:
            if sticker not in color_to_face:
                raise ValueError(
                    f"Bilinmeyen renk: '{sticker}'. Gecerli renkler: {list(color_to_face.keys())}"
                )
            result += color_to_face[sticker]

    logger.info(f"Kociemba string ({len(result)} char): {result}")
    return result


def get_anthropic_client():
    if not ANTHROPIC_AVAILABLE:
        return None
    key = os.getenv("ANTHROPIC_API_KEY", "")
    if not key:
        return None
    try:
        return anthropic.Anthropic(api_key=key)
    except Exception:
        return None


def generate_ai_hint(moves, language="tr"):
    client = get_anthropic_client()
    if not client:
        return "Anthropic API anahtari tanimli degil. ANTHROPIC_API_KEY ortam degiskenini ayarlayin."

    move_str = " -> ".join(moves[:15])
    more = f" ... ve {len(moves)-15} hamle daha" if len(moves) > 15 else ""

    if language == "tr":
        prompt = (
            f"Sen bir Rubik Kup uzmanisın. "
            f"Asagidaki cozum adimlari icin kisa ve anlasılir Turkce bir ipucu ver. "
            f"Maksimum 2-3 cumle olsun.\n\n"
            f"Hamleler: {move_str}{more}\n\nIpucu:"
        )
    else:
        prompt = (
            f"You are a Rubik's Cube expert. "
            f"Give a short, clear hint for the following solution moves. "
            f"Maximum 2-3 sentences.\n\n"
            f"Moves: {move_str}{more}\n\nHint:"
        )

    try:
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}]
        )
        return message.content[0].text.strip()
    except Exception as e:
        return f"AI ipucu hatasi: {e}"


@app.get("/")
def root():
    return {"status": "ok", "version": "2.4.0"}

@app.get("/api/v1/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        version="2.4.0",
        ai_available=ANTHROPIC_AVAILABLE and bool(os.getenv("ANTHROPIC_API_KEY"))
    )

@app.post("/api/v1/solve", response_model=SolveResponse)
def solve(request: SolveRequest):
    if len(request.faces) != 6:
        return SolveResponse(solution=[], move_count=0,
                             error=f"6 yuz gerekli, {len(request.faces)} geldi")

    total = {}
    for i, face in enumerate(request.faces):
        logger.info(f"Yuz[{i}] {FACE_NAMES[i]}: {face}")
        for c in face:
            total[c] = total.get(c, 0) + 1
    logger.info(f"Dagilim: {total}")

    working_faces = request.faces
    if (request.grid_size or 9) != 9:
        try:
            working_faces = [extract_center_9(f, request.grid_size) for f in request.faces]
        except Exception as e:
            return SolveResponse(solution=[], move_count=0, error=f"Izgara hatasi: {e}")

    for i, face in enumerate(working_faces):
        if len(face) != 9:
            return SolveResponse(
                solution=[], move_count=0,
                error=f"{FACE_NAMES[i]} yuzunde {len(face)} kare var, tam 9 olmali."
            )

    dist_str = ", ".join(f"{k}:{v}" for k, v in sorted(total.items()))
    wrong_colors = {k: v for k, v in total.items() if v != 9}
    if wrong_colors:
        bad_str = ", ".join(f"{k}={v}" for k, v in sorted(wrong_colors.items()))
        missing = [c for c in ["R", "G", "B", "Y", "O", "W"] if c not in total]
        missing_str = (f" Eksik renkler: {', '.join(missing)}." if missing else "")
        return SolveResponse(
            solution=[], move_count=0,
            error=(
                f"Renk dagilimi hatali - her renkten tam 9 kare olmali. "
                f"Hatali renkler: {bad_str}.{missing_str} "
                f"Preview ekraninda tum renkleri manuel duzeltın."
            )
        )

    try:
        cube_string = faces_to_kociemba_string(working_faces)
        solution_str = kociemba.solve(cube_string)

        if not solution_str or solution_str.startswith("Error") or not solution_str.strip():
            logger.error(f"Kociemba hata stringi: {solution_str!r}")
            return SolveResponse(
                solution=[], move_count=0,
                error=(
                    f"Kup durumu cozulemez. Preview ekraninda renkleri kontrol edin. "
                    f"Her yuzun merkez karesi o yuzun rengini temsil etmeli. "
                    f"(Kociemba: {str(solution_str).strip()})"
                )
            )

        moves = solution_str.strip().split()
        logger.info(f"Cozum: {len(moves)} hamle - {solution_str.strip()}")
        return SolveResponse(solution=moves, move_count=len(moves))

    except ValueError as e:
        logger.error(f"Veri hatasi: {e}")
        return SolveResponse(solution=[], move_count=0, error=str(e))
    except Exception as e:
        err = str(e)
        logger.error(f"Kociemba hatasi: {err}")
        return SolveResponse(
            solution=[], move_count=0,
            error=(
                f"Kup fiziksel olarak cozulemez durumda (renk dagilimi dogru: {dist_str}). "
                f"Kup elle karistirilmis olabilir veya tarama sirasinda yuzler yanlis yonde tutulmus olabilir. "
                f"Preview ekraninda renkleri kontrol edip tekrar deneyin."
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
