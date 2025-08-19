 
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, HttpUrl
import subprocess, os, json
from pathlib import Path

app = FastAPI(title="Analyzer API")

 
SCRIPT_DIR = Path(r"C:\Users\RARES LENOVO\Desktop\MUNCA\test").resolve()
RULARE = SCRIPT_DIR / "Rulare_analiza.py"
OUT_PATH = SCRIPT_DIR / "REZULTATE_FINALE.json"

 
PYTHON_EXE = str((SCRIPT_DIR / ".venv" / "Scripts" / "python.exe").resolve())

 
F_REZULTATE_LLAMACOPIE = SCRIPT_DIR / "rezultate_llama_copie.json"
F_PRODUSE_TIPURI       = SCRIPT_DIR / "produse_tipuri.json"
F_SERVICII_SUBCAT      = SCRIPT_DIR / "servicii_subcategorii.json"
F_CONTACTE_EXTRASE     = SCRIPT_DIR / "contacte_extrase.json"

class AnalyzeIn(BaseModel):
    url: HttpUrl

@app.post("/analyze")
def analyze(body: AnalyzeIn, no_db: bool = Query(False, description="Nu rula inserarea in DB")):
    if not RULARE.exists():
        raise HTTPException(status_code=500, detail=f"Error script: {RULARE}")

   
    for p in [F_REZULTATE_LLAMACOPIE, F_PRODUSE_TIPURI, F_SERVICII_SUBCAT, F_CONTACTE_EXTRASE, OUT_PATH]:
        try:
            if p.exists():
                p.unlink()
        except Exception:
            pass

   
    cmd = [PYTHON_EXE, str(RULARE), str(body.url), "--print"]
    if no_db:
        cmd.append("--no-db")

    env = os.environ.copy()
    env.update({
        "PYTHONIOENCODING": "utf-8",
        "PYTHONUTF8": "1",
        "HTTP_PROXY": "",
        "HTTPS_PROXY": "",
        "ALL_PROXY": "",
        "NO_PROXY": "*",
    })

    try:
        proc = subprocess.run(
            cmd,
            cwd=str(SCRIPT_DIR),
            env=env,
            capture_output=True,
            text=True,
            timeout=900
        )
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="Timeout la Rulare_analiza.py (15 min)")
 
    result_json = {}
    if OUT_PATH.exists():
        try:
            result_json = json.loads(OUT_PATH.read_text(encoding="utf-8"))
        except Exception as e:
            result_json = {"_error": f"Nu am putut citi {OUT_PATH.name}: {e}"}

    return {
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "result": result_json
    }
