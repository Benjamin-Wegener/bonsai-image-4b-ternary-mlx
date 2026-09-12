"""FastAPI Server Entrypoint for FLUX.2 Klein 4B Ternary MLX Photoshop Studio."""

import logging
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from app.config import WEB_DIR
from app.routes import router

# ── Suppress per-request access log spam for the progress-poll endpoint ──
class _SuppressProgressLogs(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return "GET /v1/progress" not in msg

for _logger_name in ("uvicorn.access", "uvicorn"):
    logging.getLogger(_logger_name).addFilter(_SuppressProgressLogs())

app = FastAPI(title="FLUX.2 Klein 4B Ternary MLX Photoshop Studio")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")

app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server_12345:app", host="0.0.0.0", port=12345, reload=False)
