from __future__ import annotations

import base64
import threading
import uuid
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Jalankan Uvicorn dari root project: uvicorn api.main:app --reload --port 8000
from dashboard import core, newrepo


app = FastAPI(
    title="DefectLens API",
    description="REST API untuk analisis risiko defect pada repository Java.",
    version="1.0.0",
)

# Ganti origin ini sesuai alamat frontend Next.js.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Penyimpanan sementara untuk development.
# Catatan: job dan hasil akan hilang ketika proses API restart.
# ---------------------------------------------------------------------------
JOBS: dict[str, dict[str, Any]] = {}
JOBS_LOCK = threading.RLock()

MODEL = None
MODEL_ROWS = 0
MODEL_LOCK = threading.Lock()

BENCHMARK_CACHE: dict[str, dict[str, Any]] = {}
BENCHMARK_LOCK = threading.Lock()
ANALYSIS_LOCK = threading.Lock()


class AnalysisRequest(BaseModel):
    source: str = Field(
        min_length=5,
        description="URL HTTPS repository Git atau path folder repository lokal.",
        examples=["https://github.com/ProjectGopi/StaticGuard"],
    )


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_value(value: Any) -> Any:
    """Ubah nilai NumPy/Pandas menjadi tipe JSON native."""
    if value is None:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if pd.isna(value):
        return None
    return value


def encode_file_id(path: str) -> str:
    return base64.urlsafe_b64encode(path.encode("utf-8")).decode("ascii").rstrip("=")


def decode_file_id(file_id: str) -> str:
    try:
        padded = file_id + "=" * (-len(file_id) % 4)
        return base64.urlsafe_b64decode(padded.encode("ascii")).decode("utf-8")
    except Exception as exc:
        raise HTTPException(status_code=400, detail="fileId tidak valid.") from exc


def get_model():
    """Latih model satu kali, saat analisis repository pertama diminta."""
    global MODEL, MODEL_ROWS
    if MODEL is not None:
        return MODEL

    with MODEL_LOCK:
        if MODEL is None:
            MODEL, MODEL_ROWS = newrepo.train_model()
    return MODEL


def job_summary(job: dict[str, Any]) -> dict[str, Any]:
    result = job.get("result")
    return {
        "id": job["id"],
        "source": job["source"],
        "status": job["status"],
        "created_at": job["created_at"],
        "updated_at": job["updated_at"],
        "completed_at": job.get("completed_at"),
        "error": job.get("error"),
        "summary": (
            {
                "repository": result["name"],
                "n_files": result["n_files"],
                "commits": result["commits"],
                "warnings": result["warnings"],
            }
            if result is not None
            else None
        ),
    }


def run_analysis_job(analysis_id: str, source: str) -> None:
    with JOBS_LOCK:
        job = JOBS.get(analysis_id)
        if not job:
            return
        job["status"] = "running"
        job["updated_at"] = utc_now()

    try:
        model = get_model()
        # newrepo.py memakai folder kerja berdasarkan nama repository;
        # serialisasi mencegah dua job menimpa folder kerja yang sama.
        with ANALYSIS_LOCK:
            result = newrepo.analyze_repo(source, model)

        with JOBS_LOCK:
            job = JOBS[analysis_id]
            job["result"] = result
            job["status"] = "completed"
            job["updated_at"] = utc_now()
            job["completed_at"] = utc_now()
    except Exception as exc:
        with JOBS_LOCK:
            job = JOBS[analysis_id]
            job["status"] = "failed"
            job["error"] = str(exc)
            job["updated_at"] = utc_now()
            job["completed_at"] = utc_now()


def get_completed_job(analysis_id: str) -> dict[str, Any]:
    with JOBS_LOCK:
        job = JOBS.get(analysis_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Analysis tidak ditemukan.")
    if job["status"] != "completed":
        raise HTTPException(
            status_code=409,
            detail=f"Analysis belum selesai. Status saat ini: {job['status']}.",
        )
    return job


def get_benchmark_result(benchmark_id: str) -> dict[str, Any]:
    # ID benchmark menggunakan format project::version.
    if "::" not in benchmark_id:
        raise HTTPException(
            status_code=400,
            detail="ID benchmark harus menggunakan format project::version.",
        )
    project, version = benchmark_id.split("::", 1)

    with BENCHMARK_LOCK:
        cached = BENCHMARK_CACHE.get(benchmark_id)
        if cached is not None:
            return cached

    versions = core.list_versions()
    if project not in versions or version not in versions[project]:
        raise HTTPException(status_code=404, detail="Versi benchmark tidak ditemukan.")

    result = core.analyze(project, version)
    if result is None:
        raise HTTPException(
            status_code=422,
            detail="Rilis ini belum dapat dianalisis karena tidak memiliki data latih yang layak.",
        )

    table = core.build_table(result)
    payload = {"project": project, "version": version, "result": result, "table": table}

    with BENCHMARK_LOCK:
        BENCHMARK_CACHE[benchmark_id] = payload
    return payload


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------
@app.get("/health", tags=["Health"])
def health():
    return {
        "status": "healthy",
        "message": "DefectLens API is running",
        "model_loaded": MODEL is not None,
        "model_training_rows": MODEL_ROWS if MODEL is not None else None,
    }


# ---------------------------------------------------------------------------
# Repository analyses
# ---------------------------------------------------------------------------
@app.post("/api/analyses", status_code=202, tags=["Analyses"])
def create_analysis(
    request: AnalysisRequest,
    background_tasks: BackgroundTasks,
):
    source = request.source.strip().strip('"')
    if not (source.startswith("https://") or source.startswith("http://") or source.startswith(("/", "\\"))):
        # Path relatif juga diterima untuk development lokal.
        if "://" in source:
            raise HTTPException(status_code=422, detail="Skema URL tidak didukung.")

    analysis_id = str(uuid.uuid4())
    now = utc_now()

    with JOBS_LOCK:
        JOBS[analysis_id] = {
            "id": analysis_id,
            "source": source,
            "status": "queued",
            "created_at": now,
            "updated_at": now,
            "completed_at": None,
            "error": None,
            "result": None,
        }

    background_tasks.add_task(run_analysis_job, analysis_id, source)
    return {
        "id": analysis_id,
        "source": source,
        "status": "queued",
        "created_at": now,
        "message": "Analysis job created.",
    }


@app.get("/api/analyses", tags=["Analyses"])
def list_analyses(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
):
    with JOBS_LOCK:
        items = sorted(
            (job_summary(job) for job in JOBS.values()),
            key=lambda item: item["created_at"],
            reverse=True,
        )
    return {"items": items[offset : offset + limit], "total": len(items), "limit": limit, "offset": offset}


@app.get("/api/analyses/{analysis_id}", tags=["Analyses"])
def get_analysis(analysis_id: str):
    with JOBS_LOCK:
        job = JOBS.get(analysis_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Analysis tidak ditemukan.")
    return job_summary(job)


@app.get("/api/analyses/{analysis_id}/files", tags=["Analyses"])
def list_analysis_files(
    analysis_id: str,
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
):
    job = get_completed_job(analysis_id)
    table = job["result"]["table"]
    records = []

    for row in table.itertuples(index=False):
        file_path = str(row.file)
        records.append({
            "id": encode_file_id(file_path),
            "file": file_path,
            "risk_score": float(row.risiko_ml),
            "percentile": float(row.persentil),
            "pmd_total": int(row.pmd_total),
            "pmd_high": int(row.pmd_high),
            "reason": str(row.alasan),
        })

    return {
        "analysis_id": analysis_id,
        "items": records[offset : offset + limit],
        "total": len(records),
        "limit": limit,
        "offset": offset,
    }


@app.get("/api/analyses/{analysis_id}/files/{file_id}", tags=["Analyses"])
def get_analysis_file(analysis_id: str, file_id: str):
    job = get_completed_job(analysis_id)
    result = job["result"]
    file_path = decode_file_id(file_id)

    matches = result["table"][result["table"]["file"] == file_path]
    if matches.empty:
        raise HTTPException(status_code=404, detail="File tidak ditemukan dalam hasil analisis.")

    row = matches.iloc[0]
    original_idx = int(row["idx"])
    contributions = newrepo.contributions(result, original_idx)
    findings = newrepo.pmd_for_file(result, file_path)

    return {
        "analysis_id": analysis_id,
        "id": file_id,
        "file": file_path,
        "risk_score": float(row["risiko_ml"]),
        "percentile": float(row["persentil"]),
        "pmd_total": int(row["pmd_total"]),
        "pmd_high": int(row["pmd_high"]),
        "reason": str(row["alasan"]),
        "shap": [
            {
                "feature": str(item.fitur),
                "value": json_value(item.nilai_asli),
                "contribution": float(item.kontribusi),
            }
            for item in contributions.itertuples(index=False)
        ],
        "pmd_findings": [
            {key: json_value(value) for key, value in record.items()}
            for record in findings.to_dict(orient="records")
        ],
    }


# ---------------------------------------------------------------------------
# PROMISE benchmark
# ---------------------------------------------------------------------------
@app.get("/api/benchmark", tags=["Benchmark"])
def list_benchmarks():
    versions = core.list_versions()
    items = [
        {
            "id": f"{project}::{version}",
            "project": project,
            "version": version,
        }
        for project, project_versions in versions.items()
        for version in project_versions
    ]
    return {"items": items, "total": len(items)}


@app.get("/api/benchmark/{benchmark_id}/files", tags=["Benchmark"])
def list_benchmark_files(
    benchmark_id: str,
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
):
    payload = get_benchmark_result(benchmark_id)
    table = payload["table"]
    items = []

    for idx, row in table.iterrows():
        file_path = str(row["file"])
        items.append({
            "id": encode_file_id(file_path),
            "file": file_path,
            "class_name": json_value(row["kelas"]),
            "risk_score": float(row["risiko_ml"]),
            "combined_score": float(row["skor_gabungan"]),
            "pmd_total": int(row["pmd_total"]),
            "pmd_high": int(row["pmd_high"]),
            "reason": str(row["alasan"]),
        })

    return {
        "benchmark_id": benchmark_id,
        "project": payload["project"],
        "version": payload["version"],
        "items": items[offset : offset + limit],
        "total": len(items),
        "limit": limit,
        "offset": offset,
    }


@app.get("/api/benchmark/{benchmark_id}/files/{file_id}", tags=["Benchmark"])
def get_benchmark_file(benchmark_id: str, file_id: str):
    payload = get_benchmark_result(benchmark_id)
    result = payload["result"]
    file_path = decode_file_id(file_id)

    raw_matches = result["raw"].index[result["raw"]["file"] == file_path].tolist()
    if not raw_matches:
        raise HTTPException(status_code=404, detail="File benchmark tidak ditemukan.")
    raw_idx = int(raw_matches[0])

    table_matches = payload["table"][payload["table"]["file"] == file_path]
    if table_matches.empty:
        raise HTTPException(status_code=404, detail="Hasil file benchmark tidak ditemukan.")
    row = table_matches.iloc[0]

    contributions = core.contributions(result, raw_idx)
    findings = core.pmd_findings(payload["project"], payload["version"], file_path)

    return {
        "benchmark_id": benchmark_id,
        "id": file_id,
        "project": payload["project"],
        "version": payload["version"],
        "file": file_path,
        "class_name": json_value(row["kelas"]),
        "risk_score": float(row["risiko_ml"]),
        "combined_score": float(row["skor_gabungan"]),
        "pmd_total": int(row["pmd_total"]),
        "pmd_high": int(row["pmd_high"]),
        "reason": str(row["alasan"]),
        "actual_bug": int(row["bug_asli"]),
        "shap": [
            {
                "feature": str(item.fitur),
                "value": json_value(item.nilai_asli),
                "contribution": float(item.kontribusi),
            }
            for item in contributions.itertuples(index=False)
        ],
        "pmd_findings": [
            {key: json_value(value) for key, value in record.items()}
            for record in findings.to_dict(orient="records")
        ],
    }
