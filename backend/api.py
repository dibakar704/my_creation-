
import sqlite3
from pathlib import Path
from uuid import uuid4
from typing import Literal

from fastapi import (
    FastAPI,
    HTTPException,
    UploadFile,
    File,
    Form
)
from fastapi.openapi.utils import get_openapi
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from .agent import IncidentResponseAgent
from .memory import Hindsight


# ------------------------------------------
# DATABASE CONFIGURATION
# ------------------------------------------

DB_PATH = Path(__file__).resolve().parent.parent / "runbook.db"

MAX_IMAGE_SIZE = 6 * 1024 * 1024
MAX_LOG_SIZE = 1 * 1024 * 1024
MAX_TOTAL_SIZE = 20 * 1024 * 1024


def init_db():
    connection = sqlite3.connect(DB_PATH)
    try:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS incidents (
                incident_id TEXT PRIMARY KEY,
                description TEXT NOT NULL,
                remediation_plan TEXT NOT NULL,
                status TEXT NOT NULL
            )
        """)
        connection.commit()
    finally:
        connection.close()


init_db()


# ------------------------------------------
# FASTAPI CONFIGURATION
# ------------------------------------------

app = FastAPI(
    title="RunBook AI",
    description="AI-powered incident response and multimodal memory platform",
    version="1.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------------------
# REQUEST MODELS
# ------------------------------------------

class IncidentRequest(BaseModel):
    description: str = Field(min_length=5, max_length=5000)


class IncidentResponse(BaseModel):
    incident_id: str
    description: str
    remediation_plan: str
    status: str


class ResolutionRequest(BaseModel):
    status: Literal["resolved", "failed"]
    resolution_details: str = Field(
        min_length=5,
        max_length=5000
    )


# ------------------------------------------
# HELPER: SAVE INCIDENT TO SQLITE
# ------------------------------------------

def save_incident(
    incident_id,
    description,
    remediation_plan,
    status="analyzed"
):
    connection = sqlite3.connect(DB_PATH)
    try:
        connection.execute(
            """
            INSERT INTO incidents
            (incident_id, description, remediation_plan, status)
            VALUES (?, ?, ?, ?)
            """,
            (
                incident_id,
                description,
                remediation_plan,
                status
            )
        )
        connection.commit()
    finally:
        connection.close()


# ------------------------------------------
# HELPER: PROCESS MULTIMODAL INCIDENT
# OUTSIDE THE ASYNC EVENT LOOP
# ------------------------------------------

def process_multimodal_with_agent(
    incident_description,
    image_inputs,
    log_texts
):
    agent = IncidentResponseAgent()

    return agent.handle_incident(
        incident_description=incident_description,
        image_inputs=image_inputs,
        log_texts=log_texts
    )


# ------------------------------------------
# BASIC ENDPOINTS
# ------------------------------------------

@app.get("/")
def root():
    return {
        "message": "Welcome to RunBook AI",
        "status": "running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "RunBook AI API"
    }


# ------------------------------------------
# TEXT-ONLY INCIDENT ENDPOINT
# ------------------------------------------

@app.post("/incidents", response_model=IncidentResponse)
def create_incident(request: IncidentRequest):

    incident_id = str(uuid4())

    try:
        agent = IncidentResponseAgent()

        remediation_plan = agent.handle_incident(
            request.description
        )

        if not remediation_plan:
            raise HTTPException(
                status_code=502,
                detail="The AI agent could not generate a remediation plan."
            )

        save_incident(
            incident_id,
            request.description,
            remediation_plan
        )

        return IncidentResponse(
            incident_id=incident_id,
            description=request.description,
            remediation_plan=remediation_plan,
            status="analyzed"
        )

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Incident processing failed: {str(error)}"
        )


# ------------------------------------------
# MULTIMODAL INCIDENT ENDPOINT
# ------------------------------------------

@app.post(
    "/incidents/multimodal",
    response_model=IncidentResponse
)
async def create_multimodal_incident(
    description: str = Form(default=""),
    images: list[UploadFile] = File(default=[]),
    logs: list[UploadFile] = File(default=[])
):

    image_inputs = []
    log_texts = []
    total_size = 0

    try:
        if not description.strip() and not images and not logs:
            raise HTTPException(
                status_code=400,
                detail="Provide an incident description or upload evidence."
            )

        if len(images) > 3:
            raise HTTPException(
                status_code=400,
                detail="Maximum 3 screenshots are allowed."
            )

        if len(logs) > 10:
            raise HTTPException(
                status_code=400,
                detail="Maximum 10 log files are allowed."
            )

        # Process screenshots
        for image_file in images:

            filename = image_file.filename or ""
            extension = Path(filename).suffix.lower()

            if extension not in (".png", ".jpg", ".jpeg"):
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported image format: {filename}"
                )

            image_bytes = await image_file.read()

            if not image_bytes:
                raise HTTPException(
                    status_code=400,
                    detail=f"{filename} is empty."
                )

            if len(image_bytes) > MAX_IMAGE_SIZE:
                raise HTTPException(
                    status_code=400,
                    detail=f"{filename} exceeds the 6 MB limit."
                )

            total_size += len(image_bytes)

            if extension == ".png":
                valid_image = image_bytes.startswith(
                    b"\x89PNG\r\n\x1a\n"
                )
                mime_type = "image/png"
            else:
                valid_image = image_bytes.startswith(
                    b"\xff\xd8\xff"
                )
                mime_type = "image/jpeg"

            if not valid_image:
                raise HTTPException(
                    status_code=400,
                    detail=f"{filename} is not a valid image."
                )

            image_inputs.append(
                (mime_type, image_bytes)
            )

        # Process log files
        for log_file in logs:

            filename = log_file.filename or ""
            extension = Path(filename).suffix.lower()

            if extension not in (".txt", ".log"):
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported log format: {filename}"
                )

            log_bytes = await log_file.read()

            if not log_bytes:
                raise HTTPException(
                    status_code=400,
                    detail=f"{filename} is empty."
                )

            if len(log_bytes) > MAX_LOG_SIZE:
                raise HTTPException(
                    status_code=400,
                    detail=f"{filename} exceeds the 1 MB limit."
                )

            total_size += len(log_bytes)

            log_content = log_bytes.decode(
                "utf-8",
                errors="replace"
            )

            log_texts.append(
                f"Filename: {filename}\n{log_content}"
            )

        # Check total upload size
        if total_size > MAX_TOTAL_SIZE:
            raise HTTPException(
                status_code=400,
                detail="Total evidence upload exceeds 20 MB."
            )

        incident_description = (
            description.strip()
            if description.strip()
            else "Incident submitted using uploaded evidence."
        )

        # Run synchronous Hindsight and Groq processing
        # in a worker thread to avoid event-loop conflicts.
        remediation_plan = await run_in_threadpool(
            process_multimodal_with_agent,
            incident_description,
            image_inputs,
            log_texts
        )

        if not remediation_plan:
            raise HTTPException(
                status_code=502,
                detail="AI could not generate a remediation plan."
            )

        # Save incident to SQLite
        incident_id = str(uuid4())

        save_incident(
            incident_id,
            incident_description,
            remediation_plan
        )

        return IncidentResponse(
            incident_id=incident_id,
            description=incident_description,
            remediation_plan=remediation_plan,
            status="analyzed"
        )

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Multimodal incident processing failed: {str(error)}"
        )

    finally:
        for uploaded_file in images + logs:
            await uploaded_file.close()


# ------------------------------------------
# INCIDENT RESOLUTION ENDPOINT
# ------------------------------------------

@app.post("/incidents/{incident_id}/resolution")
def record_resolution(
    incident_id: str,
    request: ResolutionRequest
):

    memory = None

    try:
        connection = sqlite3.connect(DB_PATH)
        connection.row_factory = sqlite3.Row

        try:
            incident = connection.execute(
                """
                SELECT * FROM incidents
                WHERE incident_id = ?
                """,
                (incident_id,)
            ).fetchone()
        finally:
            connection.close()

        if incident is None:
            raise HTTPException(
                status_code=404,
                detail="Incident ID not found."
            )

        memory = Hindsight()

        memory.track_resolution(
            incident_id=incident_id,
            incident=incident["description"],
            status=request.status,
            resolution_details=request.resolution_details
        )

        connection = sqlite3.connect(DB_PATH)

        try:
            connection.execute(
                """
                UPDATE incidents
                SET status = ?
                WHERE incident_id = ?
                """,
                (request.status, incident_id)
            )
            connection.commit()
        finally:
            connection.close()

        return {
            "success": True,
            "incident_id": incident_id,
            "resolution_status": request.status,
            "incident_description": incident["description"],
            "message": "Incident resolution recorded successfully."
        }

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Could not record resolution: {str(error)}"
        )

    finally:
        if memory is not None:
            memory.close()


# ------------------------------------------
# SWAGGER FILE UPLOAD COMPATIBILITY FIX
# ------------------------------------------

def custom_openapi():

    if app.openapi_schema:
        return app.openapi_schema

    schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes
    )

    schema["openapi"] = "3.0.3"

    components = schema.get(
        "components", {}
    ).get("schemas", {})

    for component in components.values():

        properties = component.get(
            "properties", {}
        )

        for prop in properties.values():

            if prop.get("type") == "array":

                items = prop.get("items", {})

                if items.get("contentMediaType") == "application/octet-stream":
                    items.pop("contentMediaType", None)
                    items["type"] = "string"
                    items["format"] = "binary"

            elif prop.get("contentMediaType") == "application/octet-stream":

                prop.pop("contentMediaType", None)
                prop["type"] = "string"
                prop["format"] = "binary"

    app.openapi_schema = schema

    return app.openapi_schema


app.openapi = custom_openapi
