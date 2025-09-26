from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, JSONResponse
import shutil
import uuid
import os
import logging

# Import pipeline functions
from .pipeline import (
    load_document,
    chunk_texts,
    build_retriever,
    generate_full_icf,
    verify_output,
    judge_output,
    fill_template,
    export_logs,
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(title="Enhanced ICF Generator API", version="1.0.0")

# Directories for I/O
UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

TEMPLATE_PATH = "ICF-template-original (3).docx"  # Updated to match your template


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload a clinical trial protocol and generate the ICF."""
    try:
        # Validate file type
        if not file.filename.endswith(('.pdf', '.docx')):
            raise HTTPException(status_code=400, detail="Only PDF and DOCX files are supported")
        
        file_id = str(uuid.uuid4())
        input_path = os.path.join(UPLOAD_DIR, f"{file_id}_{file.filename}")
        
        # Save uploaded file
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f"Processing file: {file.filename}")
        
        # Run enhanced pipeline
        raw_texts = load_document(input_path)
        chunks = chunk_texts(raw_texts)
        retriever = build_retriever(chunks, namespace=file_id)  # Use file_id as namespace
        
        sections, results = generate_full_icf(retriever)  # Updated: results instead of citations
        
        icf_sections = {}
        for sec, text in sections.items():
            verification = verify_output(text, retriever, sec)
            judgement = judge_output(sec, text, retriever)
            icf_sections[sec] = {
                "text": text,
                "verification": verification,
                "judgement": judgement,
                "citations": [
                    f"Page {doc.metadata.get('page', 'Unknown')} ({doc.metadata.get('section', 'Unknown')}) - Type: {doc.metadata.get('chunk_type', 'general')}"
                    for doc, _ in results if hasattr(doc, 'metadata') and doc.metadata
                ],
                "chunk_types_used": list(set([
                    doc.metadata.get('chunk_type', 'general') 
                    for doc, _ in results if hasattr(doc, 'metadata') and doc.metadata
                ]))
            }
        
        # Save outputs
        docx_path = os.path.join(OUTPUT_DIR, f"{file_id}_ICF.docx")
        logs_path = os.path.join(OUTPUT_DIR, f"{file_id}_logs.json")
        
        # Check if template exists and handle accordingly
        if not os.path.exists(TEMPLATE_PATH):
            logger.warning(f"Template not found: {TEMPLATE_PATH}. Creating basic document.")
        
        fill_template(TEMPLATE_PATH, docx_path, icf_sections)
        export_logs(icf_sections, logs_path)
        
        return {
            "file_id": file_id,
            "docx_url": f"/download/icf/{file_id}",
            "logs_url": f"/download/logs/{file_id}",
            "sections_generated": list(sections.keys()),
            "retrieval_strategy": "Enhanced context-aware BM25 with semantic chunk classification",
            "agentic_features": [
                "Context-aware chunk classification",
                "Section-specific retrieval filtering", 
                "Planned vs completed action distinction",
                "Domain-aware metadata enrichment"
            ]
        }
        
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@app.get("/download/icf/{file_id}")
async def download_icf(file_id: str):
    """Download generated ICF .docx file"""
    path = os.path.join(OUTPUT_DIR, f"{file_id}_ICF.docx")
    if os.path.exists(path):
        return FileResponse(
            path,
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            filename="Generated_ICF.docx",
        )
    raise HTTPException(status_code=404, detail="File not found")


@app.get("/download/logs/{file_id}")
async def download_logs(file_id: str):
    """Download logs .json file"""
    path = os.path.join(OUTPUT_DIR, f"{file_id}_logs.json")
    if os.path.exists(path):
        return FileResponse(path, media_type="application/json", filename="logs.json")
    raise HTTPException(status_code=404, detail="File not found")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy", 
        "message": "Enhanced ICF Generator API is running",
        "features": [
            "Context-aware retrieval",
            "Semantic chunk classification", 
            "Section-specific filtering",
            "Planned vs completed action distinction"
        ]
    }


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Enhanced ICF Generator API",
        "description": "Generates Informed Consent Forms from clinical trial protocols using advanced retrieval strategies",
        "endpoints": {
            "upload": "POST /upload - Upload protocol and generate ICF",
            "download_icf": "GET /download/icf/{file_id} - Download generated ICF",
            "download_logs": "GET /download/logs/{file_id} - Download processing logs",
            "health": "GET /health - Health check"
        },
        "agentic_features": [
            "Context-aware chunk classification",
            "Section-specific retrieval filtering",
            "Medical domain awareness",
            "Planned vs completed action distinction"
        ]
    }