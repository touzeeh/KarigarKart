from __future__ import annotations

import base64
import binascii
import logging
import mimetypes
import os
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy import Column, Float, Integer, String, Text
from sqlalchemy.orm import Session

from backend.ai_service import AIServiceError, GeminiAIService, compact_image, parse_json
from backend.database import Base, SessionLocal, engine

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
LOGGER = logging.getLogger("karigarkart.api")
ai = GeminiAIService()

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        Base.metadata.create_all(bind=engine)
    except Exception:
        LOGGER.exception("Database initialization failed")
    yield
    await ai.close()

app = FastAPI(title="KarigarKart Backend", version="1.1.0", lifespan=lifespan)

class ProductDB(Base):
    __tablename__ = "products"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    category = Column(String(100), nullable=False)
    description = Column(Text, nullable=False)
    price = Column(Float, nullable=False)

class Product(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    category: str = Field(min_length=1, max_length=100)
    description: str = Field(min_length=1, max_length=5000)
    price: float = Field(ge=0)

class CatalogRequest(BaseModel):
    artisan_text: str = Field(min_length=1, max_length=5000)
    language: str = Field(default="English", max_length=40)
    raw_material_cost: float = Field(default=0, ge=0)

class PricingRequest(BaseModel):
    image_base64: str | None = None
    image_url: str | None = None
    description: str = Field(min_length=1, max_length=5000)
    raw_material_cost: float = Field(default=0, ge=0)
    labor_hours: float = Field(default=0, ge=0, le=1000)
    labor_rate: float = Field(default=0, ge=0)
    labor_cost: float = Field(default=0, ge=0)
    category: str = Field(default="Handicrafts", max_length=100)

class EnhanceRequest(BaseModel):
    text: str | None = Field(default=None, max_length=5000)
    description: str | None = Field(default=None, max_length=5000)
    image_base64: str | None = None
    image_url: str | None = None
    category: str = Field(default="Handicrafts", max_length=100)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def _decode_data_uri(value: str):
    raw = value.strip()
    mime = "image/jpeg"
    if raw.startswith("data:"):
        header, separator, encoded = raw.partition(",")
        if not separator:
            raise AIServiceError("Invalid image data.", 400)
        mime = header[5:].split(";", 1)[0] or mime
        raw = encoded
    try:
        data = base64.b64decode(raw, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise AIServiceError("Invalid base64 image data.", 400) from exc
    return compact_image(data, mime)

def _fallback_catalog(text: str, language: str) -> dict[str, Any]:
    title = "Handcrafted Artisan Product"
    regional = "हस्तनिर्मित कारीगर उत्पाद" if language.lower().startswith("hindi") else title
    return {"title_en": title, "desc_en": text, "title_regional": regional, "desc_regional": text, "bullets_en": ["Handcrafted artisan product", "Made with care", "Suitable for everyday use"], "bullets_hi": ["हस्तनिर्मित कारीगर उत्पाद", "सावधानी से बनाया गया", "दैनिक उपयोग के लिए उपयुक्त"]}

def _fallback_price(material: float, labor: float) -> float:
    return round(max(199.0, (material + labor) * 1.35), 0)

@app.middleware("http")
async def request_logging(request: Request, call_next):
    request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
    try:
        response = await call_next(request)
    except Exception:
        LOGGER.exception("Unhandled request failure request_id=%s path=%s", request_id, request.url.path)
        response = JSONResponse(status_code=500, content={"success": False, "detail": "Internal server error.", "request_id": request_id})
    response.headers["x-request-id"] = request_id
    return response

@app.exception_handler(AIServiceError)
async def ai_error_handler(request: Request, exc: AIServiceError):
    LOGGER.warning("AI request error path=%s status=%s detail=%s", request.url.path, exc.status_code, exc.message)
    return JSONResponse(status_code=exc.status_code, content={"success": False, "detail": exc.message})

@app.get("/")
def home():
    return {"message": "KarigarKart Backend is running!", "ai": True}

@app.post("/products")
def create_product(product: Product, db: Session = Depends(get_db)):
    try:
        new_product = ProductDB(name=product.name, category=product.category, description=product.description, price=product.price)
        db.add(new_product)
        db.commit()
        db.refresh(new_product)
        return {"success": True, "message": "Product saved successfully", "product": {"id": new_product.id, "name": new_product.name, "category": new_product.category, "description": new_product.description, "price": new_product.price}}
    except Exception as exc:
        db.rollback()
        LOGGER.exception("Product creation failed")
        raise HTTPException(status_code=500, detail="Product could not be saved.") from exc

@app.get("/products")
def get_products(db: Session = Depends(get_db)):
    try:
        return db.query(ProductDB).all()
    except Exception as exc:
        LOGGER.exception("Product listing failed")
        raise HTTPException(status_code=500, detail="Products could not be loaded.") from exc

@app.post("/ai/transcribe")
async def transcribe(file: UploadFile = File(...), language: str = Query(default="English", max_length=40)):
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Recorded audio is empty.")
    if len(data) > 10_000_000:
        raise HTTPException(status_code=413, detail="Recorded audio is too large.")
    mime = file.content_type or mimetypes.guess_type(file.filename or "")[0] or "audio/m4a"
    prompt = f"Transcribe this artisan's speech exactly. Spoken language is {language}. Return JSON only with one key: text. Preserve product details, quantities, materials and names. Do not invent missing words."
    parsed = parse_json(await ai.generate(prompt, audio=data, audio_mime=mime, max_tokens=500))
    transcript = str(parsed.get("text", "")).strip()
    if not transcript:
        raise AIServiceError("No speech was detected.", 422)
    return {"success": True, "text": transcript}

@app.post("/ai/catalog")
async def generate_catalog(payload: CatalogRequest):
    prompt = f"""Create marketplace catalog copy for an Indian artisan product. Input language: {payload.language}. Artisan description: {payload.artisan_text}. Raw material cost: {payload.raw_material_cost}. Return JSON only with keys title_en, desc_en, title_regional, desc_regional, bullets_en, bullets_hi. English title must be concise and SEO-friendly. Descriptions must be truthful and based only on the input. Bullets are 3-5 short features. Regional fields should be Hindi when the input language is Hindi; otherwise provide a natural regional rendering when possible."""
    try:
        parsed = parse_json(await ai.generate(prompt, max_tokens=700))
    except AIServiceError as exc:
        LOGGER.warning("Catalog AI failed; using fallback: %s", exc.message)
        parsed = _fallback_catalog(payload.artisan_text.strip(), payload.language)
        parsed.update({"fallback": True, "message": exc.message})
    parsed.setdefault("title_en", "Handcrafted Artisan Product")
    parsed.setdefault("desc_en", payload.artisan_text.strip())
    parsed.setdefault("title_regional", parsed["title_en"])
    parsed.setdefault("desc_regional", parsed["desc_en"])
    parsed.setdefault("bullets_en", ["Handcrafted artisan product", "Made with care"])
    parsed.setdefault("bullets_hi", ["हस्तनिर्मित कारीगर उत्पाद", "सावधानी से बनाया गया"])
    return {"success": True, "catalog": parsed}

@app.post("/ai/pricing")
async def pricing(payload: PricingRequest):
    labor = payload.labor_cost if payload.labor_cost > 0 else payload.labor_hours * payload.labor_rate
    material = payload.raw_material_cost
    floor = material + labor
    image = None
    mime = "image/jpeg"
    if payload.image_base64:
        image, mime = _decode_data_uri(payload.image_base64)
    prompt = f"""Estimate a practical Indian artisan marketplace price using the supplied product photo and description. Category: {payload.category}. Description: {payload.description}. Raw material cost: {material}. Labor hours: {payload.labor_hours}. Labor rate: {payload.labor_rate}. Calculated labor cost: {labor}. Cost floor: {floor}. Return JSON only with suggested_price, b2c_price, b2b_price and reasoning. Do not claim live market data unless supplied. Keep suggested_price above the cost floor when possible."""
    try:
        parsed = parse_json(await ai.generate(prompt, image=image, image_mime=mime, max_tokens=450))
        suggested = float(parsed.get("suggested_price", 0))
        if suggested <= 0:
            raise AIServiceError("AI returned an invalid price.", 502)
        parsed["suggested_price"] = suggested
        parsed.setdefault("b2c_price", suggested)
        parsed.setdefault("b2b_price", max(floor * 1.10, suggested * 0.90))
        parsed.setdefault("reasoning", "AI estimate based on supplied product information and cost inputs.")
        parsed["cost_floor"] = round(floor, 2)
        return {"success": True, **parsed}
    except AIServiceError as exc:
        fallback = _fallback_price(material, labor)
        LOGGER.warning("Pricing AI failed; using fallback: %s", exc.message)
        return {"success": True, "suggested_price": fallback, "b2c_price": fallback, "b2b_price": round(max(floor * 1.10, fallback * 0.90), 0), "reasoning": "Fallback estimate based on supplied material and labor costs.", "cost_floor": round(floor, 2), "fallback": True, "message": exc.message}

@app.post("/ai/enhance")
async def enhance(payload: EnhanceRequest):
    source = (payload.text or payload.description or "").strip()

    if not source:
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "detail": "Text or description is required.",
            },
        )

    # Enhancement is text-based. Do not decode or forward the uploaded
    # image here: this endpoint previously allowed binary JPEG bytes to
    # enter the response path and trigger FastAPI's UTF-8 encoder.
    prompt = (
        "Improve this artisan marketplace description without inventing facts. "
        f"Category: {payload.category}. "
        f"Source: {source}. "
        "Return JSON only with key enhanced_text. "
        "Keep it concise, clear and SEO-friendly."
    )

    try:
        parsed = parse_json(
            await ai.generate(
                prompt,
                max_tokens=450,
            )
        )

        raw_enhanced = parsed.get("enhanced_text", "")

        if isinstance(raw_enhanced, bytes):
            enhanced = raw_enhanced.decode(
                "utf-8",
                errors="replace",
            ).strip()
        else:
            enhanced = str(raw_enhanced or "").strip()

        if not enhanced:
            raise AIServiceError("AI returned no enhanced text.", 502)

        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "enhanced_text": enhanced,
                "enhanced_description": enhanced,
                "text": enhanced,
            },
        )

    except AIServiceError as exc:
        LOGGER.warning(
            "Enhance AI failed; returning original text fallback: %s",
            exc.message,
        )

        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "enhanced_text": source,
                "enhanced_description": source,
                "text": source,
                "fallback": True,
                "message": str(exc.message),
            },
        )

    except Exception as exc:
        LOGGER.exception("Unexpected enhance failure: %s", exc)

        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "enhanced_text": source,
                "enhanced_description": source,
                "text": source,
                "fallback": True,
                "message": "Enhancement service temporarily unavailable.",
            },
        )
