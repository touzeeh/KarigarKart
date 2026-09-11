from fastapi import FastAPI, Depends
from pydantic import BaseModel
from sqlalchemy import Column, Integer, String, Float, Text
from sqlalchemy.orm import Session

from backend.database import Base, engine, SessionLocal


app = FastAPI()


# Database table
class ProductDB(Base):
    __tablename__ = "products"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    category = Column(String(100), nullable=False)
    description = Column(Text, nullable=False)
    price = Column(Float, nullable=False)


# Create tables if they don't exist
Base.metadata.create_all(bind=engine)


# Request model
class Product(BaseModel):
    name: str
    category: str
    description: str
    price: float


# Database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.get("/")
def home():
    return {
        "message": "KarigarKart Backend is running!"
    }


@app.post("/products")
def create_product(product: Product, db: Session = Depends(get_db)):

    new_product = ProductDB(
        name=product.name,
        category=product.category,
        description=product.description,
        price=product.price
    )

    db.add(new_product)
    db.commit()
    db.refresh(new_product)

    return {
        "success": True,
        "message": "Product saved successfully",
        "product": {
            "id": new_product.id,
            "name": new_product.name,
            "category": new_product.category,
            "description": new_product.description,
            "price": new_product.price
        }
    }
    
    
@app.get("/products")
def get_products(db: Session = Depends(get_db)):
    products = db.query(ProductDB).all()

    return products