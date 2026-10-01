import os
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker

DATABASE_DIR = "data"
os.makedirs(DATABASE_DIR, exist_ok=True)
DATABASE_URL = f"sqlite:///{DATABASE_DIR}/history.db"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False}  # Required for SQLite multithreading in FastAPI
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class PropertyEstimateRecord(Base):
    __tablename__ = "property_estimates"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    property_name = Column(String(100), nullable=True, default="Untitled Property")
    square_footage = Column(Float, nullable=False)
    bedrooms = Column(Integer, nullable=False)
    bathrooms = Column(Float, nullable=False)
    year_built = Column(Integer, nullable=False)
    lot_size = Column(Float, nullable=False)
    distance_to_city_center = Column(Float, nullable=False)
    school_rating = Column(Float, nullable=False)
    predicted_price = Column(Float, nullable=False)
    currency = Column(String(10), default="USD")
    created_at = Column(DateTime, default=datetime.utcnow)

def init_db():
    Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
