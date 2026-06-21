from fastapi import FastAPI

from app.database import Base, SessionLocal, engine
from app.routers import predictions
from app.seed import seed_demo_sample


app = FastAPI(
    title="ECG Arrhythmia Classifier API",
    description="API para clasificar latidos ECG con una CNN 1D y guardar resultados en base de datos.",
    version="1.0.0",
)


@app.on_event("startup")
def on_startup():
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        seed_demo_sample(db)
    finally:
        db.close()


@app.get("/")
def root():
    return {
        "message": "ECG Arrhythmia Classifier API",
        "docs": "/docs",
        "manual_prediction": "/predictions/manual",
        "file_prediction": "/predictions/file",
    }


@app.get("/health")
def health():
    return {"status": "ok"}


app.include_router(predictions.router)
