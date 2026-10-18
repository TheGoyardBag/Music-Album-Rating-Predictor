"""Local inference API and browser demo. Train the model before starting this service."""
import os
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field, ConfigDict
from .inference import Predictor


class PredictionInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tags: list[str] = Field(min_length=1, max_length=64)


def create_app(model_path: str | Path | None = None) -> FastAPI:
    predictor = Predictor(model_path or os.getenv("MODEL_PATH", "artifacts/model.joblib"))
    app = FastAPI(title="Album Rating Predictor", version="0.1.0")

    @app.get("/health")
    def health():
        return {"status": "ok", "model": predictor.report["selected_model"]}

    @app.get("/", response_class=HTMLResponse)
    def home():
        return (Path(__file__).parent / "static" / "index.html").read_text()

    @app.get("/metadata")
    def metadata():
        return {"model": predictor.report["selected_model"],
                "dataset_kind": predictor.report["dataset_kind"],
                "tags": sorted(predictor.vocabulary), "test": predictor.report["test"]}

    @app.post("/predict")
    def predict(body: PredictionInput):
        try:
            return predictor.predict(body.tags)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
    return app
