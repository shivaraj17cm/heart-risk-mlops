"""Request / response contracts for the prediction API (validated by Pydantic)."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from heartrisk.schema import ALLOWED_CODES, NUMERIC_RANGES

R = NUMERIC_RANGES


class PatientFeatures(BaseModel):
    """The 13 UCI attributes. `ca` and `thal` may be null - they are imputed,
    mirroring the missing values present in the source data."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"example": {
            "age": 58, "sex": 1, "cp": 4, "trestbps": 132, "chol": 224, "fbs": 0,
            "restecg": 2, "thalach": 173, "exang": 0, "oldpeak": 3.2, "slope": 1,
            "ca": 2, "thal": 7}},
    )

    age: int = Field(..., ge=R["age"][0], le=R["age"][1], description="Age in years")
    sex: int = Field(..., description="1 = male, 0 = female")
    cp: int = Field(..., description="Chest pain type (1-4)")
    trestbps: float = Field(..., ge=R["trestbps"][0], le=R["trestbps"][1], description="Resting BP, mm Hg")
    chol: float = Field(..., ge=R["chol"][0], le=R["chol"][1], description="Serum cholesterol, mg/dl")
    fbs: int = Field(..., description="Fasting blood sugar > 120 mg/dl (1 = true)")
    restecg: int = Field(..., description="Resting ECG result (0-2)")
    thalach: float = Field(..., ge=R["thalach"][0], le=R["thalach"][1], description="Max heart rate")
    exang: int = Field(..., description="Exercise-induced angina (1 = yes)")
    oldpeak: float = Field(..., ge=R["oldpeak"][0], le=R["oldpeak"][1], description="ST depression")
    slope: int = Field(..., description="Slope of peak exercise ST segment (1-3)")
    ca: Optional[int] = Field(None, description="Major vessels coloured (0-3); null = unknown")
    thal: Optional[int] = Field(None, description="3 normal, 6 fixed, 7 reversible; null = unknown")

    @field_validator("sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal")
    @classmethod
    def _allowed_code(cls, v, info):
        # ints and whole-number floats (1.0) are coerced by pydantic; codes are checked here
        if v is not None and v not in ALLOWED_CODES[info.field_name]:
            allowed = sorted(ALLOWED_CODES[info.field_name])
            raise ValueError(f"must be one of {allowed}")
        return v


class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="1 = heart disease present, 0 = absent")
    label: str
    probability_disease: float = Field(..., description="Model probability of class 1")
    confidence: float = Field(..., description="Probability of the predicted class")
    model_version: str
    request_id: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: Optional[str] = None
