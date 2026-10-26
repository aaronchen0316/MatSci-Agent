from pydantic import BaseModel, Field


class Settings(BaseModel):
    app_name: str = "MatSci-Agent"
    default_top_k: int = Field(default=5, ge=1, le=30)
    mlflow_experiment: str = "matsci-agent-discovery"
    matgl_max_recalc_entries: int = Field(default=10, ge=1, le=100)
    matgl_max_atoms: int = Field(default=50, ge=1, le=500)
    matgl_enable_relaxation: bool = False
    matgl_relaxation_max_steps: int = Field(default=200, ge=1, le=5000)


settings = Settings()
