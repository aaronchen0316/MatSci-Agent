from pydantic import BaseModel, Field


class Settings(BaseModel):
    app_name: str = "MatSci-Agent"
    max_iterations: int = Field(default=3, ge=1, le=20)
    default_top_k: int = Field(default=5, ge=1, le=100)
    mlflow_experiment: str = "matsci-agent-discovery"


settings = Settings()
