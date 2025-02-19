"""Define response model for the endpoint version."""
from pydantic import BaseModel, Field  # type: ignore


class VersionResponse(BaseModel):
    """Response for version endpoint."""
    version: str = Field(..., example="1.0.0")


class GenerationRequest(BaseModel):
    """Request format for generating image."""

    prompt: str = Field("mri brain scan", example="mri brain scan",
                        title="Prompt")
    a_prompt: str = Field("good quality",  example="good quality",
                          title="Added prompt")
    n_prompt: str = Field("animal, drawing, painting, vivid colors, longbody,"
                          "lowres, bad anatomy, bad hands, missing fingers, "
                          "extra digit, fewer digits, cropped, "
                          "worst quality, low quality")
    num_samples: int = Field(default=1,  example=3,
                             title="Images to generate")
    image_resolution: int = Field(default=256, example=512, ge=256,
                                  title="Image resolution in pixels. "
                                        "Values need to be at least 256 pixels.")
    ddim_steps: int = Field(default=5, example=10, title="Steps")
    guess_mode: bool = Field(default=False, example=False, title="Guess Mode")
    strength: float = Field(default=1.0, example=1.0, title="Control Strength",
                            ge=0.0, le=2.0)
    scale: float = Field(default=9.0, example=9.0, title="Guidance Scale",
                         ge=0.1, le=30)
    seed: int = Field(default=1, example=42, ge=-1, title="Seed")
    eta: float = Field(default=0.0, example=0.0, title="eta (DDIM)", ge=0)
    low_threshold: int = Field(default=50, example=50, ge=1, le=255,
                               title="Canny low threshold")
    high_threshold: int = Field(default=100, example=100, ge=1, le=255,
                                title="Canny high threshold")
