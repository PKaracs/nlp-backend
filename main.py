from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
from api import router
import torch

app = FastAPI(title="MarketMood NLP API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set device for PyTorch
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device set to use {device}")

# Include our API router
app.include_router(router, prefix="/api/v1")

class TextAnalysisRequest(BaseModel):
    text: str
    history: Optional[List[str]] = []

@app.get("/")
async def root():
    return {"status": "healthy", "service": "MarketMood NLP API"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 