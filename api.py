from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Union
from nlp_service import MarketNLPAnalyzer
import pandas as pd

router = APIRouter()
nlp_analyzer = MarketNLPAnalyzer()

# Load historical stock data
try:
    stock_data = pd.read_csv("processed_stock_data.csv")
except Exception as e:
    print(f"Warning: Could not load stock data: {e}")
    stock_data = None

class AnalysisRequest(BaseModel):
    text: str
    history: Optional[List[str]] = []

class SentimentResponse(BaseModel):
    score: float
    polarity: float
    subjectivity: float
    finbert_label: str
    finbert_score: float

class MarketSentimentResponse(BaseModel):
    score: float
    confidence: float
    detected_terms: List[str]

class EntityResponse(BaseModel):
    text: str
    type: str
    confidence: float

class TechnicalIndicatorResponse(BaseModel):
    detected: bool
    confidence: float

class TechnicalIndicatorsResponse(BaseModel):
    trend_reversal: TechnicalIndicatorResponse
    breakout: TechnicalIndicatorResponse
    momentum: TechnicalIndicatorResponse
    volume: TechnicalIndicatorResponse
    volatility: TechnicalIndicatorResponse

class ConfidenceFactors(BaseModel):
    length: float
    specificity: float
    source_reliability: float

class ConfidenceMetrics(BaseModel):
    overall_confidence: float
    factors: Dict[str, float]

class HistoricalCorrelation(BaseModel):
    correlation: float
    similar_headlines: List[Dict[str, Union[str, float]]]

class MarketContext(BaseModel):
    sentiment_contribution: float
    technical_contribution: float
    emotion_contribution: float
    bullish_indicators: int
    bearish_indicators: int

class HistoricalTrend(BaseModel):
    avg_30d_change: float
    volatility: float

class StockPrediction(BaseModel):
    direction: str  # "LONG" or "SHORT"
    confidence: float
    score: float
    market_context: MarketContext
    historical_trend: HistoricalTrend

class AnalysisResponse(BaseModel):
    sentiment: SentimentResponse
    market_sentiment: MarketSentimentResponse
    entities: List[EntityResponse]
    technical_indicators: Dict[str, TechnicalIndicatorResponse]
    emotion_analysis: Dict[str, float]
    bias_analysis: Dict[str, Any]
    historical_correlation: HistoricalCorrelation
    confidence_metrics: ConfidenceMetrics
    stock_prediction: Optional[StockPrediction] = None

@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_text(request: AnalysisRequest):
    try:
        # Get NLP analysis
        results = nlp_analyzer.analyze_text(request.text, request.history)
        
        # Debug logging
        print(f"Raw NLP results: {results}")
        
        # Ensure all technical indicators are present
        if 'technical_indicators' not in results:
            results['technical_indicators'] = {
                'trend_reversal': {'detected': False, 'confidence': 0.0},
                'breakout': {'detected': False, 'confidence': 0.0},
                'momentum': {'detected': False, 'confidence': 0.0},
                'volume': {'detected': False, 'confidence': 0.0},
                'volatility': {'detected': False, 'confidence': 0.0}
            }
        
        # Ensure emotion analysis is present and properly formatted
        if 'emotion_analysis' not in results:
            results['emotion_analysis'] = {
                'fear': 0.0,
                'greed': 0.0,
                'uncertainty': 0.0,
                'confidence': 0.0
            }
        else:
            # Ensure all emotion keys are present
            required_emotions = ['fear', 'greed', 'uncertainty', 'confidence']
            for emotion in required_emotions:
                if emotion not in results['emotion_analysis']:
                    results['emotion_analysis'][emotion] = 0.0
            
            # Convert any non-float values to float
            for emotion in results['emotion_analysis']:
                if not isinstance(results['emotion_analysis'][emotion], float):
                    try:
                        results['emotion_analysis'][emotion] = float(results['emotion_analysis'][emotion])
                    except (ValueError, TypeError):
                        results['emotion_analysis'][emotion] = 0.0
        
        # Add stock prediction if we have historical data
        if stock_data is not None and results.get('stock_prediction'):
            # Get the most recent stock data trends
            recent_data = stock_data.tail(30)
            avg_change = recent_data['Portfolio_%_Change'].mean()
            volatility = recent_data['Portfolio_%_Change'].std()
            
            # Update historical trend data
            results['stock_prediction']['historical_trend'].update({
                'avg_30d_change': float(avg_change),
                'volatility': float(volatility)
            })
            
        # Debug logging
        print(f"Processed results: {results}")
        
        return results
    except Exception as e:
        print(f"Error in analyze_text: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/batch-analyze")
async def batch_analyze(texts: List[str]):
    try:
        results = []
        for text in texts:
            analysis = nlp_analyzer.analyze_text(text)
            results.append(analysis)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": True,
        "stock_data_loaded": stock_data is not None
    } 