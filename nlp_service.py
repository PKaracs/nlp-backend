from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from textblob import TextBlob
import spacy
import numpy as np
from typing import List, Dict, Any, Optional
import yfinance as yf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import sent_tokenize
import torch
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('maxent_ne_chunker', quiet=True)
    nltk.download('words', quiet=True)
    logger.info("NLTK data downloaded successfully")
except Exception as e:
    logger.error(f"Error downloading NLTK data: {e}")

class MarketNLPAnalyzer:
    def __init__(self):
        self.sentiment_analyzer = None
        self.nlp = None
        self.vectorizer = None
        self.initialize_components()

    def initialize_components(self):
        try:
            # Initialize transformers
            logger.info("Initializing FinBERT sentiment analyzer...")
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="ProsusAI/finbert",
                tokenizer="ProsusAI/finbert"
            )
            logger.info("FinBERT sentiment analyzer initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing FinBERT sentiment analyzer: {e}")
            raise

        try:
            # Load SpaCy model
            logger.info("Loading spaCy model...")
            self.nlp = spacy.load("en_core_web_trf")
            logger.info("spaCy model loaded successfully")
        except OSError:
            logger.warning("spaCy model not found, attempting to download...")
            try:
                import subprocess
                subprocess.run(["python", "-m", "spacy", "download", "en_core_web_trf"], check=True)
                self.nlp = spacy.load("en_core_web_trf")
                logger.info("spaCy model downloaded and loaded successfully")
            except Exception as e:
                logger.error(f"Error downloading spaCy model: {e}")
                raise

        # Initialize TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer()
        
        # Market-specific terms with sentiment scores
        self.market_terms = {
            'bullish': 1.0,
            'bearish': -1.0,
            'rally': 0.8,
            'surge': 0.9,
            'plunge': -0.9,
            'crash': -1.0,
            'correction': -0.5,
            'recovery': 0.6,
            'boom': 0.9,
            'bust': -0.9,
            'soar': 0.95,
            'tumble': -0.8,
            'slump': -0.7,
            'rebound': 0.7
        }

    def analyze_text(self, text: str, history: List[str] = None) -> Dict[str, Any]:
        if not text:
            logger.warning("Empty text provided for analysis")
            raise ValueError("Text cannot be empty")

        if history is None:
            history = []

        try:
            logger.info("Starting text analysis...")
            logger.info(f"Analyzing text: {text[:100]}...")
            
            # Get all analysis components
            sentiment_results = self._analyze_sentiment(text)
            market_sentiment = self._analyze_market_specific(text)
            emotion_results = self._analyze_emotions(text)
            technical_results = self._analyze_technical_patterns(text)
            
            # Calculate trading signal
            signal_prediction = self._calculate_trading_signal(
                sentiment=sentiment_results,
                market_sentiment=market_sentiment,
                emotions=emotion_results,
                technical_indicators=technical_results
            )
            
            results = {
                'sentiment': sentiment_results,
                'market_sentiment': market_sentiment,
                'entities': self._extract_entities(text),
                'technical_indicators': technical_results,
                'emotion_analysis': emotion_results,
                'bias_analysis': self._analyze_bias(text),
                'historical_correlation': self._analyze_historical_correlation(text, history),
                'confidence_metrics': self._calculate_confidence_metrics(text),
                'stock_prediction': signal_prediction
            }
            
            logger.info(f"Final analysis results: {results}")
            return results
        except Exception as e:
            logger.error(f"Error during text analysis: {e}")
            raise

    def _analyze_sentiment(self, text: str) -> Dict[str, Any]:
        try:
            # Use FinBERT for financial sentiment
            finbert_results = self.sentiment_analyzer(text)[0]
            
            # Use TextBlob for additional sentiment metrics
            blob = TextBlob(text)
            
            # Combine both analyses
            sentiment_score = (blob.sentiment.polarity + finbert_results['score']) / 2
            
            return {
                'score': round(sentiment_score * 100, 2),  # Convert to percentage
                'polarity': round(blob.sentiment.polarity * 100, 2),
                'subjectivity': round(blob.sentiment.subjectivity * 100, 2),
                'finbert_label': finbert_results['label'],
                'finbert_score': round(finbert_results['score'] * 100, 2)
            }
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {e}")
            raise

    def _analyze_market_specific(self, text: str) -> Dict[str, Any]:
        text_lower = text.lower()
        scores = []
        detected_terms = []
        
        for term, score in self.market_terms.items():
            if term in text_lower:
                scores.append(score)
                detected_terms.append(term)
                
        if not scores:
            return {
                'score': 0.0,
                'confidence': 0.0,
                'detected_terms': []
            }
            
        return {
            'score': round(np.mean(scores) * 100, 2),
            'confidence': round(len(scores) / len(self.market_terms) * 100, 2),
            'detected_terms': detected_terms
        }

    def _extract_entities(self, text: str) -> List[Dict[str, Any]]:
        doc = self.nlp(text)
        entities = []
        
        for ent in doc.ents:
            confidence = self._calculate_entity_confidence(ent)
            entities.append({
                'text': ent.text,
                'type': ent.label_,
                'confidence': round(confidence * 100, 2)
            })
            
        return entities

    def _analyze_technical_patterns(self, text: str) -> Dict[str, Dict[str, Any]]:
        patterns = {
            'trend_reversal': ['reversal', 'bottom', 'top', 'turnaround', 'recovery', 'rebound', 'bounce'],
            'breakout': ['breakout', 'breakdown', 'resistance', 'support', 'breakthrough', 'surge', 'plunge'],
            'momentum': ['momentum', 'oversold', 'overbought', 'RSI', 'MACD', 'trend', 'strength'],
            'volume': ['volume', 'trading', 'liquidity', 'activity', 'flow', 'interest'],
            'volatility': ['volatile', 'swing', 'range', 'fluctuation', 'uncertainty', 'risk']
        }
        
        results = {}
        text_lower = text.lower()
        
        for pattern_name, pattern_terms in patterns.items():
            matches = sum(term in text_lower for term in pattern_terms)
            # Calculate confidence based on number of matches
            confidence = min((matches / len(pattern_terms)) * 100, 100)
            results[pattern_name] = {
                'detected': matches > 0,
                'confidence': round(float(confidence), 2)
            }
            
        return results

    def _analyze_emotions(self, text: str) -> Dict[str, float]:
        # Market-specific emotion terms with weights (1-10 scale)
        emotions = {
            'fear': [
                ('worry', 5), ('risk', 4), ('concern', 3), ('afraid', 7), ('panic', 9), ('doubt', 4), 
                ('uncertain', 5), ('volatile', 6), ('danger', 7), ('threat', 7), ('fear', 8), ('scared', 8), 
                ('nervous', 6), ('anxious', 7), ('crash', 10), ('plunge', 9), ('plummet', 9), ('crisis', 10), 
                ('warning', 6), ('bearish', 7), ('sell-off', 8), ('downturn', 7), ('decline', 6), 
                ('negative', 5), ('weak', 5), ('losses', 7), ('downgrade', 6)
            ],
            'greed': [
                ('opportunity', 5), ('profit', 7), ('gain', 6), ('upside', 6), ('boom', 9), ('growth', 7), 
                ('success', 6), ('win', 6), ('bullish', 8), ('optimistic', 7), ('surge', 9), ('breakthrough', 8), 
                ('innovation', 6), ('advance', 6), ('rally', 8), ('upgrade', 7), ('outperform', 7), ('beat', 6), 
                ('strong', 5), ('positive', 5), ('momentum', 6), ('soar', 9), ('jump', 7), ('rise', 6), 
                ('climb', 6), ('higher', 5), ('record', 8)
            ],
            'uncertainty': [
                ('maybe', 5), ('possibly', 6), ('might', 5), ('could', 4), ('perhaps', 5), ('unclear', 7), 
                ('ambiguous', 7), ('mixed', 5), ('unpredictable', 8), ('speculative', 7), ('uncertain', 8), 
                ('doubtful', 7), ('questionable', 6), ('volatile', 7), ('fluctuate', 6), ('unstable', 8),
                ('unknown', 7), ('variable', 5), ('tentative', 6), ('unsure', 7), ('hesitant', 6), ('cautious', 5)
            ],
            'confidence': [
                ('certain', 7), ('definitely', 8), ('surely', 7), ('confident', 8), ('guaranteed', 9), 
                ('strong', 6), ('clear', 5), ('positive', 5), ('bullish', 7), ('optimistic', 7), ('sure', 7), 
                ('trust', 6), ('belief', 5), ('conviction', 7), ('assured', 8), ('proven', 8), ('reliable', 7), 
                ('stable', 6), ('solid', 6), ('robust', 7), ('resilient', 6), ('breakthrough', 8), ('innovative', 6)
            ]
        }
        
        results = {}
        text_lower = text.lower()
        words = text_lower.split()
        word_count = len(words)
        
        # Edge case - empty text
        if word_count == 0:
            return {emotion: 0.0 for emotion in emotions.keys()}
        
        # Get word combinations (for phrases)
        word_pairs = [' '.join(words[i:i+2]) for i in range(len(words)-1)]
        word_triples = [' '.join(words[i:i+3]) for i in range(len(words)-2)]
        
        # Calculate sentence-level contextual analysis
        sentences = sent_tokenize(text)
        
        # Process each emotion
        for emotion, terms_with_weights in emotions.items():
            # Extract terms and weights for easier processing
            terms = [term for term, _ in terms_with_weights]
            weights = {term: weight for term, weight in terms_with_weights}
            
            # Count matches with weighted values
            term_matches = []
            
            # Process individual words
            for word in words:
                for term in terms:
                    if term == word:
                        term_matches.append((term, weights[term]))
                    elif term in word and len(word) < len(term) + 3:  # Partial match with limited extra chars
                        term_matches.append((term, weights[term] * 0.7))  # Reduce weight for partial matches
            
            # Process word pairs (phrases)
            for phrase in word_pairs + word_triples:
                for term in terms:
                    if term == phrase:
                        term_matches.append((term, weights[term] * 1.2))  # Boost weight for exact phrase matches
            
            # Calculate contextual amplification for each sentence
            sentence_scores = []
            for sentence in sentences:
                sentence_lower = sentence.lower()
                sentence_strength = 0
                
                # Check for intensifiers and negations
                intensifiers = ['very', 'extremely', 'highly', 'significantly', 'substantially', 'strongly']
                negations = ['not', "n't", 'no', 'never', 'neither', 'nor']
                
                intensifier_present = any(intensifier in sentence_lower.split() for intensifier in intensifiers)
                negation_present = any(negation in sentence_lower.split() for negation in negations)
                
                # Count term mentions in this sentence
                sentence_terms = []
                for term in terms:
                    if term in sentence_lower:
                        sentence_terms.append((term, weights[term]))
                
                # Calculate sentence emotion strength
                if sentence_terms:
                    # Base strength
                    base_strength = sum(weight for _, weight in sentence_terms) / len(sentence_terms)
                    
                    # Apply contextual modifiers
                    if intensifier_present:
                        base_strength *= 1.5
                    if negation_present:
                        # Negation reverses emotion in most cases
                        if emotion in ['fear', 'uncertainty']:
                            # Negating fear or uncertainty often implies confidence
                            base_strength *= 0.2  # Significantly reduce, but not eliminate
                        elif emotion in ['confidence', 'greed']:
                            # Negating confidence or greed often implies fear or uncertainty
                            base_strength *= 0.3
                    
                    sentence_scores.append(base_strength)
            
            # Calculate normalized scores based on matches
            if term_matches:
                # Get unique terms (count each term only once with its highest weight)
                unique_terms = {}
                for term, weight in term_matches:
                    if term not in unique_terms or weight > unique_terms[term]:
                        unique_terms[term] = weight
                
                # Sum the weights of unique terms
                weighted_sum = sum(unique_terms.values())
                
                # Normalize by the possible maximum (if all terms were present at max weight)
                max_possible = sum(max(weights.values()) for _ in range(min(len(weights), 10)))  # Cap at 10 terms
                
                # Emotion density (based on word count)
                emotion_density = len(unique_terms) / min(word_count, 50)  # Cap at 50 words
                
                # Base score
                base_score = (weighted_sum / max_possible) * 70  # Scale to ~70 max
                
                # Add density factor (max 30 points)
                density_score = min(emotion_density * 30, 30)
                
                # Add sentence contextual score
                contextual_score = 0
                if sentence_scores:
                    # Average sentence emotion strength (scale to max 20 points)
                    contextual_score = min((sum(sentence_scores) / len(sentence_scores)) * 2, 20)
                
                # Final score is base score + density bonus + contextual score
                results[emotion] = round(min(base_score + density_score + contextual_score, 100), 2)
            else:
                # No exact matches, but check for semantic similarity
                # This provides low but non-zero scores for semantically related content
                if emotion == 'fear' and any(term in text_lower for term in ['problem', 'issue', 'difficult']):
                    results[emotion] = round(15 + (len(text_lower) / 1000), 2)
                elif emotion == 'greed' and any(term in text_lower for term in ['increase', 'better', 'improve']):
                    results[emotion] = round(10 + (len(text_lower) / 1000), 2)
                elif emotion == 'uncertainty' and any(term in text_lower for term in ['new', 'change', 'future']):
                    results[emotion] = round(8 + (len(text_lower) / 1000), 2)
                elif emotion == 'confidence' and any(term in text_lower for term in ['good', 'great', 'well']):
                    results[emotion] = round(12 + (len(text_lower) / 1000), 2)
                else:
                    # Provide a minimal score based on text characteristics
                    results[emotion] = round(min(5 + (len(text_lower) / 2000), 10), 2)
        
        # Ensure all emotions are present
        for emotion in emotions.keys():
            if emotion not in results:
                results[emotion] = 0.0
        
        # Apply emotion relationship adjustments
        # High fear typically reduces confidence and vice versa
        if results['fear'] > 60 and results['confidence'] > 50:
            results['confidence'] = results['confidence'] * 0.7
        if results['confidence'] > 70 and results['fear'] > 40:
            results['fear'] = results['fear'] * 0.8
        
        # High greed typically reduces uncertainty and vice versa
        if results['greed'] > 60 and results['uncertainty'] > 50:
            results['uncertainty'] = results['uncertainty'] * 0.75
        if results['uncertainty'] > 70 and results['greed'] > 40:
            results['greed'] = results['greed'] * 0.85
            
        # Final rounding
        results = {k: round(v, 2) for k, v in results.items()}
            
        logger.info(f"Emotion analysis input text: {text[:100]}...")
        logger.info(f"Emotion analysis detailed results: {results}")
        
        return results

    def _analyze_bias(self, text: str) -> Dict[str, Any]:
        doc = self.nlp(text)
        
        extreme_terms = ['always', 'never', 'definitely', 'absolutely', 'certainly']
        extreme_count = sum(token.text.lower() in extreme_terms for token in doc)
        
        hedge_terms = ['might', 'maybe', 'perhaps', 'possibly', 'probably']
        hedge_count = sum(token.text.lower() in hedge_terms for token in doc)
        
        bias_score = min((extreme_count * 20) + (hedge_count * 10), 100)
        
        return {
            'bias_score': round(bias_score, 2),
            'extreme_language_detected': extreme_count > 0,
            'hedging_detected': hedge_count > 0
        }

    def _analyze_historical_correlation(self, text: str, history: List[str]) -> Dict[str, Any]:
        if not history:
            return {
                'correlation': 0.0,
                'similar_headlines': []
            }
            
        all_texts = [text] + history
        tfidf_matrix = self.vectorizer.fit_transform(all_texts)
        similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])[0]
        
        top_indices = np.argsort(similarities)[-5:][::-1]
        similar_headlines = [
            {
                'text': history[i],
                'similarity': round(float(similarities[i] * 100), 2)
            }
            for i in top_indices if similarities[i] > 0.3
        ]
        
        return {
            'correlation': round(float(np.mean(similarities) * 100), 2),
            'similar_headlines': similar_headlines
        }

    def _calculate_confidence_metrics(self, text: str) -> Dict[str, Any]:
        # Calculate various confidence factors
        length_score = self._calculate_length_score(text)
        specificity_score = self._calculate_specificity(text)
        source_reliability = self._estimate_source_reliability(text)
        
        # Calculate overall confidence
        overall_confidence = (length_score + specificity_score + source_reliability) / 3
        
        return {
            'overall_confidence': round(overall_confidence * 100, 2),
            'factors': {
                'length': round(length_score * 100, 2),
                'specificity': round(specificity_score * 100, 2),
                'source_reliability': round(source_reliability * 100, 2)
            }
        }

    def _calculate_specificity(self, text: str) -> float:
        doc = self.nlp(text)
        
        named_entities = len(doc.ents)
        numbers = len([token for token in doc if token.like_num])
        technical_terms = len([token for token in doc if token.text.lower() in self.market_terms])
        
        specificity = (named_entities + numbers + technical_terms) / max(len(doc), 1)
        return min(specificity, 1.0)

    def _estimate_source_reliability(self, text: str) -> float:
        reliable_sources = ['reuters', 'bloomberg', 'wsj', 'financial times']
        text_lower = text.lower()
        
        if any(source in text_lower for source in reliable_sources):
            return 1.0
        return 0.7

    def _calculate_entity_confidence(self, entity) -> float:
        length_factor = min(len(entity.text.split()) / 3, 1.0)
        context_factor = 0.8  # Default context confidence
        return (length_factor + context_factor) / 2

    def _calculate_length_score(self, text: str) -> float:
        """Calculate a confidence score based on text length."""
        word_count = len(text.split())
        # Normalize score between 0 and 1
        # Consider text between 10 and 100 words as optimal
        if word_count < 10:
            return word_count / 10
        elif word_count > 100:
            return 1.0
        else:
            return 0.5 + (word_count - 10) / 180  # Linear scaling between 10 and 100 words 

    def _calculate_trading_signal(self, sentiment: Dict[str, Any], 
                                market_sentiment: Dict[str, Any],
                                emotions: Dict[str, float],
                                technical_indicators: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        # Initialize score components
        sentiment_score = 0.0
        technical_score = 0.0
        emotion_score = 0.0
        
        # Calculate sentiment component (0-100 scale)
        finbert_score = sentiment['finbert_score']  # Already 0-100
        general_sentiment = sentiment['score']  # Already 0-100
        sentiment_score = (finbert_score + general_sentiment) / 2
        
        # Calculate technical component and count bullish/bearish indicators
        bullish_indicators = 0
        bearish_indicators = 0
        
        # Check sentiment
        if sentiment['finbert_label'] == 'positive' or sentiment_score > 50:
            bullish_indicators += 1
        else:
            bearish_indicators += 1
            
        # Check emotions
        if (emotions.get('greed', 0) + emotions.get('confidence', 0)) > (emotions.get('fear', 0) + emotions.get('uncertainty', 0)):
            bullish_indicators += 1
        else:
            bearish_indicators += 1
            
        # Check market-specific terms
        if market_sentiment['score'] > 0:
            bullish_indicators += 1
        elif market_sentiment['score'] < 0:
            bearish_indicators += 1
            
        # Calculate technical indicator contribution
        for indicator_name, indicator in technical_indicators.items():
            if indicator['detected'] and indicator['confidence'] > 50:
                if indicator_name in ['breakout', 'momentum'] and 'surge' in indicator_name:
                    bullish_indicators += 1
                elif indicator_name in ['trend_reversal'] and sentiment_score > 50:
                    bullish_indicators += 1
                elif indicator_name in ['volatility']:
                    # Volatility is bearish in most contexts unless sentiment is very positive
                    if sentiment_score > 70:
                        bullish_indicators += 1
                    else:
                        bearish_indicators += 1
                else:
                    # Default: use sentiment to determine if indicator is bullish or bearish
                    if sentiment_score > 50:
                        bullish_indicators += 1
                    else:
                        bearish_indicators += 1
        
        # Calculate technical score based on indicator balance
        technical_score = ((bullish_indicators - bearish_indicators) / max(bullish_indicators + bearish_indicators, 1)) * 100
        
        # Calculate emotion component
        greed_confidence = emotions.get('greed', 0) + emotions.get('confidence', 0)
        fear_uncertainty = emotions.get('fear', 0) + emotions.get('uncertainty', 0)
        emotion_score = ((greed_confidence - fear_uncertainty) / 2)  # Scale to roughly -100 to 100
        
        # Combine all components with weights
        final_score = (
            sentiment_score * 0.4 +      # 40% weight to sentiment
            technical_score * 0.3 +      # 30% weight to technical
            emotion_score * 0.3          # 30% weight to emotions
        )
        
        # Determine direction and confidence
        direction = "LONG" if final_score > 0 else "SHORT"
        confidence = min(abs(final_score), 100)  # Cap at 100
        
        # Add market context
        market_context = {
            'sentiment_contribution': round(sentiment_score, 2),
            'technical_contribution': round(technical_score, 2),
            'emotion_contribution': round(emotion_score, 2),
            'bullish_indicators': bullish_indicators,
            'bearish_indicators': bearish_indicators
        }
        
        # Check for strong signals that could override the general analysis
        bullish_terms = ['surge', 'soar', 'jump', 'rally', 'breakthrough', 'boom']
        bearish_terms = ['crash', 'plunge', 'tumble', 'slump', 'collapse', 'crisis']
        
        # Check if any strong bullish terms are detected
        for term in market_sentiment.get('detected_terms', []):
            if term.lower() in bullish_terms:
                direction = "LONG"
                confidence = max(confidence, 70)  # Minimum 70% confidence
                bullish_indicators += 1
            elif term.lower() in bearish_terms:
                direction = "SHORT"
                confidence = max(confidence, 70)  # Minimum 70% confidence
                bearish_indicators += 1
        
        return {
            'direction': direction,
            'confidence': round(confidence, 2),
            'score': round(final_score, 2),
            'market_context': market_context,
            'historical_trend': {
                'avg_30d_change': 0.0,  # Placeholder for historical data
                'volatility': 0.0       # Placeholder for historical data
            }
        } 