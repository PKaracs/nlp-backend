# NLP Backend for Mood Market Whisperer

This is the backend service for the Mood Market Whisperer application, providing NLP analysis for stock sentiment and market prediction.

## Features

- News sentiment analysis for stocks
- Market trend prediction based on text analysis
- RESTful API for frontend integration

## NLP Technologies

This backend implements advanced natural language processing techniques:

- **Transformer Models**: Leveraging state-of-the-art transformer architecture for deep contextual understanding of financial texts
- **Sentiment Analysis**: Using specialized financial sentiment analysis models to detect market sentiment
- **Named Entity Recognition**: Identifying companies, products, and financial terms in text data
- **Text Classification**: Categorizing news and social media content based on market impact
- **NLTK**: Natural Language Toolkit for tokenization and text preprocessing
- **TextBlob**: For simplified sentiment analysis and text classification
- **Scikit-learn**: For feature extraction and model evaluation
- **PyTorch**: Powering the core of our deep learning models

## Setup and Installation

### Prerequisites

- Python 3.9+
- pip (Python package manager)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/PKaracs/nlp-backend.git
   cd nlp-backend
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Server

Start the server with:

```bash
python main.py
```

By default, the server runs on port 8000.

## API Endpoints

- `POST /analyze_sentiment`: Analyze text sentiment
- `POST /analyze_stock`: Analyze stock sentiment based on news
- `GET /get_prediction`: Get market predictions

## Development

### Project Structure

- `main.py`: Entry point for the application
- `api.py`: FastAPI routes and endpoint definitions
- `nlp_service.py`: Core NLP processing functionality
- `requirements.txt`: Project dependencies

## Deployment Checklist

✅ Environment variables configuration  
✅ Dependencies listed in requirements.txt  
✅ Error handling and logging  
✅ CORS configuration for frontend integration  
✅ API documentation  
✅ Rate limiting for public endpoints  

## License

This project is licensed under the MIT License. 