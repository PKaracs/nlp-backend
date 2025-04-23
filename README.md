# NLP Backend for Mood Market Whisperer

This is the backend service for the Mood Market Whisperer application, providing NLP analysis for stock sentiment and market prediction.

## Features

- News sentiment analysis for stocks
- Market trend prediction based on text analysis
- RESTful API for frontend integration

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

## License

This project is licensed under the MIT License. 