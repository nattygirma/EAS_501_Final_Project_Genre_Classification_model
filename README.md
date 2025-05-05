# BERT Genre Classification API

A FastAPI-based service that uses BERT to classify movie descriptions into genres. The model predicts multiple genres for a given movie description with confidence scores.

## Features

- Multi-label genre classification using BERT
- RESTful API endpoints for predictions
- Support for both full genre predictions and top-3 genres
- Docker support for easy deployment

## Prerequisites

- Python 3.10+
- Docker (optional, for containerized deployment)

## Installation

### Local Setup

1. Clone the repository:
```bash
git clone https://github.com/nattygirma/EAS_501_Final_Project_Genre_Classification_model.git
cd bert-demo
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
# On Windows
.\venv\Scripts\activate
# On Unix/MacOS
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Docker Setup

1. Build the Docker image:
```bash
docker build -t bert-genre-api .
```

2. Run the container:
```bash
docker run -p 80:80 bert-genre-api
```

## API Endpoints

### 1. Health Check
- **GET** `/`
- Returns a simple health check message

### 2. Predict All Genres
- **POST** `/predict`
- Input: Movie description text
- Returns: Dictionary of all genres with confidence scores

Example request:
```bash
curl -X POST "http://localhost:80/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "A young wizard discovers his magical heritage and battles dark forces while attending a school of witchcraft and wizardry."}'
```

### 3. Predict Top 3 Genres
- **POST** `/predict/top3`
- Input: Movie description text
- Returns: Top 3 genres with confidence scores

Example request:
```bash
curl -X POST "http://localhost:80/predict/top3" \
     -H "Content-Type: application/json" \
     -d '{"text": "A young wizard discovers his magical heritage and battles dark forces while attending a school of witchcraft and wizardry."}'
```

## Model Information

- Base Model: BERT (bert-base-uncased)
- Architecture: Custom classifier head on top of BERT
- Training: Fine-tuned on movie descriptions dataset
- Output: Multi-label classification with sigmoid activation

## Docker Images

The API is available as Docker images:

- Latest version: `nattygirma27/bert-demo:latest`
- Specific version: `nattygirma27/bert-demo:bert-genre-api-1`

Pull the image:
```bash
docker pull nattygirma27/bert-demo:bert-genre-api-1
```

## Development

### Project Structure
```
bert-genre-api/
├── app/
│   ├── main.py
│   ├── model/
│   │   ├── genre_classifier.pt
│   │   ├── metadata.json
│   │   └── mlb.pkl
│   └── __init__.py
├── requirements.txt
├── Dockerfile
└── README.md
```

### Running Tests
```bash
# Add test commands here when implemented
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

Natnael Fisseha - [@natnaelgirma27](https://x.com/natnaelgirma27)

Project Link: [https://github.com/nattygirma/EAS_501_Final_Project_Genre_Classification_model](https://github.com/nattygirma/EAS_501_Final_Project_Genre_Classification_model) 