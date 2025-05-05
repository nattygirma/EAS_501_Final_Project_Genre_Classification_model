# Movie Genre Classification API

This API uses a BERT model to classify movies into multiple genres based on their descriptions.

## Setup

1. Place your trained BERT model in the `model` directory. The model should be in the Hugging Face format.

2. Build the Docker image:
```bash
docker build -t movie-genre-api .
```

3. Run the container:
```bash
docker run -p 8000:8000 -v $(pwd)/model:/app/model movie-genre-api
```

## API Usage

### Predict Genres

**Endpoint:** `POST /predict`

**Request Body:**
```json
{
    "text": "A thrilling action movie about a secret agent who must save the world from a terrorist organization."
}
```

**Response:**
```json
{
    "genres": {
        "action": 0.95,
        "comedy": 0.1,
        "drama": 0.3,
        "horror": 0.05,
        "romance": 0.1,
        "sci-fi": 0.2,
        "thriller": 0.8
    }
}
```

## AWS Deployment

To deploy this API to AWS:

1. Create an ECR repository:
```bash
aws ecr create-repository --repository-name movie-genre-api
```

2. Build and push the Docker image:
```bash
docker build -t movie-genre-api .
docker tag movie-genre-api:latest <your-account-id>.dkr.ecr.<region>.amazonaws.com/movie-genre-api:latest
aws ecr get-login-password --region <region> | docker login --username AWS --password-stdin <your-account-id>.dkr.ecr.<region>.amazonaws.com
docker push <your-account-id>.dkr.ecr.<region>.amazonaws.com/movie-genre-api:latest
```

3. Deploy to ECS or EC2:
   - For ECS: Create a task definition and service using the pushed image
   - For EC2: Launch an EC2 instance and run the container

## Environment Variables

- `MODEL_PATH`: Path to the model directory (default: "model") 