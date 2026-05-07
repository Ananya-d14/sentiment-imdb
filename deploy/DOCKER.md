# Docker

```bash
docker build -t sentiment-imdb .

docker run --rm -p 8501:8501 \
  -v "$(pwd)/models:/app/models:ro" \
  sentiment-imdb
```

Or with compose:

```bash
docker compose up --build
```

Open http://localhost:8501.

## Notes

- Base image: `python:3.11-slim`
- TensorFlow CPU only. For GPU, swap `tensorflow-cpu` for `tensorflow`
  in `requirements.txt` and use an nvidia/cuda base image.
- NLTK stopwords are downloaded at build time.
- Healthcheck hits `/_stcore/health` on port 8501.

## Hosting providers

- Render: new Web Service, Docker, point at this repo
- Railway: `railway up`
- Fly.io: `fly launch` then `fly deploy`
- Cloud Run: `gcloud run deploy --source .`
- AWS App Runner: push image to ECR, point App Runner at it

Mount or COPY the trained `models/` directory into `/app/models`,
otherwise the app shows the setup banner.
