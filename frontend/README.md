# Flood Risk Dashboard

## Environment

Create a `.env` file from `.env.example` and set:

```bash
VITE_API_BASE_URL=http://localhost:8000
```

If omitted, the frontend defaults to `http://localhost:8000`.

## Run Frontend

```bash
source ~/.nvm/nvm.sh
nvm use 18
npm install
npm run dev
```

The local Vite development server runs at [http://localhost:5173](http://localhost:5173/).

## Run With Backend

1. Start the FastAPI backend on `http://localhost:8000`
2. Start the frontend:

```bash
source ~/.nvm/nvm.sh
nvm use 18
npm run dev
```

The frontend expects these backend endpoints:

- `GET /api/grids`
- `GET /api/grids/{grid_id}`
- `GET /api/summary`
- `GET /api/grids/filter`
- `GET /api/risk/static`

## Docker

Build the production image with the backend API base URL baked into the Vite build:

```bash
docker build --build-arg VITE_API_BASE_URL=http://localhost:8000 -t flood-risk-frontend .
```

Run the container and expose it on [http://localhost:3000](http://localhost:3000/):

```bash
docker run -p 3000:80 flood-risk-frontend
```

When running in Docker, the frontend is served by Nginx on port `80` inside the container and is available on port `3000` on your machine.

## Docker Compose

Start the frontend container with Docker Compose:

```bash
docker compose up --build
```

Stop the running container:

```bash
docker compose down
```

View container logs:

```bash
docker compose logs -f
```

Rebuild the image without cache:

```bash
docker compose build --no-cache
```

The Compose setup serves the frontend at [http://localhost:3000](http://localhost:3000/) and builds the app with `VITE_API_BASE_URL=http://localhost:8000`, so browser requests still go to the separately running FastAPI backend at [http://localhost:8000](http://localhost:8000/).
