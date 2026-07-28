# flood-risk-api

FastAPI backend for serving Bharatpur flood-risk grid GeoJSON data to a React GIS dashboard.

## Project Structure

```text
flood-risk-api/
├── app/
│   ├── main.py
│   ├── config.py
│   ├── utils.py
│   └── routes/
│       ├── grids.py
│       ├── risk.py
│       └── summary.py
├── data/
│   └── bharatpur_static_grid_250m.geojson
├── requirements.txt
├── README.md
├── Dockerfile
└── docker-compose.yml
```

## Requirements

- Python 3.10+
- GeoJSON file at `data/bharatpur_static_grid_250m.geojson`

## Setup

1. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Place your QGIS-exported GeoJSON file here:

```text
data/bharatpur_static_grid_250m.geojson
```

4. Run the API:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Available Endpoints

- `GET /`
- `GET /health`
- `GET /api/grids`
- `GET /api/grids/{grid_id}`
- `GET /api/risk/static`
- `GET /api/summary`
- `GET /api/filters/lulc`
- `GET /api/filters/risk-classes`
- `GET /api/grids/filter`
- `GET /api/models`
- `POST /api/predict`
- `POST /api/predict/compare`
- `GET /api/risk-map`
- `GET /api/risk-map/affected-area`
- `GET /api/layers/rivers`
- `GET /api/conditions/live`

See `FRONTEND_CONTEXT.md` for full request/response shapes, the live-prediction data flow, and known data-quality caveats.

## Filter Examples

```bash
curl "http://localhost:8000/api/grids/filter?risk_class=High"
curl "http://localhost:8000/api/grids/filter?lulc=Cropland"
curl "http://localhost:8000/api/grids/filter?max_distance=1000"
curl "http://localhost:8000/api/grids/filter?min_elevation=150&max_elevation=300"
```

## Frontend CORS

The API allows requests from:

- `http://localhost:3000`
- `http://127.0.0.1:3000`
- `http://localhost:5173`
- `http://127.0.0.1:5173`

## Docker

Build the image:

```bash
docker build -t flood-risk-backend .
```

Run the container:

```bash
docker run -p 8000:8000 flood-risk-backend
```

The container serves the API on:

- `http://localhost:8000/health`
- `http://localhost:8000/docs`

Notes:

- The image uses `python:3.11-slim`.
- The Docker build copies the application code and the `data/` directory into the container when that folder exists in the project.
- If you want to swap datasets locally without rebuilding, you can still mount `data/` at runtime with `-v "$(pwd)/data:/app/data"`.

## Docker Compose

Start the backend:

```bash
docker compose up --build
```

Stop the backend:

```bash
docker compose down
```

View logs:

```bash
docker compose logs -f
```

Rebuild clean:

```bash
docker compose build --no-cache
```

With Docker Compose running, the backend is available at:

- `http://localhost:8000/`
- `http://localhost:8000/docs`
- `http://localhost:8000/health`

## Notes

- The backend uses file-based GeoJSON storage only.
- GeoJSON data is cached in memory after first load.
- If the dataset file is missing or invalid, the API returns JSON errors.
