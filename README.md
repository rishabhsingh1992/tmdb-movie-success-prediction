# Movie Success Classification (Logistic Regression)

Binary classification project that predicts whether a movie is financially successful based on three features:
- `budget`
- `popularity`
- `runtime`

The model is trained with scikit-learn and served through a FastAPI endpoint.

## Tech Stack

- Python 3.11+
- scikit-learn
- pandas / numpy
- FastAPI + Uvicorn
- joblib

## Project Structure

```text
movie-success-classification/
├── api/
│   └── routes/
│       └── predict.py              # FastAPI app and prediction endpoint
├── models/
│   ├── movie_success_classification_model.pkl
│   └── movie_success_classification_scaler.pkl
├── main.py                         # Training script (creates .pkl artifacts)
├── requirements.txt
└── README.md
```

## How It Works

1. Load movie dataset.
2. Create target label:
	 - `profit = revenue - budget`
	 - `is_successful = profit > 0`
3. Train logistic regression using:
	 - `budget`, `popularity`, `runtime`
4. Save trained model and scaler to `models/`.
5. Serve predictions with FastAPI via `POST /predict`.

## Setup

### 1. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

## Train the Model

Run the training script from the repository root:

```powershell
python main.py
```

Expected output artifacts:
- `models/movie_success_classification_model.pkl`
- `models/movie_success_classification_scaler.pkl`

## Run the API

From the repository root:

```powershell
fastapi dev api/routes/predict.py
```

The API will start locally (default development port is shown in terminal output).

## API Contract

### Endpoint

- `POST /predict`

### Request Body

```json
{
	"budget": 150000000,
	"popularity": 120.5,
	"runtime": 130
}
```

### Response

```json
{
	"is_successful": true
}
```

## Quick Test (PowerShell)

```powershell
$body = @{
	budget = 150000000
	popularity = 120.5
	runtime = 130
} | ConvertTo-Json

Invoke-RestMethod -Method Post `
	-Uri "http://127.0.0.1:8000/predict" `
	-ContentType "application/json" `
	-Body $body
```

## Notes for Production

- Keep training and serving paths stable and configurable.
- Add input validation constraints (minimum/maximum ranges) to reduce bad requests.
- Add automated tests for model loading and endpoint behavior.
- Version model artifacts outside Git if file sizes grow.

## Troubleshooting

- If model files are missing, run `python main.py` again.
- If FastAPI command is unavailable, ensure the venv is activated and `fastapi-cli` is installed.
- If you get path errors for the dataset, verify dataset location used in `main.py`.

