from fastapi import FastAPI, Request, Form
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
import numpy as np
from cred_card_proj.pipeline.prediction import PredictionPipeline
from schemas.user_input import UserInput

app = FastAPI()
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(request, "index.html", {})


@app.get("/health")
def health_check():
    return {"status": "OK"}


# ---- JSON API (unchanged) — for programmatic / Swagger use ----
@app.post("/predict")
def predict_fraud(data: UserInput):
    features = [
        data.Time,
        data.V1,
        data.V2,
        data.V3,
        data.V4,
        data.V5,
        data.V6,
        data.V7,
        data.V8,
        data.V9,
        data.V10,
        data.V11,
        data.V12,
        data.V13,
        data.V14,
        data.V15,
        data.V16,
        data.V17,
        data.V18,
        data.V19,
        data.V20,
        data.V21,
        data.V22,
        data.V23,
        data.V24,
        data.V25,
        data.V26,
        data.V27,
        data.V28,
        data.Amount,
    ]
    user_input = np.array(features).reshape(1, -1)

    try:
        pipeline = PredictionPipeline()
        prediction = pipeline.predict(user_input)
        return JSONResponse(status_code=200, content={"response": prediction})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# ---- HTML form flow — for the browser console (index.html / result.html) ----
@app.post("/predict-form", response_class=HTMLResponse)
def predict_form(
    request: Request,
    Time: float = Form(...),
    V1: float = Form(...),
    V2: float = Form(...),
    V3: float = Form(...),
    V4: float = Form(...),
    V5: float = Form(...),
    V6: float = Form(...),
    V7: float = Form(...),
    V8: float = Form(...),
    V9: float = Form(...),
    V10: float = Form(...),
    V11: float = Form(...),
    V12: float = Form(...),
    V13: float = Form(...),
    V14: float = Form(...),
    V15: float = Form(...),
    V16: float = Form(...),
    V17: float = Form(...),
    V18: float = Form(...),
    V19: float = Form(...),
    V20: float = Form(...),
    V21: float = Form(...),
    V22: float = Form(...),
    V23: float = Form(...),
    V24: float = Form(...),
    V25: float = Form(...),
    V26: float = Form(...),
    V27: float = Form(...),
    V28: float = Form(...),
    Amount: float = Form(...),
):
    features = [
        Time,
        V1,
        V2,
        V3,
        V4,
        V5,
        V6,
        V7,
        V8,
        V9,
        V10,
        V11,
        V12,
        V13,
        V14,
        V15,
        V16,
        V17,
        V18,
        V19,
        V20,
        V21,
        V22,
        V23,
        V24,
        V25,
        V26,
        V27,
        V28,
        Amount,
    ]
    user_input = np.array(features).reshape(1, -1)

    try:
        pipeline = PredictionPipeline()
        result = pipeline.predict(user_input)  # {"predicted_category": 0 or 1}
        return templates.TemplateResponse(request, "result.html", {})
    except Exception as e:
        return HTMLResponse(content=f"Something went wrong: {e}", status_code=500)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8080)
