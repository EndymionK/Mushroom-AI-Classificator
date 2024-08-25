from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
import io
import pandas as pd
from predict import predict_model

app = FastAPI()

@app.post("/train/")
async def train(file: UploadFile = File(...)):
    try:
        df = pd.read_csv(io.BytesIO(await file.read()))
        model_filename = train_model(df)  # Necesitas definir train_model
        with open(model_filename, "rb") as model_file:
            content = model_file.read()
        return StreamingResponse(io.BytesIO(content), media_type="application/octet-stream", headers={"Content-Disposition": f"attachment; filename={model_filename.split('/')[-1]}"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/")
async def predict(model_file: UploadFile = File(...), data_file: UploadFile = File(...)):
    try:
        # Leer el archivo CSV de datos
        df = pd.read_csv(io.BytesIO(await data_file.read()))
        
        # Leer el archivo del modelo
        model_file_content = await model_file.read()
        
        # Realizar la predicción
        predictions_csv = predict_model(df, model_file_content)
        
        # Enviar el archivo CSV de predicciones como respuesta
        return StreamingResponse(io.BytesIO(predictions_csv.encode()), media_type="text/csv", headers={"Content-Disposition": "attachment; filename=predictions.csv"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
