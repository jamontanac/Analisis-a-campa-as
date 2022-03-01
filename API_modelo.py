import uvicorn
from fastapi import FastAPI
from typing import Optional
from typing import List
from pydantic import BaseModel
import pickle
import numpy as np
import pandas as pd

class RandomForest_modelBase(BaseModel):

    """
    Clase utilizada para representar lo datos que vamos a usar en nuestro modelo
    ...
    Attributes
    ----------
    Ciudad : str, representa la ciudad del cliente
    Patrimonio : float, representa el patrimonio del cliente
    Ingresos_Mensuales : float, representa los ingresos del cliente mensuales
    No_hijos : int, representa el numero de hijos del cliente
    monto_credito : float, representa el monto del credito solicitado por el cliente
    tasa: float, representa la tasa de interes del credito
    saldo_capital : float, representa el saldo capital del cliente
    saldo_Ahorro : float, representa el saldo de ahorro del cliente
    Antiguedad_en_meses : int, representa la antiguedad del cliente en meses
    Max_dias_mora : int, representa el numero de dias de mora del cliente
    plazo_dias : int, representa el plazo del credito en dias
    Edad: int, representa la edad del cliente
    """
    Ciudad:str
    Patrimonio:float
    Ingresos_Mensuales:float
    No_hijos:int
    monto_credito:float
    tasa:float
    saldo_capital:float
    saldo_Ahorro:float
    Antiguedad_en_meses:int
    Max_dias_mora:int
    plazo_dias:int
    Edad:int

filename = "Tree_model.pk"
with open(filename,"rb") as f:
    loaded_model = pickle.load(f)


app = FastAPI()




@app.get("/api")
def read_main():
    return {
        "routes": [
            {"method": "GET", "path": "/api/status", "summary": "Estado de la API"},
            {"method": "POST", "path": "/api/predict", "predict": "Obtener una prediccion"}
        ]
    }
@app.get("/api/status")
def get_status():
    return {"status": "ok"}

@app.post("/api/predict")
def predict_taxonomy(campaign:RandomForest_modelBase):
    data = campaign.dict()
    new_data=dict([(key,data[key]) for key in sorted(data)])
    df=pd.DataFrame(data=np.array(list(new_data.values())).reshape(1, -1),columns=list(new_data.keys()))
    prediction=loaded_model.predict(df)
    classes_names=loaded_model.classes_.T
    probability=zip(
        [str(classes) for classes in classes_names],loaded_model.predict_proba(df)[0])
    return {"prediction":int(prediction[0]),"probability":list(probability)}

if __name__ == "__main__":

    # Run the app with uvicorn ASGI server asyncio frameworks. That basically responds to request on parallel and faster

    uvicorn.run("API_modelo:app", host="0.0.0.0", port=8000, reload=True)