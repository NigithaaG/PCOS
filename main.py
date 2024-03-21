import os
import uvicorn
import warnings
from fastapi import FastAPI,Query,Body
from fastapi.responses import RedirectResponse
from pydantic import BaseModel,Field
app=FastAPI()
warnings.filterwarnings("ignore")
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

xgb_model = joblib.load('xgb_model_top_10.pkl')
class Vals(BaseModel):
    follicle_no_r :float=Field(examples=[20])
    follicle_no_l :float=Field(examples=[19])
    hair_growth :str=Field(examples=['Y'])
    skin_darkening :str=Field(examples=['Y'])
    weight_gain :str=Field(examples=['Y'])
    cycle :str=Field(examples=['R'])
    lh :float=Field(examples=[2.31])
    fast_food :str=Field(examples=['Y'])
    fsh_lh :float=Field(examples=[2.41])
    cycle_length :float=Field(examples=[7])


@app.post("/predict")
async def predict(vals:Vals=Body(...)):
    input_data = pd.DataFrame({
        'Follicle No. (R)': [vals.follicle_no_r],
        'Follicle No. (L)': [vals.follicle_no_l],
        'hair growth(Y/N)': [vals.hair_growth],
        'Skin darkening (Y/N)': [vals.skin_darkening],
        'Weight gain(Y/N)': [vals.weight_gain],
        'Cycle(R/I)': [vals.cycle],
        'LH(mIU/mL)': [vals.lh],
        'Fast food (Y/N)': [vals.fast_food],
        'FSH/LH': [vals.fsh_lh],
        'Cycle length(days)': [vals.cycle_length]
    })
    label_encoder = LabelEncoder()
    for column in input_data.columns:
        if input_data[column].dtype == 'object':
            input_data[column] = label_encoder.fit_transform(input_data[column])

    prediction = xgb_model.predict(input_data)

    if prediction[0] == 1:
        return{"The prediction is PCOS (Y)"}
    else:
        return{"The prediction is not PCOS (N)"}




if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)