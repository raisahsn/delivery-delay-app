# predict.py -- Helper untuk memprediksi keterlambatan pengiriman
# menggunakan artefak model yang tersimpan di folder ini.
#
# Cara pakai:
#     python predict.py
#
# Atau dari script/notebook lain:
#     import joblib, pandas as pd
#     pipeline = joblib.load('model.pkl')
#     new_data = pd.DataFrame([{ ...fitur... }])  # lihat sample_input.json
#     proba = pipeline.predict_proba(new_data)[:, 1]
#     pred = pipeline.predict(new_data)

import json
import joblib
import pandas as pd
from pathlib import Path

HERE = Path(__file__).parent


def load_model():
    return joblib.load(HERE / 'model.pkl')


def load_config():
    with open(HERE / 'config.json', 'r', encoding='utf-8') as f:
        return json.load(f)


def predict(data, pipeline=None):
    # data: dict fitur untuk 1 pengiriman (lihat sample_input.json untuk skema)
    if pipeline is None:
        pipeline = load_model()
    X_new = pd.DataFrame([data])
    proba = pipeline.predict_proba(X_new)[:, 1][0]
    pred = int(pipeline.predict(X_new)[0])
    return {
        'delayed_prediction': 'Delayed' if pred == 1 else 'On-time',
        'delay_probability': float(proba),
    }


if __name__ == '__main__':
    pipeline = load_model()
    with open(HERE / 'sample_input.json', 'r', encoding='utf-8') as f:
        sample = json.load(f)
    result = predict(sample, pipeline)
    print('Contoh input:', sample)
    print('Hasil prediksi:', result)
