import joblib
import numpy as np
import os
from helper import get_pgx_pred, get_sa_pred
import tensorflow as tf


def load_model(file_name):
    return joblib.load(os.path.join("models", f"{file_name}.pkl"))


pipelines = {
    "PGA": load_model("pga_pipeline"),
    "PGV": load_model("pgv_pipeline"),
    "PGD": load_model("pgd_pipeline"),
    "SA_process": load_model("sa_pipeline"),
    "SA": tf.keras.models.load_model("models/SA.h5"),
}


def predict_response_spectrum(Mw, Rrup, VS30, fault_type):
    periods = np.arange(0.02, 10.02, 0.02)
    spectral_accelerations = []
    process_pipeline = pipelines["SA_process"]
    tf_model = pipelines["SA"]
    spectral_accelerations = get_sa_pred(
        process_pipeline, tf_model, Mw, Rrup, VS30, fault_type, periods
    )

    return {
        "periods": periods.tolist(),
        "spectral_accelerations": spectral_accelerations.tolist(),
    }


def predict_gmpe(prediction_data):
    Mw = prediction_data.Mw
    Rrup = prediction_data.Rrup
    VS30 = prediction_data.VS30
    fault_type = prediction_data.fault_type
    data_type = prediction_data.data_type
    period = prediction_data.period

    if data_type in ["PGA", "PGV", "PGD"]:
        pipeline = pipelines[data_type]
        output = get_pgx_pred(pipeline, Mw, Rrup, VS30, fault_type, data_type)
        return {data_type: output}
    elif data_type == "SA":
        process_pipeline = pipelines["SA_process"]
        tf_model = pipelines["SA"]
        output = get_sa_pred(
            process_pipeline, tf_model, Mw, Rrup, VS30, fault_type, [period]
        )[-1]
        return {"spectral_acceleration": output}
    elif data_type == "RS":
        output = predict_response_spectrum(Mw, Rrup, VS30, fault_type)
        return output
