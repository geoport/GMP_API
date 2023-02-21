import joblib
import numpy as np
import os
from helper import get_pgx_pred, get_sa_pred


def load_model(file_name):
    return joblib.load(os.path.join("models", f"{file_name}.pkl"))


pipelines = {
    "PGA": load_model("pga_pipeline"),
    "PGV": load_model("pgv_pipeline"),
    "PGD": load_model("pgd_pipeline"),
    "SA_process": load_model("sa_pipeline"),
}


def predict_response_spectrum(process_pipeline, Mw, Rrup, VS30, fault_type):
    periods = np.arange(0.04, 10.04, 0.04)
    spectral_accelerations = [
        get_sa_pred(process_pipeline, Mw, Rrup, VS30, fault_type, p) for p in periods
    ]

    return {
        "periods": periods.tolist(),
        "spectral_accelerations": spectral_accelerations,
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
        output = get_sa_pred(process_pipeline, Mw, Rrup, VS30, fault_type, period)
        return {"spectral_acceleration": output}
    elif data_type == "RS":
        process_pipeline = pipelines["SA_process"]
        output = predict_response_spectrum(process_pipeline, Mw, Rrup, VS30, fault_type)
        return output
