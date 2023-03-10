from pydantic import BaseModel


class Prediction(BaseModel):
    Mw: float
    Rrup: float
    VS30: float
    fault_type: str
    period: float
    data_type: str
