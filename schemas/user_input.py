from pydantic import BaseModel, Field, computed_field, field_validator
from typing import Literal, Annotated

# pydantic model to validate incoming data


class UserInput(BaseModel):
    Time: Annotated[
        float,
        Field(
            ...,
            description="Its the seconds elapsed between each transaction and the first transaction",
        ),
    ]
    V1: Annotated[float, Field(...)]
    V2: Annotated[float, Field(...)]
    V3: Annotated[float, Field(...)]
    V4: Annotated[float, Field(...)]
    V5: Annotated[float, Field(...)]
    V6: Annotated[float, Field(...)]
    V7: Annotated[float, Field(...)]
    V8: Annotated[float, Field(...)]
    V9: Annotated[float, Field(...)]
    V10: Annotated[float, Field(...)]
    V11: Annotated[float, Field(...)]
    V12: Annotated[float, Field(...)]
    V13: Annotated[float, Field(...)]
    V14: Annotated[float, Field(...)]
    V15: Annotated[float, Field(...)]
    V16: Annotated[float, Field(...)]
    V17: Annotated[float, Field(...)]
    V18: Annotated[float, Field(...)]
    V19: Annotated[float, Field(...)]
    V20: Annotated[float, Field(...)]
    V21: Annotated[float, Field(...)]
    V22: Annotated[float, Field(...)]
    V23: Annotated[float, Field(...)]
    V24: Annotated[float, Field(...)]
    V25: Annotated[float, Field(...)]
    V26: Annotated[float, Field(...)]
    V27: Annotated[float, Field(...)]
    V28: Annotated[float, Field(...)]
    Amount: Annotated[float, Field(..., description="It is the transaction Amount")]
