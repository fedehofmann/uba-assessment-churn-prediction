import pandas as pd
from datetime import datetime


# Define your preprocessing steps
def preprocess_data(new_data: pd.DataFrame):

    # Drop the first column 'X' that looks like an unnecessary index and the customer ID since it doesn't provide any useful information for our future models
    new_data = new_data.drop(["X", "numero_de_cliente"], axis=1)

    # Rename columns: lowercase and replace spaces with underscores
    new_data.columns = new_data.columns.str.lower().str.replace(" ", "_")

    # Convert target variable into numeric format that is more suitable for most machine learning models
    new_data["clase_binaria"] = new_data["clase_binaria"].map(
        {"BAJA": 1, "CONTINUA": 0}
    )

    # Change Visa_fechaalta data type to datetime.
    new_data["visa_fechaalta"] = pd.to_numeric(
        new_data["visa_fechaalta"], errors="coerce"
    )
    new_data["visa_fechaalta"] = pd.to_datetime(
        new_data["visa_fechaalta"], format="%Y%m%d", errors="coerce"
    )

    # Change categorical fields into category data type instead of str since it's more efficient
    metadata = {"tcuentas": "category", "visa_cuenta_estado": "category"}

    for column, dtype in metadata.items():
        if column in new_data.columns:
            try:
                new_data[column] = new_data[column].astype(dtype)
            except ValueError:
                print(f"Error converting column {column} to {dtype}")

    # Transform visa_fechaalta into days of tenure
    new_data["visa_tenure_days"] = (datetime.now() - new_data["visa_fechaalta"]).dt.days
    new_data = new_data.drop(columns=["visa_fechaalta"])

    # Convert null categorical fields into 'Unknown'
    new_data["visa_cuenta_estado"] = (
        new_data["visa_cuenta_estado"].cat.add_categories("unknown").fillna("unknown")
    )

    # Convert nulls to 0
    columns_to_fill = [
        "visa_marca_atraso",
        "visa_mfinanciacion_limite",
        "visa_msaldototal",
        "visa_msaldopesos",
        "visa_msaldodolares",
        "visa_mconsumospesos",
        "visa_mconsumosdolares",
        "visa_mlimitecompra",
        "visa_mpagado",
        "visa_mpagospesos",
        "visa_mpagosdolares",
        "visa_mconsumototal",
        "visa_cconsumos",
        "visa_mpagominimo",
        "visa_tenure_days",
    ]

    new_data[columns_to_fill] = new_data[columns_to_fill].fillna(0)

    # Apply one-hot encoding to the categorical fields ("tcuentas" and "visa_cuenta_estado")
    new_data_processed = pd.get_dummies(new_data, drop_first=True)

    return new_data_processed
