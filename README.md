# Churn Prediction Project

## Project Overview

This project is a final assignment for my data science course, aiming to create a model that predicts customer churn. The objective is to explore the data, engineer features, train different models, and evaluate them to determine the best-performing model for predicting churn.

## Dataset

The dataset includes information on customer behavior and attributes, which can be used to analyze and predict customer churn. The data was provided in a CSV file, and a separate XLSX file contains additional metadata describing the dataset fields.

* **Data files**:
  * `data/raw/Modelo_Clasificacion_Dataset.csv`: Original dataset
  * `data/raw/Modelo_Clasificacion_Diccionario_de_Datos.xlsx`: Description of dataset fields and data types
* **Additional file**:
  * `docs/TRABAJO_CLASIFICACION.pdf`: PDF containing the main task of the assignment

## Project Structure

Here’s an overview of the folder structure in this repository:

```
uba-assessment-churn-prediction/
├── data/
│   ├── raw/                   # Original data files (e.g., CSV and XLSX files)
│   │   ├── Modelo_Clasificacion_Dataset.csv
│   │   └── Modelo_Clasificacion_Diccionario_de_Datos.xlsx
│   └── processed/             # Cleaned and preprocessed data files
│       └── churn_prediction_dataset.parquet
│
├── notebooks/                 # Jupyter notebooks for data exploration, EDA, etc.
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_refinement.ipynb
│   └── 04_final_model.ipynb
│
├── src/                       # Source code for the project
│   └── preprocessing.py       # Function used for data processing
│
├── models/                    # Serialized models (e.g., saved model files)
│   └── final_model.pkl        # Final model in pickle
│
├── reports/                   # Folder for reports
│   └── figures/               # Figures and images to include in the report
│
├── docs/                      # Documentation files
│   └── assessment_task.pdf    # Task description for the final assignment
│
├── README.md                  # Project overview and setup instructions
├── requirements.txt           # List of dependencies
└── .gitignore                 # Files and folders to ignore in Git
```

## Setup Instructions

1. **Clone the repository**:
```
git clone https://github.com/juanignaciomonge/uba-assessment-churn-prediction.git
cd uba-assessment-churn-prediction
```
2. **Install dependencies**:
```
pip install -r requirements.txt
```
3. **Run Notebooks**:
    * Open Jupyter Notebook and run the notebooks in the `notebooks/` folder for exploratory data analysis, feature engineering, model refinement and final model evaluation.
 

## Project Workflow

## In order to run the trained model over a new set of data, run ../scripts/predict_new_data.py with the updated path to the new data or follow these steps:

1. ### Load the new dataset.

        import pickle

        # Load the trained model from the file
        model_path = "../models/final_model.pkl"
        with open(model_path, "rb") as file:
            loaded_model = pickle.load(file)

        print("Model loaded successfully")
        
2. ### Preprocess the new data to match the format of the training data (e.g., same features, transformations, etc.).

        import sys
        sys.path.append("../src")  # Adjust path if necessary
        from preprocessing import preprocess_data

        new_data = pd.read_csv(".../data.csv", index_col=[0])

        new_data = preprocess_data(new_data)
        
3. ### Use the trained model to make predictions on the new data.
        
        X_new = new_data.drop(columns=["clase_binaria"])
        y_true = new_data["clase_binaria"]

        y_new_pred = loaded_model.predict(X_new)
        print("Predictions for the new data:", y_new_pred)

4. ### Evaluate the New Predictions.

        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

        # Calculate the accuracy of the model
        accuracy = accuracy_score(y_true, y_new_pred)
        print(f"Accuracy: {accuracy:.4f}")

        # Precision, Recall, F1 Score
        precision = precision_score(y_true, y_new_pred, pos_label=1)
        recall = recall_score(y_true, y_new_pred, pos_label=1)
        f1 = f1_score(y_true, y_new_pred, pos_label=1)

        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")

        # ROC-AUC Score
        roc_auc = roc_auc_score(y_true, y_new_pred)
        print(f"ROC AUC: {roc_auc:.4f}")


## Results

The final report with model evaluation results can be found in the `reports/` folder. Key findings and model metrics will be summarized here after analysis is complete.

## Dependencies

List of dependencies is provided in `requirements.txt`. To install, run:
```
pip install -r requirements.txt
```

## License

This project is for educational purposes as part of a data science course.

