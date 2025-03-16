# Bangalore House Price Prediction

## Introduction
The Bangalore House Price Prediction project develops a machine learning model to estimate house prices in Bangalore based on key factors like location, size, and amenities. Using historical data (`Bengaluru_House_Data.csv`), the model helps buyers and sellers make informed decisions.

## Project Overview
This project follows a structured approach:
1. **Data Preprocessing**: Cleaned missing values, converted size to `bhk`, and standardized `total_sqft`.
2. **Feature Engineering**: Selected key features (`location`, `total_sqft`, `bath`, `bhk`), applied one-hot encoding, and scaled numerical data.
3. **Model Selection**: Ridge regression (`alpha=1`) was chosen for its balance between accuracy and generalization, achieving R² = 0.914.
4. **Deployment**: The trained model was saved (`RidgeModel.pkl`) and prepared for API deployment via Flask.

## Dataset
- **File**: `Bengaluru_House_Data.csv`
- **Rows**: 13,320
- **Columns**: `area_type`, `availability`, `location`, `size`, `society`, `total_sqft`, `bath`, `balcony`, `price`


# Animation of UI

![HomepageUI](./House Price Predictor.mp4)


## Steps in the Project

### 1. Data Preprocessing and Feature Engineering
- **Loading the Dataset**: Used `pandas` to load the dataset and analyzed structure, missing values, and duplicates.
- **Handling Missing Values**: 
  - `location` (1 missing), `size` (16 missing), `society` (5,502 missing), `bath` (73 missing), `balcony` (609 missing)
  - Dropped less informative columns (`society`, `availability`, `balcony`)
- **Feature Engineering**:
  - Converted `size` into `bhk`
  - Standardized `total_sqft`
  - One-hot encoded `location`

### 2. Model Selection and Optimization
- Considered multiple models (`LinearRegression`, `Lasso`, `Ridge`, `RandomForestRegressor`, `XGBRegressor`)
- Selected **Ridge Regression** (`alpha=1`) for its balance of accuracy and regularization
- **Performance Metrics**:
  - MAE = 15.87
  - RMSE = 24.53
  - R² = 0.914

### 3. Deployment Strategy
- **Model Serialization**: Saved `RidgeModel.pkl` using `pickle`
- **API Deployment**:
  - Uses Flask to create an API endpoint
  - Accepts JSON input (`location`, `total_sqft`, `bath`, `bhk`)
  - Returns predicted price in JSON format

## API Usage Guide
### **Endpoint**
```
POST /predict
```

### **Request Format**
```json
{
  "location": "string",
  "total_sqft": float,
  "bath": int,
  "bhk": int
}
```

### **Response Format**
```json
{
  "predicted_price": float
}
```

## Requirements
To run the project, install the required dependencies:
```bash
pip install -r requirements.txt
```

## Running the Project
### **1. Train the Model**
Run the Jupyter notebook to preprocess the data and train the model:
```bash
jupyter notebook House_Price_Predictor.ipynb
```
### **2. Usage:**

1. conda create -p std python=3.8 -y
2. conda activate std/
3. pip install -r requirements.txt
4. Execute main.py
5. Access http://127.0.0.1:5001/


## Conclusion
The project successfully preprocesses the Bangalore housing dataset, engineers relevant features, and trains a Ridge regression model with strong performance (R² = 0.914). The model is deployment-ready and can be used as an API for real-world price predictions.

 ## Deployment on Render
 https://house-price-predictor-9frq.onrender.com
