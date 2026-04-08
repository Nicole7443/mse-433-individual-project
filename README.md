# Public Safety Resource Allocation: Brooklyn Crime Forecasting & Optimization
**MSE 433 Individual Project**: An end-to-end decision support tool for stochastic crime forecasting and prescriptive patrol scheduling.

## Project Overview
This project aims to optimize the allocation of public safety resources in Brooklyn, NY. It combines a machine learning pipeline to forecast crime incidents with a **Gurobi-based Linear Programming (LP)** model to determine optimal patrol schedules across different precincts and time shifts.

## Repository Structure & Workflow

The project is designed to be executed in the following sequence:

1.  **`Data_Cleaning.ipynb`**: Processes raw NYPD Complaint Data. It handles missing values, renames columns for consistency, and filters for Brooklyn-specific incidents.
2.  **`EDA.ipynb`**: Visualizes crime density, temporal trends (hour/day/month), and identifies high-risk precincts.
3.  **`ML.ipynb`**: The forecasting engine. Uses feature engineering (including holiday tracking) and LightGBM to predict incident volume. Outputs `crime_predictions.csv`.
4.  **`Optimization.ipynb`**: An optimization notebook that hosts the optimization model and its analysis
5.  **`app.py`**: The final police scheduling dashboard. It provides an interactive interface to run the optimization model in real-time.

## Getting Started

### Prerequisites
* **Python 3.9+**
* **Gurobi Optimizer**: A valid license is required
* **Dependencies**: Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

### How to Run
1.  **Generate Predictions**: Ensure `crime_predictions.csv` is present or run `ML.ipynb` to regenerate it.
2.  **Launch the Dashboard**:
    ```bash
    streamlit run app.py
    ```

## Model Technical Details

### Machine Learning (Forecasting)
The model treats crime forecasting as a regression problem. Features include:
* Temporal: Hour of day, Day of week, Month, and Statutory Holidays.
* Spatial: Precinct IDs (categorical) and historical crime density.
* **Target**: Predicted incident volume per precinct/shift.

### Optimization (Prescriptive Analytics)
The Linear Programming model is formulated as follows:
* **Objective**: Minimize the "Total Weighted Residual Risk" across all Brooklyn precincts.
* **Decision Variables**: Number of officers assigned to each precinct per shift.
* **Constraints**: 
    * Total budget/officer headcount limits.
    * Minimum (floor) and maximum (cap) staffing requirements per precinct.
    * Priority weighting for "High Risk" locations identified by the ML model.

## Results
The Gurobi-optimized deployment strategy consistently demonstrates a significant reduction in unmitigated risk compared to uniform staffing, as validated in the `Optimization.ipynb` results.

---
**Author**: Nicole Thapa
**Course**: MSE 433 - Advanced Topics in Management Engineering  
**Institution**: University of Waterloo
