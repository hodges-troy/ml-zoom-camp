import logging

import numpy as np
import pandas as pd



# For fun, set up logger to print out results
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    # Question 1: Pandas version
    logger.info(f"Pandas version: {pd.__version__}")

    # Question 2: Number of records
    eff = pd.read_csv("week_one/car_fuel_efficiency.csv")
    logger.info(f"No. of records: {eff.shape[0]}")

    # Question 3: Number of fuel types
    fuel_type_count = len(eff.fuel_type.unique())
    logger.info(f"No. of fuel types: {fuel_type_count}")

    # Question 4: Number of columns with missing values
    nul_col_count = sum(eff.isna().sum() > 0)
    logger.info(f"No. of cols with nulls: {nul_col_count}")

    # Question 5: Max fuel efficiency of cars from Asia
    max_eff_asia = eff[eff["origin"] == "Asia"]["fuel_efficiency_mpg"].max()
    logger.info(f"Max efficiency Asia origin cars: {max_eff_asia}")

    # Question 6: Median Horsepower

    # Find the median value of horsepower column in the dataset.
    # Next, calculate the most frequent value of the same horsepower column.
    # Use fillna method to fill the missing values in horsepower column with the most frequent value from the previous step.
    # Now, calculate the median value of horsepower once again.

    hp_med = eff["horsepower"].median()
    hp_mode = eff["horsepower"].mode()[0]
    eff["horsepower_fill"] = eff["horsepower"].fillna(hp_mode)
    hp_fill_med = eff["horsepower_fill"].median()
    logger.info(f"Old HP Median: {hp_med}")
    logger.info(f"New HP Median: {hp_fill_med}")

    # Question 7:

    # Select all the cars from Asia
    # Select only columns vehicle_weight and model_year
    # Select the first 7 values
    # Get the underlying NumPy array. Let's call it X.
    # Compute matrix-matrix multiplication between the transpose of X and X. To get the transpose, use X.T. Let's call the result XTX.
    # Invert XTX.
    # Create an array y with values [1100, 1300, 800, 900, 1000, 1100, 1200].
    # Multiply the inverse of XTX with the transpose of X, and then multiply the result by y. Call the result w.
    # What's the sum of all the elements of the result?
    
    eff_asia = eff[eff["origin"] == "Asia"][["vehicle_weight", "model_year"]].reset_index(drop=True)
    X = eff_asia.iloc[:7,].values
    XTX = X.T @ X
    XTX_inverse = np.linalg.inv(XTX)
    logging.info(f"Check inverse math: {XTX @ XTX_inverse}")
    y = np.array([1100, 1300, 800, 900, 1000, 1100, 1200])
    w = XTX_inverse @ X.T @ y
    logger.info(f"Sum of weights: {w.sum()}")