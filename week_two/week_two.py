import logging

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    # Question 1: There's one column with missing values. What is it?
    eff = pd.read_csv("week_one/car_fuel_efficiency.csv")
    na_cols = eff.columns[eff.isna().sum() > 0]
    logger.info(f"Cols with nulls: {na_cols}")

    # Question 2: Median horsepower
    hp_med = eff['horsepower'].median()

    logger.info(f"Median horsepower: {hp_med}")
