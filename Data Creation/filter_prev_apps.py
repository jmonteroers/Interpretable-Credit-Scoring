"""
This script takes the original previous application tables 
and only keeps the applications whose contract status equal
'Approved', saving them to a new compressed csv file
"""

import pandas as pd
from utils import PARENT_DIR

prev_apps = pd.read_csv(PARENT_DIR / "previous_application.csv.zip", compression="zip")
prev_apps = prev_apps.loc[prev_apps.NAME_CONTRACT_STATUS == "Approved", ]
prev_apps.to_csv(PARENT_DIR / "processed" / "approved_previous_application.csv.zip")