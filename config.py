"""
This module contains the configuration settings
"""

DAILY_TRAIN = "/Users/philipp/000_work/060_SciScry/school/UNIC/MSc - Data Science/\
COMP-592DL_Project_in_Data_Science/M4-methods/Dataset/Train/Daily-train.csv"

DAILY_TEST = "/Users/philipp/000_work/060_SciScry/school/UNIC/MSc - Data Science/\
COMP-592DL_Project_in_Data_Science/M4-methods/Dataset/Test/Daily-test.csv"

SAMPLING = 0.01

LAGS = [1,2,3,4,5,6,7]

STEPS_AHEAD = 7

MAX_CLUSTER = 20

MODELS_PATH = 'models'

SCORES_FOLDER_PATH = 'scores'

SIL_SCORES_FILE_PATH = 'silhouette_scores.txt'

INERTIA_FILE_PATH = 'inertias.txt'