import os
from datetime import datetime as dt


DATE_MARK:str = dt.now().strftime('%y%m%d')
TIME_MARK:str = dt.now().strftime('%y%m%d-%H%M%S')
WORKERS:int = None
WORKERS_PREFIX:str = ""
WORK_PATH:str = os.getcwd()
INPUT_PATH:str = f"{WORK_PATH}/input"
OUTPUT_PATH:str = f"{WORK_PATH}/output"
PROJ_PER_PIXEL:int = 9
