
from pathlib import Path


TMP_FOLDER = Path(__file__).resolve().parent.parent / 'tmp'
if not TMP_FOLDER.exists():
    TMP_FOLDER.mkdir()

# Define this centrally here so there is only one place it needs to be changed
DATA_FOLDER = Path(__file__).resolve().parent.parent / 'data'
DATA_PATH = DATA_FOLDER / 'embeddings.csv.gz'
META_DATA_PATH = DATA_FOLDER / 'df_wo_vec.pkl'

MODEL_NAME = 'hsmw_index_20230416T171916'
