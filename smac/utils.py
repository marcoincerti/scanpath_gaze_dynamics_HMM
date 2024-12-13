import pandas as pd
import logging

# From .xls
def read_fixation_data(file_path):
    """Reads fixation data from an Excel file."""
    try:
        data = pd.read_excel(file_path)
        subjects = data['SubjectID'].unique()
        logging.info(f"Data read successfully from {file_path}. Found {len(subjects)} subjects.")
        return data, subjects
    except Exception as e:
        logging.error(f"Error reading data from {file_path}: {e}")
        return None, None
    
# From txt
def read_and_convert_txt_to_dataframe(file_path):
    columns_mapping = {
        "RECORDING_SESSION_LABEL": "SubjectID",
        "TRIAL_LABEL": "TrialID",
        "CURRENT_FIX_X": "FixX",
        "CURRENT_FIX_Y": "FixY"
    }
    
    with open(file_path, 'r') as file:
        header = file.readline().strip().split("\t")
        new_columns = [columns_mapping.get(col, col) for col in header]
        data = [line.strip().split("\t") for line in file.readlines()]
    
    df = pd.DataFrame(data, columns=new_columns)
    df['SubjectID'] = df['SubjectID'].str.extract(r'Look(\d+)').astype(int)
    df['TrialID'] = df['TrialID'].str.extract(r'Trial: (\d+)').astype(int)
    df['FixX'] = df['FixX'].str.replace(",", ".").astype(float)
    df['FixY'] = df['FixY'].str.replace(",", ".").astype(float)
    
    df = df.drop(columns=['CURRENT_FIX_START', 'CURRENT_FIX_END'])
    subjects = df['SubjectID'].unique()
    
    return df, subjects
