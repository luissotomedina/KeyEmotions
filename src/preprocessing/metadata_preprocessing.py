import os
import json
import pandas as pd

def load_metadata(metadata_path):
    """
    Load metadata from csv file

    Parameters:
        metadata_path: str, path to the metadata file
    """

    try: 
        return pd.read_csv(metadata_path)
    except Exception as e:
        print(f"Error loading {metadata_path}: {e}")
        return None
    
def process_metadata_vgmidi(df):
    """
    Process metadata from VGMIDI dataset

    Parameters:
        df: pd.DataFrame, metadata
    
    Returns:
        df: pd.DataFrame, metadata with Q label
    """
    valence_arousal_mapping = {
        (1, 1): 1, 
        (-1, 1): 2,
        (-1, -1): 3,
        (1, -1): 4
    }

    df['label'] = df[['valence', 'arousal']].apply(lambda x: valence_arousal_mapping[(x['valence'], x['arousal'])], axis=1)

    df['name'] = df['series'] + '_' + df['console'] + '_' + df['game'] + '_' + df['piece'] # + '.mid'

    df = df[['name', 'label']]

    return df

def process_metadata_emopia(df):
    """
    Process metadata from EMOPIA dataset

    Parameters:
        df: pd.DataFrame, metadata

    Returns:
        df: pd.DataFrame, metadata with Q label
    """
    # df['name'] = df['name'] + '.mid'

    df['label'] = df['label'].apply(lambda x: 1 if x == 'Q1' else 2 if x == 'Q2' else 3 if x == 'Q3' else 4)

    df = df[['name', 'label', 'keyname', 'tempo']]

    return df

def process_metadata(df):
    """
    Process metadata from vgmidi and EMOPIA dataset
    
    Parameters:
        df: pd.DataFrame, metadata
        
    Returns:
        df: pd.DataFrame
    """
    if {'valence', 'arousal'}.issubset(df.columns):
        df = process_metadata_vgmidi(df)
    else:
        df = process_metadata_emopia(df)

    required_columns = ['name', 'label', 'keyname', 'tempo']
    for col in required_columns:
        if col not in df.columns:
            df.loc[:, col] = ''
    
    return df


    
