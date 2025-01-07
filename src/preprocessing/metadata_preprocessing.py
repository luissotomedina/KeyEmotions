import os
import json
import pandas as pd

def load_metadata(metadata_path):
    """
    Load metadata from csv file

    Input:
        metadata_path: str, path to the metadata file
    
    Output: 
        None
    """

    try: 
        return pd.read_csv(metadata_path)
    except Exception as e:
        print(f"Error loading {metadata_path}: {e}")
        return None
    
def process_vgmidi(df):
    """
    Process metadata from VGMIDI dataset

    Input:
        df: pd.DataFrame, metadata
    
    Output:
        df: pd.DataFrame, metadata with Q label
    """
    valence_arousal_mapping = {
        (1, 1): "Q1", 
        (-1, 1): "Q2",
        (-1, -1): "Q3",
        (1, -1): "Q4"
    }

    df['label'] = df[['valence', 'arousal']].apply(lambda x: valence_arousal_mapping[(x['valence'], x['arousal'])], axis=1)

    df['name'] = df['midi'].apply(lambda x: x.split('/')[-1])

    # Extract name and Q_label
    df = df[['name', 'label']]

    return df

def process_emopia(df):
    """
    Process metadata from EMOPIA dataset

    Input:
        df: pd.DataFrame, metadata

    Output:
        df: pd.DataFrame, metadata with Q label
    """
    df['name'] = df['name'] + '.mid'

    df = df[['name', 'label', 'keyname', 'tempo']]

    return df

def process_metadata(df):
    """
    Process metadata from vgmidi and EMOPIA dataset
    
    Input:
        df: pd.DataFrame, metadata
        
    Output:
        None
    """
    if {'valence', 'arousal'}.issubset(df.columns):
        df = process_vgmidi(df)
    else:
        df = process_emopia(df)

    required_columns = ['name', 'label', 'keyname', 'tempo']
    for col in required_columns:
        if col not in df.columns:
            df[col] = ''
    
    return df

def save_metadata_to_json(metadata_list, output_path):
    """
    Save combined metadata to a JSON file

    Input:
        metadata_list: list of pd.DataFrame, list of metadata
        output_path: str, path to save the metadata
    """
    combined_metadata = pd.concat(metadata_list, ignore_index=True)
    metadata_dict = combined_metadata.to_dict(orient='records')
    with open(output_path, 'w') as f:
        json.dump(metadata_dict, f, indent=4)
        print(f"Metadata saved to {output_path}")

if __name__=='__main__':
    raw_data_path = './data/raw'
    processed_data_path = './data/processed'
    metadata_list = []

    files = os.listdir(raw_data_path)
    for file in files:
        if file.endswith('.csv'):
            metadata = load_metadata(os.path.join('./data/raw', file))
            if metadata is not None:
                metadata = process_metadata(metadata)
                metadata_list.append(metadata)

    save_metadata_to_json(metadata_list, os.path.join(processed_data_path, 'metadata.json'))




    
