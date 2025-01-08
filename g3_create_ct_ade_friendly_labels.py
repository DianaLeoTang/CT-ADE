import pandas as pd
import os
from src.meddra_graph import MedDRA, Node

def rename_columns_by_prefix(input_file, output_file, string_start, rename_dict):
    """
    Renames columns in a CSV file that start with a specific prefix and uses a dictionary to map suffixes.

    Parameters:
        input_file (str): Path to the input CSV file.
        output_file (str): Path to save the updated CSV file.
        string_start (str): The prefix to identify columns that need renaming.
        rename_dict (dict): Mapping of suffixes (after the prefix) to their new names.
    """
    try:
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # Load the CSV file into a DataFrame
        df = pd.read_csv(input_file)

        # Identify columns to rename
        columns_to_rename = [col for col in df.columns if col.startswith(string_start)]

        # Create a new mapping for renaming based on the suffix and the dictionary
        new_column_names = {
            col: string_start + rename_dict[col[len(string_start):]]
            for col in columns_to_rename
            if col[len(string_start):] in rename_dict
        }

        # Rename columns in the DataFrame
        df.rename(columns=new_column_names, inplace=True)

        # Save the updated DataFrame to a new CSV file
        df.to_csv(output_file, index=False)
        print(f"Columns renamed successfully. Updated file saved to '{output_file}'.")

    except FileNotFoundError:
        print(f"Error: The file '{input_file}' does not exist.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    meddra = MedDRA()
    meddra.load_data("./data/MedDRA_25_0_English/MedAscii")
    
    level = "SOC"
    rename_dict = {key[1]: value.term for key, value in meddra.nodes.items() if key[0] == level}
    rename_columns_by_prefix("./data/ct_ade/soc/train.csv", "./data/ct_ade/soc_friendly/train.csv", "label_", rename_dict)
    rename_columns_by_prefix("./data/ct_ade/soc/val.csv", "./data/ct_ade/soc_friendly/val.csv", "label_", rename_dict)
    rename_columns_by_prefix("./data/ct_ade/soc/test.csv", "./data/ct_ade/soc_friendly/test.csv", "label_", rename_dict)
    rename_columns_by_prefix("./data/ct_ade/soc/train_frequencies.csv", "./data/ct_ade/soc_friendly/train_frequencies.csv", "frequency_", rename_dict)
    rename_columns_by_prefix("./data/ct_ade/soc/val_frequencies.csv", "./data/ct_ade/soc_friendly/val_frequencies.csv", "frequency_", rename_dict)
    rename_columns_by_prefix("./data/ct_ade/soc/test_frequencies.csv", "./data/ct_ade/soc_friendly/test_frequencies.csv", "frequency_", rename_dict)
    
    level = "HLGT"
    rename_dict = {key[1]: value.term for key, value in meddra.nodes.items() if key[0] == level}
    rename_columns_by_prefix("./data/ct_ade/hlgt/train.csv", "./data/ct_ade/hlgt_friendly/train.csv", "label_", rename_dict)
    rename_columns_by_prefix("./data/ct_ade/hlgt/val.csv", "./data/ct_ade/hlgt_friendly/val.csv", "label_", rename_dict)
    rename_columns_by_prefix("./data/ct_ade/hlgt/test.csv", "./data/ct_ade/hlgt_friendly/test.csv", "label_", rename_dict)
    rename_columns_by_prefix("./data/ct_ade/hlgt/train_frequencies.csv", "./data/ct_ade/hlgt_friendly/train_frequencies.csv", "frequency_", rename_dict)
    rename_columns_by_prefix("./data/ct_ade/hlgt/val_frequencies.csv", "./data/ct_ade/hlgt_friendly/val_frequencies.csv", "frequency_", rename_dict)
    rename_columns_by_prefix("./data/ct_ade/hlgt/test_frequencies.csv", "./data/ct_ade/hlgt_friendly/test_frequencies.csv", "frequency_", rename_dict)
    
    level = "HLT"
    rename_dict = {key[1]: value.term for key, value in meddra.nodes.items() if key[0] == level}
    rename_columns_by_prefix("./data/ct_ade/hlt/train.csv", "./data/ct_ade/hlt_friendly/train.csv", "label_", rename_dict)
    rename_columns_by_prefix("./data/ct_ade/hlt/val.csv", "./data/ct_ade/hlt_friendly/val.csv", "label_", rename_dict)
    rename_columns_by_prefix("./data/ct_ade/hlt/test.csv", "./data/ct_ade/hlt_friendly/test.csv", "label_", rename_dict)
    rename_columns_by_prefix("./data/ct_ade/hlt/train_frequencies.csv", "./data/ct_ade/hlt_friendly/train_frequencies.csv", "frequency_", rename_dict)
    rename_columns_by_prefix("./data/ct_ade/hlt/val_frequencies.csv", "./data/ct_ade/hlt_friendly/val_frequencies.csv", "frequency_", rename_dict)
    rename_columns_by_prefix("./data/ct_ade/hlt/test_frequencies.csv", "./data/ct_ade/hlt_friendly/test_frequencies.csv", "frequency_", rename_dict)
    
    level = "PT"
    rename_dict = {key[1]: value.term for key, value in meddra.nodes.items() if key[0] == level}
    rename_columns_by_prefix("./data/ct_ade/pt/train.csv", "./data/ct_ade/pt_friendly/train.csv", "label_", rename_dict)
    rename_columns_by_prefix("./data/ct_ade/pt/val.csv", "./data/ct_ade/pt_friendly/val.csv", "label_", rename_dict)
    rename_columns_by_prefix("./data/ct_ade/pt/test.csv", "./data/ct_ade/pt_friendly/test.csv", "label_", rename_dict)
    rename_columns_by_prefix("./data/ct_ade/pt/train_frequencies.csv", "./data/ct_ade/pt_friendly/train_frequencies.csv", "frequency_", rename_dict)
    rename_columns_by_prefix("./data/ct_ade/pt/val_frequencies.csv", "./data/ct_ade/pt_friendly/val_frequencies.csv", "frequency_", rename_dict)
    rename_columns_by_prefix("./data/ct_ade/pt/test_frequencies.csv", "./data/ct_ade/pt_friendly/test_frequencies.csv", "frequency_", rename_dict)