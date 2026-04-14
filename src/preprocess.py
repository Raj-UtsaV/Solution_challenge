import pickle
import joblib
import pandas as pd
from config.paths_config import *

class PREPROCESSING:
    """
    Orchestrates the data transformation pipeline for raw user input.
    
    This class handles the conversion of dictionary-based input into a format 
    suitable for machine learning inference by applying saved encoding and 
    scaling transformations.
    """

    def __init__(self, user_input: dict):
        """
        Initializes the preprocessing pipeline with user data and artifact paths.

        Args:
            user_input (dict): Key-value pairs representing a single data record.
        """
        self.user_input = user_input
        self.encoding_file_path = ENCODER_FILE_PATH
        self.scaling_file_path = SCALER_FILE_PATH

    def converting_to_dataframe(self):
        """
        Converts the raw input dictionary into a pandas DataFrame.

        Returns:
            pd.DataFrame: A single-row DataFrame containing the user input.
        """
        input_df = pd.DataFrame([self.user_input])
        return input_df

    def encoding(self):
        """
        Loads the serialized OneHotEncoder object from the local file system.

        Returns:
            sklearn.preprocessing.OneHotEncoder: The fitted encoder instance.
        """
        with open(self.encoding_file_path, 'rb') as f:
            loaded_encoder = pickle.load(f)
        return loaded_encoder
    
    def scaling(self):
        """
        Loads the serialized StandardScaler/MinMaxScaler object.

        Returns:
            sklearn.preprocessing.Scaler: The fitted scaler instance.
        """
        return joblib.load(self.scaling_file_path)
    
    def run(self):
        """
        Executes the full preprocessing workflow: DataFrame conversion, 
        categorical encoding, and numerical scaling.

        Returns:
            numpy.ndarray: The final processed and scaled feature vector.
        """
        # 1. Initialize data and load transformation artifacts
        input_df = self.converting_to_dataframe()
        loaded_encoder = self.encoding()
        scaler = self.scaling()
        
        # 2. Identify categorical columns and apply encoding
        cat_cols = loaded_encoder.feature_names_in_
        encoded_cats = loaded_encoder.transform(input_df[cat_cols])
        
        # 3. Format encoded categories back into a DataFrame
        encoded_df = pd.DataFrame(
            encoded_cats, 
            columns=loaded_encoder.get_feature_names_out(cat_cols)
        )
        
        # 4. Merge encoded features with numeric features and drop original categories
        encoded_input = pd.concat(
            [input_df.drop(cat_cols, axis=1).reset_index(drop=True), encoded_df], 
            axis=1
        )
        
        # 5. Apply final numerical scaling
        scaled_input = scaler.transform(encoded_input)
        
        return scaled_input

# Example usage for integration testing
# if __name__=="__main__":
#     user_input = {
#         'Age': 25,
#         'Sex': 'male',
#         'Job': 2,
#         'Housing': 'own',
#         'Saving accounts': 'little',
#         'Checking account': 'moderate',
#         'Credit amount': 1500,
#         'Duration': 12,
#         'Purpose': 'radio/TV'
#     }
#     preprocess = PREPROCESSING(user_input)
#     print(preprocess.run())