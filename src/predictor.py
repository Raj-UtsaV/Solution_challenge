from config.paths_config import *
import joblib

class PREDICTOR:
    """
    Handles the loading of the machine learning model and execution of 
    inference on pre-processed data.
    """
    
    def __init__(self, scaled_output: dict):
        """
        Initializes the predictor with the file path from configuration 
        and the data to be predicted.

        Args:
            scaled_output (dict/list): The processed numerical features 
                                       ready for the model.
        """
        self.model_file_path = MODEL_FILE_PATH
        self.scaled_output = scaled_output
    
    def run(self):
        """
        Loads the serialized model artifact and returns the prediction.

        Returns:
            numpy.ndarray: The predicted class or value from the model.
        """
        # Load the trained model object from the specified directory
        self.model = joblib.load(self.model_file_path)
        
        # Execute the model's prediction method and return the result
        return self.model.predict(self.scaled_output)

# Example usage for testing and local validation
# if __name__=="__main__":
#     # Sample scaled and encoded input vector
#     scaled_output = [[-0.92754658,0.14694918,-0.62781066,-0.73866754,-0.67028006,0.67028006
#         ,-0.3479601 ,  0.63444822 ,-0.4669334 ,  0.81140298 , -0.33886163 , -0.25929878
#         ,-0.22454436 , -0.47327604 , -0.61433742 , 1.6484757 , -0.25929878 , -0.80632811
#         ,-0.32774947 , -0.71294854 , -0.11020775 ,  -0.2503982 , -0.47010767 , 1.60356745
#         ,-0.14998296 , -0.11020775]]
    
#     # Initialize the predictor instance with the sample data
#     predictor = PREDICTOR(scaled_output)
    
#     # Execute and print the prediction result
#     print(predictor.run())