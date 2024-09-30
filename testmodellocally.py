import pickle
import pandas as pd

# Load your model locally
with open('D:/ML/random_forest_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Sample input data
input_data = [[5.0, 3.5, 1.5, 0.2]]  # Adjust based on what your model expects
print("Input Data for Prediction:", input_data)

# Prepare the input DataFrame
input_df = pd.DataFrame(input_data)

# Make prediction
prediction = model.predict(input_df)
print("Local Prediction Result:", prediction)

# Mapping output if necessary
target_mapping = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}  # Adjust as per your dataset
predicted_species = target_mapping.get(prediction[0], "Unknown")
print(f"Predicted Species: {predicted_species}")
