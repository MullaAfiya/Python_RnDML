import json
import boto3

# Create a SageMaker runtime client
client = boto3.client('sagemaker-runtime')

# Prepare the payload for prediction
payload = json.dumps({
    "instances": [[5.0, 3.5, 1.5, 0.2]]  # Example input features
})

# Define the correct endpoint name
endpoint_name = 'sagemaker-scikit-learn-2024-10-02-07-25-36-713'  # Update this with your SageMaker endpoint name

# Mapping of class indices to species names in the Iris dataset
iris_target_names = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

# Invoke the SageMaker endpoint
try:
    response = client.invoke_endpoint(
        EndpointName=endpoint_name,
        Body=payload,
        ContentType='application/json'
    )

    # Read the response
    result = json.loads(response['Body'].read().decode())
    predicted_class = result['predictions'][0]
    
    # Map the predicted class to the corresponding species
    predicted_species = iris_target_names[predicted_class]
    
    print(f"Prediction Result: {predicted_species} (Class {predicted_class})")

except Exception as e:
    print(f"Error invoking endpoint: {e}")
