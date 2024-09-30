import json
import boto3

# Create a SageMaker runtime client
client = boto3.client('sagemaker-runtime')

# Prepare the payload for prediction
payload = json.dumps({
    "instances": [[5.0, 3.5, 1.5, 0.2]]  # Example input features; adjust based on your model
})

# Replace 'your-endpoint-name' with the actual endpoint name you created
endpoint_name = 'sagemaker-scikit-learn-2024-09-30-17-24-55-369'  # Update this with your SageMaker endpoint name

# Invoke the SageMaker endpoint
try:
    response = client.invoke_endpoint(
        EndpointName=endpoint_name,
        Body=payload,
        ContentType='application/json'
    )

    # Read the response
    result = json.loads(response['Body'].read().decode())
    print("Prediction Result:", result)

except Exception as e:
    print(f"Error invoking endpoint: {e}")
