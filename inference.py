import json
import pickle
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier

def model_fn(model_dir):
    """Load the model from the model_dir."""
    model_path = os.path.join(model_dir, 'random_forest_model.pkl')
    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)
    return model

def input_fn(request_body, request_content_type='application/json'):
    """Process the input data."""
    if request_content_type == 'application/json':
        data = json.loads(request_body)
        return np.array(data['instances'])
    raise ValueError("Unsupported content type: {}".format(request_content_type))

def predict_fn(input_data, model):
    """Make predictions using the loaded model."""
    predictions = model.predict(input_data)
    return predictions

def output_fn(prediction, response_content_type='application/json'):
    """Format the prediction results."""
    if response_content_type == 'application/json':
        return json.dumps({'predictions': prediction.tolist()})
    raise ValueError("Unsupported content type: {}".format(response_content_type))
