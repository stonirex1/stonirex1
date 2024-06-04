import numpy as np
from sklearn.metrics import mean_squared_error

# Observational data and model predictions
observations = np.array([15.2, 16.1, 14.5, 15.8, 25.0])
model_output = np.array([14.8, 15.5, 14.0, 16.0, 25.3])

# Function to perform Optimal Interpolation
def optimal_interpolation(observations, model_output, obs_error=1.0, model_error=2.0):
    # Calculate the weights
    weight_obs = model_error**2 / (obs_error**2 + model_error**2)
    weight_model = obs_error**2 / (obs_error**2 + model_error**2)
    
    # Calculate the analysis
    analysis = weight_obs * observations + weight_model * model_output
    
    return analysis

# Generate analysis fields using Optimal Interpolation
analysis = optimal_interpolation(observations, model_output)
print("Analysis:", analysis)

# Function to calculate RMSE
def rmse(predictions, targets):
    return np.sqrt(mean_squared_error(targets, predictions))

# Function to calculate Bias
def bias(predictions, targets):
    return np.mean(predictions - targets)

# Compute RMSE and Bias
rmse_obs = rmse(observations, observations)
rmse_model = rmse(model_output, observations)
rmse_analysis = rmse(analysis, observations)

bias_obs = bias(observations, observations)
bias_model = bias(model_output, observations)
bias_analysis = bias(analysis, observations)

print("RMSE (Observations):", rmse_obs)
print("RMSE (Model):", rmse_model)
print("RMSE (Analysis):", rmse_analysis)
print("Bias (Observations):", bias_obs)
print("Bias (Model):", bias_model)
print("Bias (Analysis):", bias_analysis
