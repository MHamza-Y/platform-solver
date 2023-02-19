# platform-solver
This repo aims to train agent for solving the [gym-platform](https://github.com/cycraig/gym-platform) environment.

# Results

| Converged policy with PPO|
| --- |
| ![](https://github.com/MHamza-Y/platform-solver/blob/main/media/converged_platform_env_without_lstm.gif) | 

| Converged policy with PPO + LSTM |
| --- |
| ![](https://github.com/MHamza-Y/platform-solver/blob/main/media/converged_platform_env.gif) | 


# Setup
Install dependencies by executing:
```
pip install -r requirements.txt
```
# Train Agent
To reproduce this the agent can be trained by running the [pipeline](pipeline.ipynb) notebook. The pipeline does the following:
* Search for the best hyper-parameters
* Train using the chosen hyper-parameters
* Evaluate the trained agent

This pipeline outputs following artifacts:
* Training checkpoints containing policy
* Evaluation results in dataframe format

# Model Explainability 
Executing [explainability](explainability.ipynb) notebook creates visualizations to explain the policy behaviour.

# Watch Trained Agent
To watch the trained agent in action, execute:
```
python watch_trained_agent.py -c <Checkpoint Path> -e <Number of Epochs (optional)>
```

