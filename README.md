# KAN-DQN

## Requirements
We assume you have access to a GPU that can run CUDA 11.8. Then, the simplest way to install all required dependencies is to create an anaconda environment by running
```
  conda env create -f requirements.yaml
```
After the installation ends, you can activate your environment with
```
  conda activate kan
```

## Instructions
To run a single run, use the ```scripts/dqn/kan_dqn.sh``` script
```
  cd scripts/dqn
  bash kan_dqn.sh
```
