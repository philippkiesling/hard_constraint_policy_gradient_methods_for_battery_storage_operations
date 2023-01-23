# master_thesis_philipp_kiesling
# 0. Setup
#### 0.1. Install Python 3.6 or higher (tested under 3.10)
#### 0.2. _(Optional) If you want to use conda , install the conda environment.yml first_
```conda env create -f environment.yml```
#### 0.3. Install package batterytrading with pip in editable mode
```pip install -e .```

_This will install all dependencies with pip (if not installed via conda already)_

#### _[NOT WORKING YET]: 0.4. Run the tests with pytest_
```pytest```

# 1. Project Structure

```
batterytrading
├── data_loader
│   ├── data_loader.py
├── environment
│   ├── environment.py
├── ppo
│   ├── train.py
│   ├── policies.py
│   ├── model_setup.py
│   ├── cfg.yml
├── webscraping
│   ├── webscrape_energy_charts.py
│   ├── webscrape_SMARD.py
│   ├── concat-files.py

data
├── energy_charts_data 

DRL_Multi_Purpose [Git Submodule]

notebooks

tests
├── test_environment.py

```
 **batterytrading**: Contains the main code for the project

 **data_loader**: Contains the code for loading the data

 **environment**: Contains the code for creating an environment from the data provided by the data_loader

 **ppo**: Contains the code for training PPO agent and adaptions to the policies

 **webscraping**: Contains the code for webscraping the energy charts and SMARD data


# 2. Running the code
The main script is the train.py script in the ppo folder. It contains the code for training the PPO agent. The script can be run with the following command:
```python batterytrading/ppo/train.py```
_Note:_ If your run the code from the an IDE, please make sure that the working directory is set to the root directory of the project. (This is the directory where the README.md is located)

# 3. Configuration
To configure environment, training and policies, the cfg.yml file in the ppo folder can be used.

# 4. Webscraping
The webscraping folder contains the code for webscraping the energy charts and SMARD data. The webscraping scripts can be run with the following command:
```python webscraping/webscrape_energy_charts.py```
```python webscraping/webscrape_SMARD.py```
(Or from an IDE with the working directory set to the root directory of the project)
