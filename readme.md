# airfoil_surrogate

Implementing a workflow with parametric airfoil generation, coefficient simulation using XFoil and training/prediction via surrogate modelling.

## Geometry

The `airfoil.py` module provides methods for reading, transforming and creating airfoil geometric parameters.

## Evaluation

The `evaluate.py` module makes use of `aerosandbox` to execute `XFoil` simulations of profiles. It additionally parses the simulation information to extract relevant parameters, which will be the desired outputs of the surrogate models.

## Training data generation

The script `main_generate.py` loads an initial airfoil and morphs it into many variations which are then evaluated. The results are saved to a csv file and are used for surrogate model training. The module `train_data.py` provides useful methods for saving, cleaning and loading training data.

## Surrogate model

The methods for creating and training surrogate models are defined in the module `surrogate.py`. Additionally, useful methods for saving and loading models are defined in `model_manager.py`.

## Training the surrogate model

The script `main_train.py` contains the workflow for loading and cleaning training data, initializing and training the surrogate models, and saving the resulting models for further use.

## Scoring the surrogate models

The script `score_surrogate_pack.py` provides code to score/validate all models trained.