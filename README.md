# NoBoxNeuralNetAttack

This repository contains code to demonstrate a method of generating "no-box" neural network attacks. Unlike "white-box" attacks (requiring knowledge of the internal workings of the target network) and "black-box" attacks (requiring query access to the target network), no-box attacks are generated without any knowledge of the target network (e.g. imagine if the attacks are intended for target networks that do not yet exist). Instead, it is possible to extract effective designs for adversarial attacks purely from an analysis of the statistics of the input space. In this way, the weights of a yet-to-be-trained neural network can be modeled as random variables whose statistics are calculable from knowledge of the input space.

## Running the experiment

To run the experiment, navigate to the directory and run
```
python significance_experiment.py
```
You may need to install numpy, tensorflow, and matplotlib, if you do not have them installed already.

## Customizing the experiment

`significance_experiment.py` is pretty simple and easily manipulable, but there are three fields at the top (`LOAD_SIGNIFICANCE`, `SAVE_AVERAGE_WEIGHTS`, `GENERATE_FRESH_AVERAGE_WEIGHTS`) which can be set to different boolean values to achieved the desired behavior (view comments above these fields to learn their effect).
