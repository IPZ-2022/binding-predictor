# README

## Overview
There are three files that present three steps of our methodology:

**data.py** gives an example of dataset preparation for training the model.

**main.py** is the main training loop that trains the model.

**find-ligand.py** gives an example of how a trained model would be utilized to predict likely candidates for 
binding to a given protein.

## Usage
- Use requirements.txt to install the neccesary packages
- Run {data,main,find-ligand}.py to run your script of choice to see how it works

## Notes
- Each of the three files downloads all the data it requires independently, so you don't have to run through the whole pipeline.
- data.py gives a commented-out example of how to calculate protein encoding for your own dataset, along with info on the 
docker environment prepared for this task.
- main.py is 