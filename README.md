# Self-Driving Car Project
This is a project inspired by the recent announcement of the Automated Vehicles ( AV ) Act becoming law (20 May 2024) and the possiblility of self-driving cars being on the road in 2026!

Python code which uses video input to analyse situations when driving and make descisions.

# Functionalities
I want this project to cover most important aspects of driving, these are:
  + Recognising lane lines
  + Recognising pedestrians, other cars as well as roadsigns
  + Predict the angle of steer needed to stay within the road lines
  + Predict hazards on the road and make descisions to prevent them

It would be nice to have:
  + Making descisions on when to give way on a road with cars park on either side
  + Creation of a "virtual driving space" which contains elements on the road, similar to the Tesla FSD

# Predicting Lane Lines
The first peice of the puzzle, allowing the car to follow a path. My approach was to use a Convolutional Neural Network
