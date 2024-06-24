# Self-Driving Car Project
This is a project inspired following the recent announcement of the Automated Vehicles ( AV ) Act becoming law (20 May 2024) and the possiblility of self-driving cars being on the road in 2026!

Python code which uses sensor and video input to analyse situations when driving and make descisions.

# Functionalities
I want this project to cover most important aspects of driving, these are:
  + Recognising lane lines
  + Recognising pedestrians, other cars as well as roadsigns
  + Predict the angle of steer needed to stay within the road lines
  + Predict hazards on the road and make descisions to prevent them
  + Park in bays and parallel parking spots

It would be nice to have:
  + Making descisions on when to give way on a road with cars parked on either side
  + Creation of a "virtual driving space" which contains elements on the road, similar to the Tesla FSD

# Predicting Lane Lines ()
The first peice of the puzzle, allowing the car to follow a path. My approach was to use a Convolutional Neural Network, feeding in greyscaled, blurred and masked images to the network would simplify the learning process for the network. 

# Automatic Parellel Parking (23/06/24)
After the neural network in the lane lines part of the project, I wanted a break from the deep learning side and to try a completely new domain to me: pygame and visualisations. By having a direct birds eye view of the car, it allows for a great visual of the road around the car and to see its parking abilities in action. 
I have started with a simple simulation of a road, with a simple algorithm for only parallel parking. This is something I'd like to change, possibly for reinforcement learning, and ultimately let sensor and video input aid the computer to park the car. 

This section also required the use of some code from my repository: Python-Projects and was one of my first projects, so it was in desparate need of some revamping.
