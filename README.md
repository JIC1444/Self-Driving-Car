# Self-Driving Car Project
This is a project inspired following the recent announcement of the Automated Vehicles ( AV ) Act becoming law (20 May 2024) and the possiblility of self-driving cars being on the road in 2026!

Generally, the C++ scripts will take in sensor data, using SLAM algorithms to create a map for a pathfinding algorithm to then be used. Python then takes over to do higher level descision-making based on the road provided by the neural networks (NNs) detecting traffic, pedestrians and signs.

This project is limited by computational power, time and money for equipment, I want to see how close I can get to Tesla FSD as possible in one summer (3 months).

# Functionalities
I want this project to cover most important aspects of driving, these are:
  + Creating a "virtual roadspace" which contains elements on the road, similar to the Tesla FSD - C++
  + Driving on the road and predicting the angle of steer needed to stay within lane lines - C++
  + Recognising lane lines - Python / C++?
  + Recognising pedestrians, other cars as well as roadsigns - Python
  + Predict hazards on the road and make descisions to prevent them - Python
  + Park in bays and parallel parking spots - C++

It would be nice to have:
  + Making descisions on when to give way on a road with cars parked on either side

# Plans and setbacks
All of the functionalities were originally set to be in Python, however, C++ is much faster and the more work I can make it do, the better. For this reason, the script I made for predicting lane lines was originally a convoulional NN, trained on ~2000 images. However, I am hoping to first do this in C++ then have the lane line prediction to guide the angle of steer as well. (29/06/2024)

The automatic parking was originally in Pygame (Python), however I think I can make something better in C++, but I will finish the Python version first and come back to the C++ implementation later. (29/06/2024)

# Predicting Lane Lines
The first peice of the puzzle, allowing the car to follow a path. My first approach was to use a PyTorch (Python) to write a convolutional NN, feeding in greyscaled, blurred and masked images to the network would simplify the learning process for the network. However, my second approach will be using the same image transforms as well as a hough transform, 

# Automatic Parellel Parking (23/06/24)
After the neural network in the lane lines part of the project, I wanted a break from the deep learning side and to try a completely new domain to me: pygame and visualisations. By having a direct birds eye view of the car, it allows for a great visual of the road around the car and to see its parking abilities in action. 
I have started with a simple simulation of a road, with a simple algorithm for only parallel parking. This is something I'd like to change, possibly for reinforcement learning, and ultimately let sensor and video input aid the computer to park the car. 

This section also required the use of some code from my repository: Python-Projects and was one of my first projects, so it was in desparate need of some revamping.
