# Self-Driving Car Project
This is a project inspired following the recent announcement of the Automated Vehicles ( AV ) Act becoming law (20 May 2024) and the possiblility of self-driving cars being on the road in 2026!

Generally, the C++ scripts will take in sensor data, using SLAM algorithms to create a map for a pathfinding algorithm to then be used. Python then takes over to do higher level descision-making based on the road provided by the neural networks (NNs) detecting traffic, pedestrians and signs.

This project is limited by computational power, time and money for equipment, I want to see how close I can get to Tesla FSD as possible in one summer (3 months).

# Functionalities
I want this project to cover most important aspects of driving, these are:
  + Recognising lane lines - Python
  + Recognising pedestrians, other cars as well as roadsigns - Python
  + Driving on the road and predicting the angle of steer needed to stay within lane lines - C++

Then if possible, some extra goals
  + Creating a "virtual roadspace" which contains elements on the road, similar to the Tesla FSD - C++/Python?
  + Park in bays and parallel parking spots - C++


# Plans and setbacks
This project is proving difficult and I can't seem to find a great place to start, being about a month in and having written a ConvNet for a lane line idenifier but not being happy with it and also having almost written a "virtual parking environment" in PyGame. I feel as if this is not the correct place to start - this I think the project is better started with the object identifier, this way I can incorperate a coordinate system and attribute coordinates to cars, lane lines and other objects. These coordinates can then be relayed to the pathfinding and mapping script in C++.

I feel as if this is a better approach to the project as the modules like the pathfinding and mapping/SLAM build off of the data processed by the object identifier.
While the project has been undertaken, I have been completing the Deep Learning Specialization on Coursera so I am hoping to implement alot of the methods I have learnt from this.

# Object Identification
This part of the project uses the NuScenes dataset (American) and aims to use a sliding window/YOLO algorithm 
