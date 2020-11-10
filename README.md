# robotics_GraphOptimization
Graph optimization project to estimate the state of a robot using its sensor data.

This project is organized into two classes in `graph_opt.py`: one to implement the g2o library for creating nodes and edges, and another to implement the graph optimization by subscribing to topics from a bag file. The PoseGraphOptimization class the methods to account for two dimensions. There are functions to add a landmark node and an edge from a pose node to a landmark node. This project depends on the g2o library to run. Optimizaiton can be seen in `plot.png`. There are two extraneous points from a timing issue as a result of running off a bag file. 
