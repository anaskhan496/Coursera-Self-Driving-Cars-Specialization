# Self Driving Cars Specialization
This four course Specialization gave a comprehensive understanding of state-of-the-art engineering practices used in the self-driving car industry. It provided opportunities to interact with real data sets from an autonomous vehicle (AV)―all through hands-on projects using the open source simulator CARLA.

## Objectives of the course
- Understand the detailed architecture and components of a self-driving car software stack.
- Implement methods for static and dynamic object detection, localization and mapping, behaviour and maneuver planning, and vehicle control.
- Use realistic vehicle physics, complete sensor suite: camera, LIDAR, GPS/INS, wheel odometry, depth map, semantic segmentation, object bounding boxes.
- Demonstrate skills in CARLA and build programs with Python.


## Part 1 - Introduction to Self-Driving Cars
###### **Self Driving Vehicle Control** (https://rb.gy/220yta)
<img src="1_Introduction_to_Self_Driving_Cars/videos/Carla-Simulator.png" alt="Carla" style="width:20 height:20">
                                    

## Part 2 - State Estimation and Localization
###### **Trajectory Estimation using fusion of range and bearing measurements through Extended Kalman Filter** (https://rb.gy/7xagpm)
![](2_State_Estimation_and_Localization/Module%202%20Extended%20Kalman%20Filter/images/Trajectory-Plot.png) 
![](2_State_Estimation_and_Localization/Module%202%20Extended%20Kalman%20Filter/images/Angular-Change-vs-Time.png)

###### **Vehicle State Estimation on a Roadway using Error-State EKF** (https://rb.gy/7rgac1)
<p float="left">
  <img src="2_State_Estimation_and_Localization/Final%20-%20Vehicle%20State%20Estimation%20on%20a%20Roadway/images/Part1-Ground-Truth-vs-Estimated.png" width="800" />
  <img src="2_State_Estimation_and_Localization/Final%20-%20Vehicle%20State%20Estimation%20on%20a%20Roadway/images/Part1-Error-Plots.png" width="650" /> 
</p>


## Part 3 - Visual Perception for Self-Driving Cars
###### **Applying Stereo Depth to a Driving Scenario** (https://rb.gy/spligv)
- Obstacle detection using disparity map, depth map and, cross-corelation matrix computations.
<p float="left">
  <img src="3_Visual_Perception_for_Self_Driving_Cars/module1/images/2-disparity-map.png" width="220" />
  <img src="3_Visual_Perception_for_Self_Driving_Cars/module1/images/3-depth-map.png" width="220" />
  <img src="3_Visual_Perception_for_Self_Driving_Cars/module1/images/4-cross-corelation.png" width="240" />
  <img src="3_Visual_Perception_for_Self_Driving_Cars/module1/images/5-obstacle-detection.png" width="240" />
</p>

###### **Visual Odometry for Localization in Autonomous Driving** (https://rb.gy/9vqt9k)
<p float="left">
  <img src="3_Visual_Perception_for_Self_Driving_Cars/module2/images/4-Visual-Odometry.png" />
</p>

###### Environment Perception For Self-Driving Cars (https://rb.gy/vbx5cw)
- Estimation of ground plane using RANSAC algorithm
<p float="left">
  <img src="3_Visual_Perception_for_Self_Driving_Cars/module6/output_images/1-Test-Image.png" width="400" />
  <img src="3_Visual_Perception_for_Self_Driving_Cars/module6/output_images/2-Ground-Plane-RANSAC.png" width="400" />
</p>

- Lane line detection using Canny edge detection, Hough transform and,merging & filtering.
<p float="left">
  <img src="3_Visual_Perception_for_Self_Driving_Cars/module6/output_images/3_0-Canny-Edge-Hough-Transform.png" width="300" />
  <img src="3_Visual_Perception_for_Self_Driving_Cars/module6/output_images/3_1-Merging-Lane-Lines.png" width="300" />
  <img src="3_Visual_Perception_for_Self_Driving_Cars/module6/output_images/3_2-Filtered-Lane-Line.png" width="300" />
</p>

- Filtering 2D detection output using Semantic Segmentation results.
<p float="left">
  <img src="3_Visual_Perception_for_Self_Driving_Cars/module6/output_images/4_0-Unfiltered-Bounding-Box.png" width="330" />
  <img src="3_Visual_Perception_for_Self_Driving_Cars/module6/output_images/4_1-Filtered-Bounding-Box.png" width="330" />
</p>

- Distance to impact estimation.
<p float="left">
  <img src="3_Visual_Perception_for_Self_Driving_Cars/module6/output_images/5-Distance-to-impact.png" width="350" />
</p>


## Part 4 - Motion Planning for Self-Driving Cars 
###### **Occupany Grid Mapping for Planning using Laser Scanner** (https://rb.gy/cj1i2o)
![](4_Motion_Planning_for_Self_Driving_Cars/Module%202%20Mapping%20for%20Planning/True-vs-Occupancy-Maps%20.gif)

###### **Motion Planning Pipeline for Self-Driving Cars** (https://rb.gy/9nkqjx)
![](4_Motion_Planning_for_Self_Driving_Cars/Module%206%20Final%20Motion%20Planning/videos/Planner_in_motion.gif)
