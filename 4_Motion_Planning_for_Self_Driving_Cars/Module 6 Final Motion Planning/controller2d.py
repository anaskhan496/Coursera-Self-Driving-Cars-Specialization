#!/usr/bin/env python3

"""
2D Controller Class to be used for the CARLA waypoint follower demo.
All units are in SI (meters, seconds, radians), and CARLA works in the left-handed coordinate system 
(due to the Unreal Engine adopting the left-handed coordinate system). 
This generally shouldn't be an issue for this assignment since the controller operates in the x, y and yaw space.
The waypoints variable is a Python list of waypoints to track where each row denotes a waypoint of format [x, y, v],
which are the x and y positions as well as the desired speed at that position, respectively.
the waypoints are a linearly interpolated (for location and speed) subset of the entire set of waypoints (from racetrack_waypoints.txt).
In other words the waypoints variable is an enhanced (finer resolution) portion of the entire set of waypoints that is near the vehicle.
"""
import cutils
import numpy as np

class Controller2D(object):
    def __init__(self, waypoints):
        self.vars                   = cutils.CUtils()
        self._lookahead_distance    = 2.0
        self._current_x             = 0
        self._current_y             = 0
        self._current_yaw           = 0
        self._current_speed         = 0
        self._desired_speed         = 0
        self._current_frame         = 0
        self._current_timestamp     = 0
        self._start_control_loop    = False
        self._set_throttle          = 0
        self._set_brake             = 0
        self._set_steer             = 0
        self._waypoints             = waypoints
        self._conv_rad_to_steer     = 180.0 / 70.0 / np.pi
        self._pi                    = np.pi
        self._2pi                   = 2.0 * np.pi

    def update_values(self, x, y, yaw, speed, timestamp, frame):
        self._current_x         = x
        self._current_y         = y
        self._current_yaw       = yaw
        self._current_speed     = speed
        self._current_timestamp = timestamp
        self._current_frame     = frame
        if self._current_frame:
            self._start_control_loop = True

    def get_lookahead_index(self, lookahead_distance):
        min_idx       = 0
        min_dist      = float("inf")
        for i in range(len(self._waypoints)):
            dist = np.linalg.norm(np.array([
                    self._waypoints[i][0] - self._current_x,
                    self._waypoints[i][1] - self._current_y]))
            if dist < min_dist:
                min_dist = dist
                min_idx = i

        total_dist = min_dist #setting the distance between the current vehicle position and the waypoints to a variable
        lookahead_idx = min_idx
        for i in range(min_idx + 1, len(self._waypoints)):
            if total_dist >= lookahead_distance:
                break
            total_dist += np.linalg.norm(np.array([
                    self._waypoints[i][0] - self._waypoints[i-1][0],
                    self._waypoints[i][1] - self._waypoints[i-1][1]]))
            lookahead_idx = i
        return lookahead_idx


    def update_desired_speed(self):
        min_idx       = 0
        min_dist      = float("inf")
        desired_speed = 0
        for i in range(len(self._waypoints)):
            dist = np.linalg.norm(np.array([
                    self._waypoints[i][0] - self._current_x,
                    self._waypoints[i][1] - self._current_y]))
            if dist < min_dist:
                min_dist = dist
                min_idx = i
        if min_idx < len(self._waypoints)-1:
            desired_speed = self._waypoints[min_idx][2]
            self.minidx = min_idx
        else:
            desired_speed = self._waypoints[-1][2]
            self.minidx = -1 #if the minimum index at which the current vehicle position is beyond the length of the waypoints list, then we use the last index of waypoints
        self._desired_speed = desired_speed

    def update_waypoints(self, new_waypoints):
        self._waypoints = new_waypoints

    def get_commands(self):
        return self._set_throttle, self._set_steer, self._set_brake

    def set_throttle(self, input_throttle):
        # Clamp the throttle command to valid bounds
        throttle           = np.fmax(np.fmin(input_throttle, 1.0), 0.0)
        self._set_throttle = throttle

    def set_steer(self, input_steer_in_rad):
        # Covnert radians to [-1, 1]
        input_steer = self._conv_rad_to_steer * input_steer_in_rad

        # Clamp the steering command to valid bounds
        steer           = np.fmax(np.fmin(input_steer, 1.0), -1.0)
        self._set_steer = steer

    def set_brake(self, input_brake):
        # Clamp the steering command to valid bounds
        brake           = np.fmax(np.fmin(input_brake, 1.0), 0.0)
        self._set_brake = brake

    def update_controls(self):
        ######################################################
        # RETRIEVE SIMULATOR FEEDBACK
        ######################################################
        x               = self._current_x
        y               = self._current_y
        yaw             = self._current_yaw
        v               = self._current_speed
        self.update_desired_speed()
        v_desired       = self._desired_speed
        t               = self._current_timestamp
        waypoints       = self._waypoints
        throttle_output = 0
        steer_output    = 0
        brake_output    = 0

        self.vars.create_var('kp', 0.50)
        self.vars.create_var('ki', 0.30)
        self.vars.create_var('integrator_min', 0.0)
        self.vars.create_var('integrator_max', 10.0)
        self.vars.create_var('kd', 0.13)
        self.vars.create_var('kp_heading', 8.00)
        self.vars.create_var('k_speed_crosstrack', 0.00)
        self.vars.create_var('cross_track_deadband', 0.01)
        self.vars.create_var('x_prev', 0.0)
        self.vars.create_var('y_prev', 0.0)
        self.vars.create_var('yaw_prev', 0.0)
        self.vars.create_var('v_prev', 0.0)
        self.vars.create_var('t_prev', 0.0)
        self.vars.create_var('v_error', 0.0)
        self.vars.create_var('v_error_prev', 0.0)
        self.vars.create_var('v_error_integral', 0.0)
        
        # Skip the first frame to store previous values properly
        if self._start_control_loop:

            self.vars.v_error           = v_desired - v # proportional error
            self.vars.v_error_integral += self.vars.v_error *  (t - self.vars.t_prev) #integral error (integr(e))
            v_error_rate_of_change      = (self.vars.v_error - self.vars.v_error_prev) /\
                                          (t - self.vars.t_prev) #derivative error (de/dt)

            # cap the integrator sum to a min/max
            self.vars.v_error_integral = \
                    np.fmax(np.fmin(self.vars.v_error_integral, 
                                    self.vars.integrator_max), 
                            self.vars.integrator_min)
            
            # adding the proportional, derivative and integral errors 
            throttle_output = self.vars.kp * self.vars.v_error +\
                              self.vars.ki * self.vars.v_error_integral +\
                              self.vars.kd * v_error_rate_of_change

            # Find cross track error (assume point with closest distance)
            crosstrack_error = float("inf")
            crosstrack_vector = np.array([float("inf"), float("inf")])

            ce_idx = self.get_lookahead_index(self._lookahead_distance)
            crosstrack_vector = np.array([waypoints[ce_idx][0] - \
                                         x - self._lookahead_distance*np.cos(yaw), 
                                          waypoints[ce_idx][1] - \
                                         y - self._lookahead_distance*np.sin(yaw)])
            crosstrack_error = np.linalg.norm(crosstrack_vector)

            # set deadband to reduce oscillations
            # print(crosstrack_error)
            if crosstrack_error < self.vars.cross_track_deadband:
                crosstrack_error = 0.0

            # Compute the sign of the crosstrack error
            crosstrack_heading = np.arctan2(crosstrack_vector[1], 
                                            crosstrack_vector[0])
            crosstrack_heading_error = crosstrack_heading - yaw
            crosstrack_heading_error = \
                    (crosstrack_heading_error + self._pi) % \
                    self._2pi - self._pi

            crosstrack_sign = np.sign(crosstrack_heading_error)
    
            # Compute heading relative to trajectory (heading error)
            # First ensure that we are not at the last index. If we are,
            # flip back to the first index (loop the waypoints)
            if ce_idx < len(waypoints)-1:
                vect_wp0_to_wp1 = np.array(
                        [waypoints[ce_idx+1][0] - waypoints[ce_idx][0],
                         waypoints[ce_idx+1][1] - waypoints[ce_idx][1]])
                trajectory_heading = np.arctan2(vect_wp0_to_wp1[1], 
                                                vect_wp0_to_wp1[0])
            else:
                vect_wp0_to_wp1 = np.array(
                        [waypoints[0][0] - waypoints[-1][0],
                         waypoints[0][1] - waypoints[-1][1]])
                trajectory_heading = np.arctan2(vect_wp0_to_wp1[1], 
                                                vect_wp0_to_wp1[0])

            heading_error = trajectory_heading - yaw
            heading_error = \
                    (heading_error + self._pi) % self._2pi - self._pi

            # Compute steering command based on error
            steer_output = heading_error + \
                    np.arctan(self.vars.kp_heading * \
                              crosstrack_sign * \
                              crosstrack_error / \
                              (v + self.vars.k_speed_crosstrack))
            np.seterr(all='ignore') 
            ######################################################
            # SET CONTROLS OUTPUT
            ######################################################
            self.set_throttle(throttle_output)  # in percent (0 to 1)
            self.set_steer(steer_output)        # in rad (-1.22 to 1.22)
            self.set_brake(brake_output)        # in percent (0 to 1)

        self.vars.x_prev       = x
        self.vars.y_prev       = y
        self.vars.yaw_prev     = yaw
        self.vars.v_prev       = v
        self.vars.v_error_prev = self.vars.v_error
        self.vars.t_prev       = t
        
