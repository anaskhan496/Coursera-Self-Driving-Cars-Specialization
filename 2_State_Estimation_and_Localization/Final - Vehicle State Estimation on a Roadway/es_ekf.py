# Starter code for the Coursera SDC Course 2 final project.
#
# Author: Trevor Ablett and Jonathan Kelly
# University of Toronto Institute for Aerospace Studies
import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import distance as rel_dist
from rotations import angle_normalize, rpy_jacobian_axis_angle, skew_symmetric, Quaternion

#### 1. Data ###################################################################################

################################################################################################
# This is where you will load the data from the pickle files. For parts 1 and 2, you will use
# p1_data.pkl. For Part 3, you will use pt3_data.pkl.
################################################################################################
with open('data/pt1_data.pkl', 'rb') as file:
    data = pickle.load(file)
# with open('data/pt3_data.pkl', 'rb') as file:
#     data = pickle.load(file)

################################################################################################
# Each element of the data dictionary is stored as an item from the data dictionary, which we
# will store in local variables, described by the following:
#   gt: Data object containing ground truth. with the following fields:
#     a: Acceleration of the vehicle, in the inertial frame
#     v: Velocity of the vehicle, in the inertial frame
#     p: Position of the vehicle, in the inertial frame
#     alpha: Rotational acceleration of the vehicle, in the inertial frame
#     w: Rotational velocity of the vehicle, in the inertial frame
#     r: Rotational position of the vehicle, in Euler (XYZ) angles in the inertial frame
#     _t: Timestamp in ms.
#   imu_f: StampedData object with the imu specific force data (given in vehicle frame).
#     data: The actual data
#     t: Timestamps in ms.
#   imu_w: StampedData object with the imu rotational velocity (given in the vehicle frame).
#     data: The actual data
#     t: Timestamps in ms.
#   gnss: StampedData object with the GNSS data.
#     data: The actual data
#     t: Timestamps in ms.
#   lidar: StampedData object with the LIDAR data (positions only).
#     data: The actual data
#     t: Timestamps in ms.
################################################################################################
gt = data['gt']
imu_f = data['imu_f']
imu_w = data['imu_w']
gnss = data['gnss']
lidar = data['lidar']

# print("gt acc size", np.shape(gt.a))
# print("gt vel size", np.shape(gt.v))
# print("gt position size", np.shape(gt.p))
# print("gt rot acc size", np.shape(gt.alpha))
# print("gt rot vel size", np.shape(gt.w))
# print("gt rot pos size", np.shape(gt.r))
# print("gt timestamp size", np.shape(gt._t))
# print("imu specific force data size",np.shape(imu_f.data))
# print("imu specific force timestamp size",np.shape(imu_f.t))
# print("imu rot vel data size",np.shape(imu_w.data))
# print("imu rot vel Timestamp size",np.shape(imu_w.t))
# print("GNSS data size", np.shape(gnss.data))
# print("GNSS timestamp size", np.shape(gnss.t))
# print("LIDAR position data size", np.shape(lidar.data))
# print("LIDAR position timestamp size", np.shape(lidar.t))

################################################################################################
# Let's plot the ground truth trajectory to see what it looks like. When you're testing your
# code later, feel free to comment this out.
################################################################################################
# gt_fig = plt.figure()
# ax = gt_fig.add_subplot(111, projection='3d')
# ax.plot(gt.p[:,0], gt.p[:,1], gt.p[:,2])
# ax.set_xlabel('x [m]')
# ax.set_ylabel('y [m]')
# ax.set_zlabel('z [m]')
# ax.set_title('Ground Truth trajectory')
# ax.set_zlim(-1, 5)
# plt.show()

################################################################################################
# Remember that our LIDAR data is actually just a set of positions estimated from a separate
# scan-matching system, so we can insert it into our solver as another position measurement,
# just as we do for GNSS. However, the LIDAR frame is not the same as the frame shared by the
# IMU and the GNSS. To remedy this, we transform the LIDAR data to the IMU frame using our
# known extrinsic calibration rotation matrix C_li and translation vector t_i_li.
#
# THIS IS THE CODE YOU WILL MODIFY FOR PART 2 OF THE ASSIGNMENT.
################################################################################################
# Correct calibration rotation matrix, corresponding to Euler RPY angles (0.05, 0.05, 0.1).
C_li = np.array([
   [ 0.99376, -0.09722,  0.05466],
   [ 0.09971,  0.99401, -0.04475],
   [-0.04998,  0.04992,  0.9975 ]
])

# Incorrect calibration rotation matrix, corresponding to Euler RPY angles (0.05, 0.05, 0.05).
# C_li = np.array([
#      [ 0.9975 , -0.04742,  0.05235],
#      [ 0.04992,  0.99763, -0.04742],
#      [-0.04998,  0.04992,  0.9975 ]
# ])

t_i_li = np.array([0.5, 0.1, 0.5])

# Transform from the LIDAR frame to the vehicle (IMU) frame.
lidar.data = (C_li @ lidar.data.T).T + t_i_li # @ is the matrix multiplication

#### 2. Constants ##############################################################################

################################################################################################
# Now that our data is set up, we can start getting things ready for our solver. One of the
# most important aspects of a filter is setting the estimated sensor variances correctly.
# We set the values here.
################################################################################################
#Original
# var_imu_f = 0.10 # imu acceleration variance
# var_imu_w = 0.25 # imu gyroscopic/rotational velocity variance
# var_gnss  = 0.01 # gnss variance
# var_lidar = 1.00 # lidar variance

#Part 1
var_imu_f = 0.1 # imu acceleration variance
var_imu_w = 0.2 # imu gyroscopic/rotational velocity variance
var_gnss  = 0.01 # gnss variance
var_lidar = 0.25 # lidar variances

#Part 2
# var_imu_f = 0.05 # imu acceleration variance
# var_imu_w = 0.1 # imu gyroscopic/rotational velocity variance
# var_gnss  = 0.008 # gnss variance
# var_lidar = 15 # lidar variances

#Part 3
# var_imu_f = 0.01 # imu acceleration variance
# var_imu_w = 0.025 # imu gyroscopic/rotational velocity variance
# var_gnss  = 5 # gnss variance
# var_lidar = 1 # lidar variances


################################################################################################
# We can also set up some constants that won't change for any iteration of our solver.
################################################################################################
g = np.array([0, 0, -9.81])  # gravity
l_jac = np.zeros([9, 6])  # We have 9 states combined with imu specific force and imu rotational vel data
l_jac[3:, :] = np.eye(6)  # motion model noise jacobian
h_jac = np.zeros([3, 9])
h_jac[:, :3] = np.eye(3)  # measurement model jacobian. This matrix has been assigned identity elements like this because we only have position measurements (x,y and z)
                          # H_jac is a 3*3 matrix because we have 3-D measurements combined with 9 states


'''if your function takes a three-dimensional vector and
spits out a two-dimensional vector, the Jacobian would be
a two by three matrix. Intuitively, the Jacobian matrix
tells you how fast each output of your function is changing
along each input dimension, just like how the derivative
of a scalar function tells you how fast the output is
changing as you vary the input.
no of variables to be differentiated is the number of columns for a jacobian'''
#### 3. Initial Values #########################################################################

################################################################################################
# Let's set up some initial values for our ES-EKF solver.
################################################################################################
p_est = np.zeros([imu_f.data.shape[0], 3])  # position estimates
v_est = np.zeros([imu_f.data.shape[0], 3])  # velocity estimates
q_est = np.zeros([imu_f.data.shape[0], 4])  # orientation estimates as quaternions
p_cov = np.zeros([imu_f.data.shape[0], 9, 9])  # covariance matrices at each timestep
# print(np.shape(p_est))

# Set initial values.
p_est[0] = gt.p[0]
v_est[0] = gt.v[0]
q_est[0] = Quaternion(euler=gt.r[0]).to_numpy()
p_cov[0] = np.zeros(9)  # covariance of estimate
gnss_i  = 0 # a counter that will be used to check the availability of gnss measurement
lidar_i = 0 # a counter that will be used to check the availability of lidar measurement

#### 4. Measurement Update #####################################################################

################################################################################################
# Since we'll need a measurement update for both the GNSS and the LIDAR data, let's make
# a function for it.
################################################################################################
def measurement_update(sensor_var, p_cov_check, y_k, p_check, v_check, q_check):

    # 3.1 Compute Kalman Gain
    K =  p_cov_check@h_jac.T@(np.linalg.inv(h_jac@p_cov_check@h_jac.T + sensor_var * np.eye(3))) # (9*9).(9*3) yields a (9*3) vector and np.inv((3*9).(9*9).(9*3)+(3*9).(9*3)) yields a (3*3) vector
    # print(np.shape(K))                                                          # The Kalman gain is (9*3) matrix
    # print(np.shape(p_check))
    # print(np.shape(y_k))                                                      #Both of the above are a vector with 3 elements

    # 3.2 Compute error state
    del_xk = K@(y_k-p_check)                                                    #(9*3).(3*1) yields a (9*1) vector
    del_pk = del_xk[:3]                                                         #Assigning x,y and z position
    del_vk = del_xk[3:6]                                                        #Assigning x,y and z velocity
    del_qk = del_xk[6:]                                                         #Assigning x,y and z orientation

    # 3.3 Correct predicted state
    p_hat = p_check + del_pk
    v_hat = v_check + del_vk
    q_hat = Quaternion(euler=del_qk).quat_mult_right(q_check)                   #Pass the euler angles to convert them to quaternion and then do a quaternion multiplication

    # 3.4 Compute corrected covariance
    p_cov_hat = (np.eye(9) - K.dot(h_jac)).dot(p_cov_check)                     #((9*9) - (9*3).(9*9)).(9*9) yields a (9*9) matrix
    # print(np.shape(p_cov_hat))
    return p_hat, v_hat, q_hat, p_cov_hat

#### 5. Main Filter Loop #######################################################################

################################################################################################
# Now that everything is set up, we can start taking in the sensor data and creating estimates
# for our state in a loop.
################################################################################################

for k in range(1, imu_f.data.shape[0]):  # start at 1 b/c we have initial prediction from gt

    delta_t = imu_f.t[k] - imu_f.t[k - 1]
    Cns = Quaternion(*q_est[k-1]).to_mat()                                      # Here we are passing the quaternion as a tuple which consists of (w,x,y,z) to
                                                                                #the 'Quaternion' class and then using the rotation function to_mat.
                                                                                # The rot function uses the formula for Rotating quaternions given in C2M4L1

    # 1. Update state with IMU inputs
    p_est[k] = p_est[k-1] + delta_t*v_est[k-1] + (delta_t**2/2)*(Cns@imu_f.data[k-1] + g) #Update position
    v_est[k] = v_est[k-1] + delta_t*(Cns@imu_f.data[k-1] + g)                             #Update velocity
    # q_est[k] = Quaternion(euler = imu_w.data[k-1]*delta_t).quat_mult_right(q_est[k-1])
    q_est[k] = Quaternion(euler = imu_w.data[k-1]*delta_t).quat_mult_right(q_est[k-1])    #Update orientation

    # 1.1 Linearize the motion model and compute Jacobians
    F = np.eye(9)                                                               # Linearizing motion model jacobian. It has 9 states and thus there are 9 differentiables
    F[0:3,3:6] = delta_t*np.eye(3)                                              # Based on the position jacobian when differentiated w.r.t velocities
    # F[3:6,6:9] = -skew_symmetric(Cns@imu_f.data[k-1])*delta_t                 # This representation is wrong
    F[3:6, 6:] = -(Cns.dot(skew_symmetric(imu_f.data[k-1].reshape((3,1)))))*delta_t            #relation between velocity and orientation
    Q = np.eye(6)                                                               #IMU has 6 inputs of specific force and rotational velocity respectively
    Q[0:3,0:3] *= delta_t**2*var_imu_f                                          #Assignning specific force variance
    Q[3:6,3:6] *= delta_t**2*var_imu_w                                          #Assigning rotational velocity variance

    # 2. Propagate uncertainty/predicted error covariance
    p_cov[k] = F@p_cov[k-1]@F.T + l_jac@Q@l_jac.T                               #(9*9).(9*9).(9*9) + (9*6).(6*6).(6*9) eventually yields a 1*9*9 vector
    # print(np.shape(p_cov))

    # 3. Check availability of GNSS and LIDAR measurements
    # Check 1: If both lidar and gnss data are available and their counters are less than the maximum number of measurements available
    if (lidar_i < lidar.t.shape[0] and gnss_i < gnss.t.shape[0] and gnss.t[gnss_i]==imu_f.t[k-1] and lidar.t[lidar_i]==imu_f.t[k-1]):
        print("Both lidar and gnss measurements are available")
        y_k = gnss.data[gnss_i]                                                 #assign gnss measurements
        sensor_var = var_gnss                                                   #assign gnss variance (x,y and z positions)
        p_est[k], v_est[k], q_est[k], p_cov[k] = measurement_update(sensor_var,p_cov[k],y_k,p_est[k],v_est[k],q_est[k])

        y_k = lidar.data[lidar_i]                                               #assign lidar measurements
        sensor_var = var_lidar                                                  #assign lidar variance (x,y and z positions)
        p_est[k], v_est[k], q_est[k], p_cov[k] = measurement_update(sensor_var,p_cov[k],y_k,p_est[k],v_est[k],q_est[k])

        gnss_i+=1                                                               #updating the measurement loop counter
        lidar_i+=1

    #Check 2 : Only when gnss measurements are available
    elif (gnss_i < gnss.t.shape[0] and gnss.t[gnss_i]==imu_f.t[k-1]):
        y_k = gnss.data[gnss_i]
        sensor_var = var_gnss
        p_est[k], v_est[k], q_est[k], p_cov[k] = measurement_update(sensor_var,p_cov[k],y_k,p_est[k],v_est[k],q_est[k])
        gnss_i+=1

    #Check 3 : Onle when lidar measurements are available
    elif  (lidar_i < lidar.t.shape[0] and lidar.t[lidar_i]==imu_f.t[k-1]):
        y_k = lidar.data[lidar_i]
        sensor_var = var_lidar
        p_est[k], v_est[k], q_est[k], p_cov[k] = measurement_update(sensor_var,p_cov[k],y_k,p_est[k],v_est[k],q_est[k])
        lidar_i+=1

    # Update states (save) P.S - Not a very required step
    p_est[k,:] = p_est[k]
    v_est[k,:] = v_est[k]
    q_est[k,:] = q_est[k]
    p_cov[k,:,:] = p_cov[k]

print("Measurement update count using gnss", gnss_i)
print("Measurement update count using lidar",lidar_i)
errors = np.sqrt((gt.p[:,:] - p_est[len(gt.p),:])**2)
print(np.average(errors[:,0]))
print(np.average(errors[:,1]))
print(np.average(errors[:,2]))

# def optimize_var():
#     var_gnss = 0.01
#     var_lidar = 0.1
#
#     # self.var_imu_f = 0.3
#     # self.var_imu_w = 6.0
#
#     var_1_inc = 0.02 # Size of the Increment of the var_gnss on every step
#     var_1_num = 10 # Num of steps to try.
#
#     var_2_inc = 0.1 # Size of the Increment of the var_lidar on every step
#     var_2_num = 10 # Num of steps to try.
#
#     size = (var_1_num, var_2_num)
#     avg_sqr_errors = np.zeros(size)
#
#     for i in range(0, var_1_num):
#         var_gnss += var_1_inc * i
#         for j in range(0, var_2_num):
#             var_lidar += var_2_inc * i
#             prediction_function() # here you run you ekf main loop
#
#             avg_sqr_errors[i][j]  = calculate_error()
#
#     best_idxs = np.argwhere(avg_sqr_errors == np.min(avg_sqr_errors))
#
#     best_var_1_idx = best_idxs[0][0]
#     best_var_2_idx = best_idxs[0][1]
#
#     print("best var_1, best var_2:", best_var_1_idx*var_1_inc + self.var_imu_f , best_var_2_idx*var_2_inc +  self.var_imu_w)
#
# def calculate_error():
#     num_gt = self.gt.p.shape[0]
#     p_est_euler = []
#     size = (num_gt,6)
#     errors = np.zeros(size)
#
#     # Convert estimated quaternions to euler angles
#     for q in self.q_est:
#         p_est_euler.append(Quaternion(*q).to_euler())
#     p_est_euler = np.array(p_est_euler)
#
#     # Get uncertainty estimates from P matrix
#     #p_cov_diag_std = np.sqrt(np.diagonal(self.p_cov, axis1=1, axis2=2))
#
#     errors[:,:3] = (self.gt.p - self.p_est[:num_gt])**2
#     errors[:,3:6] = (self.gt.p - p_est_euler[:num_gt])**2
#     avg_sqr_error = np.average(errors)
#
#     return avg_sqr_error


#### 6. Results and Analysis ###################################################################

################################################################################################
# Now that we have state estimates for all of our sensor data, let's plot the results. This plot
# will show the ground truth and the estimated trajectories on the same plot. Notice that the
# estimated trajectory continues past the ground truth. This is because we will be evaluating
# your estimated poses from the part of the trajectory where you don't have ground truth!
################################################################################################
est_traj_fig = plt.figure()
ax = est_traj_fig.add_subplot(111, projection='3d')
ax.plot(p_est[:,0], p_est[:,1], p_est[:,2], label='Estimated')
ax.plot(gt.p[:,0], gt.p[:,1], gt.p[:,2], label='Ground Truth')
ax.set_xlabel('Easting [m]')
ax.set_ylabel('Northing [m]')
ax.set_zlabel('Up [m]')
ax.set_title('Ground Truth and Estimated Trajectory')
ax.set_xlim(0, 200)
ax.set_ylim(0, 200)
ax.set_zlim(-2, 2)
ax.set_xticks([0, 50, 100, 150, 200])
ax.set_yticks([0, 50, 100, 150, 200])
ax.set_zticks([-2, -1, 0, 1, 2])
ax.legend(loc=(0.62,0.77))
ax.view_init(elev=45, azim=-50)
plt.show()

################################################################################################
# We can also plot the error for each of the 6 DOF, with estimates for our uncertainty
# included. The error estimates are in blue, and the uncertainty bounds are red and dashed.
# The uncertainty bounds are +/- 3 standard deviations based on our uncertainty (covariance).
################################################################################################
error_fig, ax = plt.subplots(2, 3)
error_fig.suptitle('Error Plots')
num_gt = gt.p.shape[0]
p_est_euler = []
p_cov_euler_std = []

# Convert estimated quaternions to euler angles
for i in range(len(q_est)):
    qc = Quaternion(*q_est[i, :])
    p_est_euler.append(qc.to_euler())

    # First-order approximation of RPY covariance
    J = rpy_jacobian_axis_angle(qc.to_axis_angle())
    p_cov_euler_std.append(np.sqrt(np.diagonal(J @ p_cov[i, 6:, 6:] @ J.T)))

p_est_euler = np.array(p_est_euler)
p_cov_euler_std = np.array(p_cov_euler_std)

# Get uncertainty estimates from P matrix
p_cov_std = np.sqrt(np.diagonal(p_cov[:, :6, :6], axis1=1, axis2=2))

titles = ['Easting', 'Northing', 'Up', 'Roll', 'Pitch', 'Yaw']
for i in range(3):
    ax[0, i].plot(range(num_gt), gt.p[:, i] - p_est[:num_gt, i])
    ax[0, i].plot(range(num_gt),  3 * p_cov_std[:num_gt, i], 'r--')
    ax[0, i].plot(range(num_gt), -3 * p_cov_std[:num_gt, i], 'r--')
    ax[0, i].set_title(titles[i])
ax[0,0].set_ylabel('Meters')

for i in range(3):
    ax[1, i].plot(range(num_gt), \
        angle_normalize(gt.r[:, i] - p_est_euler[:num_gt, i]))
    ax[1, i].plot(range(num_gt),  3 * p_cov_euler_std[:num_gt, i], 'r--')
    ax[1, i].plot(range(num_gt), -3 * p_cov_euler_std[:num_gt, i], 'r--')
    ax[1, i].set_title(titles[i+3])
ax[1,0].set_ylabel('Radians')
plt.show()

#### 7. Submission #############################################################################

################################################################################################
# Now we can prepare your results for submission to the Coursera platform. Uncomment the
# corresponding lines to prepare a file that will save your position estimates in a format
# that corresponds to what we're expecting on Coursera.
################################################################################################

# Pt. 1 submission
p1_indices = [9000, 9400, 9800, 10200, 10600]
p1_str = ''
for val in p1_indices:
    for i in range(3):
        p1_str += '%.3f ' % (p_est[val, i])
with open('pt1_submission.txt', 'w') as file:
    file.write(p1_str)

# Pt. 2 submission
# p2_indices = [9000, 9400, 9800, 10200, 10600]
# p2_str = ''
# for val in p2_indices:
#     for i in range(3):
#         p2_str += '%.3f ' % (p_est[val, i])
# with open('pt2_submission.txt', 'w') as file:
#     file.write(p2_str)

# Pt. 3 submission
# p3_indices = [6800, 7600, 8400, 9200, 10000]
# p3_str = ''
# for val in p3_indices:
#     for i in range(3):
#         p3_str += '%.3f ' % (p_est[val, i])
# with open('pt3_submission.txt', 'w') as file:
#     file.write(p3_str)
