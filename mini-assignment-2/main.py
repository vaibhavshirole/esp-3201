import numpy as np
import matplotlib.pyplot as plt

def main():
    # read in the file

    # initialize kalman stuff for doing math

    # loop through measurements from csv
        # take measurement value and create prediction  -- prediction step
        # update & feed back predictions                -- update step
        # save into array to plot it

    # plot ground truth
    # plot array of predictions

#   ==============================================================================================
#   ==============================================================================================

    ## Define parts used for calculations

    # define constant velocity model parameters
    dt = 1  # Time step between measurements
    velocity = 12.5  # Constant velocity in m/s

    # set initial state
    with open('MeasGT.csv', 'r') as file:
        # Read the first line to get the initial measurement
        initial_data = file.readline().strip().split(',')
        initial_state = np.array([float(initial_data[1]), 
                                  float(initial_data[2]), 
                                  velocity              , 
                                  velocity              ])  # Use the first measurement
    state_estimate = initial_state

    # set P matrix
    state_covariance = np.array([[10**2, 0, 0, 0],
                                 [0, 10**2, 0, 0],
                                 [0, 0, ((velocity/3)**2), 0],
                                 [0, 0, 0, ((velocity/3)**2)]]) # Initial state covariance identity matrix (P matrix)
    
    # set Q matrix
    process_noise_cov = np.array([[0.01, 0, 0, 0],
                                  [0, 0.01, 0, 0],
                                  [0, 0, 0.01, 0],
                                  [0, 0, 0, 0.01]]) # Process noise covariance identity matrix

    # set R matrix
    measurement_noise_cov = np.array([[10**2, 0],
                                    [0, 10**2]])  # Measurement noise covariance (R matrix)

    # set K matrix 
    kalman_gain = np.zeros((4, 2))  #kalman gain matrix

    # lists to store ground truth and predicted values
    ground_truth_x = []
    ground_truth_y = []
    predicted_x = []
    predicted_y = []
    measured_x = []
    measured_y = []

    # Initialize RMSE variables
    rmse_x = 0
    rmse_y = 0

    ## The final state_estimate contains the filtered position and velocity information

    # read the CSV file and iterate through measurements
    with open('MeasGT.csv', 'r') as file:
        for line in file:
            data = line.strip().split(',')
            time = float(data[0])
            measurement = np.array([float(data[1]), float(data[2])])  # create [x,y] of the measurement data from csv

            # PREDICTION STEP
            A = np.array([[1, 0, dt, 0],
                          [0, 1, 0, dt],
                          [0, 0, 1, 0],
                          [0, 0, 0, 1]])
            #eqn1
            state_estimate = np.dot(A, state_estimate) # can assume G_(k-1)*u_(k-1) is 0 for this assignment

            #eqn2
            state_covariance = np.dot(np.dot(A, state_covariance), A.T) + process_noise_cov

            # UPDATE STEP
            H = np.array([[1, 0, 0, 0],
                          [0, 1, 0, 0]])
            
            # get the innovation
            measurement_residual = measurement - np.dot(H, state_estimate)

            #eqn2a
            measurement_residual_cov = np.dot(np.dot(H, state_covariance), H.T) + measurement_noise_cov #bottom part of eqn2
            kalman_gain = np.dot(np.dot(state_covariance, H.T), np.linalg.inv(measurement_residual_cov))

            #eqn2b
            state_estimate = state_estimate + np.dot(kalman_gain, measurement_residual)
            state_covariance = np.dot(np.eye(4) - np.dot(kalman_gain, H), state_covariance)

            # store ground truth values and predicted values
            ground_truth_x.append(float(data[3]))
            ground_truth_y.append(float(data[4]))
            predicted_x.append(state_estimate[0])
            predicted_y.append(state_estimate[1])
            measured_x.append(measurement[0])
            measured_y.append(measurement[1])

            # calculate RMSE for x and y after each correction step
            rmse_x += (float(data[3]) - state_estimate[0])**2
            rmse_y += (float(data[4]) - state_estimate[1])**2

            # print data at the current predicted (x, y) coordinates
            current_predicted_x = state_estimate[0]
            current_predicted_y = state_estimate[1]
            print(f"Predicted x,y: ({current_predicted_x:.2f} m, {current_predicted_y:.2f} m)")
            print(f"Measured x,y: ({measurement[0]:.2f} m, {measurement[1]:.2f} m)")
            print(f"Ground truth x,y: ({float(data[3])} m, {float(data[4])} m)")
            print()
    
    # calculate and print the final RMSE
    rmse_x = np.sqrt(rmse_x / len(ground_truth_x))
    rmse_y = np.sqrt(rmse_y / len(ground_truth_y))
    print(f"RMSE for x: {rmse_x:.2f} m")
    print(f"RMSE for y: {rmse_y:.2f} m")

    # plot ground truth and predicted values
    plt.figure(figsize=(10, 6))
    plt.plot(ground_truth_x, ground_truth_y, label='Ground Truth', marker='o', linestyle='-', color='green')
    plt.plot(predicted_x, predicted_y, label='Predicted', marker='x', linestyle='-', color='blue')
    plt.plot(measured_x, measured_y, label='Measured', marker='.', linestyle='', color='red')
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.title(f'Kalman Filter Tracking - RMSE for (x,y): {rmse_x:.2f} m, {rmse_y:.2f} m ')
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    main()