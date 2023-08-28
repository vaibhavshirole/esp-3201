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

    # Define constant velocity model parameters
    dt = 1  # Time step between measurements
    velocity = 12.5  # Constant velocity in m/s

    # Initialize Kalman filter parameters
    with open('MeasGT.csv', 'r') as file:
        # Read the first line to get the initial measurement
        initial_data = file.readline().strip().split(',')
        initial_state = np.array([float(initial_data[1]), float(initial_data[2]), velocity, velocity])  # Use the first measurement    state_estimate = initial_state
    state_estimate = initial_state
    #print(state_estimate)
    state_covariance = np.eye(4)  # Initial state covariance
    process_noise_cov = np.eye(4)  # Process noise covariance
    #print(state_covariance)
    measurement_noise_cov = np.array([[10**2, 0],
                                    [0, 10**2]])  # Measurement noise covariance (R)

    # Kalman gain matrix
    kalman_gain = np.zeros((4, 2))

    # Lists to store ground truth and predicted values
    ground_truth_x = []
    ground_truth_y = []
    predicted_x = []
    predicted_y = []


    ## The final state_estimate contains the filtered position and velocity information.

    # Read the CSV file and iterate through measurements
    with open('MeasGT.csv', 'r') as file:
        for line in file:
            data = line.strip().split(',')
            time = float(data[0])
            measurement = np.array([float(data[1]), float(data[2])])  # [x, y]

            # Prediction Step (constant velocity model)
            A = np.array([[1, 0, dt, 0],
                        [0, 1, 0, dt],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])
            state_estimate = np.dot(A, state_estimate)
            state_covariance = np.dot(np.dot(A, state_covariance), A.T) + process_noise_cov

            # Update Step
            H = np.array([[1, 0, 0, 0],
                        [0, 1, 0, 0]])
            measurement_residual = measurement - np.dot(H, state_estimate)
            measurement_residual_cov = np.dot(np.dot(H, state_covariance), H.T) + measurement_noise_cov
            kalman_gain = np.dot(np.dot(state_covariance, H.T), np.linalg.inv(measurement_residual_cov))
            state_estimate = state_estimate + np.dot(kalman_gain, measurement_residual)
            state_covariance = np.dot(np.eye(4) - np.dot(kalman_gain, H), state_covariance)

            # Store ground truth values and predicted values
            ground_truth_x.append(float(data[3]))
            ground_truth_y.append(float(data[4]))
            predicted_x.append(state_estimate[0])
            predicted_y.append(state_estimate[1])

    # Plot ground truth and predicted values
    plt.figure(figsize=(10, 6))
    plt.plot(ground_truth_x, ground_truth_y, label='Ground Truth', marker='o')
    plt.plot(predicted_x, predicted_y, label='Predicted', marker='x')
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.title('Kalman Filter Tracking')
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    main()