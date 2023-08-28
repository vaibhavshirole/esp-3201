import numpy as np
import math
import random
import scipy.stats

def initialize_particles(num_particles, map_limits):
    # randomly initialize the particles inside the map limits
    particles = []

    for i in range(num_particles):
        particle = dict()

        # draw x,y and theta coordinate from uniform distribution
        # inside map limits
        particle['x'] = np.random.uniform(map_limits[0], map_limits[1])
        particle['y'] = np.random.uniform(map_limits[2], map_limits[3])
        particle['theta'] = np.random.uniform(-np.pi, np.pi)

        particles.append(particle)

    return particles #each particle has an x,y,theta


def mean_pose(particles):
    # calculate the mean pose of a particle set.
    #
    # for x and y, the mean position is the mean of the particle coordinates
    #
    # for theta, we cannot simply average the angles because of the wraparound 
    # (jump from -pi to pi). Therefore, we generate unit vectors from the 
    # angles and calculate the angle of their average 

    # save x and y coordinates of particles
    xs = []
    ys = []

    # save unit vectors corresponding to particle orientations 
    vxs_theta = []
    vys_theta = []

    for particle in particles:
        xs.append(particle['x'])
        ys.append(particle['y'])

        #make unit vector from particle orientation
        vxs_theta.append(np.cos(particle['theta']))
        vys_theta.append(np.sin(particle['theta']))

    #calculate average coordinates
    mean_x = np.mean(xs)
    mean_y = np.mean(ys)
    mean_theta = np.arctan2(np.mean(vys_theta), np.mean(vxs_theta))

    return [mean_x, mean_y, mean_theta]

def sample_motion_model(odometry, particles):
    # Samples new particle positions, based on old positions, the odometry
    # measurements and the motion noise 

    delta_rot1 = odometry['r1']
    delta_trans = odometry['t']
    delta_rot2 = odometry['r2']

    # the motion noise parameters: [alpha1, alpha2, alpha3, alpha4]
    noise = [0.1, 0.1, 0.05, 0.05]
    
    # standard deviations of motion noise
    sigma_delta_rot1 = noise[0] * abs(delta_rot1) + noise[1] * delta_trans
    sigma_delta_trans = noise[2] * delta_trans + noise[3] * (abs(delta_rot1) + abs(delta_rot2))
    sigma_delta_rot2 = noise[0] * abs(delta_rot2) + noise[1] * delta_trans
    
    # generate new particle set after motion update
    new_particles = []

    for particle in particles:
        new_particle = dict()
        #sample noisy motions
        noisy_delta_rot1 = delta_rot1 + np.random.normal(0, sigma_delta_rot1)
        noisy_delta_trans = delta_trans + np.random.normal(0, sigma_delta_trans)
        noisy_delta_rot2 = delta_rot2 + np.random.normal(0, sigma_delta_rot2)
        
        #calculate new particle pose
        new_particle['x'] = particle['x'] + \
            noisy_delta_trans * np.cos(particle['theta'] + noisy_delta_rot1)
        new_particle['y'] = particle['y'] + \
            noisy_delta_trans * np.sin(particle['theta'] + noisy_delta_rot1)
        new_particle['theta'] = particle['theta'] + \
            noisy_delta_rot1 + noisy_delta_rot2
        new_particles.append(new_particle)
    return new_particles


def eval_sensor_model(sensor_data, particles, landmarks):
    # Computes the observation likelihood of all particles, given the
    # particle and landmark positions and sensor measurements
    #
    # The employed sensor model is range only.

    #sensor data ---- array of distance from each particle
    #particles ---- x, y, theta
    #landmarks ---- dict size 4 of x, y

    sigma_r = 0.2

    #measured landmark ids and ranges
    ids = sensor_data['id']
    ranges = sensor_data['range']

    weights = []
    
    '''your code here'''

    #print(landmarks)
    
    for particle in particles:
        #store position of measured sensor readings for particle
        p_x = particle['x']
        p_y = particle['y']
        d = []

        sum_error = 0   #add to array of weights after checking goodness of particle
        num_landmarks = len(landmarks)

        for i in range(num_landmarks):
            landmark = landmarks.get(i+1)   # gets landmark and skips the one at index 0
            #print(landmark)

            #store position of the landmark
            l_x = landmark[0]
            l_y = landmark[1]

            compare_particle = [p_x, p_y]
            compare_landmark = [l_x, l_y]

            #compare position of landmarks with measured sensor readings  
            dist = math.dist(compare_particle, compare_landmark)
            d.append(dist)

            #calculate weight based on pos of landmarks  
            x = sensor_data['range'][i]
            # print((sensor_data)
            miu = dist
            sigma_r_squared = sigma_r**2

            cur_error = (1 / (math.sqrt(2 * math.pi * sigma_r_squared))) * math.exp(-((x - miu)**2) / (2 * sigma_r_squared)) #eqn
            sum_error += cur_error

        sum_error = sum_error/len(landmarks)
        
        weights.append(sum_error)
    
    '''***        ***'''

    #normalize the weights and return
    normalizer = sum(weights)
    weights = [weights / normalizer for weights in weights] #normalization

    return weights

def resample_particles(particles, weights):
    # Returns a new set of particles obtained by performing
    # stochastic universal sampling, according to the particle weights.

    new_particles = []

    ################################## systematic resampling

    '''your code here'''
    # N = len(particles)
    # new_particles = []

    # #compute cumulative sum of weights
    # cum_sum = [0.0]
    # for weight in weights:
    #     cum_sum.append(cum_sum[-1] + weight)
    # #print("length: ",len(cum_sum))
    # # Initialize variables
    # index = random.uniform(0, 1 / N)
    # j = 0

    # #resampling loop
    # for i in range(N):
    #     #print("index: ", index)
    #     while j < N - 1 and index > cum_sum[j]:
    #         j += 1
    #     new_particles.append(particles[j])
    #     index += 1 / N

    # #normalize weights
    # total_weight = sum(weights)
    # normalized_weights = [weight / total_weight for weight in weights]

    ################################## monte carlo resampling

    N = len(particles)
    new_particles = []

    #normalize weights
    total_weight = sum(weights)
    normalized_weights = [weight / total_weight for weight in weights]

    #create cumulative distribution
    cumulative_distribution = [0.0]
    for weight in normalized_weights:
        cumulative_distribution.append(cumulative_distribution[-1] + weight)

    #resampling loop
    for i in range(N):
        random_value = random.uniform(0, 1)
        for j in range(N):
            if random_value <= cumulative_distribution[j+1]:
                new_particles.append(particles[j])
                break

    '''***        ***'''

    return new_particles
