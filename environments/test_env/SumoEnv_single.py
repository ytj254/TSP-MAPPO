import gymnasium as gym
from gymnasium import spaces

import timeit

import numpy as np
# import traci
import libsumo as traci
import traci.constants as tc

cell_length = 7
detection_length = 350
n_channels = 2
width = 16
height = int(detection_length / cell_length)
car_occupancy = 1.0
bus_occupancy = 1.0
yellow_time = 3
all_red_time = 2
min_green_time = 10
max_green_time = 50

# edges = {
#     'east_edge': (0, '-E2'),
#     'south_edge': (4, '-E3'),
#     'west_edge': (8, 'E0'),
#     'north_edge': (12, 'E1')
# }
incoming_edges = {'E1': 0, '-E2': 4, '-E3': 8, 'E0': 12}

action_state_map = {
    0: 'grrrgrrGGgrrrgrrGG',  # WL EL
    1: 'grrrgrrrrgrrrgGGGG',  # WL WT
    2: 'grrrgGGrrgrrrgGGrr',  # WT ET
    3: 'grrrgGGGGgrrrgrrrr',  # EL ET
    4: 'grrGgrrrrgrrGgrrrr',  # SL NL
    5: 'grrrgrrrrgGGGgrrrr',  # SL ST
    6: 'gGGrgrrrrgGGrgrrrr',  # ST NT
    7: 'gGGGgrrrrgrrrgrrrr'  # NL NT
}


class SumoEnv(gym.Env):
    """Custom Environment that follows gym interface
    :param sumo_cmd: The command for the sumo.
    :param obs_type: Sets the output type ('img': image, 'vec': vector, 'comb': combined)
    for observations in the scenarios.
    :param cv_only: Controls whether only the CV can be detected.
    """

    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, sumo_cmd, obs_type='img', cv_only=False):
        super(SumoEnv, self).__init__()
        self.last_tot_person_delay = None
        self.time_since_last_phase_switch = None
        self.green_state = None
        self.last_tot_person_delays = None
        self.ep_step = None
        self.sim_step = None
        self.start_time = None
        self.ep_reward = None

        # Set map configs
        self.phase_number = len(action_state_map)
        self.phase_state_len = len(action_state_map[0])  # Count the number of lanes
        # Define action and observation space
        # They must be gym.spaces objects
        # Initiate action space:
        self.action_space = spaces.Discrete(8)

        # Initiate observation space:
        self.obs_type = obs_type

        if self.obs_type == 'comb':
            self.observation_space = spaces.Dict(
                {
                    'img': spaces.Box(low=0, high=1, shape=(n_channels, height, width), dtype=np.float64),
                    'vec': spaces.Box(low=0, high=1, shape=(width,), dtype=np.float64),
                }
            )

        elif self.obs_type == 'vec':
            self.observation_space = spaces.Box(low=0, high=1, shape=(width,), dtype=np.float64)

        else:
            self.observation_space = spaces.Box(low=0, high=1, shape=(n_channels, height, width), dtype=np.float64)

        self.cv_det = cv_only
        self.episode = 0
        self.total_rewards = []
        self.sumo_cmd = sumo_cmd

    def reset(self, seed=None, options=None):
        self.episode += 1

        self.done = False
        self.terminated = False
        self.ep_reward = 0
        self.start_time = timeit.default_timer()
        self.sim_step = 0
        self.ep_step = 0

        # Reset signal control related variables
        self.green_state = {'J1': action_state_map[0]}
        self.time_since_last_phase_switch = {'J1': 0}

        # print(f'---Episode: {self.episode}--- Simulating...')
        traci.start(self.sumo_cmd)

        # Warm up 10 minutes
        while self.sim_step <= 600:
            if self.sim_step == 600:
                observation, self.last_tot_person_delay = self.observe('J1')
                return observation, {}
            traci.simulationStep()
            self.sim_step += 1
        # print(self.last_state)

    def step(self, action):
        # Take the action: Signal control
        # current_action = action
        self.set_next_phase('J1', action)
        self.simulate()

        # Get the info after taking the action
        observation, current_tot_person_delay = self.observe('J1')
        reward = self.last_tot_person_delay - current_tot_person_delay

        # Update the last action and total person delay for the next step
        self.last_tot_person_delay = current_tot_person_delay

        self.ep_reward += reward
        # terminated = False
        if self.sim_step > 4400:
            self.done = True
            self.terminated = True
            # print(f'Episode: {self.episode}---Total Steps: {self.ep_step}---Total Sim Steps: {self.sim_step}')
            # simulation_time = round(timeit.default_timer() - self.start_time, 1)
            # info.update({'Simulation_time': simulation_time})
            # print(f'Simulation time: {simulation_time} seconds -- '
            #       f'Total reward: {self.ep_reward} -- ')
            # traci.close()
        self.ep_step += 1

        return observation, reward, self.terminated, self.done, {'ep_step': self.ep_step}

    def render(self):
        pass

    def close(self):
        traci.close()

    def observe(self, agent):
        img_observation = np.zeros((n_channels, height, width))
        # print(img_observation.dtype)
        tot_person_delay = 0
        veh_set = []
        detect_veh_set = {}
        lanes = traci.trafficlight.getControlledLanes(agent)

        # Get vehicle id list
        for lane in lanes:
            veh_set += traci.lane.getLastStepVehicleIDs(lane)

        # Identify the set of detected vehicles
        for veh in veh_set:
            veh_type = traci.vehicle.getTypeID(veh)
            if not self.cv_det:
                detect_veh_set[veh] = veh_type
            else:
                if veh_type == 'cv' or veh_type == 'bus':
                    detect_veh_set[veh] = veh_type
            # print(len(detect_veh_set))

        # Output image-like observation and calculate total person delay
        for veh, veh_type in detect_veh_set.items():
            veh_next_tls = traci.vehicle.getNextTLS(veh)
            # print(veh_next_tls)
            distance_to_tls = veh_next_tls[0][2]  # get the distance to the corresponding signal

            # get the lane index
            # lane_id = traci.vehicle.getLaneID(veh)
            edge, ln_idx = traci.vehicle.getLaneID(veh).split('_')
            # if edge in incoming_edges.keys():
            lane_index = int(ln_idx) + incoming_edges[edge]

            # Get the speed and normalize it
            spd = traci.vehicle.getSpeed(veh) / 30  # Suppose the max speed is 50 mph

            if distance_to_tls > 0:  # vehicle not crossing the stop line
                delay = traci.vehicle.getTimeLoss(veh)
            else:  # vehicle already crossing the stop line
                delay = 0

            # get the vehicle type and assign the occupancy
            if veh_type == 'cv':
                v_occupancy = car_occupancy
            else:
                v_occupancy = bus_occupancy

            person_delay = delay * v_occupancy
            tot_person_delay += person_delay
            # print(tot_person_delay)

            # get the position in state array
            if 0 < distance_to_tls < detection_length:
                height_index = int(distance_to_tls / cell_length)
                # print(lane_index)
                img_observation[:, height_index, lane_index] = (v_occupancy, spd)

        if self.obs_type == 'img':
            observation = img_observation

        else:
            # Count the stopped vehicles on each lane and normalize it, speed <= 0.1
            queue_observation = [
                traci.lane.getLastStepHaltingNumber(lane) /
                (traci.lane.getLength(lane) / (2.5 + traci.lane.getLastStepLength(lane)))  # 2.5 is the gap
                for lane in lanes
            ]

            if self.obs_type == 'comb':
                # # Concatenate the queue array with the image state, after this, the dimension is (3, 50, 16)
                observation = {
                    'img': img_observation,
                    'vec': queue_observation
                }

            else:  # obs_type = vec
                # Divided by the maximum number of vehicles in a lane to normalize
                observation = queue_observation

        return observation, tot_person_delay

    # Execute the designated simulation step
    def simulate(self):
        traci.simulationStep()
        self.sim_step += 1

    def set_next_phase(self, agent, action):
        new_phase_state = action_state_map[action]
        current_state = traci.trafficlight.getRedYellowGreenState(agent)
        if self.time_since_last_phase_switch[agent] < yellow_time:  # 3
            self.time_since_last_phase_switch[agent] += 1
        elif self.time_since_last_phase_switch[agent] < yellow_time + all_red_time:  # 5
            red_state = self.create_red(current_state)
            traci.trafficlight.setRedYellowGreenState(agent, red_state)
            self.time_since_last_phase_switch[agent] += 1
        elif self.time_since_last_phase_switch[agent] < yellow_time + all_red_time + min_green_time:  # 15
            traci.trafficlight.setRedYellowGreenState(agent, self.green_state[agent])
            self.time_since_last_phase_switch[agent] += 1
        else:
            if new_phase_state != self.green_state[agent]:
                yellow_state = self.create_yellow(new_phase_state, current_state)
                traci.trafficlight.setRedYellowGreenState(agent, yellow_state)
                self.time_since_last_phase_switch[agent] = 0
                self.green_state[agent] = new_phase_state
            else:
                self.time_since_last_phase_switch[agent] += 1

    def create_yellow(self, phase_state, current_state):
        yellow_state = []
        for i in range(self.phase_state_len):
            if current_state[i] == 'G' and current_state[i] != phase_state[i]:
                yellow_state.append('Y')
            else:
                yellow_state.append(current_state[i])
        yellow_state = ''.join(yellow_state)
        return yellow_state

    # Create the corresponding red phase state
    @staticmethod
    def create_red(current_state):
        return current_state.replace('Y', 'r')
