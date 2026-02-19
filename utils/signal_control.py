import numpy as np
import libsumo as traci
from utils.sig_config import sig_configs

cell_length = 7
detection_length = 350
n_channels = 2
height = int(detection_length / cell_length)
car_occupancy = 1.0
bus_occupancy = 1.0
# min_left_green_time = 5
# min_through_green_time = 12  # Used in DQN or Double-DQN to speed up the training process
yellow_time = 3
all_red_time = 2
min_green_time = 10
max_green_time = 50


class Signal:
    """(Not implemented) THis is called in multi-agents scenarios for controlling single signal.
    :param map_name: the map this signal belongs to.
    :param sig_id: the signal to be controlled.
    :param obs_type: Sets the output type ('img': image, 'vec': vector, 'comb': combined)
    for observations in the scenarios.
    :param cv_det: Controls whether only the CV can be detected.
    """
    def __init__(self, map_name, sig_id, obs_type='img', cv_det=False):
        self.sig_configs = sig_configs[map_name]
        self.sig_id = sig_id
        self.phase_number = len(self.sig_configs['phase_pairs'])
        self.phase_state_len = len(self.sig_configs['action_phase_map'][0])
        self.incoming_edges = self.sig_configs[sig_id]['incoming_edges']
        self.num_lanes = self.sig_configs[sig_id]['number_lanes']
        print(self.num_lanes)
        self.sim_step = 0
        self.obs_type = obs_type
        self.cv_det = cv_det

    def act(self, sig_id, action):
        # Take the action: Signal control
        # current_action = action
        self.set_next_phase(sig_id, action)
        # self.simulate()

        # Get the info after taking the action
        observation, current_tot_person_delay = self.observe(sig_id)
        reward = self.last_tot_person_delay - current_tot_person_delay

        # Update the last action and total person delay for the next step
        self.last_tot_person_delay = current_tot_person_delay

        self.ep_reward += reward
        # terminated = False
        if self.sim_step > 4400:
            self.done = True
            self.terminated = True

        return observation, reward, self.terminated, self.done, {}

    def observe(self, agent):
        img_observation = np.zeros((n_channels, height, self.num_lanes))
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
            lane_index = int(ln_idx) + self.incoming_edges[edge]

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