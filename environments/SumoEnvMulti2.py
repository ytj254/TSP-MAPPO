import functools
from pettingzoo import ParallelEnv
from gymnasium import spaces
import timeit
import numpy as np
import traci
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


class SumoEnvMulti(ParallelEnv):
    """Custom Environment that follows pettingzoo interface
    :param sumo_cmd: The command for the sumo.
    :param map_name: the map this signal belongs to.
    :param obs_type: Sets the output type ('img': image, 'vec': vector, 'comb': combined)
    for observations in the scenarios.
    :param cv_only: Controls whether only the CV can be detected.
    """

    metadata = {
        'render_modes': ['human', 'rgb_array'],
        'name': 'sumo_env_multi',
    }
    render_mode = 'rgb_array'

    def __init__(self, sumo_cmd, map_name='h_corridor', obs_type='img', cv_only=False):
        # To obey format rules
        self.active_agents = None
        self.time_since_last_phase_switch = None
        self.green_state = None
        self.last_tot_person_delays = None
        self.act_step = None
        self.sim_step = None
        self.start_time = None
        # self.render_mode = render_mode

        # Set map configs
        self.sig_configs = sig_configs[map_name]
        self.sig_ids = self.sig_configs['sig_ids']
        self.phase_number = len(self.sig_configs['phase_pairs'])
        self.action_phase_map = self.sig_configs['action_phase_map']
        self.num_lanes = 12
        self.phase_state_len = len(self.action_phase_map[0])  # Count the length of phase state

        # Set traffic configs
        self.sumo_cmd = sumo_cmd
        self.cv_det = cv_only
        self.episode = 0
        self.possible_agents = self.sig_configs['sig_ids']
        self.agents = self.possible_agents[:]
        self.obs_type = obs_type
        # For rllib ParallelPettingZooEnv
        self.action_spaces = {a: self.action_space(a) for a in self.agents}
        self.observation_spaces = {a: self.observation_space(a) for a in self.agents}

        """
        Note: as of v1.18.1, the action_spaces and observation_spaces attributes are deprecated.
        Spaces should be defined in the action_space() and observation_space() methods.
        self._action_spaces = {a: spaces.Discrete(self.phase_number) for a in self.agents}
        self._observation_spaces = {a: self.observation_space_selection(obs_type) for a in self.agents}
        """

    def reset(self, seed=None, options=None):

        self.agents = self.possible_agents[:]
        self.episode += 1
        self.start_time = timeit.default_timer()
        self.sim_step = 0
        self.act_step = {a: 0 for a in self.agents}

        # Reset signal control related variables
        self.active_agents = {a: True for a in self.agents}
        self.green_state = {a: self.action_phase_map[0] for a in self.agents}
        self.time_since_last_phase_switch = {a: 15 for a in self.agents}

        # print(f'---Episode: {self.episode}--- Simulating...')
        # print(f'-----------self.sumo_cmd: {self.sumo_cmd}--------------')
        traci.start(self.sumo_cmd)
        # print(self.sumo_cmd)

        # Warm up 10 minutes
        while self.sim_step <= 600:
            if self.sim_step == 600:
                observations = {a: self.observe(a)[0] for a in self.agents}
                self.last_tot_person_delays = {a: self.observe(a)[1] for a in self.agents}
                infos = {a: {} for a in self.agents}
                # print(observations[0].dtype)
                return observations, infos
            traci.simulationStep()
            self.sim_step += 1

    def step(self, actions):

        # print(f'-------actions: {actions}--------')
        self.act(actions)
        self.simulate()
        # print(f'-------{self.sim_step}, {self.active_agents}----------')
        observations = {a: self.observe(a)[0] for a in self.agents if self.active_agents[a]}
        rewards = {a: self.last_tot_person_delays[a] - self.observe(a)[1] for a in self.agents if self.active_agents[a]}
        # Update the last action and total person delay for the next step
        self.last_tot_person_delays.update(
            {a: self.observe(a)[1] for a in self.agents if self.active_agents[a]}
        )

        terminations = {agent: False for agent in self.agents}
        truncations = {agent: False for agent in self.agents}

        self.act_step.update(
            {a: self.act_step[a] + 1 for a in self.agents if self.active_agents[a]}
        )
        infos = {}
        # print(f'-------{self.sim_step}: {rewards}----------')
        if self.sim_step > 4400:
            terminations = {a: True for a in self.agents}
            # print(f'Episode: {self.episode}---Total Steps: {self.act_step}---Total Sim Steps: {self.sim_step}')

        return observations, rewards, terminations, truncations, infos

    def render(self):
        pass

    def close(self):
        traci.close()

    def observation_space_selection(self):
        if self.obs_type == 'comb':
            observation_space = spaces.Dict(
                {
                    'img': spaces.Box(low=0, high=1, shape=(n_channels, height, self.num_lanes), dtype=np.float64),
                    'vec': spaces.Box(low=0, high=1, shape=(self.num_lanes,), dtype=np.float64),
                }
            )

        elif self.obs_type == 'vec':
            observation_space = spaces.Box(low=0, high=1, shape=(self.num_lanes,), dtype=np.float64)

        else:
            observation_space = spaces.Box(low=0, high=1,
                                           shape=(n_channels, height, self.num_lanes), dtype=np.float64)
        return observation_space

    # lru_cache allows observation and action spaces to be memoized, reducing clock cycles required to get each
    # agents' space. If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return spaces.Discrete(self.phase_number)

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self.observation_space_selection()

    # Observe the scenarios and output observations and delays
    def observe(self, agent):
        img_observation = np.zeros((n_channels, height, self.num_lanes))
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
            distance_to_tls = veh_next_tls[0][2]  # get the distance to the corresponding signal
            # get the lane index
            edge, ln_idx = traci.vehicle.getLaneID(veh).split('_')
            lane_index = int(ln_idx) + self.sig_configs[agent]['incoming_edges'][edge]

            # spd = traci.vehicle.getSpeed(veh)
            # Get the speed and normalize it
            spd = traci.vehicle.getSpeed(veh) / 30  # Suppose the max speed is 30 m/s

            if distance_to_tls > 0:  # vehicle not crossing the stop line
                delay = traci.vehicle.getTimeLoss(veh)
            else:  # vehicle already crossing the stop line
                delay = 0

            # get the vehicle type and assign the occupancy
            if veh_type == 'DEFAULT_VEHTYPE':
                v_occupancy = car_occupancy
            else:
                v_occupancy = bus_occupancy

            person_delay = delay * v_occupancy
            tot_person_delay += person_delay
            # print(tot_person_delay)

            # get the position in state array
            if 0 < distance_to_tls < detection_length:
                height_index = int(distance_to_tls / cell_length)
                img_observation[:, height_index, lane_index] = (v_occupancy, spd)

        if self.obs_type == 'img':
            observation = img_observation
            # observation = observation.astype(np.uint8)

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
        # print(observation, observation.dtype)
        # print(f'----{agents}: {tot_person_delay}-------')
        return observation, tot_person_delay

    # Execute the designated simulation step
    def simulate(self):
        traci.simulationStep()
        self.sim_step += 1

    def act(self, actions):
        for agent in self.agents:
            if self.time_since_last_phase_switch[agent] < yellow_time + all_red_time + min_green_time:  # 15
                self.active_agents[agent] = False
                self.time_since_last_phase_switch[agent] += 1
                # Reach yellow time limit, switch to red
                if self.time_since_last_phase_switch[agent] == yellow_time:  # 3
                    red_state = self.create_red(traci.trafficlight.getRedYellowGreenState(agent))
                    traci.trafficlight.setRedYellowGreenState(agent, red_state)
                # Reach all red time limit, switch to green
                elif self.time_since_last_phase_switch[agent] == yellow_time + all_red_time:  # 5
                    traci.trafficlight.setRedYellowGreenState(agent, self.green_state[agent])
            # Agent should act now
            else:
                self.active_agents[agent] = True
                if agent in actions.keys():
                    new_phase_state = self.action_phase_map[actions[agent]]
                    current_state = traci.trafficlight.getRedYellowGreenState(agent)
                    if new_phase_state != current_state:
                        yellow_state = self.create_yellow(new_phase_state, current_state)
                        traci.trafficlight.setRedYellowGreenState(agent, yellow_state)  # Switch to yellow
                        self.time_since_last_phase_switch[agent] = 0
                        self.green_state[agent] = new_phase_state
                    else:
                        self.time_since_last_phase_switch[agent] += 1
                else:
                    self.time_since_last_phase_switch[agent] += 1

    # Create the corresponding yellow phase state
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

