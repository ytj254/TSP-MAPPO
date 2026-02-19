import traci
from utils.sig_config import sig_configs

saturation_left = 1.5  # Saturation flow ratio of left turn to through
yellow_time = 3
all_red_time = 2
min_green_time = 23  # h_corridor high demand 23 s low demand 11 s; r_corridor peak 20 s offpeak 9 s
max_green_time = 50
car_occupancy = 1.0
bus_occupancy = 30.0


class MaxPressure:
    def __init__(self, scenario='corridor', tsp=False):

        if scenario == 'intersection':
            map_name = 'h_intersection'
        elif scenario == 'real':
            map_name = 'r_corridor'
        else:
            map_name = 'h_corridor'

        self.tsp = tsp
        self.sig_configs = sig_configs[map_name]
        self.sig_ids = self.sig_configs['sig_ids']
        # self.phase_number = len(self.sig_configs['phase_pairs'])
        # self.action_phase_map = self.sig_configs['action_phase_map']
        # self.num_lanes = len(self.action_phase_map[0])
        self.sim_step = 0
        self.new_green_state = {}
        self.yellow_phase, self.red_phase, self.green_phase = {}, {}, {}
        self.yellow_durations, self.red_durations, self.green_durations = {}, {}, {}

        for sig_id in self.sig_ids:
            self.yellow_phase[sig_id] = False
            self.red_phase[sig_id] = False
            self.green_phase[sig_id] = True
            self.yellow_durations[sig_id] = 0
            self.red_durations[sig_id] = 0
            self.green_durations[sig_id] = 0
            self.new_green_state[sig_id] = self.sig_configs[sig_id]['action_phase_map'][0]
        # print(self.yellow_durations, self.yellow_phase)

    @staticmethod
    def count_queued(objective):
        queued_veh = 0
        if '_' in objective:
            veh_set = traci.lane.getLastStepVehicleIDs(objective)
        else:
            veh_set = traci.edge.getLastStepVehicleIDs(objective)
        for veh in veh_set:
            if traci.vehicle.getSpeed(veh) <= 5:
                queued_veh += 1
        return queued_veh

    @staticmethod
    def count_person(objective):
        num_person = 0
        if '_' in objective:
            veh_set = traci.lane.getLastStepVehicleIDs(objective)
        else:
            veh_set = traci.edge.getLastStepVehicleIDs(objective)
        for veh in veh_set:
            if traci.vehicle.getTypeID(veh) == 'BUS':
                num_person += bus_occupancy
            else:
                num_person += car_occupancy
        return num_person

    def move_pressure(self, sig_id, move_index):
        """ calculations of other metrics:
            num_v_in += self.count_queued(in_lane)
            num_v_out = self.count_queued(out_edge)
            num_v_out = traci.lane.getLastStepVehicleNumber(f'{out_edge}_0') * 0.15 \
                      + traci.lane.getLastStepVehicleNumber(f'{out_edge}_1') * 0.75 \
                      + traci.lane.getLastStepVehicleNumber(f'{out_edge}_2') * 0.1"""
        # print(self.sig_configs[self.id])
        in_lane_sets = self.sig_configs[sig_id]['lane_sets']
        out_edge_sets = self.sig_configs[sig_id]['downstream']
        in_lanes = in_lane_sets[move_index]
        # print('out-edge:', move_index.split('-')[1])
        out_edge = out_edge_sets[move_index.split('-')[1]]
        # print(in_lanes, type(in_lanes))
        if self.tsp:
            num_in = sum(self.count_person(in_lane) for in_lane in in_lanes)
            # num_out = self.count_person(out_edge) / 3

        else:
            num_in = sum(traci.lane.getLastStepVehicleNumber(in_lane) for in_lane in in_lanes)
        num_out = traci.edge.getLastStepVehicleNumber(out_edge) / 3

        if move_index in self.sig_configs['left_turns']:
            return num_in * saturation_left - num_out
        # print('move pressure:', sig_id, move_index, num_v_in - num_v_out)
        return num_in - num_out

    def phase_pressure(self, sig_id):
        phase_pairs = self.sig_configs[sig_id]['phase_pairs']
        # phase_pressures = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0}
        phase_pressures = {phase: 0.0 for phase in phase_pairs.keys()}
        for phase in range(len(phase_pairs)):
            for move in phase_pairs[phase]:
                # print(move, type(move))
                phase_pressures[phase] += self.move_pressure(sig_id, move)
        # print(phase_pressures)
        return phase_pressures

    def act(self, sig_id):
        phase_pressures = self.phase_pressure(sig_id)
        # print('phase pressure:', sig_id, phase_pressures)
        return max(phase_pressures, key=phase_pressures.get)

    def actions(self):
        actions = {}
        for sig_id in self.sig_ids:
            actions[sig_id] = self.act(sig_id)
        return actions

    def control(self):
        """
        Each signal must execute only one sim step within a control step
        """
        # print(self.phase_durations)
        for sig_id in self.sig_ids:

            current_state = traci.trafficlight.getRedYellowGreenState(sig_id)
            action_phase_map = self.sig_configs[sig_id]['action_phase_map']
            # print(current_state)
            if self.green_phase[sig_id]:
                if min_green_time < self.green_durations[sig_id] <= max_green_time:
                    act = self.act(sig_id)
                    self.new_green_state[sig_id] = action_phase_map[act]
                    if self.new_green_state[sig_id] != current_state:
                        self.switch_phase(sig_id, current_state, 'yellow')

                # Max green time check
                elif self.green_durations[sig_id] > max_green_time:
                    act = self.act(sig_id)
                    if action_phase_map[act] != current_state:
                        self.new_green_state[sig_id] = action_phase_map[act]
                    else:
                        new_act = act + 1 if act < 7 else 0
                        self.new_green_state[sig_id] = action_phase_map[new_act]
                    self.switch_phase(sig_id, current_state, 'yellow')

                self.green_durations[sig_id] += 1
                # self.green_phase[sig_id] = True

            elif self.yellow_phase[sig_id]:
                if self.yellow_durations[sig_id] > yellow_time:
                    self.switch_phase(sig_id, current_state, 'red')

                self.yellow_durations[sig_id] += 1

            elif self.red_phase[sig_id]:
                if self.red_durations[sig_id] > all_red_time:
                    self.switch_phase(sig_id, current_state, 'green')

                self.red_durations[sig_id] += 1

        traci.simulationStep()

    def switch_phase(self, sig_id, current_state, to_color):
        if to_color == 'green':
            traci.trafficlight.setRedYellowGreenState(sig_id, self.new_green_state[sig_id])
            self.red_phase[sig_id] = False
            self.green_phase[sig_id] = True
            self.green_durations[sig_id] = 1
        if to_color == 'yellow':
            yellow_state = self.create_yellow(sig_id, self.new_green_state[sig_id], current_state)
            traci.trafficlight.setRedYellowGreenState(sig_id, yellow_state)
            self.green_phase[sig_id] = False
            self.yellow_phase[sig_id] = True
            self.yellow_durations[sig_id] = 1
        if to_color == 'red':
            red_state = self.create_red(current_state)
            traci.trafficlight.setRedYellowGreenState(sig_id, red_state)
            self.yellow_phase[sig_id] = False
            self.red_phase[sig_id] = True
            self.red_durations[sig_id] = 1

    # Create the corresponding yellow and red phase state
    def create_yellow(self, sig, phase_state, current_state):
        yellow_state = []
        phase_state_len = len(self.sig_configs[sig]['action_phase_map'][0])
        for i in range(phase_state_len):
            if current_state[i] == 'G' and current_state[i] != phase_state[i]:
                yellow_state.append('Y')
            else:
                yellow_state.append(current_state[i])
        yellow_state = ''.join(yellow_state)

        return yellow_state

    @staticmethod
    def create_red(current_state):
        return current_state.replace('Y', 'r')
