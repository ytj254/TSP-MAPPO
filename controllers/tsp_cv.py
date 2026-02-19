import traci
from utils.sig_config import sig_configs

yellow_time = 3
all_red_time = 2


class TspCv:
    def __init__(self, scenario='corridor'):
        if scenario == 'intersection':
            map_name = 'h_intersection'
        elif scenario == 'real':
            map_name = 'r_corridor'
        else:
            map_name = 'h_corridor'

        self.sig_configs = sig_configs[map_name]
        self.sig_ids = self.sig_configs['sig_ids']
        # self.action_phase_map = self.sig_configs['action_phase_map']
        # self.phase_state_len = len(self.action_phase_map[0])
        self.time_since_tsp_activated = {sig: 0 for sig in self.sig_ids}
        self.tsp_activation = {sig: False for sig in self.sig_ids}

        if scenario == 'real':
            self.bus_green_state = {
                'h2': 'GGrgrgGG',
                'h3': 'grrgGGrgrrgGGr',
                'h4': 'grrrgGGrrgrrrgGGrr',
                'h5': 'grrgGGrgrrgGGr',
                'h6': 'grrgGGrgrrgGGr',
                                    }
        else:
            self.bus_green_state = {sig: 'grrgGrgrrgGr' for sig in self.sig_ids}

    def control(self):
        bus_infos = self.get_bus_info()
        tsp_sig = []
        # print(self.time_since_tsp_activated)

        for info in bus_infos:
            if info[2] <= 100:
                sig = info[0]
                tsp_sig.append(sig)
                if not self.tsp_activation[sig]:
                    if info[3] == 'G':
                        traci.trafficlight.setRedYellowGreenState(sig, traci.trafficlight.getRedYellowGreenState(sig))
                    # Red or yellow state and corresponding signal is not activated for tsp
                    else:
                        self.tsp_activation[sig] = True
                        self.time_since_tsp_activated[sig] = 0
                        # print(f'------{sig}: tsp triggered-----')
                        yellow_state = self.create_yellow(sig)
                        traci.trafficlight.setRedYellowGreenState(sig, yellow_state)

                else:
                    # Greater than yellow and all red time limit, switch to green
                    if self.time_since_tsp_activated[sig] >= yellow_time + all_red_time:
                        traci.trafficlight.setRedYellowGreenState(sig, self.bus_green_state[sig])

                    # Greater than yellow time limit, switch to red
                    elif self.time_since_tsp_activated[sig] >= yellow_time:
                        red_state = self.create_red(sig)
                        traci.trafficlight.setRedYellowGreenState(sig, red_state)

                    self.time_since_tsp_activated[sig] += 1

        for sig in self.sig_ids:
            if sig not in tsp_sig and traci.trafficlight.getProgram(sig) != 'NEMA':
                traci.trafficlight.setProgram(sig, 'NEMA')
                self.tsp_activation[sig] = False

        traci.simulationStep()

    def create_yellow(self, sig):
        current_state = traci.trafficlight.getRedYellowGreenState(sig)
        yellow_state = []
        phase_state_len = len(self.sig_configs[sig]['action_phase_map'][0])
        for i in range(phase_state_len):
            if current_state[i] == 'G' and current_state[i] != self.bus_green_state[sig][i]:
                yellow_state.append('Y')
            else:
                yellow_state.append(current_state[i])
        yellow_state = ''.join(yellow_state)
        return yellow_state

    # Create the corresponding red phase state
    @staticmethod
    def create_red(sig):
        current_state = traci.trafficlight.getRedYellowGreenState(sig)
        return current_state.replace('Y', 'r')

    @staticmethod
    def get_bus_info():
        bus_list = []
        for veh in traci.vehicle.getIDList():
            if traci.vehicle.getVehicleClass(veh) == 'bus' and traci.vehicle.getNextTLS(veh):
                bus_list.append(traci.vehicle.getNextTLS(veh)[0])
        return bus_list
