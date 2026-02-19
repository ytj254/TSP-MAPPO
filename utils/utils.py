import os
import sys
import pandas as pd
from datetime import date
from math import pi
from matplotlib import pyplot as plt
from shutil import copyfile
from sumolib import checkBinary
# import seaborn as sns
import traci


def generate_config():
    sig_ids = ['h2', 'h3', 'h4', 'h5', 'h6']  # h_corridor
    print('GENERATING CONFIG')
    # TODO raise Exception('Invalid signal config')
    index_to_movement = {0: 'N-W', 1: 'N-S', 2: 'N-E', 3: 'E-N', 4: 'E-W', 5: 'E-S', 6: 'S-E',
                         7: 'S-N', 8: 'S-W', 9: 'W-S', 10: 'W-E', 11: 'W-N'}
    lane_sets = {}
    for sig_id in sig_ids:

        for idx, movement in index_to_movement.items():
            lane_sets[movement] = []

        links = traci.trafficlight.getControlledLinks(sig_id)
        # print(sig_id, links)
        for i, link in enumerate(links):
            link = link[0]  # unpack so link[0] is inbound, link[1] outbound
            lane_sets[index_to_movement[i]].append(link[0])
        print("'" + sig_id + "'" + ": {")
        print("'lane_sets':" + str(lane_sets) + '},')


def create_folder(folders_name, alg):
    today = date.today()
    dirs = []
    folders_path = os.path.join(os.getcwd(), folders_name)
    if not os.path.exists(folders_path):
        os.mkdir(folders_path)
    # print(os.listdir(folders_path))
    for n in os.listdir(folders_path):
        d, index = n.split('_')
        if d == f'{alg}-{str(today)}':
            dirs.append(index)
    dirs = sorted(int(i) for i in dirs)
    if dirs:
        new_dir = dirs[-1] + 1
    else:
        new_dir = 1
    folder_path = os.path.join(folders_path, f'{alg}-{str(today)}_{str(new_dir)}')
    os.mkdir(folder_path)
    return folder_path


def create_result_folder(folder_path):
    result_path = os.path.join(os.getcwd(), folder_path)
    if not os.path.exists(result_path):
        os.makedirs(folder_path)


def list_folders(path):
    return [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]


def set_sumo(gui=False, sumocfg_path='scenarios/r_intersection/Eastway-Central.sumocfg', random=True, log_path=None,
             seed=-1):
    # we need to import python modules from the $SUMO_HOME/tools directory
    if 'SUMO_HOME' in os.environ:
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        sys.path.append(tools)
    else:
        sys.exit("please declare scenarios variable 'SUMO_HOME'")

    # cmd mode or visual mode
    if gui:
        sumoBinary = checkBinary('sumo-gui')
    else:
        sumoBinary = checkBinary('sumo')

    # setting the cmd to run sumo
    if random and not log_path:
        if seed < 0:
            sumo_cmd = [sumoBinary, '-c', sumocfg_path, '--random', '--time-to-teleport', '-1',
                        '--no-warnings', '--no-step-log']
        else:
            sumo_cmd = [sumoBinary, '-c', sumocfg_path, "--seed", "%d" % seed, '--time-to-teleport', '-1',
                        '--no-warnings', '--no-step-log']
    elif random and log_path:
        if seed < 0:
            sumo_cmd = [sumoBinary, '-c', sumocfg_path, '--random', '--time-to-teleport', '-1',
                        '--no-warnings', '--no-step-log',
                        "--duration-log.statistics",
                        '--tripinfo-output', log_path + '_tripinfo.xml']
        else:
            sumo_cmd = [sumoBinary, '-c', sumocfg_path, '--seed', '%d' % seed, '--time-to-teleport', '-1',
                        '--no-warnings', '--no-step-log',
                        '--tripinfo-output', log_path + '_tripinfo.xml']
    elif not random and log_path:
        sumo_cmd = [sumoBinary, '-c', sumocfg_path, '--time-to-teleport', '-1',
                    '--no-warnings', '--no-step-log',
                    '--tripinfo-output', log_path + '_tripinfo.xml']
    else:
        sumo_cmd = [sumoBinary, '-c', sumocfg_path, '--time-to-teleport', '-1', '--no-warnings', '--no-step-log']

    return sumo_cmd
