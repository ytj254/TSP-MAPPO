#!/usr/bin/env python

import os
import sys
import time
import datetime
import numpy as np
import argparse
from utils.analysis import analysis_tripinfo, analysis_queue, analysis_stopinfo
from agents.MaxPressure import MaxPressure
from agents.tsp_cv import TspCv
from agents import MaxPressure, tsp_cv, LongestQueueFirst

# we need to import some python modules from the $SUMO_HOME/tools directory
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare scenarios variable 'SUMO_HOME")

from sumolib import checkBinary  # Check for the binary in environ vars
import traci


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--test', action='store_true')
    ap.add_argument('--gui', action='store_true')
    ap.add_argument('-s', '--scenario', type=str, default='corridor', choices=['corridor', 'intersection', 'real'])
    ap.add_argument('--model', type=str, default='actuated', choices=['actuated', 'mp', 'atsp', 'lqf', 'ltsp', 'mptsp'])
    args = ap.parse_args()

    if args.test:
        test(args)
    else:
        iteration(args)

    # print(args)


#  contains TraCI control loop
def run(args):
    if args.model == 'mp':
        model = MaxPressure.MaxPressure(args.scenario)
    elif args.model == 'mptsp':
        model = MaxPressure.MaxPressure(args.scenario, tsp=True)
    elif args.model == 'atsp':
        model = tsp_cv.TspCv(args.scenario)
    elif args.model == 'lqf':
        model = LongestQueueFirst.LongestQueueFirst(args.scenario)
    elif args.model == 'ltsp':
        model = LongestQueueFirst.LongestQueueFirst(args.scenario, tsp=True)
    else:
        model = None

    while traci.simulation.getTime() <= 4400:
        if model:
            model.control()
        else:
            # generate_config()
            traci.simulationStep()
    traci.close()
    sys.stdout.flush()


def test(args):
    # choose gui mode
    if args.gui:
        sumoBinary = checkBinary('sumo-gui')
    else:
        sumoBinary = checkBinary('sumo')

    # choose scenario
    if args.scenario == 'intersection':
        config_file = 'scenarios/h_intersection/h.sumocfg'
    elif args.scenario == 'real':
        config_file = 'scenarios/r_corridor/r.sumocfg'
    else:
        config_file = 'scenarios/h_corridor/h.sumocfg'
    # print(args.scenario)
    # print(args.agents)
    # traci starts sumo as a subprocess and then this script connects and runs
    # "--step-length", "0.1",
    traci.start(
        [
            sumoBinary, "-c", config_file, '--no-warnings', '--time-to-teleport', '-1',
            '--random', '--no-step-log',
            # '--queue-output', f'{data_path}/results/queue.xml', '--fcd-output', f'{data_path}/results/fcd.xml',
            # "--tripinfo-output", f"{data_path}/results/tripinfo.xml",
            # '--stop-output', 'scenarios/h_corridor/results/stop.xml',
            "--duration-log.statistics",
            # "--log", f"{data_path}/results/logfile.xml"
        ]
    )
    run(args)


def iteration(args):
    start_time = time.time()
    tot_trip_res = []
    # tot_queue_res = []
    # tot_stop_res = {}
    episode = 0
    n = 50
    if args.gui:
        sumoBinary = checkBinary('sumo-gui')
    else:
        sumoBinary = checkBinary('sumo')

    # choose scenario
    if args.scenario == 'intersection':
        config_file = 'scenarios/h_intersection/h.sumocfg'
        result_path = 'scenarios/h_intersection/results'
    elif args.scenario == 'real':
        config_file = 'scenarios/r_corridor/r.sumocfg'
        result_path = 'scenarios/r_corridor/results'
    else:
        config_file = 'scenarios/h_corridor/h.sumocfg'
        result_path = 'scenarios/h_corridor/results'

    while episode < n:  # run n simulations
        #  check binary
        print(f'------ Episode {str(episode + 1)} of {n} ------', end='\r', flush=True)

        # traci starts sumo as a subprocess and then this script connects and runs
        # "--step-length", "0.1",
        traci.start(
            [
                sumoBinary, "-c", config_file, "--seed", "%d" % episode, '--time-to-teleport', '-1',
                '--no-warnings', '--no-step-log',
                "--tripinfo-output", f'{result_path}/tripinfo.xml',
                '--queue-output', f'{result_path}/queue.xml',
                '--stop-output', f'{result_path}/stop.xml',
                # "--duration-log.statistics",
                # "--log", "logfile.xml",
            ]
        )
        run(args)
        trip_res = analysis_tripinfo(f'{result_path}/tripinfo.xml')
        # queue_res = analysis_queue(f'{result_path}/queue.xml')
        # stop_res = analysis_stopinfo(f'{result_path}/stop.xml')
        # print(stop_res)
        tot_trip_res.append(trip_res)
        # tot_queue_res.append(queue_res)
        # for stop, headways in stop_res.items():
        #     if stop not in tot_stop_res:
        #         tot_stop_res[stop] = headways
        #     else:
        #         tot_stop_res[stop] += headways
        episode += 1

    # Change list to array and transpose
    # print(tot_stop_res)
    # for k, v in tot_stop_res.items():
    #     print(len(v))
    trip_ares = np.array(tot_trip_res)
    # queue_ares = np.array(tot_queue_res).T
    # Save to csv files
    np.savetxt(f'{result_path}/trip_result.csv', trip_ares, delimiter=',')
    # np.savetxt(f'{result_path}/queue_result.csv', queue_ares, delimiter=',')
    # max_len_headways = max(len(v) for v in tot_stop_res.values())
    # np.savetxt(f'{result_path}/stop_result.csv',
    #            np.column_stack([v + [np.nan] * (max_len_headways - len(v)) for v in tot_stop_res.values()]),
    #            delimiter=',', header=','.join(tot_stop_res.keys()), comments='')
    # print(ares)
    trip_mean = np.mean(trip_ares, axis=0)
    [print(f'trip mean: {trip_mean[i: i + 4]}') for i in range(0, len(trip_mean), 4)]
    # for i in range(0, len(trip_mean), 4):
    #     print(f'trip mean: {trip_mean[i]}, {trip_mean[i + 1]}, {trip_mean[i + 2]}, {trip_mean[i + 3]}')
    print(f'Evaluating time: {datetime.timedelta(seconds=int(time.time() - start_time))}')


# main entry point
if __name__ == "__main__":
    main()
