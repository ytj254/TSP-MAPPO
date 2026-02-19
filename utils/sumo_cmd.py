import os
import argparse

folder_path = '/usr/share/sumo/tools/'


def get_options():
    op = argparse.ArgumentParser()
    op.add_argument('-rt', '--randomTrips', type=bool, default=False)
    op.add_argument('-rs', '--routeSampler', type=bool, default=False)
    op.add_argument('-pt', '--plot_trajectories', type=bool, default=False)
    op.add_argument('-s', '--scenario', type=str, default='corridor')
    op.add_argument('-f', '--filterIDs', type=str, default='0.*')
    ops = op.parse_args()
    return ops


def main(options):
    cmd = 'None'
    if options.scenario == 'corridor':
        data_path = 'scenarios/h_corridor'
    else:
        data_path = 'scenarios/h_intersection'

    # Generate trips for routeSampler
    """
    Weights files:
    .src.xml contains the probabilities for an edge to be selected as from-edge
    .dst.xml contains the probabilities for an edge to be selected as to-edge
    .via.xml contains the probabilities for an edge to be selected as via-edge (only used when 
    option --intermediate is set).
    """
    if options.randomTrips:
        cmd = (
            f'python "{folder_path}randomTrips.py" '
            f'-n {data_path}/h.net.xml '
            # f'-o {data_path}/h.trips.xml '
            f'-r {data_path}/h.rou.xml '
            # '--fringe-factor 2000 '  # Increase the probability that trips start/end at the fringe edges 
            f'--weights-prefix {data_path}/h '  # Load weight files
            # f'--weights-output-prefix {data_path}/h'  # Output wepyight files that define edge selection probability
        )

    # Generate route for simulation
    if options.routeSampler:
        cmd = (
            f'python "{folder_path}routeSampler.py" '  # File path
            f'-r {data_path}/h.rou.xml '  # Route files
            f'--edgedata-files {data_path}/h.enter.xml '  # Enter data
            f'--turn-ratio-files {data_path}/h.turn.xml '  # Turn ratio data
            f'-o {data_path}/sample.rou.xml '  # output file name
            '--write-flows probability --write-route-ids '
            '--attributes="departLane=\\"best\\" departSpeed=\\"max\\"" '
        )

    """Plot trajectory figure
    td: time vs distance 
    ts: time vs speed
    ta: time vs acceleration
    ds: distance vs speed
    da: distance vs acceleration
    xy: Spatial plot of driving path
    kt: kilometrage vs time (combine with option --invert-yaxis to get a classic railway diagram).
    """
    if options.plot_trajectories:
        cmd = (
            f'py "{folder_path}plot_trajectories.py" '  # File path
            f'{data_path}/results/fcd.xml '  # fcd file 
            f'-t td '  
            f'-o {data_path}/results/fcd.png -s '
            f'--filter-ids {options.filterIDs} '
        )

    os.system(cmd)
    # print(cmd, type(cmd))


if __name__ == '__main__':
    main(get_options())
