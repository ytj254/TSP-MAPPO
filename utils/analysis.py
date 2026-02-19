import xml.etree.ElementTree as ET
import numpy as np

car_occupancy = 1.0
bus_occupancy = 30.0


def analysis_tripinfo(input_path):
    # Open xml
    tree = ET.parse(input_path)
    tripinfos = tree.getroot()

    travel_times = []
    travel_times_bus = []
    travel_times_car = []
    delays = []
    delays_bus = []
    delays_car = []
    travel_speeds = []
    travel_speeds_bus = []
    travel_speeds_car = []
    n_stops = []
    n_stops_bus = []
    n_stops_car = []
    # n_trips = []
    n_trips = 0

    for tripinfo in tripinfos:
        depart_time = float(tripinfo.attrib.get('depart'))
        if 600 <= depart_time <= 4200:
            n_trips += 1
            if 'bus' in tripinfo.attrib.get('id'):
                travel_time = float(tripinfo.attrib.get('duration'))
                route_len = float(tripinfo.attrib.get('routeLength'))
                travel_times_bus.append(travel_time)
                travel_speeds_bus.append(route_len / travel_time)
                delays_bus.append(float(tripinfo.attrib.get('timeLoss')))
                n_stops_bus.append(int(tripinfo.attrib.get('waitingCount')))
            else:
                travel_time = float(tripinfo.attrib.get('duration'))
                route_len = float(tripinfo.attrib.get('routeLength'))
                travel_times_car.append(travel_time)
                travel_speeds_car.append(route_len / travel_time)
                delays_car.append(float(tripinfo.attrib.get('timeLoss')))
                n_stops_car.append(int(tripinfo.attrib.get('waitingCount')))
    travel_times = travel_times_bus + travel_times_car
    delays = delays_bus + delays_car
    travel_speeds = travel_speeds_bus + travel_speeds_car
    n_stops = n_stops_bus + n_stops_car

    # Calculate the mean of metrics
    ave_travel_time = np.mean(travel_times)
    ave_delay = np.mean(delays)
    ave_speed = np.mean(travel_speeds)
    ave_stops = np.mean(n_stops)

    ave_travel_time_bus = np.mean(travel_times_bus)
    ave_delay_bus = np.mean(delays_bus)
    ave_speed_bus = np.mean(travel_speeds_bus)
    ave_stops_bus = np.mean(n_stops_bus)

    ave_travel_time_car = np.mean(travel_times_car)
    ave_delay_car = np.mean(delays_car)
    ave_speed_car = np.mean(travel_speeds_car)
    ave_stops_car = np.mean(n_stops_car)

    ave_travel_time_person = (sum(travel_times_bus) * bus_occupancy + sum(travel_times_car) * car_occupancy) / \
                             (len(travel_times_bus) * bus_occupancy + len(travel_times_car) * car_occupancy)

    ave_delay_person = (sum(delays_bus) * bus_occupancy + sum(delays_car) * car_occupancy) / \
                       (len(delays_bus) * bus_occupancy + len(delays_car) * car_occupancy)

    ave_speed_person = (sum(travel_speeds_bus) * bus_occupancy + sum(travel_speeds_car) * car_occupancy) / \
                       (len(travel_speeds_bus) * bus_occupancy + len(travel_speeds_car) * car_occupancy)

    ave_stops_person = (sum(n_stops_bus) * bus_occupancy + sum(n_stops_car) * car_occupancy) / \
                       (len(n_stops_bus) * bus_occupancy + len(n_stops_car) * car_occupancy)

    # ave_trips = np.mean(n_trips)

    return ave_travel_time, ave_delay, ave_speed, ave_stops, \
        ave_travel_time_bus, ave_delay_bus, ave_speed_bus, ave_stops_bus, \
        ave_travel_time_car, ave_delay_car, ave_speed_car, ave_stops_car, \
        ave_travel_time_person, ave_delay_person, ave_speed_person, ave_stops_person, \
        n_trips


def analysis_queue(input_path):
    # Open xml
    tree = ET.parse(input_path)
    queue_data = tree.getroot()
    queue_length = []

    for data in queue_data.findall(".//data"):
        timestep = float(data.attrib.get('timestep'))

        if 600 <= timestep <= 4200:
            total_queueing_length = sum(
                float(lane.attrib.get('queueing_length')) for lane in data.findall(".//lane"))

            queue_length.append(total_queueing_length)
    return queue_length


def analysis_stopinfo(input_path):
    tree = ET.parse(input_path)
    stopinfos = tree.getroot()
    headway_dic = {}
    stop_arrival = {}

    for stopinfo in stopinfos.findall(".//stopinfo"):
        bus_stop = stopinfo.attrib.get('busStop')
        arrival_time = stopinfo.attrib.get('started')
        if bus_stop not in stop_arrival:
            stop_arrival[bus_stop] = [arrival_time]
        else:
            stop_arrival[bus_stop].append(arrival_time)

    for stop, arrival_time in stop_arrival.items():
        headways = [float(arrival_time[i + 1]) - float(arrival_time[i]) for i in range(len(arrival_time) - 1)]
        headway_dic[stop] = headways
    return headway_dic


if __name__ == '__main__':
    file_path = 'D:/Paper Publish/MARL_TSC/MARL_TSC_code/scenarios/h_corridor/results/stop.xml'
    # print(analysis_tripinfo(file_path))
    # print(analysis_queue(file_path))
    analysis_stopinfo(file_path)
