# utils.py
# module: vespy.utils
# General purpose utilities for seismic array analysis

import numpy as np
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
from obspy.core import Trace
from obspy.taup import TauPyModel
from obspy.clients.fdsn import Client
from obspy.clients.fdsn.header import FDSNException

G_KM_DEG = 111.19 # km / deg, Conversion factor for converting angular great circle distance (in degrees) into km on the surface

class Phase:
    '''
    Class for handling phase arrivals.

    Similar to obspy.core.event.Arrival, which does the same thing but with a few more parameters I don't need

    :type name: string
    :param name: Name of seismic phase
    :type time: float
    :param time: Time after event origin of arrival in seconds
    :type slowness: float
    :param slowness: Slowness of seismic phase in seconds / kilometre

    '''

    def __init__(self, phase_name, arrival_time, slowness_s_km):
        self.name = phase_name
        self.time = arrival_time
        self.slowness = slowness_s_km

    def __str__(self):
        return_str = "%s, %.1f, %.3f" % (self.name, self.time, self.slowness)
        return return_str

def traceify(data_array, ref_trace, name="NEW_TRACE", channel=None):
    '''
    Turns an array into an ObsPy Trace object, using the reference trace.

    The new trace inherits the time data all stats (except 'station' and 'channel', which are left for the user to define) from the reference trace.

    Parameters
    ----------
    data_array: NumPy array-like object
        The data to be turned into an ObsPy Trace. Should be 1D time series numerical data of the same length as the reference trace, e.g. F-statistic, power, envelope etc.
    ref_trace: ObsPy Trace object
        Reference time series data to use when creating the new trace
    name: string
        String to identify the new trace, e.g. 'F-STAT', 'POWER', 'ENVELOPE'. Occupies the 'station' SAC header variable.
    channel: string
        Data channel to identify the component of the time series data the new trace represents (if applicable)

    Returns
    -------
    new_trace: ObsPy Trace object
        An ObsPy Trace containg  the data in data_array, and the time data and stats from ref_trace

    '''

    new_trace = Trace(data_array)
    new_trace.stats = ref_trace.stats.copy()
    new_trace.stats.station = name

    if channel is not None:
        new_trace.stats.channel = channel

    return new_trace

def get_station_coordinates(stream):
    '''
    Calculates the x, y, z coordinates of stations in a seismic array relative to a reference point for a given stream of SAC seismographic data files.

    The reference point will be taken as the coordinates of the first station in the stream.

    Parameters
    ----------
    stream : ObsPy Stream object
        The stream of seismograms for the array

    Returns
    -------
    xyz_rel_coords : NumPy Array
        x, y, z coordinates for each station in the array, measured in metres relative to the first station in the stream
    '''

    # Get coordinates of stations in array. Looks for SAC headers stla, stlo, stel.
    coords = []

    # Get station name, longitude, latitude, and elevation
    for trace in stream:
        coords.append((trace.stats.sac.stlo, trace.stats.sac.stla, trace.stats.sac.stel))

    coords = np.array(coords)

    # Convert latitude/longitude into a local rectilinear grid (in km) at the array. Uses Basemap.

    # Define the corners of the rectangular grid using the minimum and maximum longitudes and latitudes of stations in the array
    lon_min = coords.min(axis=0)[0]
    lon_max = coords.max(axis=0)[0]
    lat_min = coords.min(axis=0)[1]
    lat_max = coords.max(axis=0)[1]

    # Define keyword arguments for Basemap: mapping to a Mercator-projected rectangular box
    kwargs = dict(
        projection='merc',
        resolution='l',
        lat_ts=20,
        llcrnrlon=lon_min,
        llcrnrlat=lat_min,
        urcrnrlon=lon_max,
        urcrnrlat=lat_max
    )

    map = Basemap(**kwargs)

    # Assemble coordinates
    xyz_coords = []

    for lon, lat, elev in coords:
        x, y = map(lon, lat)
        z = elev
        xyz_coords.append((x, y, z))

    # Get coordinates relative to the first station
    x_0, y_0, z_0 = xyz_coords[0]
    xyz_rel_coords = []
    for x, y, z in xyz_coords:
        x -= x_0
        y -= y_0
        z -= z_0
        xyz_rel_coords.append((x, y, z))
    xyz_rel_coords = np.array(xyz_rel_coords)

    return xyz_rel_coords

def plot_array_map(stream, outfile=None, format='PDF'):

    '''
    Plots a map of the seismic array from an ObsPy stream of time series data.

    Uses Basemap.

    Parameters
    ----------
    stream : ObsPy Stream object
        The stream of seismograms for the array
    '''

    # Assemble longitude and latitude coordinates of each station in the stream, and also a list of station names for labelling the map
    coords = []
    labels = []

    for trace in stream:
        coords.append([trace.stats.sac.stlo, trace.stats.sac.stla])
        labels.append(trace.stats.station)

    # Convert coordinates to NumPy array for easier finding of min/max
    coords = np.array(coords)

    # Find max, min, range and centre for longitude and latitude
    lon_max = max(coords[:, 0])
    lon_min = min(coords[:, 0])

    lon_range = lon_max - lon_min
    lon_0 = lon_min + lon_range/2.

    lat_max = max(coords[:, 1])
    lat_min = min(coords[:, 1])

    lat_range = lat_max - lat_min
    lat_0 = lat_min + lat_range/2.

    # Set keyword arguments for Basemap.
    kwargs = dict(
        projection='merc',
        #lat_0=lat0,
        #lon_0=lon0,
        resolution='f',
        lat_ts=20,
        llcrnrlon=lon_min - 0.1*lon_range,
        llcrnrlat=lat_min - 0.1*lat_range,
        urcrnrlon=lon_max + 0.1*lon_range,
        urcrnrlat=lat_max + 0.1*lat_range,
    )

    # Draw map
    plt.figure(figsize=(12, 10))

    map = Basemap(**kwargs)
    map.drawcoastlines()
    map.fillcontinents(color='white')
    map.drawcountries()
    map.drawmapboundary(fill_color='aqua')
    map.drawrivers()
    map.drawparallels(np.round(np.arange(np.round(lat_min, 2), np.round(lat_max, 2), round(lat_range/5., 2)), 2), dashes=[1, 5], labels=[1, 0, 1, 0])
    map.drawmeridians(np.round(np.arange(np.round(lon_min, 2), np.round(lon_max, 2), round(lon_range/5., 2)), 2), dashes=[1, 5], labels=[0, 1, 0, 1])
    map.drawmapscale(lon=lon_0, lat=lat_min, lon0=lon_0, lat0=lat_0, length=int(round(lon_range * 111.19)), barstyle='simple', labelstyle='simple', yoffset=200, fontsize=14);

    xy_coords = []

    i = 0

    # Plot stations and label them
    for lon, lat in coords:
        x, y = map(lon, lat)
        xy_coords.append((x, y))

        map.plot(x, y, 'b^', markersize=10)
        plt.text(x + 50, y + 50, labels[i], fontsize=14)
        i += 1

    if not outfile == None:
        plt.savefig(outfile, format=format, bbox_inches='tight', pad_inches=0.1)

    plt.show()

def find_event(st, timebefore=5, timeafter=5, service="IRIS"):
    '''
    Uses the selected webservice to search for an event matching the stream's starttime plus/minus the specified time window.

    If multiple streams match, lists them.

    Parameters
    ----------
    st : ObsPy Stream object
        Stream of SAC format seismograms for the event in question
    timebefore : float
        Time in seconds before stream start time from which to search catalog for events
    timeafter : float
        Time insseconds after stream start time up to which to search catalog for events
    service : String
        Web service to use to search for events. Same options as for obspy.fdsn.Client. Default is IRIS.

    Returns
    -------
    event : ObsPy Event object
        Downloaded information for the event, if found.
    '''

    webservice = Client(service)

    try:
        cat = webservice.get_events(starttime=st[0].stats.starttime - timebefore, endtime=st[0].stats.starttime + timeafter, minmagnitude=st[0].stats.sac.mag - 1.0, maxmagnitude=st[0].stats.sac.mag + 1.0)

    except FDSNException:
        print "No event found for stream startttime. Try adjusting time window."
        return

    except AttributeError:
        print "No stats.sac dictionary, attempting search based on time window alone..."

        try:
            cat = webservice.get_events(starttime=st[0].stats.starttime - timebefore, endtime=st[0].stats.starttime + timeafter)

        except FDSNException:
            print "No event found for stream startttime. Try adjusting time window."
            return

    if len(cat) > 1:
        print "Multiple events found for stream starttime. Try adjusting time window."
        print cat
        return

    event = cat[0]

    print event

    return event

def get_first_arrival(st, model='ak135'):
    '''
    Returns first arrival information for a particular stream and a theoretical velocity model ak135 or iasp91.

    Output is phase name, arrival time (in s after origin) and slowness (in s / km).

    Parameters
    ----------
    st : ObsPy Stream object
        Stream of SAC format seismograms for the seismic array
    model : string
        Model to use for the travel times, either 'ak135' or 'iasp91'.

    Returns
    -------
    phase : Phase object
        Phase object containing the phase name, arrival time, and slowness of the first arrival
    '''

    # Read event depth and great circle distance from SAC header
    depth = st[0].stats.sac.evdp
    delta = st[0].stats.sac.gcarc

    taup = TauPyModel(model)
    first_arrival = taup.get_travel_times(depth, delta)[0]

    phase = Phase(first_arrival.name, first_arrival.time, first_arrival.ray_param_sec_degree / G_KM_DEG)

    return phase

def get_arrivals(st, model='ak135'):
    '''
    Returns complete arrival information for a particular stream and a theoretical velocity model ak135 or iasp91.

    Output is phase name, arrival time (in s after origin) and slowness (in s / km).

    Parameters
    ----------
    st : ObsPy Stream object
        Stream of SAC format seismograms for the seismic array
    model : string
        Model to use for the travel times, either 'ak135' or 'iasp91'.

    Returns
    -------
    phase_list : list
        List containing Phase objects containing the phase name, arrival time, and slowness of each arrival
    '''

    # Read event depth and great circle distance from SAC header
    depth = st[0].stats.sac.evdp
    delta = st[0].stats.sac.gcarc

    tau_model = TauPyModel(model)
    arrivals = tau_model.get_travel_times(depth, delta)

    phase_list = []

    for arrival in arrivals:
        phase = Phase(arrival.name, arrival.time, arrival.ray_param_sec_degree / G_KM_DEG)
        phase_list.append(phase)

    return phase_list
