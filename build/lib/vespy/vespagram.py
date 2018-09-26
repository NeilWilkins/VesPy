# vespagram.py
# module: vespy.vespagram
# Module for performing velocity spectral analysis echniques on seismic datasets, and plotting vespagrams

import numpy as np
import matplotlib.pyplot as plt
from obspy.taup import TauPyModel
from obspy.geodetics import locations2degrees

from vespy.utils import G_KM_DEG
from vespy.stacking import linear_stack, nth_root_stack, phase_weighted_stack
from vespy.stats import n_power_vespa, f_vespa, pw_power_vespa

def vespagram(st, smin, smax, ssteps, baz, winlen, stat='power', phase_weighting=False, n=1):
    '''
    Calculates the vespagram for a seismic array over a given slowness range, for a single backazimuth, using the statistic specified.

    The chosen statistic is calculated as a function of time (in s) and slowness (in s/km). This may be:-

    * 'amplitude' - the raw amplitude of the linear or nth root stack at each time and slowness step;
    * 'power' - the power in the linear or nth root beam calculated over a time window (length winlen) around each time step for each slowness;
    * 'F' - the F-statistic of the beam calculated over a time window (length winlen) around each time step for each slowness.

    Parameters
    ----------
    st : ObsPy Stream object
        Stream of SAC format seismograms for the seismic array, length K = no. of stations in array
    smin  : float
        Minimum magnitude of slowness vector, in s / km
    smax  : float
        Maximum magnitude of slowness vector, in s / km
    ssteps  : int
        Integer number of steps between smin and smax for which to calculate the vespagram
    baz : float
        Backazimuth of slowness vector, (i.e. angle from North back to epicentre of event)
    winlen : int
        Length of Hann window over which to calculate the power.
    stat : string
        Statistic to use for plotting the vespagram, either 'amplitude', 'power', or 'F'
    phase_weighting : Boolean
        Whether or not to apply phase-weighting to the stacks in the vespagram.
    n : int
        Order for the stack, or if phase_weighting==True, order of the weighting applied.


    Returns
    -------
    vespagram_data : NumPy array
        Array of values for the chosen statistic at each slowness and time step. Dimensions: ssteps*len(tr) for traces tr in st.
    '''

    assert stat == 'amplitude' or stat == 'power' or stat == 'F', "'stat' argument must be one of 'amplitude', 'power' or 'F'"

    vespagram_data = np.array([])

    try:
        if stat == 'amplitude':
            if phase_weighting:
                vespagram_data = np.array([phase_weighted_stack(st, s, baz, n) for s in np.linspace(smin, smax, ssteps)])
            else:
                vespagram_data = np.array([nth_root_stack(st, s, baz, n) for s in np.linspace(smin, smax, ssteps)])

        elif stat == 'power':
            if phase_weighting:
                vespagram_data = np.array([pw_power_vespa(st, s, baz, n, winlen) for s in np.linspace(smin, smax, ssteps)])
            else:
                vespagram_data = np.array([n_power_vespa(st, s, baz, n, winlen) for s in np.linspace(smin, smax, ssteps)])

        elif stat == 'F':
            vespagram_data = np.array([f_vespa(st, s, baz, winlen, n) for s in np.linspace(smin, smax, ssteps)])

    except AssertionError as err:
        raise err

    return vespagram_data

def plot_vespagram(st, smin, smax, ssteps, baz, winlen, stat='power', phase_weighting=False, n=1, display='contourf', outfile=None):
    '''
    Plots the vespagram for a seismic array over a given slowness range, for a single backazimuth, using the statistic specified.

    The chosen statistic is plotted as a function of time (in s) and slowness (in s/km). This may be:-

    * 'amplitude' - the raw amplitude of the linear or nth root stack at each time and slowness step;
    * 'power' - the power in the linear or nth root beam calculated over a time window (length winlen) around each time step for each slowness;
    * 'F' - the F-statistic of the beam calculated over a time window (length winlen) around each time step for each slowness.

    Parameters
    ----------
    st : ObsPy Stream object
        Stream of SAC format seismograms for the seismic array, length K = no. of stations in array
    smin  : float
        Minimum magnitude of slowness vector, in s / km
    smax  : float
        Maximum magnitude of slowness vector, in s / km
    ssteps  : int
        Integer number of steps between smin and smax for which to calculate the vespagram
    baz : float
        Backazimuth of slowness vector, (i.e. angle from North back to epicentre of event)
    winlen : int
        Length of Hann window over which to calculate the power.
    stat : string
        Statistic to use for plotting the vespagram, either 'amplitude', 'power', or 'F'
    phase_weighting : Boolean
        Whether or not to apply phase-weighting to the stacks in the vespagram.
    n : int
        Order for the stack, or if phase_weighting==True, order of the weighting applied.
    display: string
        Option for plotting: either 'contourf' for filled contour plot, or 'contour' for contour plot. See matplotlib documentation for more details.
    outfile : string
        Filename for saving plot.
    '''

    assert display == 'contourf' or display == 'contour', "Invalid display option; must be 'contourf' or 'contour'"

    vespagram_data = np.array([])

    try:
        vespagram_data = vespagram(st, smin, smax, ssteps, baz, winlen, stat, phase_weighting, n)
    except AssertionError as err:
        print(err.args[0])
        return None

    timestring = str(st[0].stats.starttime.datetime)

    if stat == 'amplitude':
        label = "Amplitude"
        title = timestring + ": " + label + " Vespagram, n=" + str(n)
    elif stat == 'power':
        label = "Power"
        title = timestring + ": " + label + " Vespagram, n=" + str(n)
    elif stat == 'F':
        label = 'F'
        title = timestring + ": " + label + " Vespagram, n=" + str(n)
    else:
        raise AssertionError("'stat' argument must be one of 'amplitude', 'power' or 'F'")

    plt.figure(figsize=(16, 8))

    if display == 'contourf':
        plt.contourf(st[0].times(), np.linspace(smin, smax, ssteps), vespagram_data[:, :])
    else:
        plt.contour(st[0].times(), np.linspace(smin, smax, ssteps), vespagram_data[:, :])

    cb = plt.colorbar()
    cb.set_label(label)

    plt.xlabel("Time (s)", fontsize=16)
    plt.ylabel("Slowness (s / km)", fontsize=16)
    plt.title(title, y=1.08, fontsize=18)
    plt.tick_params(axis='both', which='major', labelsize=14)

    if outfile is not None:
        plt.savefig(outfile, bbox_inches='tight')

    plt.show()

def f_vespagram_theoretical_arrivals(st, origin, smin, smax, ssteps, baz, winlen):
    '''
    Plots the F-stat vespagram for a seismic array over a given slowness range, for a single backazimuth, using the statistic specified. Also plots theoretical arrival times and slownesses for each phase.

    The chosen statistic is plotted as a function of time (in s) and slowness (in s/km). This may be:-

    * 'amplitude' - the raw amplitude of the linear or nth root stack at each time and slowness step;
    * 'power' - the power in the linear or nth root beam calculated over a time window (length winlen) around each time step for each slowness;
    * 'F' - the F-statistic of the beam calculated over a time window (length winlen) around each time step for each slowness.

    Parameters
    ----------
    st : ObsPy Stream object
        Stream of SAC format seismograms for the seismic array, length K = no. of stations in array
    origin : ObsPy Origin object
        Origin of the event in question. Should contain the origin time of the earthquake and if necessary the depth and location.
    smin  : float
        Minimum magnitude of slowness vector, in s / km
    smax  : float
        Maximum magnitude of slowness vector, in s / km
    ssteps  : int
        Integer number of steps between smin and smax for which to calculate the vespagram
    baz : float
        Backazimuth of slowness vector, (i.e. angle from North back to epicentre of event)
    winlen : int
        Length of Hann window over which to calculate the power.
    stat : string
        Statistic to use for plotting the vespagram, either 'amplitude', 'power', or 'F'
    display: string
        Option for plotting: either 'contourf' for filled contour plot, or 'contour' for contour plot. See matplotlib documentation for more details.
    '''

    starttime = st[0].stats.starttime
    tt_model = TauPyModel()

    # Arrivals are calculated from the information in origin.
    delta = locations2degrees(origin.latitude, origin.longitude, st[0].stats.sac.stla, st[0].stats.sac.stlo) # Distance in degrees from source to receiver
    arrivals = tt_model.get_travel_times(origin.depth/1000., delta)

    arrival_names = [arrival.name for arrival in arrivals]
    arrival_times = [origin.time + arrival.time - starttime for arrival in arrivals]
    arrival_slowness = [arrival.ray_param_sec_degree/G_KM_DEG for arrival in arrivals]

    plt.figure(figsize=(16, 8))

    vespagram = np.array([f_vespa(st, s, baz, winlen) for s in np.linspace(smin, smax, ssteps)])
    label = 'F'
    timestring = str(st[0].stats.starttime.datetime)
    title = timestring + ": " + label + " Vespagram"

    plt.contourf(st[0].times(), np.linspace(smin, smax, ssteps), vespagram[:, :])

    cb = plt.colorbar()
    cb.set_label(label)

    # Plot predicted arrivals
    plt.scatter(arrival_times, arrival_slowness, c='cyan', s=200, marker='+')

    plt.xlabel("Time (s)")
    plt.xlim(min(st[0].times()), max(st[0].times()))
    plt.ylim(smin, smax)

    # Thanks, Stack Overflow: http://stackoverflow.com/questions/5147112/matplotlib-how-to-put-individual-tags-for-a-scatter-plot
    for label, x, y in zip(arrival_names, arrival_times, arrival_slowness):
        plt.annotate(label, xy=(x, y), xytext=(-20, 20), textcoords='offset points', ha='right', va='bottom',
                     bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                     arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

    plt.ylabel("Slowness (s / km)")
    plt.title(title)

def vespagram_backazimuth(st, s, bazmin, bazmax, bazsteps, winlen, stat='power', phase_weighting=False, n=1):
    '''
    Calculates the vespagram for a seismic array over a given range of backazimuths, for a single scalar slowness, using the statistic specified.

    The chosen statistic is calculated as a function of time (in s) and backazimuth (in deg). This may be:-

    * 'amplitude' - the raw amplitude of the linear or nth root stack at each time and slowness step;
    * 'power' - the power in the linear or nth root beam calculated over a time window (length winlen) around each time step for each slowness;
    * 'F' - the F-statistic of the beam calculated over a time window (length winlen) around each time step for each slowness.

    Parameters
    ----------
    st : ObsPy Stream object
        Stream of SAC format seismograms for the seismic array, length K = no. of stations in array
    s : float
        Magnitude of the slowness vector, in s/km
    bazmin  : float
        Minimum backazimuth, in degrees
    bazmax  : float
        Maximum backazimuth, in degrees
    bazsteps  : int
        Integer number of steps between bazmin and bazmax for which to calculate the vespagram
    winlen : int
        Length of Hann window over which to calculate the power.
    stat : string
        Statistic to use for plotting the vespagram, either 'amplitude', 'power', or 'F'
    phase_weighting : Boolean
        Whether or not to apply phase-weighting to the stacks in the vespagram.
    n : int
        Order for the stack, or if phase_weighting==True, order of the weighting applied.

    Returns
    -------
    vespagram_data : NumPy array
        Array of values for the chosen statistic at each backazimuth and time step. Dimensions: bazsteps*len(tr) for traces tr in st.
    '''

    try:
        if stat == 'amplitude':
            if phase_weighting:
                vespagram_data = np.array([phase_weighted_stack(st, s, baz, n) for baz in np.linspace(bazmin, bazmax, bazsteps)])
            else:
                vespagram_data = np.array([nth_root_stack(st, s, baz, n) for baz in np.linspace(bazmin, bazmax, bazsteps)])

        elif stat == 'power':
            if phase_weighting:
                vespagram_data = np.array([pw_power_vespa(st, s, baz, n, winlen) for baz in np.linspace(bazmin, bazmax, bazsteps)])
            else:
                vespagram_data = np.array([n_power_vespa(st, s, baz, n, winlen) for baz in np.linspace(bazmin, bazmax, bazsteps)])

        elif stat == 'F':
            vespagram_data = np.array([f_vespa(st, s, baz, winlen, n) for baz in np.linspace(bazmin, bazmax, bazsteps)])

    except AssertionError as err:
        raise err

    return vespagram_data


def plot_vespagram_backazimuth(st, s, bazmin, bazmax, bazsteps, winlen, stat='power', phase_weighting=False, n=1, display='contourf', outfile=None):
    '''
    Plots the vespagram for a seismic array over a given range of backazimuths, for a single scalar slowness, using the statistic specified.

    The chosen statistic is plotted as a function of time (in s) and backazimuth (in deg). This may be:-

    * 'amplitude' - the raw amplitude of the linear or nth root stack at each time and slowness step;
    * 'power' - the power in the linear or nth root beam calculated over a time window (length winlen) around each time step for each slowness;
    * 'F' - the F-statistic of the beam calculated over a time window (length winlen) around each time step for each slowness.

    Parameters
    ----------
    st : ObsPy Stream object
        Stream of SAC format seismograms for the seismic array, length K = no. of stations in array
    s : float
        Magnitude of the slowness vector, in s/km
    bazmin  : float
        Minimum backazimuth, in degrees
    bazmax  : float
        Maximum backazimuth, in degrees
    bazsteps  : int
        Integer number of steps between bazmin and bazmax for which to calculate the vespagram
    winlen : int
        Length of Hann window over which to calculate the power.
    stat : string
        Statistic to use for plotting the vespagram, either 'amplitude', 'power', or 'F'
    phase_weighting : Boolean
        Whether or not to apply phase-weighting to the stacks in the vespagram.
    n : int
        Order for the stack, or if phase_weighting==True, order of the weighting applied.
    display: string
        Option for plotting: either 'contourf' for filled contour plot, or 'contour' for contour plot. See matplotlib documentation for more details.
    outfile : string
        Filename for saving plot. display: string
        Option for plotting: either 'contourf' for filled contour plot, or 'contour' for contour plot. See matplotlib documentation for more details.

    '''

    assert display == 'contourf' or display == 'contour', "Invalid display option; must be 'contourf' or 'contour'"

    vespagram_data = np.array([])

    try:
        vespagram_data = vespagram_backazimuth(st, s, bazmin, bazmax, bazsteps, winlen, stat, phase_weighting, n)
    except AssertionError as err:
        print(err.args[0])
        return None

    timestring = str(st[0].stats.starttime.datetime)

    if stat == 'amplitude':
        label = "Amplitude"
        title = timestring + ": " + label + " Vespagram, n=" + str(n)
    elif stat == 'power':
        label = "Power"
        title = timestring + ": " + label + " Vespagram, n=" + str(n)
    elif stat == 'F':
        label = 'F'
        title = timestring + ": " + label + " Vespagram, n=" + str(n)
    else:
        raise AssertionError("'stat' argument must be one of 'amplitude', 'power' or 'F'")

    plt.figure(figsize=(16, 8))

    if display == 'contourf':
        plt.contourf(st[0].times(), np.linspace(bazmin, bazmax, bazsteps), vespagram_data[:, :])
    else:
        plt.contour(st[0].times(), np.linspace(bazmin, bazmax, bazsteps), vespagram_data[:, :])

    cb = plt.colorbar()
    cb.set_label(label)

    plt.xlabel("Time (s)", fontsize=16)
    plt.ylabel("Backazimuth (deg)", fontsize=16)
    plt.title(title, y=1.08, fontsize=18)
    plt.tick_params(axis='both', which='major', labelsize=14)

    if outfile is not None:
        plt.savefig(outfile, bbox_inches='tight')

    plt.show()
