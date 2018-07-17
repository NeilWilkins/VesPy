# vespagram.py
# module: vespy.vespagram
# Module for performing velocity spectral analysis echniques on seismic datasets, and plotting vespagrams

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
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
    display: string
        Option for plotting: either 'contourf' for filled contour plot, or 'contour' for contour plot. See matplotlib documentation for more details.
    '''

    assert display == 'contourf' or display == 'contour', "Invalid display option; must be 'contourf' or 'contour'"

    vespagram_data = np.array([])

    try:
        vespagram_data = vespagram(st, smin, smax, ssteps, baz, winlen, stat, phase_weighting, n)
    except AssertionError as err:
        print err.args[0]
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
    plt.set_cmap(cm.viridis)

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
    plt.set_cmap(cm.viridis)

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

def vespagram_backazimuth(st, s, bazmin, bazmax, bazsteps, winlen, stat='power', n=1):
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

    Returns
    -------
    vespagram_data : NumPy array
        Array of values for the chosen statistic at each backazimuth and time step. Dimensions: bazsteps*len(tr) for traces tr in st.
    '''

    if stat == 'amplitude':
        vespagram_data = np.array([nth_root_stack(st, s, baz, n) for baz in np.linspace(bazmin, bazmax, bazsteps)])
    elif stat == 'power':
        vespagram_data = np.array([n_power_vespa(st, s, baz, n, winlen) for baz in np.linspace(bazmin, bazmax, bazsteps)])
    elif stat == 'F':
        vespagram_data = np.array([f_vespa(st, s, baz, winlen) for baz in np.linspace(bazmin, bazmax,bazsteps)])
    else:
        raise AssertionError("'stat' argument must be one of 'amplitude', 'power' or 'F'")

    return vespagram_data


def plot_vespagram_backazimuth(st, s, bazmin, bazmax, bazsteps, winlen, stat='power', n=1, display='contourf', outfile=None):
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
    display: string
        Option for plotting: either 'contourf' for filled contour plot, or 'contour' for contour plot. See matplotlib documentation for more details.

    '''

    assert display == 'contourf' or display == 'contour', "Invalid display option; must be 'contourf' or 'contour'"

    timestring = str(st[0].stats.starttime.datetime)

    vespagram_data = vespagram_backazimuth(st, s, bazmin, bazmax, bazsteps, winlen, stat, n=1)

    if stat == 'amplitude':
        label = "Amplitude"
        title = timestring + ": " + label + " Vespagram, n=" + str(n)
    elif stat == 'power':
        label = "Power"
        title = timestring + ": " + label + " Vespagram, n=" + str(n)
    elif stat == 'F':
        label = 'F'
        title = timestring + ": " + label + " Vespagram"
    else:
        raise AssertionError("'stat' argument must be one of 'amplitude', 'power' or 'F'")

    plt.figure(figsize=(16, 8))
    plt.set_cmap(cm.viridis)

    if display == 'contourf':
        plt.contourf(st[0].times(), np.linspace(bazmin, bazmax, bazsteps), vespagram_data[:, :])
    else:
        plt.contour(st[0].times(), np.linspace(bazmin, bazmax, bazsteps), vespagram_data[:, :])

    cb = plt.colorbar()
    cb.set_label(label)

    plt.xlabel("Time (s)", fontsize=16)
    plt.ylabel("Backazimuth (deg)", fontsize=16)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.title(title, y=1.08, fontsize=18)



    if outfile is not None:
        plt.savefig(outfile, bbox_inches='tight')

    plt.show()

def stack_vespagram(st, slowness_min, slowness_max, n_steps, baz, separation=0.5):

    '''
    Plots a stack vespagram for a seismic array over a given slowness range, for a single backazimuth.

    The individual beam traces for each slowness step are plotted as a function of time (in s) and slowness (in s/km).

    Parameters
    ----------
    st : ObsPy Stream object
        Stream of SAC format seismograms for the seismic array, length K = no. of stations in array
    slowness_min : float
        Minimum magnitude of the slowness vector, in s/km
    slowness_max : float
        Maximum magnitude of the slowness vector, in s/km
    n_steps  : int
        Maximum backazimuth, in degrees
    baz : float
        Backazimuth of slowness vector, (i.e. angle from North back to epicentre of event)
    winlen : int
        Length of Hann window over which to calculate the power.
    separation : float
        Fraction of a single beam that overlaps with its neighbours. Smaller separations make a denser vespagram.
    '''

    slowness_range = np.linspace(slowness_min, slowness_max, n_steps)

    vespas = [linear_stack(st, slowness, baz) for slowness in slowness_range]
    times = st[0].times()

    plt.figure(figsize=(14, 10))

    # Find maximum and minimum amplitudes of the beam in order to set y-axes for beam traces.
    max_y = np.max(np.array(vespas))
    min_y = np.min(np.array(vespas))

    trace_height = 1. / (1 + (n_steps - 1) * separation)

    # Plot beam trace for each slowness value
    for i, trace in enumerate(vespas):
        # rect = [left, bottom, width, height]
        trace_bottom = (1 - trace_height) + (i - n_steps + 1) * separation * trace_height
        rect = [0, trace_bottom, 1, trace_height]
        ax = plt.axes(rect, axisbg=None, frameon=False)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.plot(times, trace, alpha=0.6)
        plt.ylim(min_y, max_y);
        plt.xlim(0, 350);

    # Create external axes to contain beam traces, and set its slowness range
    plt.axes([0, 0, 1, 1], axisbg='none')
    main_ax_slowness_max = slowness_max / (1 - 0.5 * trace_height)
    main_ax_slowness_min = slowness_min - 0.5 * trace_height * main_ax_slowness_max

    plt.ylim(main_ax_slowness_min, main_ax_slowness_max)
    plt.xlim(0, 350)
    plt.xlabel('Time (s)')
    plt.yticks(fontsize=14);
    plt.xticks(fontsize=14);
    plt.ylabel('Slowness (s / km)')

    plt.show()
