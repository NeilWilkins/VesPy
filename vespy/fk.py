# fk.py
# module: vespy.fk
# FK analysis functions for seismic data

import numpy as np
import matplotlib.pyplot as plt

from vespy.utils import get_station_coordinates

def fk_analysis(st, smax, fmin, fmax, tmin, tmax, stat='power'):
    '''
    Performs frequency-wavenumber space (FK) analysis on a stream of time series data for a given slowness range, frequency band and time window.

    For an input stream of length K, the output is a K x K array with values of the chosen statistic calculated on a slowness grid in the x and y spatial dimensions. This statistic can be one of:-

    * 'power' - the power in frequency-domain stack
    * 'semblance' - the semblance calculated in the frequency domain
    * 'F' - the F-statistic calculated in the frequency domain

    Before the FK analysis is performed, the seismograms are cut to a time window between tmin and tmax, and the data is bandpass-filtered between frequencies fmin and fmax.


    Parameters
    ----------
    st : ObsPy Stream object
        Stream of SAC format seismograms for the seismic array, length K = no. of stations in array
    smax  : float
        Maximum magnitude of slowness, used for constructing the slowness grid in both x and y directions
    fmin  : float
        Lower end of frequency band to perform FK analysis over
    fmax  : float
        Upper end of frequency band to perform FK analysis over
    tmin : float
        Start of time window, seismograms are cut between tmin and tmax before FK starts
    tmax : int
        End of time window, seismograms are cut between tmin and tmax before FK starts
    stat : string
        Statistic that is to be calculated over the slowness grid, either 'power', 'semblance', or 'F'

    Returns
    -------
    fk : NumPy array
        The chosen statistic calculated at each point in a K x K grid of slowness values in the x and y directions
    '''

    assert (stat == 'power' or stat == 'semblance' or stat == 'F'), "Argument 'stat' must be one of 'power', 'semblance' or 'F'"

    st = st.copy().trim(starttime=tmin, endtime=tmax)

    # Retrieve metadata: time step and number of stations to be stacked
    delta = st[0].stats.delta
    nbeam = len(st)

    # Pre-process, and filter to frequency window
    st.detrend()
    st.taper(type='cosine', max_percentage=0.05)

    st = st.copy().filter("bandpass", freqmin=fmin, freqmax=fmax)

    npts = st[0].stats.npts

    # Computer Fourier transforms for each trace
    fft_st = np.zeros((nbeam, (npts / 2) + 1), dtype=complex) # Length of real FFT is only half that of time series data
    for i, tr in enumerate(st):
        fft_st[i, :] = np.fft.rfft(tr.data) # Only need positive frequencies, so use rfft

    freqs = np.fft.fftfreq(npts, delta)[0:(npts / 2) + 1]

    # Slowness grid
    slow_x = np.linspace(-smax, smax, nbeam)
    slow_y = np.linspace(-smax, smax, nbeam)

    # Array geometry
    x, y = np.split(get_station_coordinates(st)[:, :2], 2, axis=1)
    # convert to km
    x /= 1000.
    y /= 1000.

    # Calculate the F-K spectrum
    fk = np.zeros((nbeam, nbeam))
    for ii in range(nbeam):
        for jj in range(nbeam):
            dt = slow_x[jj] * x + slow_y[ii] * y
            beam = np.sum(np.exp(-1j * 2 * np.pi * np.outer(dt, freqs)) * fft_st / nbeam, axis=0)
            fk[ii, jj] = np.vdot(beam, beam).real

    if stat == 'semblance' or stat == 'F':
        tracepower = np.vdot(fft_st, fft_st).real

        if stat == 'semblance':
            fk_semb = nbeam * fk / tracepower
            return fk_semb

        elif stat == 'F':
            fk_F = (nbeam - 1)* nbeam * fk / (tracepower - nbeam * fk)
            return fk_F

    else:
        return fk

def fk_slowness_vector(st, smax, fmin, fmax, tmin, tmax, stat='power'):
    '''
    Returns the slowness vector (amplitude and back azimuth) for time series data from a seismic array, as calculated using FK-beamforming.

    Performs frequency-wavenumber space (FK) analysis on a stream of time series data for a given slowness range, frequency band and time window.

    The output is a tuple containing the scalar slowness and backazimuth of the incoming wave, determined using a grid search to maximise the chosen beamforming statistic. This can be one of:-
    * 'power' - the power in frequency-domain stack
    * 'semblance' - the semblance calculated in the freqency domain
    * 'F' - the F-statistic calculated in the frequency domain

    Before the FK analysis is performed, the seismograms are cut to a time window between tmin and tmax, and the data is bandpass-filtered between frequencies fmin and fmax.

    Parameters
    ----------
    st : ObsPy Stream object
        Stream of SAC format seismograms for the seismic array, length K = no. of stations in array
    smax  : float
        Maximum magnitude of slowness, used for constructing the slowness grid in both x and y directions
    fmin  : float
        Lower end of frequency band to perform FK analysis over
    fmax  : float
        Upper end of frequency band to perform FK analysis over
    tmin : float
        Start of time window, seismograms are cut between tmin and tmax before FK starts
    tmax : int
        End of time window, seismograms are cut between tmin and tmax before FK starts
    stat : string
        Statistic that is to be calculated over the slowness grid, either 'power', 'semblance', or 'F'

    Returns
    -------
    slowness : float
        The scalar magnitude, in s/km, of the slowness of the incident seismic wave, as determined by the FK analysis
    backazimuth: float
        The backazimuth, in degrees, from North back to the epicentre of the incident seismic wave, as determined by the FK analysis
    '''

    nbeam = len(st)

    fk = fk_analysis(st, smax, fmin, fmax, tmin, tmax, stat)

    # Find maximum
    fkmax = np.unravel_index(np.argmax(fk), (nbeam, nbeam))

    # Slowness ranges
    slow_x = np.linspace(-smax, smax, nbeam)
    slow_y = np.linspace(-smax, smax, nbeam)

    slow_x_max = slow_x[fkmax[1]]
    slow_y_max = slow_y[fkmax[0]]

    slowness = np.hypot(slow_x_max, slow_y_max)
    backazimuth = np.degrees(np.arctan2(slow_x_max, slow_y_max))

    if backazimuth < 0:
        backazimuth += 360. # For backazimuths in range 0 - 360 deg

    return (slowness, backazimuth)

def fk_plot(st, smax, fmin, fmax, tmin, tmax, stat='power', outfile=None):
    '''
    Plots the results of FK analysis on a stream of time series data from a seismic array.

    Performs frequency-wavenumber space (FK) analysis on a stream of time series data for a given slowness range, frequency band and time window.

    This function plots the chosen statistic on a slowness grid in the x and y directions. The statistic can be one of:-
    * 'power' - the power in frequency-domain stack
    * 'semblance' - the semblance calculated in the freqency domain
    * 'F' - the F-statistic calculated in the frequency domain

    The title of the plot also contains the slowness and backazimuth for which the chosen statistic is maximised,

    Before the FK analysis is performed, the seismograms are cut to a time window between tmin and tmax, and the data is bandpass-filtered between frequencies fmin and fmax.

    Parameters
    ----------
    st : ObsPy Stream object
        Stream of SAC format seismograms for the seismic array, length K = no. of stations in array
    smax  : float
        Maximum magnitude of slowness, used for constructing the slowness grid in both x and y directions
    fmin  : float
        Lower end of frequency band to perform FK analysis over
    fmax  : float
        Upper end of frequency band to perform FK analysis over
    tmin : float
        Start of time window, seismograms are cut between tmin and tmax before FK starts
    tmax : int
        End of time window, seismograms are cut between tmin and tmax before FK starts
    stat : string
        Statistic that is to be calculated over the slowness grid, either 'power', 'semblance', or 'F'
    '''

    nbeam = len(st)

    fk = fk_analysis(st, smax, fmin, fmax, tmin, tmax, stat)

    # Slowness ranges
    slow_x = np.linspace(-smax, smax, nbeam)
    slow_y = np.linspace(-smax, smax, nbeam)

    # Find maximum
    fkmax = np.unravel_index(np.argmax(fk), (nbeam, nbeam))

    slow_x_max = slow_x[fkmax[1]]
    slow_y_max = slow_y[fkmax[0]]

    slowness = np.hypot(slow_x_max, slow_y_max)
    backazimuth = np.degrees(np.arctan2(slow_x_max, slow_y_max))

    if backazimuth < 0:
        backazimuth += 360.

    fig = plt.figure(figsize=(16, 14))
    fig.add_axes([0.5,0.5,0.45,0.45])

    plt.contourf(slow_x, slow_y, fk, 16)
    plt.grid('on', linestyle='-')
    plt.xlabel('slowness east (s/km)', fontsize=16)
    plt.ylabel('slowness north (s/km)', fontsize=16)
    cb = plt.colorbar()
    cb.set_label(stat, fontsize=16)
    plt.xlim(-smax, smax);
    plt.ylim(-smax, smax);
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.title("FK Analysis, slowness= " + '%.4f' % slowness + " s/km,  backazimuth= " + '%.1f' % backazimuth + " deg", y=1.08, fontsize=18)

    if outfile is not None:
        plt.savefig(outfile, bbox_inches='tight')

    plt.show()
