# statics.py
# module: vespa.statics
# Functions for calculating slowness, azimuth and static corrections at seismic
# arrays using the vespagram functions

import numpy as np

from obspy.core import read, UTCDateTime
from obspy.core.event import read_events

from scipy.signal import correlate

from vespa.utils import get_arrivals, get_first_arrival
from vespa.vespagram import vespagram, vespagram_backazimuth, plot_vespagram, plot_vespagram_backazimuth
from vespa.stacking import linear_stack, get_shifts

def time_corrections(st, s_corr, inv=None, station_list=None, winlen=20):
        '''
        Calculates static corrections to the arrival time of a particular phase at a seismic array.

        Cross-correlates the best beam (calculated using the slowness correction, scorr, provided)
        for the array minus each station with the raw data trace from that station. The delay time
        that gives the maximum value in the cross-correlation is the static correction.

        Returns the adjustment (in s) to the arrival time for each station in the array.

        Parameters
        ----------
        st : ObsPy Stream object
            Seismic time series data for stations in the array
        inv : ObsPy Inventory object
            Inventory containing metadata for the seismic array
        scorr : bool
            Scalar slowness correction with which to calculate the statics
        winlen : int
            Window length to perform cross-correlation over to calculate statics

        Returns
        -------
        statics : list of floats
            Time shifts (static corrections to the arrival time for the seismic phase) for each station

        '''

        if station_list != None:
            stations = station_list

        elif inv != None:
            net = inv[0]
            stations = [sta.code for sta in net.stations]

        else:
            stations = [tr.stats.station for tr in st]

        st.sort(['station'])
        N = len(st)
        sampling_rate = st[0].stats.sampling_rate
        stime = st[0].stats.starttime
        baz_0 = st[0].stats.sac.baz

        # Calculate predicted slowness and travel time
        P_phase = get_first_arrival(st)
        s_0 = P_phase.slowness
        ptime = P_phase.time

        # Calculate shifts for best beam corrected from vespagram
        s_best = s_0 + s_corr
        shifts = get_shifts(st, s_best, baz_0)

        # Cut to narrow time window around predicted arrival
        st_trim = st.copy().trim(starttime=stime+ptime-winlen/2, endtime=stime+ptime+winlen/2)

        # Compute best beam
        best_beam = linear_stack(st_trim, s_best, baz_0)

        # Shift individual traces ready for cross-correlation
        stations_present = [] # Keep track of what stations are actually present in our data
        for i, tr in enumerate(st_trim):
            tr.data = np.roll(tr.data, shifts[i])
            stations_present.append(tr.stats.station)

        # Calculate static corrections
        statics = []
        #maxima = []
        i = 0
        for station in stations:
            if station in stations_present:
                beam = N * best_beam - st_trim[i].data # subtract individual trace from beam
                xcorr = correlate(beam, st_trim[i].data, 'full') # calculate cross-correlation with individual trace
                x_range = np.arange(-len(tr)+1, len(tr)) # want to find shift from centre, so realign the middle point to zero
                shift_corr = x_range[np.argmax(xcorr)] / sampling_rate # convert to time shift
                xcorr_max = np.max(xcorr)
                statics.append(shift_corr)
                #maxima.append(xcorr_max)
                i += 1
            else:
                statics.append(np.nan)
                #maxima.append(np.nan)

        return statics #, maxima

def amplitude_corrections(st, s_corr, inv=None, station_list=None, winlen=20):

    '''
        Calculates static corrections to the amplitude of the trace at a seismic array.

        Cross-correlates the best beam (calculated using the slowness correction, scorr, provided)
        for the array minus each station with the raw data trace from that station. The scaling for
        each trace to match the ampilitude of the best beam

        Returns the scaling for the amplitude for each station in the array.

        Parameters
        ----------
        st : ObsPy Stream object
            Seismic time series data for stations in the array
        inv : ObsPy Inventory object
            Inventory containing metadata for the seismic array
        scorr : bool
            Scalar slowness correction with which to calculate the statics
        winlen : int
            Window length to perform cross-correlation over to calculate statics

        Returns
        -------
        scalings : list of floats
    '''


    if station_list != None:
        stations = station_list

    elif inv != None:
        net = inv[0]
        stations = [sta.code for sta in net.stations]

    else:
        stations = [tr.stats.station for tr in st]

    N = len(st)
    sampling_rate = st[0].stats.sampling_rate
    stime = st[0].stats.starttime
    baz_0 = st[0].stats.sac.baz

    # Calculate predicted slowness and travel time
    P_phase = get_first_arrival(st)
    s_0 = P_phase.slowness
    ptime = P_phase.time

    # Calculate shifts for best beam corrected from vespagram
    s_corr = 0
    s_best = s_0 + s_corr
    shifts = get_shifts(st, s_best, baz_0)

    # Cut to narrow time window around predicted arrival
    st_trim = st.copy().trim(starttime=stime+ptime-winlen/2, endtime=stime+ptime+winlen/2)

    # Compute best beam
    best_beam = linear_stack(st_trim, s_best, baz_0)

    # Shift individual traces ready for cross-correlation
    stations_present = [] # Keep track of what stations are actually present in our data
    for i, tr in enumerate(st_trim):
        tr.data = np.roll(tr.data, shifts[i])
        stations_present.append(tr.stats.station)

    # Calculate amplitude correction
    scalings = []

    i = 0
    for station in stations:
        if station in stations_present:
            beam = N * best_beam - st_trim[i].data # subtract individual trace from beam
            xcorr = correlate(beam, st_trim[i].data, 'full') # calculate cross-correlation with individual trace
            autocorr = correlate(beam, beam, 'full') # calcuate autocorr to find beam energy

            # If one signal is a scaled, shifted version of another, with noise:
            # Signal 1: A, Signal 2: B = kA' + n
            #   A * B = A * (kA' + n) = k(A * A') + A * n
            # Ignoring the correlation with the noise,
            #   k = (A * B) / (A * A')
            # So to find the scaling, just divide by the power in the signal (the autocorrelation)

            xcorr_max = np.max(xcorr)
            autocorr_max = np.max(autocorr)

            scaling = (xcorr_max * (N - 1)) / autocorr_max
            scalings.append(scaling)

            i += 1
        else:
            scalings.append(np.nan)

    return scalings
