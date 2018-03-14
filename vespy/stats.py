# stats.py
# module: vespy.stats
# Power statistic functions for seismic data

import numpy as np
from obspy.core import Trace

from vespy.stacking import get_shifts, linear_stack, nth_root_stack, phase_weighted_stack

def semblance(st, s, baz, winlen, n=1):
    '''
    Returns the semblance for a seismic array, for a beam of given slowness and backazimuth.

    Parameters
    ----------
    st : ObsPy Stream object
        Stream of SAC format seismograms for the seismic array, length K = no. of stations in array
    s  : float
        Magnitude of slowness vector, in s / km
    baz : float
        Backazimuth of slowness vector, (i.e. angle from North back to epicentre of event)
    winlen : int
        Length of Hann window over which to calculate the semblance.

    Returns
    -------
    semblance : NumPy array
        The semblance at the given slowness and backazimuth, as a time series.

    '''

    # Check that each channel has the same number of samples, otherwise we can't construct the beam properly
    assert len(set([len(tr) for tr in st])) == 1, "Traces in stream have different lengths, cannot stack."

    nsta = len(st)

    stack = nth_root_stack(st, s, baz, n)

    # Taper the linear stack
    stack_trace = Trace(stack)
    stack_trace.taper(type='cosine', max_percentage=0.05)
    stack = stack_trace.data

    shifts = get_shifts(st, s, baz)

    # Smooth data with sliding Hann window (i.e. convolution of signal and window function)
    window = np.hanning(winlen)

    shifted_st = st.copy()

    for i, tr in enumerate(shifted_st):
        tr.data = np.roll(tr.data, shifts[i]) # Shift data in each trace by its offset
        tr.taper(type='cosine', max_percentage=0.05) # Taper it

    # Calculate the power in the beam
    beampower = np.convolve(stack**2, window, mode='same')

    # Calculate the summed power of each trace
    tracepower = np.convolve(np.sum([tr.data**2 for tr in shifted_st], axis=0), window, mode='same')

    # Calculate semblance
    semblance = nsta * beampower / tracepower

    return semblance

def f_stat(stack, st, s, baz, winlen):

    '''
    Generalised F-statistic function that calculates F in the stream st using the user-defined stack as a signal estimate.

    Parameters
    ----------
    stack : NumPy Array
        Stacked data for the array - if linear stack, use vespa.stats.f_vespa() function instead. For other stacks (nth root, PWS, semblance, etc.) use this function
    st : ObsPy Stream object
        Stream of SAC format seismograms for the seismic array, length K = no. of stations in array
    s  : float
        Magnitude of slowness vector, in s / km
    baz : float
        Backazimuth of slowness vector, (i.e. angle from North back to epicentre of event)
    winlen : int
        Length of Hann window over which to calculate F.

    Returns
    -------
    F : NumPy array
        The F statistic in the beam of the given slowness and backazimuth, as a time series.

    '''

    # Check that each channel has the same number of samples, otherwise we can't construct the beam properly
    assert len(set([len(tr) for tr in st])) == 1, "Traces in stream have different lengths, cannot stack."

    nsta = len(st)

    # Taper the linear stack
    stack_trace = Trace(stack)
    stack_trace.taper(type='cosine', max_percentage=0.05)
    stack = stack_trace.data

    shifts = get_shifts(st, s, baz)

    # Smooth data with sliding Hann window (i.e. convolution of signal and window function)
    window = np.hanning(winlen)

    shifted_st = st.copy()

    for i, tr in enumerate(shifted_st):
        tr.data = np.roll(tr.data, shifts[i]) # Shift data in each trace by its offset
        tr.taper(type='cosine', max_percentage=0.05) # Taper it

    # Calculate the power in the beam
    beampower = np.convolve(stack**2, window, mode='same')

    # Calculate the summed power of each trace
    tracepower = np.convolve(np.sum([tr.data**2 for tr in shifted_st], axis=0), window, mode='same')

    # Calculate F
    F = (nsta - 1) * nsta * beampower / (tracepower - nsta * beampower)

    return F

def f_vespa(st, s, baz, winlen, n=1):
    '''
    Returns the F-statistic for a seismic array, for a beam of given slowness and backazimuth.

    Parameters
    ----------
    st : ObsPy Stream object
        Stream of SAC format seismograms for the seismic array, length K = no. of stations in array
    s  : float
        Magnitude of slowness vector, in s / km
    baz : float
        Backazimuth of slowness vector, (i.e. angle from North back to epicentre of event)
    winlen : int
        Length of Hann window over which to calculate F.

    Returns
    -------
    F : NumPy array
        The F statistic in the beam of the given slowness and backazimuth, as a time series.

    '''

    # Check that each channel has the same number of samples, otherwise we can't construct the beam properly
    assert len(set([len(tr) for tr in st])) == 1, "Traces in stream have different lengths, cannot stack."

    nsta = len(st)
    stack = nth_root_stack(st, s, baz, n)

    # Taper the linear stack
    stack_trace = Trace(stack)
    stack_trace.taper(type='cosine', max_percentage=0.05)
    stack = stack_trace.data

    shifts = get_shifts(st, s, baz)

    # Smooth data with sliding Hann window (i.e. convolution of signal and window function)
    window = np.hanning(winlen)

    shifted_st = st.copy()

    for i, tr in enumerate(shifted_st):
        tr.data = np.roll(tr.data, shifts[i]) # Shift data in each trace by its offset
        tr.taper(type='cosine', max_percentage=0.05) # Taper it

    # Calculate the power in the beam
    beampower = np.convolve(stack**2, window, mode='same')

    # Calculate the summed power of each trace
    tracepower = np.convolve(np.sum([tr.data**2 for tr in shifted_st], axis=0), window, mode='same')

    # Calculate F
    F = (nsta - 1) * nsta * beampower / (tracepower - nsta * beampower)

    return F

def power_vespa(st, s, baz, winlen):
    '''
    Returns the power vespa (i.e. the power in the linear delay-and-sum beam) for a seismic array, for a beam of given slowness and backazimuth.

    Parameters
    ----------
    st : ObsPy Stream object
        Stream of SAC format seismograms for the seismic array, length K = no. of stations in array
    s  : float
        Magnitude of slowness vector, in s / km
    baz : float
        Backazimuth of slowness vector, (i.e. angle from North back to epicentre of event)
    winlen : int
        Length of Hann window over which to calculate the power.

    Returns
    -------
    power : NumPy array
        The power of the linear beam at the given slowness and backazimuth, as a time series.
    '''
    amplitude = linear_stack(st, s, baz)
    power = np.convolve(amplitude**2, np.hanning(winlen), mode='same')
    return power

def n_power_vespa(st, s, baz, n, winlen):
    '''
    Returns the nth root power vespa (i.e. the power in the nth root stack) for a seismic array, for a beam of given slowness and backazimuth.

    Parameters
    ----------
    st : ObsPy Stream object
        Stream of SAC format seismograms for the seismic array, length K = no. of stations in array
    s  : float
        Magnitude of slowness vector, in s / km
    baz : float
        Backazimuth of slowness vector, (i.e. angle from North back to epicentre of event)
    n   : int
        Order of the nth root process
    winlen : int
        Length of Hann window over which to calculate the power.

    Returns
    -------
    power : NumPy array
        The power of the nth root stack at the given slowness and backazimuth, as a time series.
    '''
    amplitude = nth_root_stack(st, s, baz, n)
    power = np.convolve(amplitude**2, np.hanning(winlen), mode='same')
    return power

def pw_power_vespa(st, s, baz, n, winlen):
        '''
        Returns the phase-weighted power vespa (i.e. the power in the phase weighted stack) for a seismic array, for a beam of given slowness and backazimuth.

        Parameters
        ----------
        st : ObsPy Stream object
            Stream of SAC format seismograms for the seismic array, length K = no. of stations in array
        s  : float
            Magnitude of slowness vector, in s / km
        baz : float
            Backazimuth of slowness vector, (i.e. angle from North back to epicentre of event)
        n   : int
            Order of the phase-weighted stacking to be applied, default 1. Should be int, n >= 0.
        winlen : int
            Length of Hann window over which to calculate the power.

        Returns
        -------
        power : NumPy array
            The power of the nth root stack at the given slowness and backazimuth, as a time series.
        '''
        amplitude = phase_weighted_stack(st, s, baz, n)
        power = np.convolve(amplitude**2, np.hanning(winlen), mode='same')
        return power
