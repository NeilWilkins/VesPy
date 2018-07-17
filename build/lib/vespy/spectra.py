# spectra.py
# module: vespy.spectra
# Functions for generating and plotting amplitude and frequency spectra for seismic data

import numpy as np
import cmath
import matplotlib.pyplot as plt

def get_spectra(tr, power=False, unwrap=True):
    '''
    Calculates the amplitude and phase spectra of a seismic signal, along with the corresponding frequencies.

    Can optionally return power spectrum rather than amplitude, and by default unwraps the phase.

    Parameters
    ----------
    tr : ObsPy Trace object
        Seismic time series data. Trace stats should include length (npts) and timestep (delta)
    power : bool
        If True, returns power spectrum instead of amplitude
    unwrap : bool
        If True, unwraps the phase spectrum by changing data values to their 2*pi complement

    Returns
    -------
    freqs : NumPy Array
        The (positive only) frequencies associated with the spectrum
    spectrum: NumPy Array
        The amplitude (or, if specified, power) part of the spectrum
    phase : NumPy Array
        The (unwrapped) phase part of the spectrum
    '''

    fft = np.fft.rfft(tr.data)
    freqs = np.fft.rfftfreq(tr.stats.npts, tr.stats.delta)

    power_spectrum = fft * fft.conjugate()
    spectrum = power_spectrum

    if power == False:
        amplitude_spectrum = np.sqrt(power_spectrum)
        spectrum = amplitude_spectrum

    phase_spectrum = np.array([cmath.phase(d) for d in fft])
    phase = phase_spectrum

    if unwrap == True:
        unwrapped_phase = np.unwrap(phase_spectrum)
        phase = unwrapped_phase

    return freqs, spectrum, phase


def plot_spectra(tr, power=False, unwrap=True, x_scale='linear', y_scale='linear'):
    '''
    Plots the amplitude and phase spectra of a seismic time series.

    Can optionally return power spectrum rather than amplitude, and by default unwraps the phase.

    Options can be specified for logarithmic axes for the amplitude/power spectrum

    Parameters
    ----------
    tr : ObsPy Trace object
        Seismic time series data. Trace stats should include length (npts) and timestep (delta)
    power : bool
        If True, plots power spectrum instead of amplitude
    unwrap : bool
        If True, unwraps the phase spectrum by changing data values to their 2*pi complement
    x_scale : String
        Either 'linear', or 'log' - specifies the scale on the x-axis for the amplitude (or power) spectrum
    y_scale : String
        Either 'linear', or 'log' - specifies the scale on the y-axis for the amplitude (or power) spectrum
    '''

    freqs, spectrum, phase = get_spectra(tr, power=False, unwrap=unwrap)

    fig = plt.figure(figsize=(16, 12))
    spectrum_ax = fig.add_subplot(2, 1, 1)
    spectrum_ax.plot(freqs, spectrum)
    spectrum_ax.set_xscale(x_scale)
    spectrum_ax.set_yscale(y_scale)
    spectrum_ax.set_xlabel('Frequency (Hz)')

    if power:
        spectrum_ax.set_ylabel('Power')
    else:
        spectrum_ax.set_ylabel('Amplitude')

    phase_ax = fig.add_subplot(2, 1, 2)
    phase_ax.plot(freqs, phase)
    phase_ax.set_xlabel('Frequency (Hz)')
    phase_ax.set_ylabel('Phase (rad)')

    plt.show()
