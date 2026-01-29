# Copyright (C) 2015-2023: The University of Edinburgh
#                 Authors: Craig Warren and Antonis Giannopoulos
#
# This file is part of gprMax.
#
# gprMax is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# gprMax is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with gprMax.  If not, see <http://www.gnu.org/licenses/>.

import argparse
import os
import sys

import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from gprMax.exceptions import CmdInputError


def power(q):
    #q[np.invert(np.isfinite(q))] = 0
    p = 20 * np.log10( np.abs(q) )
    #p[np.invert(np.isfinite(p))] = 0
    return p


def calculate_antenna_params(filename, tltxnumber=1, tlrxnumber=None, rxnumber=None, rxcomponent=None):
    """Calculates antenna parameters - incident, reflected and total volatges and currents; s11, (s21) and input impedance.

    Args:
        filename (string): Filename (including path) of output file.
        tltxnumber (int): Transmitter antenna - transmission line number
        tlrxnumber (int): Receiver antenna - transmission line number
        rxnumber (int): Receiver antenna - output number
        rxcomponent (str): Receiver antenna - output electric field component

    Returns:
        antennaparams (dict): Antenna parameters.
    """

    # Open output file and read some attributes
    f = h5py.File(filename, 'r')
    dxdydz = f.attrs['dx_dy_dz']
    dt = f.attrs['dt']
    iterations = f.attrs['Iterations']

    # Calculate time array and frequency bin spacing
    time = np.linspace(0, (iterations - 1) * dt, num=iterations)
    df = 1 / np.amax(time)

    print('Time window: {:g} s ({} iterations)'.format(np.amax(time), iterations))
    print('Time step: {:g} s'.format(dt))
    print('Frequency bin spacing: {:g} Hz'.format(df))

    # Read/calculate voltages and currents from transmitter antenna
    tltxpath = '/tls/tl' + str(tltxnumber) + '/'

    # Incident voltages/currents
    Vinc = f[tltxpath + 'Vinc'][:]
    Iinc = f[tltxpath + 'Iinc'][:]

    # Total (incident + reflected) voltages/currents
    Vtotal = f[tltxpath + 'Vtotal'][:]
    Itotal = f[tltxpath + 'Itotal'][:]

    # Reflected voltages/currents
    Vref = Vtotal - Vinc
    Iref = Itotal - Iinc

    # If a receiver antenna is used (with a transmission line or receiver), get received voltage for s21
    if tlrxnumber:
        tlrxpath = '/tls/tl' + str(tlrxnumber) + '/'
        Vrec = f[tlrxpath + 'Vtotal'][:]

    elif rxnumber:
        rxpath = '/rxs/rx' + str(rxnumber) + '/'
        availableoutputs = list(f[rxpath].keys())

        if rxcomponent not in availableoutputs:
            raise CmdInputError('{} output requested, but the available output for receiver {} is {}'.format(rxcomponent, rxnumber, ', '.join(availableoutputs)))

        rxpathorig = rxpath
        rxpath += rxcomponent

        # Received voltage
        if rxcomponent == 'Ex':
            Vrec = f[rxpath][:] * -1 * dxdydz[0]
            Irec = f[rxpathorig + 'Ix'][:]
        elif rxcomponent == 'Ey':
            Vrec = f[rxpath][:] * -1 * dxdydz[1]
            Irec = f[rxpathorig + 'Iy'][:]
        elif rxcomponent == 'Ez':
            Vrec = f[rxpath][:] * -1 * dxdydz[2]
            Irec = f[rxpathorig + 'Iz'][:]
    f.close()

    # Frequency bins
    freqs = np.fft.fftfreq(Vinc.size, d=dt)

    # Delay correction - current lags voltage, so delay voltage to match current timestep
    delaycorrection = np.exp(1j * 2 * np.pi * freqs * (dt / 2))

    # Calculate s11 and (optionally) s21
    with np.errstate(divide='ignore'):
        s11 = np.abs(np.fft.fft(Vref) / np.fft.fft(Vinc))
    if tlrxnumber or rxnumber:
        with np.errstate(divide='ignore'):
            s21 = np.abs(np.fft.fft(Vrec) / np.fft.fft(Vinc))

    # Calculate input impedance
    with np.errstate(divide='ignore'):
        zin = (np.fft.fft(Vtotal) * delaycorrection) / np.fft.fft(Itotal)

    # Calculate received impedance
    with np.errstate(divide='ignore'):
        zrec = np.divide( (np.fft.fft(Vrec) * delaycorrection), np.fft.fft(Irec), where=np.fft.fft(Irec)!=0 )

    # Calculate input admittance
    with np.errstate(divide='ignore'):
        yin = np.fft.fft(Itotal) / (np.fft.fft(Vtotal) * delaycorrection)

    # Convert to decibels (ignore warning from taking a log of any zero values)
    with np.errstate(divide='ignore'):
#        Vincp = 20 * np.log10(np.abs((np.fft.fft(Vinc) * delaycorrection)))
#        Iincp = 20 * np.log10(np.abs(np.fft.fft(Iinc)))
#        Vrefp = 20 * np.log10(np.abs((np.fft.fft(Vref) * delaycorrection)))
#        Irefp = 20 * np.log10(np.abs(np.fft.fft(Iref)))
#        Vtotalp = 20 * np.log10(np.abs((np.fft.fft(Vtotal) * delaycorrection)))
#        Itotalp = 20 * np.log10(np.abs(np.fft.fft(Itotal)))
#        Vrecp = 20 * np.log10(np.abs((np.fft.fft(Vrec) * delaycorrection)))
#        Irecp = 20 * np.log10(np.abs(np.fft.fft(Irec)))
        Vincp = np.fft.fft(Vinc) * delaycorrection
        Iincp = np.fft.fft(Iinc)
        Vrefp = np.fft.fft(Vref) * delaycorrection
        Irefp = np.fft.fft(Iref)
        Vtotalp = np.fft.fft(Vtotal) * delaycorrection
        Itotalp = np.fft.fft(Itotal)
        Vrecp = np.fft.fft(Vrec) * delaycorrection
        Irecp = np.fft.fft(Irec)
        s11 = 20 * np.log10(s11)


    # Replace any NaNs or Infs from zero division
    Vincp[np.invert(np.isfinite(Vincp))] = 0
    Iincp[np.invert(np.isfinite(Iincp))] = 0
    Vrefp[np.invert(np.isfinite(Vrefp))] = 0
    Irefp[np.invert(np.isfinite(Irefp))] = 0
    Vtotalp[np.invert(np.isfinite(Vtotalp))] = 0
    Itotalp[np.invert(np.isfinite(Itotalp))] = 0
    s11[np.invert(np.isfinite(s11))] = 0
    Vrecp[np.invert(np.isfinite(Vrefp))] = 0
    Irecp[np.invert(np.isfinite(Irefp))] = 0

    # Create dictionary of antenna parameters
    antennaparams = {'time': time, 'freqs': freqs, 'Vinc': Vinc, 'Vincp': Vincp, 'Iinc': Iinc, 'Iincp': Iincp,
                     'Vref': Vref, 'Vrefp': Vrefp, 'Iref': Iref, 'Irefp': Irefp,
                     'Vtotal': Vtotal, 'Vtotalp': Vtotalp, 'Itotal': Itotal, 'Itotalp': Itotalp,
                     's11': s11, 'zin': zin, 'yin': yin, 'zrec':zrec, 'Vrec': Vrec, 'Vrecp': Vrecp, 'Irec': Irec, 'Irecp': Irecp}
    if tlrxnumber or rxnumber:
        with np.errstate(divide='ignore'): # Ignore warning from taking a log of any zero values
            s21 = 20 * np.log10(s21)
        s21[np.invert(np.isfinite(s21))] = 0
        antennaparams['s21'] = s21

    return antennaparams


def mpl_plot(filename, time, freqs, Vinc, Vincp, Iinc, Iincp, Vref, Vrefp, Iref, Irefp, Vtotal, Vtotalp, Itotal, Itotalp, s11, zin, zrec, yin, Vrec, Vrecp, Irec, Irecp, s21=None):
    """Plots antenna parameters - incident, reflected and total volatges and currents; s11, (s21) and input impedance.

    Args:
        filename (string): Filename (including path) of output file.
        time (array): Simulation time.
        freq (array): Frequencies for FFTs.
        Vinc, Vincp, Iinc, Iincp (array): Time and frequency domain representations of incident voltage and current.
        Vref, Vrefp, Iref, Irefp (array): Time and frequency domain representations of reflected voltage and current.
        Vtotal, Vtotalp, Itotal, Itotalp (array): Time and frequency domain representations of total voltage and current.
        s11, s21 (array): s11 and, optionally, s21 parameters.
        zin, yin (array): Input impedance and input admittance parameters.

    Returns:
        plt (object): matplotlib plot object.
    """

    # Set plotting range
    pltrangemin = 1
    # To a certain drop from maximum power
    pltrangemax = np.where((np.amax(power(Vincp[1::])) - power(Vincp[1::])) > 60)[0][0] + 1
    # To a maximum frequency
    # pltrangemax = np.where(freqs > 6e9)[0][0]
    pltrange = np.s_[pltrangemin:pltrangemax]

    # Print some useful values from s11, and input impedance
    s11minfreq = np.where(s11[pltrange] == np.amin(s11[pltrange]))[0][0]
    print('s11 minimum: {:g} dB at {:g} Hz'.format(np.amin(s11[pltrange]), freqs[s11minfreq + pltrangemin]))
    print('At {:g} Hz...'.format(freqs[s11minfreq + pltrangemin]))
    print('Input impedance: {:.1f}{:+.1f}j Ohms'.format(np.abs(zin[s11minfreq + pltrangemin]), zin[s11minfreq + pltrangemin].imag))
    # print('Input admittance (mag): {:g} S'.format(np.abs(yin[s11minfreq + pltrangemin])))
    # print('Input admittance (phase): {:.1f} deg'.format(np.angle(yin[s11minfreq + pltrangemin], deg=True)))

    # Figure 1
    # Plot incident voltage
    fig1, ax = plt.subplots(num='Transmitter transmission line parameters', figsize=(30, 16), facecolor='w', edgecolor='w')
    ax.axis('off')
    gs1 = gridspec.GridSpec(8, 2, hspace=0.7)
    ax = plt.subplot(gs1[0, 0])
    ax.plot(time, Vinc, 'r', lw=2, label='Vinc')
    ax.set_title('Incident voltage')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Voltage [V]')
    ax.set_xlim([0, np.amax(time)])
    ax.grid(which='both', axis='both', linestyle='-.')

    # Plot frequency spectra of incident voltage
    ax = plt.subplot(gs1[0, 1])
    markerline, stemlines, baseline = ax.stem(freqs[pltrange], power(Vincp[pltrange]), '-.')
    plt.setp(baseline, 'linewidth', 0)
    plt.setp(stemlines, 'color', 'r')
    plt.setp(markerline, 'markerfacecolor', 'r', 'markeredgecolor', 'r')
    ax.plot(freqs[pltrange], power(Vincp[pltrange]), 'r', lw=2)
    ax.set_title('Incident voltage')
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('Power [dB]')
    ax.grid(which='both', axis='both', linestyle='-.')

    # Plot incident current
    ax = plt.subplot(gs1[1, 0])
    ax.plot(time, Iinc, 'b', lw=2, label='Vinc')
    ax.set_title('Incident current')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Current [A]')
    ax.set_xlim([0, np.amax(time)])
    ax.grid(which='both', axis='both', linestyle='-.')

    # Plot frequency spectra of incident current
    ax = plt.subplot(gs1[1, 1])
    markerline, stemlines, baseline = ax.stem(freqs[pltrange], power(Iincp[pltrange]), '-.')
    plt.setp(baseline, 'linewidth', 0)
    plt.setp(stemlines, 'color', 'b')
    plt.setp(markerline, 'markerfacecolor', 'b', 'markeredgecolor', 'b')
    ax.plot(freqs[pltrange], power(Iincp[pltrange]), 'b', lw=2)
    ax.set_title('Incident current')
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('Power [dB]')
    ax.grid(which='both', axis='both', linestyle='-.')

    # Plot total voltage
    ax = plt.subplot(gs1[2, 0])
    ax.plot(time, Vtotal, 'r', lw=2, label='Vinc')
    ax.set_title('Total (incident + reflected) voltage')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Voltage [V]')
    ax.set_xlim([0, np.amax(time)])
    ax.grid(which='both', axis='both', linestyle='-.')

    # Plot frequency spectra of total voltage
    ax = plt.subplot(gs1[2, 1])
    markerline, stemlines, baseline = ax.stem(freqs[pltrange], power(Vtotalp[pltrange]), '-.')
    plt.setp(baseline, 'linewidth', 0)
    plt.setp(stemlines, 'color', 'r')
    plt.setp(markerline, 'markerfacecolor', 'r', 'markeredgecolor', 'r')
    ax.plot(freqs[pltrange], power(Vtotalp[pltrange]), 'r', lw=2)
    ax.set_title('Total (incident + reflected) voltage')
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('Power [dB]')
    ax.grid(which='both', axis='both', linestyle='-.')

    # Plot total current
    ax = plt.subplot(gs1[3, 0])
    ax.plot(time, Itotal, 'b', lw=2, label='Vinc')
    ax.set_title('Total (incident + reflected) current')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Current [A]')
    ax.set_xlim([0, np.amax(time)])
    ax.grid(which='both', axis='both', linestyle='-.')

    # Plot frequency spectra of total current
    ax = plt.subplot(gs1[3, 1])
    markerline, stemlines, baseline = ax.stem(freqs[pltrange], power(Itotalp[pltrange]), '-.')
    plt.setp(baseline, 'linewidth', 0)
    plt.setp(stemlines, 'color', 'b')
    plt.setp(markerline, 'markerfacecolor', 'b', 'markeredgecolor', 'b')
    ax.plot(freqs[pltrange], power(Itotalp[pltrange]), 'b', lw=2)
    ax.set_title('Total (incident + reflected) current')
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('Power [dB]')
    ax.grid(which='both', axis='both', linestyle='-.')

    # Plot reflected (reflected) voltage
    ax = plt.subplot(gs1[4, 0])
    ax.plot(time, Vref, 'r', lw=2, label='Vref')
    ax.set_title('Reflected voltage')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Voltage [V]')
    ax.set_xlim([0, np.amax(time)])
    ax.grid(which='both', axis='both', linestyle='-.')

    ## Plot frequency spectra of reflected voltage
    ax = plt.subplot(gs1[4, 1])
    markerline, stemlines, baseline = ax.stem(freqs[pltrange], power(Vrefp[pltrange]), '-.')
    plt.setp(baseline, 'linewidth', 0)
    plt.setp(stemlines, 'color', 'r')
    plt.setp(markerline, 'markerfacecolor', 'r', 'markeredgecolor', 'r')
    ax.plot(freqs[pltrange], power(Vrefp[pltrange]), 'r', lw=2)
    ax.set_title('Reflected voltage')
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('Power [dB]')
    ax.grid(which='both', axis='both', linestyle='-.')

    ## Plot reflected (reflected) current
    ax = plt.subplot(gs1[5, 0])
    ax.plot(time, Iref, 'b', lw=2, label='Iref')
    ax.set_title('Reflected current')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Current [A]')
    ax.set_xlim([0, np.amax(time)])
    ax.grid(which='both', axis='both', linestyle='-.')

    ## Plot frequency spectra of reflected current
    ax = plt.subplot(gs1[5, 1])
    markerline, stemlines, baseline = ax.stem(freqs[pltrange], power(Irefp[pltrange]), '-.')
    plt.setp(baseline, 'linewidth', 0)
    plt.setp(stemlines, 'color', 'b')
    plt.setp(markerline, 'markerfacecolor', 'b', 'markeredgecolor', 'b')
    ax.plot(freqs[pltrange], power(Irefp[pltrange]), 'b', lw=2)
    ax.set_title('Reflected current')
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('Power [dB]')
    ax.grid(which='both', axis='both', linestyle='-.')

    # Plot received  voltage
    ax = plt.subplot(gs1[6, 0])
    ax.plot(time, Vrec, 'r', lw=2, label='Vref')
    ax.set_title('Received voltage')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Voltage [V]')
    ax.set_xlim([0, np.amax(time)])
    ax.grid(which='both', axis='both', linestyle='-.')

    ## Plot frequency spectra of received voltage
    ax = plt.subplot(gs1[6, 1])
    markerline, stemlines, baseline = ax.stem(freqs[pltrange], power(Vrecp[pltrange]), '-.')
    plt.setp(baseline, 'linewidth', 0)
    plt.setp(stemlines, 'color', 'r')
    plt.setp(markerline, 'markerfacecolor', 'r', 'markeredgecolor', 'r')
    ax.plot(freqs[pltrange], power(Vrecp[pltrange]), 'r', lw=2)
    ax.set_title('Received voltage')
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('Power [dB]')
    ax.grid(which='both', axis='both', linestyle='-.')


    # Plot received current
    ax = plt.subplot(gs1[7, 0])
    ax.plot(time, Irec, 'b', lw=2, label='Irec')
    ax.set_title('Received current')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Current [A]')
    ax.set_xlim([0, np.amax(time)])
    ax.grid(which='both', axis='both', linestyle='-.')

    ## Plot frequency spectra of received current
    ax = plt.subplot(gs1[7, 1])
    markerline, stemlines, baseline = ax.stem(freqs[pltrange], power(Irecp[pltrange]), '-.')
    plt.setp(baseline, 'linewidth', 0)
    plt.setp(stemlines, 'color', 'b')
    plt.setp(markerline, 'markerfacecolor', 'b', 'markeredgecolor', 'b')
    ax.plot(freqs[pltrange], power(Irecp[pltrange]), 'b', lw=2)
    ax.set_title('Received current')
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('Power [dB]')
    ax.grid(which='both', axis='both', linestyle='-.')




    # ========================================================================================================
    # Figure 2
    # Plot frequency spectra of s11
    fig2, ax = plt.subplots(num='Antenna parameters', figsize=(20, 12), facecolor='w', edgecolor='w')
    ax.axis('off')
    gs2 = gridspec.GridSpec(4, 2, hspace=0.3)
    ax = plt.subplot(gs2[0, 0])
    markerline, stemlines, baseline = ax.stem(freqs[pltrange], s11[pltrange], '-.')
    plt.setp(baseline, 'linewidth', 0)
    plt.setp(stemlines, 'color', 'g')
    plt.setp(markerline, 'markerfacecolor', 'g', 'markeredgecolor', 'g')
    ax.plot(freqs[pltrange], s11[pltrange], 'g', lw=2)
    ax.set_title('s11 $ = 20 log_{10}\\left| V_{ref} / V_{inc} \\right|$')
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('Power [dB]')
    # ax.set_xlim([0, 5e9])
    # ax.set_ylim([-25, 0])
    ax.grid(which='both', axis='both', linestyle='-.')

    # Plot frequency spectra of s21
    if s21 is not None:
        ax = plt.subplot(gs2[0, 1])
        markerline, stemlines, baseline = ax.stem(freqs[pltrange], s21[pltrange], '-.')
        plt.setp(baseline, 'linewidth', 0)
        plt.setp(stemlines, 'color', 'g')
        plt.setp(markerline, 'markerfacecolor', 'g', 'markeredgecolor', 'g')
        ax.plot(freqs[pltrange], s21[pltrange], 'g', lw=2)
        ax.set_title('s21 $ = 20 log_{10}\\left| V_{out} / V_{inc} \\right|$')
        ax.set_xlabel('Frequency [Hz]')
        ax.set_ylabel('Power [dB]')
        # ax.set_xlim([0.88e9, 1.02e9])
        # ax.set_ylim([-25, 50])
        ax.grid(which='both', axis='both', linestyle='-.')

    # Plot input resistance (real part of impedance)
    ax = plt.subplot(gs2[1, 0])
    markerline, stemlines, baseline = ax.stem(freqs[pltrange], zin[pltrange].real, '-.')
    plt.setp(baseline, 'linewidth', 0)
    plt.setp(stemlines, 'color', 'g')
    plt.setp(markerline, 'markerfacecolor', 'g', 'markeredgecolor', 'g')
    ax.plot(freqs[pltrange], zin[pltrange].real, 'g', lw=2)
    ax.set_title('Impedance (real) $ = Re Z = Re(V_{tot} / I_{tot})$')
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('Resistance [Ohms]')
    # ax.set_xlim([0.88e9, 1.02e9])
    ax.set_ylim(bottom=0)
    # ax.set_ylim([0, 300])
    ax.grid(which='both', axis='both', linestyle='-.')

    # Plot input reactance (imaginery part of impedance)
    ax = plt.subplot(gs2[1, 1])
    markerline, stemlines, baseline = ax.stem(freqs[pltrange], zin[pltrange].imag, '-.')
    plt.setp(baseline, 'linewidth', 0)
    plt.setp(stemlines, 'color', 'g')
    plt.setp(markerline, 'markerfacecolor', 'g', 'markeredgecolor', 'g')
    ax.plot(freqs[pltrange], zin[pltrange].imag, 'g', lw=2)
    ax.set_title('Impedance (imag) $= Im Z = Im(V_{tot} / I_{tot})$')
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('Reactance [Ohms]')
    # ax.set_xlim([0.88e9, 1.02e9])
    # ax.set_ylim([-300, 300])
    ax.grid(which='both', axis='both', linestyle='-.')

    # Plot input admittance (magnitude)
    ax = plt.subplot(gs2[2, 0])
    markerline, stemlines, baseline = ax.stem(freqs[pltrange], np.abs(yin[pltrange]), '-.')
    plt.setp(baseline, 'linewidth', 0)
    plt.setp(stemlines, 'color', 'g')
    plt.setp(markerline, 'markerfacecolor', 'g', 'markeredgecolor', 'g')
    ax.plot(freqs[pltrange], np.abs(yin[pltrange]), 'g', lw=2)
    ax.set_title('Admittance (magnitude) $= |Y| = |I_{tot} / V_{tot}|$')
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('Admittance [Siemens]')
    #ax.set_xlim([0.88e9, 1.02e9])
    #ax.set_ylim([0, 2e-4])
    ax.grid(which='both', axis='both', linestyle='-.')

    # Plot input admittance (phase)
    ax = plt.subplot(gs2[2, 1])
    markerline, stemlines, baseline = ax.stem(freqs[pltrange], np.angle(yin[pltrange], deg=True), '-.')
    plt.setp(baseline, 'linewidth', 0)
    plt.setp(stemlines, 'color', 'g')
    plt.setp(markerline, 'markerfacecolor', 'g', 'markeredgecolor', 'g')
    ax.plot(freqs[pltrange], np.angle(yin[pltrange], deg=True), 'g', lw=2)
    ax.set_title('Admittance (phase) $= angle(Y) = angle(I_{tot} / V_{tot})$')
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('Phase [degrees]')
    #ax.set_xlim([0.88e9, 1.02e9])
    ax.set_ylim([89, 90])
    ax.grid(which='both', axis='both', linestyle='-.')

    # Plot received impedance (magnitude)
    print(np.abs(zrec[pltrange]))
    ax = plt.subplot(gs2[3, 0])
    markerline, stemlines, baseline = ax.stem(freqs[pltrange], np.abs(zrec[pltrange]), '-.')
    plt.setp(baseline, 'linewidth', 0)
    plt.setp(stemlines, 'color', 'g')
    plt.setp(markerline, 'markerfacecolor', 'g', 'markeredgecolor', 'g')
    ax.plot(freqs[pltrange], np.abs(zrec[pltrange]), 'g', lw=2)
    ax.set_title('Received impedance (magnitude) $= |Y| = |V_{rec} / I_{rec}|$')
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('Impedance [Ohms]')
    #ax.set_xlim([0.88e9, 1.02e9])
    #ax.set_ylim([0, 2e-4])
    ax.grid(which='both', axis='both', linestyle='-.')

    # Plot received impedance (angle)
    print(np.angle(zrec[pltrange], deg=True))
    ax = plt.subplot(gs2[3, 1])
    markerline, stemlines, baseline = ax.stem(freqs[pltrange], np.angle(zrec[pltrange], deg=True), '-.')
    plt.setp(baseline, 'linewidth', 0)
    plt.setp(stemlines, 'color', 'g')
    plt.setp(markerline, 'markerfacecolor', 'g', 'markeredgecolor', 'g')
    ax.plot(freqs[pltrange], np.angle(zrec[pltrange], deg=True), 'g', lw=2)
    ax.set_title('Received impedance (phase) $= angle(Y) = angle(V_{rec} / I_{rec})$')
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('Phase [degrees]')
    #ax.set_xlim([0.88e9, 1.02e9])
    ax.set_ylim([90, 91])
    ax.grid(which='both', axis='both', linestyle='-.')

    # Save a PDF/PNG of the figure
    # fig1.savefig(os.path.splitext(os.path.abspath(filename))[0] + '_tl_params.png', dpi=150, format='png', bbox_inches='tight', pad_inches=0.1)
    # fig2.savefig(os.path.splitext(os.path.abspath(filename))[0] + '_ant_params.png', dpi=150, format='png', bbox_inches='tight', pad_inches=0.1)
    dirname = os.path.dirname(os.path.relpath(filename)).replace('/', '_') #.split('/')[-1]
    fig1.savefig(dirname + '_tl_params.pdf', dpi=None, format='pdf', bbox_inches='tight', pad_inches=0.1)
    fig2.savefig(dirname + '_ant_params.pdf', dpi=None, format='pdf', bbox_inches='tight', pad_inches=0.1)

    plt.close(fig1)
    plt.close(fig2)

    return plt


if __name__ == "__main__":

    # Parse command line arguments)
    parser = argparse.ArgumentParser(description='Plots antenna parameters (s11, s21 parameters and input impedance) from an output file containing a transmission line source.', usage='cd gprMax; python -m tools.plot_antenna_params outputfile')
    parser.add_argument('outputfile', help='name of output file including path')
    parser.add_argument('--tltx-num', default=1, type=int, help='transmitter antenna - transmission line number')
    parser.add_argument('--tlrx-num', type=int, help='receiver antenna - transmission line number')
    parser.add_argument('--rx-num', type=int, help='receiver antenna - output number')
    parser.add_argument('--rx-component', type=str, help='receiver antenna - output electric field component', choices=['Ex', 'Ey', 'Ez'])
    args = parser.parse_args()

    antennaparams = calculate_antenna_params(args.outputfile, args.tltx_num, args.tlrx_num, args.rx_num, args.rx_component)
    plthandle = mpl_plot(args.outputfile, **antennaparams)
#    plthandle.show()
