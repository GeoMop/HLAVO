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

from plot_antenna_params import calculate_antenna_params, mpl_plot, power
import argparse
import os
import sys

import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from gprMax.exceptions import CmdInputError


def mpl_plot_diff(filenames, params, output_plot_file='diff_params.pdf'):
    """Plots antenna parameters - incident, reflected and total volatges and currents; s11, (s21) and input impedance.

    Args:
        filenames (array of string): Filenames (including path) of output files from gprMax simulations.
        params (array of dict): Dictionaries containing data:
            time (array): Simulation time.
            freq (array): Frequencies for FFTs.
            Vinc, Vincp, Iinc, Iincp (array): Time and frequency domain representations of incident voltage and current.
            Vref, Vrefp, Iref, Irefp (array): Time and frequency domain representations of reflected voltage and current.
            Vtotal, Vtotalp, Itotal, Itotalp (array): Time and frequency domain representations of total voltage and current.
            s11, s21 (array): s11 and, optionally, s21 parameters.
            zin, yin (array): Input impedance and input admittance parameters.
            Vrec, Vrecp, Irec, Irecp, zrec: Received voltage, current and impedance.

    Returns:
        plt (object): matplotlib plot object.
    """

    dirnames = []
    for fn in filenames:
#        dirnames.append( os.path.dirname(os.path.relpath(fn)).split('/')[-1] )
        dirnames.append( os.path.dirname(os.path.relpath(fn)) )

    freqs = params[0]['freqs']
    times = params[0]['time']
    Irecp = []
    Vinc = []
    Vincp = []
    Vrec = []
    Vrecp = []
    Vref = []
    Zrecp = []
    for par in params:
        if np.linalg.norm(freqs - par['freqs']) > 0:
            print('Frequencies are not the same!')
            exit()

        Irecp.append( par['Irecp'] )
        Vinc.append( par['Vinc'] )
        Vincp.append( par['Vincp'] )
        Vrec.append( par['Vrec'] )
        Vrecp.append( par['Vrecp'] )
        Vref.append( par['Vref'] )
        Zrecp.append( par['zrec'] )

    # Set plotting range
    pltrangemin = 1
    # To a certain drop from maximum power
    #print('Vrecp: ', Vrecp[0][1::])
    pltrangemax = np.where((np.amax(power(Vrecp[0][1::])) - power(Vrecp[0][1::])) > 60)[0][0] + 1
    # To a maximum frequency
    # pltrangemax = np.where(freqs > 6e9)[0][0]
    pltrange = np.s_[pltrangemin:pltrangemax]
    # manually set frequency range
    xlim = [0.5e8,3e8]

    # Print some useful values from s11, and input impedance
    # s11minfreq = np.where(s11[pltrange] == np.amin(s11[pltrange]))[0][0]
    # print('s11 minimum: {:g} dB at {:g} Hz'.format(np.amin(s11[pltrange]), freqs[s11minfreq + pltrangemin]))
    # print('At {:g} Hz...'.format(freqs[s11minfreq + pltrangemin]))
    # print('Input impedance: {:.1f}{:+.1f}j Ohms'.format(np.abs(zin[s11minfreq + pltrangemin]), zin[s11minfreq + pltrangemin].imag))
    # print('Input admittance (mag): {:g} S'.format(np.abs(yin[s11minfreq + pltrangemin])))
    # print('Input admittance (phase): {:.1f} deg'.format(np.angle(yin[s11minfreq + pltrangemin], deg=True)))

    # Figure 1
    # Plot incident voltage
    fig1, ax = plt.subplots(num='Transmitter transmission line parameters', figsize=(30, 15), facecolor='w', edgecolor='w')
    ax.axis('off')
    gs1 = gridspec.GridSpec(3, 2, hspace=0.5)

    ## Plot frequency spectra of received voltage
    ax = plt.subplot(gs1[0, 0])
    for Vrec_ in Vrecp:
        ax.plot(freqs[pltrange], power(Vrec_[pltrange]), 'o-', lw=2)
    plt.legend(dirnames)
    ax.set_title('Received voltage')
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('Power [dB]')
    ax.set_xlim(xlim)
    ax.grid(which='both', axis='both', linestyle='-.')

    ## Plot frequency spectra of received current
    ax = plt.subplot(gs1[0, 1])
    for Irec_ in Irecp:
        ax.plot(freqs[pltrange], power(Irec_[pltrange]), 'o-', lw=2)
    plt.legend(dirnames)
    ax.set_title('Received current')
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('Power [dB]')
    ax.set_xlim(xlim)
    ax.grid(which='both', axis='both', linestyle='-.')

    ## Plot frequency spectra of received s11 (magnitude)
    ax = plt.subplot(gs1[1, 0])
    for Vinc_,Vrec_ in zip(Vincp,Vrecp):
        ax.plot(freqs[pltrange], power(Vrec_[pltrange]/Vinc_[pltrange]), 'o-', lw=2)
    plt.legend(dirnames)
    ax.set_title('s11 (magnitude) = |V_{rec}/V_{inc}|')
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('Power [dB]')
    ax.set_xlim(xlim)
    ax.grid(which='both', axis='both', linestyle='-.')

    ## Plot frequency spectra of received s11 (phase)
    ax = plt.subplot(gs1[1, 1])
    for Vinc_,Vrec_ in zip(Vincp,Vrecp):
        with np.errstate(divide='ignore'):
            phase = np.mod( np.angle(Vrec_[pltrange]/Vinc_[pltrange], deg=True)+0.9, 360)-0.9
        ax.plot(freqs[pltrange], phase, 'o-', lw=2)
    plt.legend(dirnames)
    ax.set_title('s11 (phase) = angle(V_{rec}/V_{inc})')
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('Angle [degrees]')
    ax.set_xlim(xlim)
    ax.grid(which='both', axis='both', linestyle='-.')

    ## Plot time evolution of received voltage
    ax = plt.subplot(gs1[2, 0])
    for Vrec_,Vref_ in zip(Vrec,Vref):
        ax.plot(times, Vrec_, '-', lw=2)
    plt.legend(dirnames, loc='upper right')
    ax.set_title('Vrec')
    ax.set_xlabel('time [s]')
    ax.set_ylabel('Voltage [V]')
    ax.set_xlim([0,5e-8])
    ax.grid(which='both', axis='both', linestyle='-.')

    ## Plot time evolution of incident voltage
    ax = plt.subplot(gs1[2, 1])
    for Vinc_,Vref_ in zip(Vinc,Vref):
        ax.plot(times, Vref_, '-', lw=2)
#        ax.plot(times, Vinc_, '-', lw=2)
    plt.legend(dirnames)
    ax.set_title('Vref')
    ax.set_xlabel('time [s]')
    ax.set_ylabel('Voltage [V]')
    ax.set_xlim([0,5e-8])
    ax.grid(which='both', axis='both', linestyle='-.')


    # Save a PDF/PNG of the figure
    # fig1.savefig(os.path.splitext(os.path.abspath(filename))[0] + '_tl_params.png', dpi=150, format='png', bbox_inches='tight', pad_inches=0.1)
    # fig2.savefig(os.path.splitext(os.path.abspath(filename))[0] + '_ant_params.png', dpi=150, format='png', bbox_inches='tight', pad_inches=0.1)
    # dirname = os.path.dirname(os.path.relpath(filename)).split('/')[-1]
    fig1.savefig(output_plot_file, dpi=None, format='pdf', bbox_inches='tight', pad_inches=0.1)
    plt.close(fig1)

    return plt


def plot_files(outputfiles, plot_file_name):
    tltx_num = 1
    tlrx_num = None
    rx_num = 1
    rx_component = 'Ez'

    antennaparams = []

    for of in outputfiles:
        print('Processing ', of)
        params = calculate_antenna_params(of, tltx_num, tlrx_num, rx_num, rx_component)
        antennaparams.append( params )
#        mpl_plot(of, **params)

    mpl_plot_diff(outputfiles, antennaparams, plot_file_name)


if __name__ == "__main__":

    # Parse command line arguments
    # parser = argparse.ArgumentParser(description='Plots antenna parameters (s11, s21 parameters and input impedance) from an output file containing a transmission line source.', usage='cd gprMax; python -m tools.plot_antenna_params outputfile')
    # parser.add_argument('outputfile1', help='name of output file including path')
    # parser.add_argument('outputfile2', help='name of output file including path')
    # parser.add_argument('--tltx-num', default=1, type=int, help='transmitter antenna - transmission line number')
    # parser.add_argument('--tlrx-num', type=int, help='receiver antenna - transmission line number')
    # parser.add_argument('--rx-num', type=int, help='receiver antenna - output number')
    # parser.add_argument('--rx-component', type=str, help='receiver antenna - output electric field component', choices=['Ex', 'Ey', 'Ez'])
    # args = parser.parse_args()

    outputfiles = [
#        'large_scale/freespace/main.out',
#        'large_scale/water/main.out',
#        'large_scale_with_rod/freespace/main.out',
#        'large_scale_with_rod/water/main.out',


#        'small_scale_x/freespace/main.out',
#        'small_scale_x/sand_uns/main.out',
#        'small_scale_x/sand_sat/main.out',
#        'small_scale_x/water/main.out',
        ]

    plot_files([
        'small_scale/freespace/main.out',
        'small_scale/sand_uns/main.out',
        'small_scale/sand_sat/main.out',
        'small_scale/water/main.out',
        ], 'diff_small_scale.pdf')

    plot_files([
        'small_scale_with_wire/freespace/main.out',
        'small_scale_with_wire/sand_uns/main.out',
        'small_scale_with_wire/sand_sat/main.out',
        'small_scale_with_wire/water/main.out',
        ], 'diff_small_scale_with_wire.pdf')


