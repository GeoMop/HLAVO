import numpy as np
from openEMS import openEMS
from openEMS.physical_constants import c0

# Simulation parameters
frequency = 300e6  # Frequency of operation (300 MHz)
wavelength = c0 / frequency
resolution = wavelength / 20  # Resolution (20 cells per wavelength)

# Simulation domain size
domain_size = np.array([2, 2, 2]) * wavelength  # 2x2x2 wavelengths

# Create the openEMS simulation object
sim = openEMS()
sim.SetDomain(domain_size, resolution)

# Define the dipole antenna
half_length = 0.5  # Half of the dipole length (1m total)
dipole_radius = 0.01  # Dipole radius (1 cm)

dipole_start = np.array([0, 0, -half_length])  # Start point of the dipole
dipole_end = np.array([0, 0, half_length])  # End point of the dipole

sim.AddWire(dipole_start, dipole_end, dipole_radius)

# Define the coaxial cable
coax_length = 1.0  # Length of the coaxial cable (1m)
coax_outer_radius = 0.02  # Outer conductor radius (2 cm)
coax_inner_radius = 0.005  # Inner conductor radius (0.5 cm)
coax_dielectric_epsr = 2.2  # Relative permittivity of the dielectric

coax_start = np.array([-coax_length, 0, 0])  # Start of the coaxial cable
coax_end = np.array([0, 0, 0])  # End of the coaxial cable

sim.AddCoaxialCable(coax_start, coax_end, coax_inner_radius, coax_outer_radius, coax_dielectric_epsr)

# Define the excitation source at the coaxial feed point
feed_point = np.array([0, 0, 0])  # Feed point at the end of the coax
sim.AddExcitation(feed_point, voltage=1.0)  # 1V excitation

# Add boundary conditions
sim.SetBoundaryCondition('PEC', 'PEC', 'PEC', 'PEC', 'PEC', 'PEC')

# Run the simulation
sim.RunSimulation()

# Post-process and visualize results
sim.PlotFarField(frequency)
