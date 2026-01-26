import numpy as np
import matplotlib.pyplot as plt
from richards import RichardsEquationSolver
from soil import VanGenuchtenParams, plot_soils
from bc_models import dirichlet_bc, neumann_bc, free_drainage_bc, seepage_bc
from plots import plot_richards_output

