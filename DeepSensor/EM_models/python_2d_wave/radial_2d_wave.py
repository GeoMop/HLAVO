import numpy as np

# --- 1) Set up spatial grid ---
Nr = 80
Nz = 300
dr = 0.01
dz = 0.01

r_vals = np.arange(Nr) * dr  # [0, dr, 2dr, ...]
z_min = -1.0
z_vals = z_min + np.arange(Nz) * dz  # e.g. from -1 up to ...
R, Z = np.meshgrid(r_vals, z_vals, indexing='ij')  # R.shape = (Nr, Nz)

# --- 2) Define wave speed c(r,z) and forcing F(r,z) ---
# For example, c might be in some 2D array c[i,j]
# Then c2 = c^2
c_rock = 1.0 / 30
c_wet = 1.0 / 10
c = np.full_like(R, 1.0)
source_r = 0.04
mask_inner = R > 0.06
mask_outer = R > 0.015
c[mask_inner,:] = c_wet
c[mask_outer,:] = c_rock
c2 = c**2

# Forcing field F(r,z), e.g. nonzero in some region:
F_field = np.zeros_like(R)
mask = (source_r - dr/2 < R) & (R < source_r + dr/2)   # some small region near axis
F_field[mask] = 1.0                # amplitude 1 in that region
omega = 100.0   # in MHz, time in us; 
Tend = 1e-7

# --- 3) Choose time step (CFL) ---
c_max = np.max(c)
CFL_factor = 0.9
dt = CFL_factor / ( c_max * np.sqrt(1.0/dr**2 + 1.0/dz**2) )
# for 1cm space step about 1e-13
# for 10 periods about 1e4 steps

# Number of time steps to run
nt = Tend // dt

# --- 4) Allocate arrays for u at time n-1, n, n+1 ---
u_nm1 = np.zeros((Nr, Nz))
u_n   = np.zeros((Nr, Nz))
u_np1 = np.zeros((Nr, Nz))

# Optional: if you want, store time history, etc.
# Or just do inline updates.
u_history = np

# -- Helper function to compute the spatial operator Lu = div(c^2 grad(u)) --
def compute_Lu(u):
    """
    Returns array L where L[i,j] ~ div(c^2 * grad(u))(i,j).
    Simple 2nd-order finite difference in r,z, ignoring boundary complexity.
    """
    L = np.zeros_like(u)
    
    # Interior points in r
    for i in range(1, Nr-1):
        # r_i = i*dr
        r_i = r_vals[i]
        for j in range(1, Nz-1):
            
            # partial_r u ~ (u[i+1,j] - u[i-1,j]) / (2 dr)
            # flux_r = r_i * c2[i,j] * partial_r u
            flux_r_ip = (r_vals[i+1]) * c2[i+1,j] * (u[i+1,j] - u[i,j]) / dr
            flux_r_im = (r_vals[i  ]) * c2[i  ,j] * (u[i,j]   - u[i-1,j]) / dr
            
            # Then d/dr of flux_r / r  ~ [flux_r(i+1/2) - flux_r(i-1/2)] / (dr * r_i)
            # We'll approximate flux_r(i+1/2) by flux_r_ip, etc.
            ddr_term = (flux_r_ip - flux_r_im) / (dr * r_i)
            
            # partial_z of c2 partial_z u
            # flux_z = c2[i,j] * (u[i,j+1] - u[i,j-1]) / (2 dz)
            # for central difference, let's do it more simply
            flux_z_p = c2[i,j+1] * (u[i,j+1] - u[i,j]) / dz
            flux_z_m = c2[i,j  ] * (u[i,j]   - u[i,j-1]) / dz
            ddz_term = (flux_z_p - flux_z_m) / dz
            
            L[i,j] = ddr_term + ddz_term
    
    # Handle r=0 axis (i=0): impose ∂u/∂r = 0
    # A simple approach: copy or mirror the values at i=1 to i=0
    L[0,:] = L[1,:]  # or a more careful axis condition

    # Similarly, handle outer boundaries or apply PML modifications
    # (not shown here).

    return L

# --- 5) Main time-stepping loop ---
for n in range(nt):
    t_n = n * dt
    
    # 5a) Compute the spatial operator at current time
    Lu = compute_Lu(u_n)
    
    # 5b) Forcing at time t_n
    forcing = F_field * np.sin(omega * t_n)
    
    # 5c) Update: explicit 2nd-order in time
    u_np1 = 2.0*u_n - u_nm1 + (dt**2)*(Lu + forcing)
    
    # 5d) (Optional) apply PML boundary conditions or damping
    #      This typically modifies u_np1 near edges to absorb waves.
    #      For a real PML, you'd incorporate extra terms
    #      in compute_Lu(...) or do here as a post-step damping.

    # 5e) Swap arrays for next step
    u_nm1, u_n, u_np1 = u_n, u_np1, u_nm1

# After this loop, u_n holds the wavefield at final time.

