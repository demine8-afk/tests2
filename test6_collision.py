# test6_collision.py
#
# MODEL: Unified NLSE (Saturation + Repulsion K=0.2)
# PURPOSE: Soliton Collision Test (Interactions)
#
# We simulate a head-on collision of two ground-state breathers.
# Expected: Quasi-elastic scattering (no merger, no collapse).

import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2, fftfreq
from scipy.ndimage import maximum_filter

# -----------------------------------------------------------------------------
# 1. Parameters
# -----------------------------------------------------------------------------
Nx, Ny = 384, 256  # Elongated domain for collision
Lx, Ly = 60.0, 40.0
dx, dy = Lx / Nx, Ly / Ny
x = np.linspace(-Lx/2, Lx/2, Nx, endpoint=False)
y = np.linspace(-Ly/2, Ly/2, Ny, endpoint=False)
X, Y = np.meshgrid(x, y, indexing='ij')

m = 1.0
G = 1.0
S = 0.5
K = 0.2          # <--- Repulsion
dt = 0.005
T_max = 60.0     # Long enough to see separation
N_steps = int(T_max / dt)
save_every = 20

# Initial State Parameters
A0 = 2.5
sigma = 2.0
dist = 10.0      # Initial separation (half-dist = 5.0)
vel = 0.5        # Velocity

# Construct two wavepackets
# Left moving right
x1 = -dist
k1 = m * vel
psi1 = A0 * np.exp(-((X - x1)**2 + Y**2) / (2 * sigma**2)) * np.exp(1j * k1 * X)

# Right moving left
x2 = dist
k2 = -m * vel
psi2 = A0 * np.exp(-((X - x2)**2 + Y**2) / (2 * sigma**2)) * np.exp(1j * k2 * X)

psi = psi1 + psi2

# Fourier
kx = fftfreq(Nx, d=dx) * 2 * np.pi
ky = fftfreq(Ny, d=dy) * 2 * np.pi
KX, KY = np.meshgrid(kx, ky, indexing='ij')
K2 = KX**2 + KY**2
kinetic_half = np.exp(-1j * K2 / (2 * m) * dt / 2)

# -----------------------------------------------------------------------------
# 2. Helper Functions
# -----------------------------------------------------------------------------
def compute_observables(psi_):
    rho = np.abs(psi_)**2
    norm = np.sum(rho) * dx * dy
    rho_max = np.max(rho)
    
    psi_k = fft2(psi_)
    E_kin = (1.0 / (2 * m)) * np.sum(K2 * np.abs(psi_k)**2) * dx * dy / (Nx * Ny)
    
    U_sat = -G * (rho / S - (1.0 / S**2) * np.log(1.0 + S * rho + 1e-30))
    U_rep = 0.5 * K * (rho**2)
    E_pot = np.sum(U_sat + U_rep) * dx * dy
    
    return norm, rho_max, E_kin + E_pot

def find_peaks_x(rho, threshold_rel=0.3):
    """Find x-coordinates of density peaks"""
    thresh = np.max(rho) * threshold_rel
    local_max = maximum_filter(rho, size=10)
    mask = (rho == local_max) & (rho > thresh)
    coords = np.argwhere(mask)
    if len(coords) == 0: return []
    return x[coords[:, 0]]

# -----------------------------------------------------------------------------
# 3. Evolution
# -----------------------------------------------------------------------------
print(f"STARTING TEST 6 (Collision). K={K}")
print(f"Initial separation: {2*dist}, Velocity: {vel}")
print("-" * 60)

times = []
rho_maxs = []
norms = []
energies = []
peak_positions = []

# Snapshots strategy: 6 frames
# 1. Start
# 2. Approach
# 3. Merger (Collision)
# 4. Separation start
# 5. Separation mid
# 6. Final
t_collision_est = dist / vel
snap_times_target = [0, 
                     t_collision_est * 0.5, 
                     t_collision_est, 
                     t_collision_est * 1.5, 
                     t_collision_est * 2.0, 
                     T_max]
snapshots = []
actual_snap_times = []

norm0, _, E0 = compute_observables(psi)

for step in range(N_steps):
    t = step * dt
    
    # Step
    psi_k = fft2(psi)
    psi_k *= kinetic_half
    psi = ifft2(psi_k)
    
    rho = np.abs(psi)**2
    V_eff = -G * rho / (1.0 + S * rho) + K * rho
    psi *= np.exp(-1j * V_eff * dt)
    
    psi_k = fft2(psi)
    psi_k *= kinetic_half
    psi = ifft2(psi_k)
    
    # Logging
    if step % save_every == 0:
        norm, r_max, E = compute_observables(psi)
        times.append(t)
        norms.append(norm)
        rho_maxs.append(r_max)
        energies.append(E)
        
        # Peak tracking
        peaks = find_peaks_x(rho)
        peak_positions.append(peaks)
        
        # Check for snapshot
        if len(snap_times_target) > 0 and t >= snap_times_target[0] - dt:
            snapshots.append(rho)
            actual_snap_times.append(t)
            snap_times_target.pop(0)
            
        if step % (save_every * 50) == 0:
            print(f"t={t:5.1f} | rho={r_max:5.2f} | N={norm:.2f}")

# -----------------------------------------------------------------------------
# 4. Visualization
# -----------------------------------------------------------------------------
times = np.array(times)
rho_maxs = np.array(rho_maxs)

# Figure 1: Evolution Plots
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Density Spike
axes[0, 0].plot(times, rho_maxs, 'b-')
axes[0, 0].set_title('Peak Density (Collision Spike)')
axes[0, 0].set_ylabel(r'$\rho_{max}$')
axes[0, 0].grid(True, alpha=0.3)

# Trajectories
for i, t in enumerate(times):
    pks = peak_positions[i]
    axes[0, 1].scatter([t]*len(pks), pks, c='k', s=5, alpha=0.5)
axes[0, 1].set_title('Peak Trajectories (X-coord)')
axes[0, 1].set_xlabel('Time')
axes[0, 1].set_ylabel('X Position')
axes[0, 1].grid(True, alpha=0.3)

# Norm
axes[1, 0].plot(times, norms, 'g-')
axes[1, 0].set_ylim(norm0*0.99, norm0*1.01)
axes[1, 0].set_title('Norm Conservation')
axes[1, 0].grid(True, alpha=0.3)

# Energy
axes[1, 1].plot(times, energies, 'r-')
axes[1, 1].set_ylim(E0*0.99, E0*1.01)
axes[1, 1].set_title('Energy Conservation')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('test6_observables.png', dpi=150)
print("Saved: test6_observables.png")

# Figure 2: 6 Snapshots (2 rows x 3 cols)
fig2, axes2 = plt.subplots(2, 3, figsize=(15, 8))
axes2 = axes2.flatten()

for i, snap in enumerate(snapshots):
    if i < 6:
        im = axes2[i].imshow(snap.T, cmap='inferno', origin='lower', 
                        extent=[-Lx/2, Lx/2, -Ly/2, Ly/2]) # Note transpose for correct orientation
        axes2[i].set_title(f"t={actual_snap_times[i]:.1f}")
        axes2[i].set_xlabel("x")
        axes2[i].set_ylabel("y")
        
plt.tight_layout()
plt.savefig('test6_collision_evolution.png', dpi=150)
print("Saved: test6_collision_evolution.png")

# -----------------------------------------------------------------------------
# 5. Verdict
# -----------------------------------------------------------------------------
print("\n" + "="*60)
print("COLLISION REPORT")
print("="*60)
final_peaks = peak_positions[-1]
print(f"Final peaks detected: {len(final_peaks)}")
if len(final_peaks) >= 2:
    print(f"Positions: {final_peaks}")
    print("[ SUCCESS ] - Quasi-elastic scattering observed.")
else:
    print("[ FAILURE ] - Merger or dispersion detected.")
print("="*60)
