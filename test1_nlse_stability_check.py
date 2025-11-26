# test1_nlse_stability_check.py
#
# MODEL: NLSE + Saturation + Repulsion (K=0.2)
# PURPOSE: Verify stability against collapse.
# AUTOMATIC SELF-CHECK INCLUDED AT THE END.

import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2, fftfreq
from scipy.signal import find_peaks

# -----------------------------------------------------------------------------
# 1. Parameters
# -----------------------------------------------------------------------------
Nx, Ny = 256, 256
Lx, Ly = 40.0, 40.0
dx, dy = Lx / Nx, Ly / Ny
x = np.linspace(-Lx/2, Lx/2, Nx, endpoint=False)
y = np.linspace(-Ly/2, Ly/2, Ny, endpoint=False)
X, Y = np.meshgrid(x, y, indexing='ij')

m = 1.0
G = 1.0
S = 0.5
K = 0.2          # Repulsion
dt = 0.005
T_max = 50.0
N_steps = int(T_max / dt)
save_every = 50

# Initial Condition
A0 = 3.0
sigma = 2.0
psi = A0 * np.exp(-(X**2 + Y**2) / (2 * sigma**2))

# Fourier Operators
kx = fftfreq(Nx, d=dx) * 2 * np.pi
ky = fftfreq(Ny, d=dy) * 2 * np.pi
KX, KY = np.meshgrid(kx, ky, indexing='ij')
K2 = KX**2 + KY**2
kinetic_half = np.exp(-1j * K2 / (2 * m) * dt / 2)

# -----------------------------------------------------------------------------
# 2. Observables & Hamiltonian
# -----------------------------------------------------------------------------
def compute_observables(psi_):
    rho = np.abs(psi_)**2
    norm = np.sum(rho) * dx * dy
    rho_max = np.max(rho)
    
    rms = np.sqrt(np.sum((X**2 + Y**2) * rho) * dx * dy / norm) if norm > 1e-12 else 0.0
    
    psi_k = fft2(psi_)
    kinetic = (1.0 / (2 * m)) * np.sum(K2 * np.abs(psi_k)**2) * dx * dy / (Nx * Ny)
    
    # Potential Energy: Saturation + Repulsion
    U_sat = -G * (rho / S - (1.0 / S**2) * np.log(1.0 + S * rho + 1e-30))
    U_rep = 0.5 * K * (rho**2)
    
    potential = np.sum(U_sat + U_rep) * dx * dy
    energy = kinetic + potential
    
    return norm, rho_max, rms, energy

# -----------------------------------------------------------------------------
# 3. Evolution
# -----------------------------------------------------------------------------
times = []
norms = []
rho_maxs = []
rms_vals = []
energies = []

norm0, rho0, rms0, E0 = compute_observables(psi)

print(f"STARTING TEST 1. Model: G={G}, S={S}, K={K}")
print(f"Initial State: Rho_max={rho0:.2f}, E={E0:.4f}")
print("-" * 60)

for step in range(N_steps):
    t = step * dt
    
    psi_k = fft2(psi)
    psi_k *= kinetic_half
    psi = ifft2(psi_k)
    
    rho = np.abs(psi)**2
    # --- UNIFIED POTENTIAL ---
    V_sat = -G * rho / (1.0 + S * rho)
    V_rep = K * rho
    psi *= np.exp(-1j * (V_sat + V_rep) * dt)
    # -------------------------
    
    psi_k = fft2(psi)
    psi_k *= kinetic_half
    psi = ifft2(psi_k)
    
    if step % save_every == 0:
        norm, rho_max, rms, energy = compute_observables(psi)
        times.append(t)
        norms.append(norm / norm0)
        rho_maxs.append(rho_max)
        rms_vals.append(rms)
        energies.append(energy / E0)
        
        if step % (save_every * 10) == 0:
            print(f"t={t:6.2f} | rho={rho_max:6.3f} | N/N0={norm/norm0:.6f}")

# Final data point
norm_f, rho_f, rms_f, E_f = compute_observables(psi)
times.append(T_max)
norms.append(norm_f / norm0)
rho_maxs.append(rho_f)
rms_vals.append(rms_f)
energies.append(E_f / E0)

times = np.array(times)
rho_maxs = np.array(rho_maxs)
norms = np.array(norms)
energies = np.array(energies)

# -----------------------------------------------------------------------------
# 4. Visualization
# -----------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

axes[0, 0].plot(times, rho_maxs, 'b-')
axes[0, 0].set_title('Density Stability')
axes[0, 0].set_ylabel(r'$\rho_{max}$')
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(times, rms_vals, 'g-')
axes[0, 1].set_title('Pulse Width (RMS)')
axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].plot(times, norms, 'r-')
axes[1, 0].set_title('Norm Conservation')
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].plot(times, energies, 'm-')
axes[1, 1].set_title('Energy Conservation')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('test1_observables.png', dpi=150)

# -----------------------------------------------------------------------------
# 5. SELF-CHECK ROUTINE
# -----------------------------------------------------------------------------
print("\n" + "="*60)
print("AUTOMATIC SELF-CHECK")
print("="*60)

# Criteria
crit_blowup = np.max(rho_maxs) < 20.0        # Did it explode?
crit_norm   = np.max(np.abs(norms - 1.0)) < 1e-3  # Is mass conserved?
crit_energy = np.max(np.abs(energies - 1.0)) < 1e-2 # Is energy conserved?

print(f"1. NO BLOWUP (Max Rho < 20):      {'[ YES ]' if crit_blowup else '[ NO ]'} -> Max Rho = {np.max(rho_maxs):.3f}")
print(f"2. NORM CONSERVED (Err < 1e-3):   {'[ YES ]' if crit_norm else '[ NO ]'} -> Max Err = {np.max(np.abs(norms - 1.0)):.2e}")
print(f"3. ENERGY CONSERVED (Err < 1e-2): {'[ YES ]' if crit_energy else '[ NO ]'} -> Max Err = {np.max(np.abs(energies - 1.0)):.2e}")

print("-" * 60)
if crit_blowup and crit_norm and crit_energy:
    print("FINAL STATUS: [ SUCCESS ] - System is stable.")
else:
    print("FINAL STATUS: [ FAILURE ] - Something is wrong.")
print("="*60)
