# test3_perturbation_response.py
#
# MODEL: Unified NLSE (Saturation + Repulsion)
# PURPOSE: Check Linear Stability.
# METHOD:
#   1. Relax to stable state (t=0..30).
#   2. Apply strong perturbation (t=30).
#   3. Watch if it blows up or oscillates (t=30..80).

import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2, fftfreq

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
K = 0.2          # <--- UNIVERSAL REPULSION
dt = 0.005
T_relax = 30.0
T_perturb = 30.0
T_monitor = 50.0
T_total = T_relax + T_monitor
N_steps_total = int(T_total / dt)
save_every = 20

# Fourier
kx = fftfreq(Nx, d=dx) * 2 * np.pi
ky = fftfreq(Ny, d=dy) * 2 * np.pi
KX, KY = np.meshgrid(kx, ky, indexing='ij')
K2 = KX**2 + KY**2
kinetic_half = np.exp(-1j * K2 / (2 * m) * dt / 2)

# Initial Condition
A0 = 3.0
sigma = 2.0
psi = A0 * np.exp(-(X**2 + Y**2) / (2 * sigma**2))

# -----------------------------------------------------------------------------
# 2. Observables
# -----------------------------------------------------------------------------
def compute_observables(psi_):
    rho = np.abs(psi_)**2
    norm = np.sum(rho) * dx * dy
    rho_max = np.max(rho)
    
    rms = np.sqrt(np.sum((X**2 + Y**2) * rho) * dx * dy / norm) if norm > 1e-12 else 0.0
    
    psi_k = fft2(psi_)
    kinetic = (1.0 / (2 * m)) * np.sum(K2 * np.abs(psi_k)**2) * dx * dy / (Nx * Ny)
    
    # Potential Energy (Unified)
    U_sat = -G * (rho / S - (1.0 / S**2) * np.log(1.0 + S * rho + 1e-30))
    U_rep = 0.5 * K * (rho**2)
    potential = np.sum(U_sat + U_rep) * dx * dy
    
    energy = kinetic + potential
    return norm, rho_max, rms, energy

# Storage
times = []
norms = []
rho_maxs = []
rms_vals = []
energies = []
delta_norms = []

# Ref values
N_ref = None
E_ref = None

print(f"STARTING TEST 3 (Linear Stability). K={K}")
print(f"Phase I: Relax to t={T_perturb}")
print(f"Phase II: Perturbation")
print(f"Phase III: Monitor to t={T_total}")
print("="*60)

# -----------------------------------------------------------------------------
# 3. Evolution Loop
# -----------------------------------------------------------------------------
perturbation_applied = False

for step in range(N_steps_total):
    t = step * dt
    
    # --- APPLY PERTURBATION ---
    if not perturbation_applied and t >= T_perturb:
        print(f"\n>>> APPLYING PERTURBATION AT t={t:.2f}")
        
        # Store reference
        N_ref, _, _, E_ref = compute_observables(psi)
        
        # Perturbation: Gaussian Bump + Phase Noise
        # This adds "kick" to density and velocity
        x_pert, y_pert = 3.0, 3.0
        sigma_pert = 1.5
        amp_pert = 0.2 * A0  # 20% amplitude kick
        
        pert_field = amp_pert * np.exp(-((X - x_pert)**2 + (Y - y_pert)**2) / (2 * sigma_pert**2))
        phase_noise = 0.2 * np.random.randn(Nx, Ny) # Random phase kick
        
        psi = psi + pert_field * np.exp(1j * phase_noise)
        
        N_new, _, _, E_new = compute_observables(psi)
        print(f"    Delta N: {N_new - N_ref:.4f}")
        print(f"    Delta E: {E_new - E_ref:.4f}")
        
        # Update reference to new state to track stability RELATIVE to perturbed state
        N_ref = N_new
        E_ref = E_new
        perturbation_applied = True
    # --------------------------
    
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
        norms.append(norm)
        rho_maxs.append(rho_max)
        rms_vals.append(rms)
        energies.append(energy)
        
        if N_ref is not None:
            delta_norms.append(norm - N_ref)
        else:
            delta_norms.append(0.0)
            
        if step % (save_every * 50) == 0:
            print(f"t={t:6.2f} | rho={rho_max:6.2f} | N={norm:.4f}")

# -----------------------------------------------------------------------------
# 4. Analysis & Plotting
# -----------------------------------------------------------------------------
times = np.array(times)
rho_maxs = np.array(rho_maxs)
delta_norms = np.array(delta_norms)

# Check stability after perturbation
idx_pert = np.searchsorted(times, T_perturb)
rho_after = rho_maxs[idx_pert:]

is_stable = True
if np.max(rho_after) > 3.0 * np.mean(rho_after[:10]): # Heuristic for blowup
    is_stable = False

print("\n" + "="*60)
print("STABILITY REPORT")
print("="*60)
print(f"Max density after perturbation: {np.max(rho_after):.2f}")
print(f"Mean density after perturbation: {np.mean(rho_after):.2f}")
if is_stable:
    print("[ SUCCESS ] - System returned to bounded oscillation.")
else:
    print("[ FAILURE ] - Unbounded growth detected.")
print("="*60)

# Plot
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Density
axes[0, 0].plot(times, rho_maxs, 'b-')
axes[0, 0].axvline(T_perturb, color='r', ls='--')
axes[0, 0].set_title('Peak Density Response')
axes[0, 0].set_ylabel(r'$\rho_{max}$')
axes[0, 0].grid(True, alpha=0.3)

# RMS
axes[0, 1].plot(times, rms_vals, 'g-')
axes[0, 1].axvline(T_perturb, color='r', ls='--')
axes[0, 1].set_title('Pulse Width Response')
axes[0, 1].grid(True, alpha=0.3)

# Norm deviation
axes[1, 0].plot(times, delta_norms, 'k-')
axes[1, 0].axvline(T_perturb, color='r', ls='--')
axes[1, 0].set_title('Norm Conservation Check')
axes[1, 0].set_ylabel(r'$\Delta N$')
axes[1, 0].grid(True, alpha=0.3)

# Phase Space
axes[1, 1].plot(rms_vals[:idx_pert], rho_maxs[:idx_pert], 'b-', alpha=0.5, label='Pre-Perturb')
axes[1, 1].plot(rms_vals[idx_pert:], rho_maxs[idx_pert:], 'r-', alpha=0.8, label='Post-Perturb')
axes[1, 1].set_xlabel('RMS Radius')
axes[1, 1].set_ylabel(r'$\rho_{max}$')
axes[1, 1].set_title('Phase Space Trajectory')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('test3_perturbation_response.png', dpi=150)
print("Saved: test3_perturbation_response.png")
