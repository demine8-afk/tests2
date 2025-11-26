# test3_stability_final.py
#
# MODEL: Unified NLSE (Saturation + Repulsion K=0.2)
# PURPOSE: Linear Stability Test with precise graphical output.
#
# OUTPUTS:
# 1. test3_density_snapshots.png (5 frames)
# 2. test3_perturbation_evolution.png (6 subplots metrics)
# 3. test3_stability_analysis.png (Phase space & Energy-Norm)

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
K = 0.2          # Repulsion
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
# 2. Functions
# -----------------------------------------------------------------------------
def compute_observables(psi_):
    rho = np.abs(psi_)**2
    norm = np.sum(rho) * dx * dy
    rho_max = np.max(rho)
    
    rms = np.sqrt(np.sum((X**2 + Y**2) * rho) * dx * dy / norm) if norm > 1e-12 else 0.0
    
    psi_k = fft2(psi_)
    kinetic = (1.0 / (2 * m)) * np.sum(K2 * np.abs(psi_k)**2) * dx * dy / (Nx * Ny)
    
    U_sat = -G * (rho / S - (1.0 / S**2) * np.log(1.0 + S * rho + 1e-30))
    U_rep = 0.5 * K * (rho**2)
    potential = np.sum(U_sat + U_rep) * dx * dy
    
    return norm, rho_max, rms, kinetic + potential

# Storage
times = []
norms = []
rho_maxs = []
rms_vals = []
energies = []
delta_norms = []     # N(t) - N_ref
delta_energies = []  # E(t) - E_ref

snapshots = []
snapshot_times = []
target_snap_times = [0.0, 30.1, 40.0, 55.0, 79.9] # Exact frames

# Reference values (updated at perturbation)
N_ref = None
E_ref = None

print(f"STARTING TEST 3. K={K}")
print("-" * 60)

# -----------------------------------------------------------------------------
# 3. Evolution
# -----------------------------------------------------------------------------
perturbation_applied = False
N_pre_pert = 0
E_pre_pert = 0

for step in range(N_steps_total):
    t = step * dt
    
    # --- PERTURBATION EVENT ---
    if not perturbation_applied and t >= T_perturb:
        print(f">>> Perturbation at t={t:.2f}")
        
        # Store pre-perturbation state for graphs
        N_pre_pert, _, _, E_pre_pert = compute_observables(psi)
        
        # Apply Perturbation (Kick)
        pert_amp = 0.2 * A0
        bump = pert_amp * np.exp(-((X - 2.0)**2 + (Y - 2.0)**2) / (2 * 1.5**2))
        noise = 0.2 * np.random.randn(Nx, Ny)
        psi = psi + bump * np.exp(1j * noise)
        
        # Store new references
        N_ref, _, _, E_ref = compute_observables(psi)
        perturbation_applied = True
    # --------------------------
    
    # Step
    psi_k = fft2(psi)
    psi_k *= kinetic_half
    psi = ifft2(psi_k)
    
    rho = np.abs(psi)**2
    V_sat = -G * rho / (1.0 + S * rho)
    V_rep = K * rho
    psi *= np.exp(-1j * (V_sat + V_rep) * dt)
    
    psi_k = fft2(psi)
    psi_k *= kinetic_half
    psi = ifft2(psi_k)
    
    # Data Collection
    if step % save_every == 0:
        norm, rho_max, rms, energy = compute_observables(psi)
        times.append(t)
        norms.append(norm)
        rho_maxs.append(rho_max)
        rms_vals.append(rms)
        energies.append(energy)
        
        # Calculate deltas relative to INITIAL state for the step plot
        # (or relative to current ref? Let's follow the visual guide)
        if not perturbation_applied:
            # Initial reference
            ref_N = norms[0]
            ref_E = energies[0]
        else:
            # Just show the jump
            ref_N = norms[0] 
            ref_E = energies[0]

        delta_norms.append(norm - ref_N)
        delta_energies.append(energy - ref_E)
        
        # Snapshots
        if len(target_snap_times) > 0 and t >= target_snap_times[0] - dt:
            snapshots.append(rho)
            snapshot_times.append(t)
            target_snap_times.pop(0)
            
        if step % (save_every * 50) == 0:
             print(f"t={t:5.1f} | rho={rho_max:5.2f}")

times = np.array(times)
rho_maxs = np.array(rho_maxs)
rms_vals = np.array(rms_vals)
norms = np.array(norms)
energies = np.array(energies)
delta_norms = np.array(delta_norms)
delta_energies = np.array(delta_energies)

# -----------------------------------------------------------------------------
# 4. Plotting
# -----------------------------------------------------------------------------

idx_pert = np.searchsorted(times, T_perturb)

# === FIGURE 1: Snapshots (5 frames) ===
fig1, axes1 = plt.subplots(2, 3, figsize=(15, 10))
axes1 = axes1.flatten()
for i, snap in enumerate(snapshots):
    im = axes1[i].contourf(X, Y, snap, levels=60, cmap='hot')
    axes1[i].set_title(f"t = {snapshot_times[i]:.1f}", fontweight='bold')
    axes1[i].set_xlabel("x")
    axes1[i].set_ylabel("y")
    plt.colorbar(im, ax=axes1[i], label=r'$\rho$')

# Hide the 6th empty plot
axes1[5].axis('off')
plt.tight_layout()
plt.savefig('test3_density_snapshots.png', dpi=150)
print("Saved: test3_density_snapshots.png")


# === FIGURE 2: Evolution Metrics (6 subplots) ===
fig2, axes2 = plt.subplots(2, 3, figsize=(18, 10))

# 1. Peak Density
axes2[0, 0].plot(times, rho_maxs, 'b-')
axes2[0, 0].axvline(T_perturb, color='r', ls='--', label='Perturbation')
axes2[0, 0].set_title('Peak Density Evolution', fontweight='bold')
axes2[0, 0].set_ylabel(r'$\rho_{max}(t)$')
axes2[0, 0].set_xlabel('Time')
axes2[0, 0].legend()
axes2[0, 0].grid(True, alpha=0.3)

# 2. RMS Width
axes2[0, 1].plot(times, rms_vals, 'g-')
axes2[0, 1].axvline(T_perturb, color='r', ls='--')
axes2[0, 1].set_title('RMS Width', fontweight='bold')
axes2[0, 1].set_ylabel('RMS Radius')
axes2[0, 1].set_xlabel('Time')
axes2[0, 1].grid(True, alpha=0.3)

# 3. Norm Normalized
axes2[0, 2].plot(times, norms / norms[0], 'purple')
axes2[0, 2].axvline(T_perturb, color='r', ls='--')
axes2[0, 2].set_title('Norm (Normalized)', fontweight='bold')
axes2[0, 2].set_ylabel(r'$N(t)/N_{ref}$')
axes2[0, 2].set_xlabel('Time')
axes2[0, 2].grid(True, alpha=0.3)

# 4. Norm Deviation (Step)
axes2[1, 0].plot(times, delta_norms, 'r-')
axes2[1, 0].axvline(T_perturb, color='r', ls='--')
axes2[1, 0].set_title('Norm Deviation', fontweight='bold')
axes2[1, 0].set_ylabel(r'$\Delta N = N(t) - N_0$')
axes2[1, 0].set_xlabel('Time')
axes2[1, 0].grid(True, alpha=0.3)

# 5. Energy Deviation (Step)
axes2[1, 1].plot(times, delta_energies, 'm-')
axes2[1, 1].axvline(T_perturb, color='r', ls='--')
axes2[1, 1].set_title('Energy Deviation', fontweight='bold')
axes2[1, 1].set_ylabel(r'$\Delta E = E(t) - E_0$')
axes2[1, 1].set_xlabel('Time')
axes2[1, 1].grid(True, alpha=0.3)

# 6. Log Deviation (Stability Check)
# We plot deviation from N_ref (POST perturbation) to show numerical stability
n_ref_post = norms[-1]
log_err = np.log10(np.abs(norms[idx_pert:] - n_ref_post) + 1e-15)
t_post = times[idx_pert:] - T_perturb
axes2[1, 2].plot(t_post, log_err, 'r-')
axes2[1, 2].set_title('Numerical Error (Log Scale)', fontweight='bold')
axes2[1, 2].set_ylabel(r'$\log_{10}|N(t) - N_{final}|$')
axes2[1, 2].set_xlabel('Time after perturbation')
axes2[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('test3_perturbation_evolution.png', dpi=150)
print("Saved: test3_perturbation_evolution.png")


# === FIGURE 3: Stability Analysis (2 subplots) ===
fig3, axes3 = plt.subplots(1, 2, figsize=(14, 6))

# 1. Phase Space
axes3[0].plot(rms_vals[:idx_pert], rho_maxs[:idx_pert], 'b-', lw=2, alpha=0.6, label='Before perturbation')
axes3[0].plot(rms_vals[idx_pert:], rho_maxs[idx_pert:], 'r-', lw=2, alpha=0.8, label='After perturbation')
axes3[0].scatter([rms_vals[idx_pert]], [rho_maxs[idx_pert]], color='red', marker='*', s=150, zorder=10, label='Perturbation point')
axes3[0].set_title('Phase Space: Density vs Width', fontweight='bold')
axes3[0].set_xlabel('RMS Radius')
axes3[0].set_ylabel(r'$\rho_{max}$')
axes3[0].legend()
axes3[0].grid(True, alpha=0.3)

# 2. Energy-Norm Diagram
axes3[1].plot(norms[:idx_pert], energies[:idx_pert], 'b.', label='Before')
axes3[1].scatter([norms[idx_pert]], [energies[idx_pert]], color='red', marker='*', s=200, label='After (Jump)')
# Plot after as a single point or small locus since N is conserved
axes3[1].plot(norms[idx_pert:], energies[idx_pert:], 'r.', alpha=0.1)
axes3[1].set_title('Energy-Norm Diagram', fontweight='bold')
axes3[1].set_xlabel('Norm N')
axes3[1].set_ylabel('Energy E')
axes3[1].legend()
axes3[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('test3_stability_analysis.png', dpi=150)
print("Saved: test3_stability_analysis.png")

print("\n[ TEST 3 COMPLETE ]")
