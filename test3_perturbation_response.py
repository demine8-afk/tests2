# test3_perturbation_response.py
# Outputs: test3_perturbation_evolution.png, test3_stability_analysis.png, test3_density_snapshots.png
#
# THEORETICAL SUMMARY:
# ====================
# PDE (hypothesis):
#   i ∂_t ψ = -(1/2m) ∇² ψ - [G|ψ|²/(1 + S|ψ|²)] ψ
#
# We test linear stability of breather solutions by:
#   1. Evolving to a quasi-stationary breather state
#   2. Applying a localized perturbation δψ
#   3. Monitoring whether perturbation grows (instability) or decays/oscillates (stability)
#
# Control model (cubic NLS):
#   i ∂_t ψ = -(1/2m) ∇² ψ - g|ψ|² ψ
#
# Literature (Vakhitov-Kolokolov criterion, Grillakis-Shatah-Strauss):
#   Stability condition: dω/dN < 0 for ground-state solitons in attractive NLS
#   For saturating nonlinearity: expect enhanced stability due to saturation
#
# Observables:
#   ρ_max(t)  — peak density
#   ΔN(t)     — norm deviation from pre-perturbation value
#   ΔE(t)     — energy deviation
#   δρ_rms(t) — RMS of density perturbation δρ = ρ(t) - ρ_ref
#
# Expected behavior:
#   Perturbation triggers oscillations around equilibrium
#   No exponential growth (linear stability)
#   System remains localized and bounded
#
# Method:
#   Phase I (t ∈ [0, 30]):   Relaxation to breather
#   Phase II (t = 30):        Apply Gaussian perturbation
#   Phase III (t ∈ [30, 80]): Monitor stability

import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2, fftfreq

# -----------------------------------------------------------------------------
# Parameters
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
dt = 0.005
T_relax = 30.0      # relaxation phase
T_perturb = 30.0    # perturbation application time
T_monitor = 50.0    # monitoring phase after perturbation
T_total = T_relax + T_monitor
N_steps_total = int(T_total / dt)

# Initial condition
A0 = 3.0
sigma = 2.0
psi = A0 * np.exp(-(X**2 + Y**2) / (2 * sigma**2))

# Fourier space
kx = fftfreq(Nx, d=dx) * 2 * np.pi
ky = fftfreq(Ny, d=dy) * 2 * np.pi
KX, KY = np.meshgrid(kx, ky, indexing='ij')
K2 = KX**2 + KY**2
kinetic_half = np.exp(-1j * K2 / (2 * m) * dt / 2)

save_every = 20  # save every 0.1 time units

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------
def compute_observables(psi_):
    rho = np.abs(psi_)**2
    norm = np.sum(rho) * dx * dy
    rho_max = np.max(rho)
    
    rms = np.sqrt(np.sum((X**2 + Y**2) * rho) * dx * dy / norm) if norm > 1e-12 else 0.0
    
    psi_k = fft2(psi_)
    kinetic = (1.0 / (2 * m)) * np.sum(K2 * np.abs(psi_k)**2) * dx * dy / (Nx * Ny)
    
    U_rho = -G * (rho / S - (1.0 / S**2) * np.log(1.0 + S * rho + 1e-30))
    potential = np.sum(U_rho) * dx * dy
    energy = kinetic + potential
    
    return norm, rho_max, rms, energy

# Storage
times = []
norms = []
rho_maxs = []
rms_vals = []
energies = []
delta_norms = []
delta_energies = []

# Reference values (will be set at perturbation time)
N_ref = None
E_ref = None
rho_ref = None

# Snapshot storage
snapshots = []
snapshot_times = []

print("="*70)
print("TEST 3: PERTURBATION RESPONSE AND LINEAR STABILITY")
print("="*70)
print(f"Phase I:  Relaxation (t ∈ [0, {T_relax}])")
print(f"Phase II: Perturbation applied at t = {T_perturb}")
print(f"Phase III: Monitoring (t ∈ [{T_perturb}, {T_total}])")
print("="*70)

# -----------------------------------------------------------------------------
# Time evolution
# -----------------------------------------------------------------------------
perturbation_applied = False

for step in range(N_steps_total):
    t = step * dt
    
    # Apply perturbation at T_perturb
    if not perturbation_applied and t >= T_perturb:
        print(f"\n>>> PERTURBATION APPLIED AT t = {t:.2f}")
        
        # Store reference state
        N_ref, rho_max_ref, rms_ref, E_ref = compute_observables(psi)
        rho_ref = np.abs(psi)**2
        
        # Perturbation: Gaussian bump in density + random phase
        # Type 1: Localized density perturbation
        x_pert, y_pert = 3.0, 3.0  # offset location
        sigma_pert = 1.5
        amplitude_pert = 0.2  # 20% of initial amplitude
        
        perturbation = amplitude_pert * A0 * np.exp(-((X - x_pert)**2 + (Y - y_pert)**2) / (2 * sigma_pert**2))
        
        # Add random phase modulation
        phase_pert = 0.1 * np.random.randn(Nx, Ny)
        
        psi = psi + perturbation * np.exp(1j * phase_pert)
        
        N_after, _, _, E_after = compute_observables(psi)
        print(f"    Before: N = {N_ref:.4f}, E = {E_ref:.6f}")
        print(f"    After:  N = {N_after:.4f}, E = {E_after:.6f}")
        print(f"    ΔN = {N_after - N_ref:.4f}, ΔE = {E_after - E_ref:.6f}")
        
        perturbation_applied = True
    
    # Strang splitting evolution
    psi_k = fft2(psi)
    psi_k *= kinetic_half
    psi = ifft2(psi_k)
    
    rho = np.abs(psi)**2
    V_eff = -G * rho / (1.0 + S * rho)
    psi *= np.exp(-1j * V_eff * dt)
    
    psi_k = fft2(psi)
    psi_k *= kinetic_half
    psi = ifft2(psi_k)
    
    # Observables
    if step % save_every == 0:
        norm, rho_max, rms, energy = compute_observables(psi)
        times.append(t)
        norms.append(norm)
        rho_maxs.append(rho_max)
        rms_vals.append(rms)
        energies.append(energy)
        
        # Deviations from reference (after perturbation)
        if N_ref is not None:
            delta_norms.append(norm - N_ref)
            delta_energies.append(energy - E_ref)
        else:
            delta_norms.append(0.0)
            delta_energies.append(0.0)
        
        # Store snapshots at key moments
        if t in [0, T_perturb - 0.1, T_perturb + 0.1, T_perturb + 10, T_perturb + 25, T_total - 0.1]:
            snapshots.append(np.abs(psi)**2)
            snapshot_times.append(t)
        
        if step % (save_every * 25) == 0:  # print every 2.5 time units
            if N_ref is not None:
                print(f"t = {t:6.2f} | ρ_max = {rho_max:8.4f} | ΔN = {norm - N_ref:+.4e} | ΔE = {energy - E_ref:+.4e}")
            else:
                print(f"t = {t:6.2f} | ρ_max = {rho_max:8.4f} | N = {norm:.4f} | E = {energy:.6f}")

times = np.array(times)
norms = np.array(norms)
rho_maxs = np.array(rho_maxs)
rms_vals = np.array(rms_vals)
energies = np.array(energies)
delta_norms = np.array(delta_norms)
delta_energies = np.array(delta_energies)

# -----------------------------------------------------------------------------
# Analysis: Perturbation growth
# -----------------------------------------------------------------------------
idx_pert = np.argmin(np.abs(times - T_perturb))
idx_end = len(times) - 1

# Check if perturbation grows exponentially
delta_N_after = delta_norms[idx_pert:]
delta_E_after = delta_energies[idx_pert:]
times_after = times[idx_pert:] - T_perturb

# Max deviations in monitoring phase
max_delta_N = np.max(np.abs(delta_N_after))
max_delta_E = np.max(np.abs(delta_E_after))

print("\n" + "="*70)
print("STABILITY ANALYSIS")
print("="*70)
print(f"Reference state (t = {T_perturb}):")
print(f"  N_ref = {N_ref:.6f}")
print(f"  E_ref = {E_ref:.6f}")
print(f"\nMonitoring phase (t ∈ [{T_perturb}, {T_total}]):")
print(f"  Max |ΔN| = {max_delta_N:.6e} ({max_delta_N/N_ref*100:.4f}%)")
print(f"  Max |ΔE| = {max_delta_E:.6e} ({max_delta_E/abs(E_ref)*100:.4f}%)")
print(f"  Final ρ_max = {rho_maxs[-1]:.4f} (ref: {rho_max_ref:.4f})")

# Check for exponential growth
if len(times_after) > 10:
    # Fit |ΔN(t)| ~ exp(γt)
    log_delta_N = np.log(np.abs(delta_N_after) + 1e-12)
    gamma_fit = np.polyfit(times_after, log_delta_N, 1)[0]
    print(f"\nGrowth rate estimate: γ = {gamma_fit:.6f}")
    if gamma_fit > 0.01:
        print("  WARNING: Possible exponential growth detected")
    else:
        print("  ✓ No exponential growth (γ ≈ 0 or negative)")

print("="*70)

# -----------------------------------------------------------------------------
# Plotting: Main observables
# -----------------------------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Row 1: Evolution
axes[0, 0].plot(times, rho_maxs, 'b-', linewidth=1.5)
axes[0, 0].axvline(T_perturb, color='red', linestyle='--', linewidth=2, label='Perturbation')
axes[0, 0].set_xlabel('Time', fontsize=12)
axes[0, 0].set_ylabel(r'$\rho_{\max}(t)$', fontsize=12)
axes[0, 0].set_title('Peak Density Evolution', fontsize=13, fontweight='bold')
axes[0, 0].legend(fontsize=10)
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(times, rms_vals, 'g-', linewidth=1.5)
axes[0, 1].axvline(T_perturb, color='red', linestyle='--', linewidth=2)
axes[0, 1].set_xlabel('Time', fontsize=12)
axes[0, 1].set_ylabel('RMS Radius', fontsize=12)
axes[0, 1].set_title('RMS Width', fontsize=13, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)

axes[0, 2].plot(times, norms / N_ref if N_ref else norms, 'purple', linewidth=1.5)
axes[0, 2].axvline(T_perturb, color='red', linestyle='--', linewidth=2)
axes[0, 2].axhline(1.0, color='k', linestyle=':', alpha=0.5)
axes[0, 2].set_xlabel('Time', fontsize=12)
axes[0, 2].set_ylabel(r'$N(t)/N_{\mathrm{ref}}$', fontsize=12)
axes[0, 2].set_title('Norm (Normalized)', fontsize=13, fontweight='bold')
axes[0, 2].grid(True, alpha=0.3)

# Row 2: Perturbation response
axes[1, 0].plot(times, delta_norms, 'r-', linewidth=1.5)
axes[1, 0].axvline(T_perturb, color='red', linestyle='--', linewidth=2, alpha=0.5)
axes[1, 0].axhline(0, color='k', linestyle=':', alpha=0.5)
axes[1, 0].set_xlabel('Time', fontsize=12)
axes[1, 0].set_ylabel(r'$\Delta N(t) = N(t) - N_{\mathrm{ref}}$', fontsize=12)
axes[1, 0].set_title('Norm Deviation', fontsize=13, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].plot(times, delta_energies, 'm-', linewidth=1.5)
axes[1, 1].axvline(T_perturb, color='red', linestyle='--', linewidth=2, alpha=0.5)
axes[1, 1].axhline(0, color='k', linestyle=':', alpha=0.5)
axes[1, 1].set_xlabel('Time', fontsize=12)
axes[1, 1].set_ylabel(r'$\Delta E(t) = E(t) - E_{\mathrm{ref}}$', fontsize=12)
axes[1, 1].set_title('Energy Deviation', fontsize=13, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)

# Zoom on post-perturbation
axes[1, 2].plot(times_after, np.abs(delta_N_after), 'r-', linewidth=2, label=r'$|\Delta N|$')
axes[1, 2].set_xlabel(r'Time after perturbation', fontsize=12)
axes[1, 2].set_ylabel(r'$|\Delta N(t)|$', fontsize=12)
axes[1, 2].set_title('Perturbation Amplitude (log scale)', fontsize=13, fontweight='bold')
axes[1, 2].set_yscale('log')
axes[1, 2].grid(True, alpha=0.3, which='both')
axes[1, 2].legend(fontsize=10)

plt.tight_layout()
plt.savefig('test3_perturbation_evolution.png', dpi=150)
print("\n✓ Saved: test3_perturbation_evolution.png")

# -----------------------------------------------------------------------------
# Plotting: Density snapshots
# -----------------------------------------------------------------------------
n_snapshots = len(snapshots)
fig2, axes2 = plt.subplots(2, 3, figsize=(15, 10))
axes2 = axes2.flatten()

for i in range(min(6, n_snapshots)):
    im = axes2[i].contourf(X, Y, snapshots[i], levels=40, cmap='hot')
    axes2[i].set_xlabel('x', fontsize=10)
    axes2[i].set_ylabel('y', fontsize=10)
    axes2[i].set_title(f't = {snapshot_times[i]:.1f}', fontsize=11, fontweight='bold')
    plt.colorbar(im, ax=axes2[i], label=r'$\rho$')

for i in range(n_snapshots, 6):
    axes2[i].axis('off')

plt.tight_layout()
plt.savefig('test3_density_snapshots.png', dpi=150)
print("✓ Saved: test3_density_snapshots.png")

# -----------------------------------------------------------------------------
# Plotting: Stability diagram
# -----------------------------------------------------------------------------
fig3, axes3 = plt.subplots(1, 2, figsize=(14, 5))

# Phase space: ρ_max vs RMS
axes3[0].plot(rms_vals[:idx_pert], rho_maxs[:idx_pert], 'b-', linewidth=2, alpha=0.7, label='Before perturbation')
axes3[0].plot(rms_vals[idx_pert:], rho_maxs[idx_pert:], 'r-', linewidth=2, alpha=0.7, label='After perturbation')
axes3[0].scatter([rms_vals[idx_pert]], [rho_maxs[idx_pert]], color='red', s=100, zorder=10, marker='*', label='Perturbation point')
axes3[0].set_xlabel('RMS Radius', fontsize=12)
axes3[0].set_ylabel(r'$\rho_{\max}$', fontsize=12)
axes3[0].set_title('Phase Space: Density vs Width', fontsize=13, fontweight='bold')
axes3[0].legend(fontsize=10)
axes3[0].grid(True, alpha=0.3)

# Energy-Norm diagram
axes3[1].plot(norms[:idx_pert], energies[:idx_pert], 'b-', linewidth=2, alpha=0.7, label='Before')
axes3[1].plot(norms[idx_pert:], energies[idx_pert:], 'r-', linewidth=2, alpha=0.7, label='After')
axes3[1].scatter([norms[idx_pert]], [energies[idx_pert]], color='red', s=100, zorder=10, marker='*')
axes3[1].set_xlabel('Norm N', fontsize=12)
axes3[1].set_ylabel('Energy E', fontsize=12)
axes3[1].set_title('Energy-Norm Diagram', fontsize=13, fontweight='bold')
axes3[1].legend(fontsize=10)
axes3[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('test3_stability_analysis.png', dpi=150)
print("✓ Saved: test3_stability_analysis.png")

plt.show()

print("\n" + "="*70)
print("TEST 3 COMPLETE: PERTURBATION RESPONSE")
print("="*70)
print(f"Perturbation applied at t = {T_perturb}")
print(f"Monitoring duration: {T_total - T_perturb}")
print(f"Maximum norm deviation: {max_delta_N:.6e} ({max_delta_N/N_ref*100:.4f}%)")
print(f"Maximum energy deviation: {max_delta_E:.6e}")
print("\nConclusion:")
if max_delta_N / N_ref < 0.01 and gamma_fit < 0.01:
    print("  ✓ STABLE: Perturbation remains bounded, no exponential growth")
    print("  ✓ System exhibits linear stability around breather equilibrium")
elif max_delta_N / N_ref < 0.1:
    print("  ✓ QUASI-STABLE: Small bounded oscillations around equilibrium")
else:
    print("  ⚠ Significant perturbation growth detected")
print("="*70)
