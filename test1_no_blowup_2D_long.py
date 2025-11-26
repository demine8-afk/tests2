# test1_2_no_blowup_2D_long.py
# Outputs: test1_2_density_evolution.png, test1_2_observables.png, test1_2_periodicity.png
#
# THEORETICAL SUMMARY:
# ====================
# PDE (hypothesis):
#   i ∂_t ψ = -(1/2m) ∇² ψ - [G|ψ|²/(1 + S|ψ|²)] ψ
#
# We test whether the saturating nonlinearity V_eff(ρ) = -Gρ/(1+Sρ)
# prevents collapse in 2+1D and supports stable breather solutions.
#
# Control model:
#   Cubic NLS: i ∂_t ψ = -(1/2m) ∇² ψ - g|ψ|² ψ
#   Literature (Zakharov, Sulem): blowup occurs for N > N_critical in 2D.
#
# Observables:
#   ρ_max(t)  — peak density
#   RMS(t)    — root mean square radius
#   N(t)      — norm ∫|ψ|² dx dy
#   E(t)      — energy (kinetic + potential)
#
# Expected behavior:
#   Long-time stability: breather with period T ≈ 6.5–7, stable amplitude.
#
# Version 1.2: Extended to t=50, periodicity analysis added.

import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2, fftfreq
from scipy.signal import find_peaks

# -----------------------------------------------------------------------------
# Parameters
# -----------------------------------------------------------------------------
Nx, Ny = 256, 256
Lx, Ly = 40.0, 40.0
dx, dy = Lx / Nx, Ly / Ny
x = np.linspace(-Lx/2, Lx/2, Nx, endpoint=False)
y = np.linspace(-Ly/2, Ly/2, Ny, endpoint=False)
X, Y = np.meshgrid(x, y, indexing='ij')

m = 1.0          # mass
G = 1.0          # interaction strength
S = 0.5          # saturation parameter
dt = 0.005
T_max = 50.0
N_steps = int(T_max / dt)
save_every = 50  # save every 0.25 time units

# Initial condition: Gaussian with amplitude chosen to be supercritical for cubic NLS
A0 = 3.0
sigma = 2.0
psi = A0 * np.exp(-(X**2 + Y**2) / (2 * sigma**2))

# Fourier space operators
kx = fftfreq(Nx, d=dx) * 2 * np.pi
ky = fftfreq(Ny, d=dy) * 2 * np.pi
KX, KY = np.meshgrid(kx, ky, indexing='ij')
K2 = KX**2 + KY**2

# Kinetic operator (evolution factor for half step)
kinetic_half = np.exp(-1j * K2 / (2 * m) * dt / 2)

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------
def compute_observables(psi_):
    rho = np.abs(psi_)**2
    norm = np.sum(rho) * dx * dy
    rho_max = np.max(rho)
    
    # RMS radius
    rms = np.sqrt(np.sum((X**2 + Y**2) * rho) * dx * dy / norm) if norm > 1e-12 else 0.0
    
    # Energy: E = ∫ [ (1/2m)|∇ψ|² + U(ρ) ] dx dy
    psi_k = fft2(psi_)
    kinetic = (1.0 / (2 * m)) * np.sum(K2 * np.abs(psi_k)**2) * dx * dy / (Nx * Ny)
    
    # Potential energy: U(ρ) = -G [ ρ/S - (1/S²)ln(1 + Sρ) ]
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

norm0, rho0, rms0, E0 = compute_observables(psi)

print("Starting long-time evolution...")
print(f"Initial: ρ_max = {rho0:.4f}, E = {E0:.6f}, N = {norm0:.6f}")
print("="*70)

# -----------------------------------------------------------------------------
# Time evolution: Strang splitting
# -----------------------------------------------------------------------------
for step in range(N_steps):
    t = step * dt
    
    # (1) Half step kinetic
    psi_k = fft2(psi)
    psi_k *= kinetic_half
    psi = ifft2(psi_k)
    
    # (2) Full step potential (nonlinear)
    rho = np.abs(psi)**2
    V_eff = -G * rho / (1.0 + S * rho)
    psi *= np.exp(-1j * V_eff * dt)
    
    # (3) Half step kinetic
    psi_k = fft2(psi)
    psi_k *= kinetic_half
    psi = ifft2(psi_k)
    
    # Observables
    if step % save_every == 0:
        norm, rho_max, rms, energy = compute_observables(psi)
        times.append(t)
        norms.append(norm / norm0)
        rho_maxs.append(rho_max)
        rms_vals.append(rms)
        energies.append(energy / E0)
        
        if step % (save_every * 10) == 0:  # print every 2.5 time units
            print(f"t = {t:6.2f} | ρ_max = {rho_max:8.4f} | RMS = {rms:6.3f} | "
                  f"N/N₀ = {norm/norm0:.6f} | E/E₀ = {energy/E0:.6f}")

# Final observables
norm_f, rho_f, rms_f, E_f = compute_observables(psi)
times.append(T_max)
norms.append(norm_f / norm0)
rho_maxs.append(rho_f)
rms_vals.append(rms_f)
energies.append(E_f / E0)

times = np.array(times)
rho_maxs = np.array(rho_maxs)
rms_vals = np.array(rms_vals)
norms = np.array(norms)
energies = np.array(energies)

# -----------------------------------------------------------------------------
# Periodicity analysis
# -----------------------------------------------------------------------------
print("\n" + "="*70)
print("PERIODICITY ANALYSIS")
print("="*70)

# Find peaks in ρ_max(t)
peaks_idx, _ = find_peaks(rho_maxs, height=20, distance=50)  # distance ~ T/dt_save
peaks_times = times[peaks_idx]
peaks_values = rho_maxs[peaks_idx]

if len(peaks_times) > 1:
    periods = np.diff(peaks_times)
    mean_period = np.mean(periods)
    std_period = np.std(periods)
    print(f"Number of peaks detected: {len(peaks_times)}")
    print(f"Peak times: {peaks_times}")
    print(f"Peak values: {peaks_values}")
    print(f"Periods between peaks: {periods}")
    print(f"Mean period: {mean_period:.3f} ± {std_period:.3f}")
    print(f"Peak amplitude drift: {(peaks_values[-1] - peaks_values[0])/peaks_values[0] * 100:.2f}%")
else:
    print("Insufficient peaks for period analysis")
    mean_period = None

# Find minima
minima_idx, _ = find_peaks(-rho_maxs, height=-15, distance=50)
minima_times = times[minima_idx]
minima_values = rho_maxs[minima_idx]

if len(minima_times) > 1:
    print(f"\nNumber of minima detected: {len(minima_times)}")
    print(f"Minima times: {minima_times}")
    print(f"Minima values: {minima_values}")
    print(f"Minima drift: {(minima_values[-1] - minima_values[0])/minima_values[0] * 100:.2f}%")

# -----------------------------------------------------------------------------
# Plotting: Observables
# -----------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].plot(times, rho_maxs, 'b-', linewidth=1.5, alpha=0.8)
if len(peaks_idx) > 0:
    axes[0, 0].plot(times[peaks_idx], rho_maxs[peaks_idx], 'ro', markersize=6, label='Peaks')
if len(minima_idx) > 0:
    axes[0, 0].plot(times[minima_idx], rho_maxs[minima_idx], 'go', markersize=6, label='Minima')
axes[0, 0].set_xlabel('Time', fontsize=12)
axes[0, 0].set_ylabel(r'$\rho_{\max}(t)$', fontsize=12)
axes[0, 0].set_title('Peak Density — Breather Oscillations', fontsize=13)
axes[0, 0].legend(fontsize=10)
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(times, rms_vals, 'g-', linewidth=1.5)
axes[0, 1].set_xlabel('Time', fontsize=12)
axes[0, 1].set_ylabel('RMS Radius', fontsize=12)
axes[0, 1].set_title('RMS Width (anti-correlated with density)', fontsize=13)
axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].plot(times, norms, 'r-', linewidth=1.5)
axes[1, 0].axhline(1.0, color='k', linestyle='--', alpha=0.5)
axes[1, 0].set_xlabel('Time', fontsize=12)
axes[1, 0].set_ylabel(r'$N(t)/N_0$', fontsize=12)
axes[1, 0].set_title('Norm Conservation', fontsize=13)
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].plot(times, energies, 'm-', linewidth=1.5)
axes[1, 1].axhline(1.0, color='k', linestyle='--', alpha=0.5)
axes[1, 1].set_xlabel('Time', fontsize=12)
axes[1, 1].set_ylabel(r'$E(t)/E_0$', fontsize=12)
axes[1, 1].set_title('Energy Conservation', fontsize=13)
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('test1_2_observables.png', dpi=150)
print("\n✓ Saved: test1_2_observables.png")

# -----------------------------------------------------------------------------
# Plotting: Periodicity detail
# -----------------------------------------------------------------------------
fig2, ax = plt.subplots(1, 1, figsize=(12, 5))
ax.plot(times, rho_maxs, 'b-', linewidth=2, label=r'$\rho_{\max}(t)$')
if len(peaks_idx) > 0:
    ax.plot(times[peaks_idx], rho_maxs[peaks_idx], 'ro', markersize=8, label='Peaks', zorder=10)
if len(minima_idx) > 0:
    ax.plot(times[minima_idx], rho_maxs[minima_idx], 'go', markersize=8, label='Minima', zorder=10)

if mean_period is not None:
    # Mark periods
    for i, t_peak in enumerate(peaks_times[:-1]):
        ax.axvspan(t_peak, t_peak + mean_period, alpha=0.1, color='gray')
        ax.text(t_peak + mean_period/2, 45, f'T≈{periods[i]:.1f}', 
                ha='center', fontsize=9, color='gray')

ax.set_xlabel('Time', fontsize=13)
ax.set_ylabel(r'$\rho_{\max}(t)$', fontsize=13)
ax.set_title('Breather Periodicity — Long-Time Stability', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('test1_2_periodicity.png', dpi=150)
print("✓ Saved: test1_2_periodicity.png")

# Density snapshot
fig3, ax = plt.subplots(1, 1, figsize=(7, 6))
rho_final = np.abs(psi)**2
im = ax.contourf(X, Y, rho_final, levels=40, cmap='hot')
ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('y', fontsize=12)
ax.set_title(rf'Density $\rho(x,y)$ at $t={T_max}$', fontsize=13)
plt.colorbar(im, ax=ax, label=r'$|\psi|^2$')
plt.tight_layout()
plt.savefig('test1_2_density_evolution.png', dpi=150)
print("✓ Saved: test1_2_density_evolution.png")

plt.show()

print("\n" + "="*70)
print("TEST 1.2 COMPLETE: LONG-TIME BREATHER STABILITY IN 2+1D")
print("="*70)
print(f"Simulation time:      {T_max}")
print(f"Initial peak density: {rho0:.4f}")
print(f"Final peak density:   {rho_f:.4f}")
print(f"Norm conservation:    {abs(norm_f/norm0 - 1.0) * 100:.3e}%")
print(f"Energy conservation:  {abs(E_f/E0 - 1.0) * 100:.3e}%")
if mean_period is not None:
    print(f"Breather period:      {mean_period:.3f} ± {std_period:.3f}")
    print(f"Number of cycles:     {T_max / mean_period:.1f}")
print("="*70)
