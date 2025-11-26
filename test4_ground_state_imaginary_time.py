# test4_ground_state_imaginary_time.py
# Outputs: test4_relaxation.png, test4_ground_state_profile.png, test4_stationarity_check.png
#
# THEORETICAL SUMMARY:
# ====================
# PDE (hypothesis):
#   i ∂_t ψ = -(1/2m) ∇² ψ - [G|ψ|²/(1 + S|ψ|²)] ψ
#
# We seek stationary solutions: ψ(x,y,t) = φ(x,y) e^(-iμt)
# where φ is real and satisfies:
#   μ φ = -(1/2m) ∇² φ - [Gφ²/(1+Sφ²)] φ
#
# Method: Imaginary time evolution
#   -∂_τ ψ = -(1/2m) ∇² ψ - [G|ψ|²/(1 + S|ψ|²)] ψ - μ ψ
#   with normalization ∫|ψ|² = N₀ at each step
#
# This minimizes energy E[ψ] at fixed norm N, converging to ground state.
#
# Control model (cubic NLS):
#   μ φ = -(1/2m) ∇² φ - g|φ|² φ
#   Ground state: Gaussian-like profile
#
# Literature (Cazenave, Sulem & Sulem):
#   Ground states exist for attractive NLS with appropriate conditions
#   Imaginary time evolution converges exponentially to ground state
#
# Observables:
#   E(τ) — energy in imaginary time (must decrease monotonically)
#   μ(τ) — chemical potential μ = dE/dN
#   φ(r) — radial profile of ground state
#   Stationarity: evolve φ e^(-iμt) in real time → check only phase changes
#
# Expected behavior:
#   Phase I:  E(τ) → E_gs (ground state energy)
#   Phase II: φ(r) becomes stationary, μ stabilizes
#   Phase III: Real-time evolution shows ρ(t) = const

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
R = np.sqrt(X**2 + Y**2)

m = 1.0
G = 1.0
S = 0.5

# Imaginary time parameters
dtau = 0.01
N_relax = 5000  # relaxation steps
norm_target = 100.0  # target norm

# Real time verification
dt_real = 0.005
T_verify = 30.0
N_verify = int(T_verify / dt_real)

# Initial guess: Gaussian
A0 = 3.0
sigma = 2.0
psi = A0 * np.exp(-R**2 / (2 * sigma**2))

# Normalize to target norm
rho = np.abs(psi)**2
norm_init = np.sum(rho) * dx * dy
psi *= np.sqrt(norm_target / norm_init)

# Fourier space
kx = fftfreq(Nx, d=dx) * 2 * np.pi
ky = fftfreq(Ny, d=dy) * 2 * np.pi
KX, KY = np.meshgrid(kx, ky, indexing='ij')
K2 = KX**2 + KY**2

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------
def compute_energy(psi_):
    """Compute total energy E = T + V"""
    psi_k = fft2(psi_)
    kinetic = (1.0 / (2 * m)) * np.sum(K2 * np.abs(psi_k)**2) * dx * dy / (Nx * Ny)
    
    rho = np.abs(psi_)**2
    U_rho = -G * (rho / S - (1.0 / S**2) * np.log(1.0 + S * rho + 1e-30))
    potential = np.sum(U_rho) * dx * dy
    
    return kinetic + potential

def compute_chemical_potential(psi_):
    """μ = δE/δψ* (evaluated as energy per particle)"""
    rho = np.abs(psi_)**2
    norm = np.sum(rho) * dx * dy
    energy = compute_energy(psi_)
    return energy / norm

def normalize(psi_, target_norm):
    """Normalize wavefunction to target norm"""
    rho = np.abs(psi_)**2
    current_norm = np.sum(rho) * dx * dy
    return psi_ * np.sqrt(target_norm / current_norm)

def radial_average(field):
    """Compute radial average of 2D field"""
    r_bins = np.linspace(0, Lx/2, 100)
    r_avg = np.zeros(len(r_bins) - 1)
    
    for i in range(len(r_bins) - 1):
        mask = (R >= r_bins[i]) & (R < r_bins[i+1])
        if np.sum(mask) > 0:
            r_avg[i] = np.mean(field[mask])
    
    r_centers = (r_bins[:-1] + r_bins[1:]) / 2
    return r_centers, r_avg

# Storage
tau_vals = []
energy_vals = []
mu_vals = []
max_density_vals = []

print("="*70)
print("TEST 4: GROUND STATE SEARCH VIA IMAGINARY TIME EVOLUTION")
print("="*70)
print(f"Target norm: N = {norm_target}")
print(f"Imaginary time step: dτ = {dtau}")
print(f"Relaxation steps: {N_relax}")
print("="*70)

# -----------------------------------------------------------------------------
# Phase I: Imaginary time evolution
# -----------------------------------------------------------------------------
print("\nPhase I: Relaxation to ground state...")

# Kinetic operator for imaginary time
kinetic_half_imag = np.exp(-K2 / (2 * m) * dtau / 2)

for step in range(N_relax):
    tau = step * dtau
    
    # Strang splitting in imaginary time: exp(-H dτ)
    # (1) Half kinetic
    psi_k = fft2(psi)
    psi_k *= kinetic_half_imag
    psi = ifft2(psi_k)
    
    # (2) Full potential
    rho = np.abs(psi)**2
    V_eff = -G * rho / (1.0 + S * rho)
    psi *= np.exp(-V_eff * dtau)
    
    # (3) Half kinetic
    psi_k = fft2(psi)
    psi_k *= kinetic_half_imag
    psi = ifft2(psi_k)
    
    # (4) Normalize to preserve norm
    psi = normalize(psi, norm_target)
    
    # Observables
    if step % 50 == 0:
        energy = compute_energy(psi)
        mu = compute_chemical_potential(psi)
        rho_max = np.max(np.abs(psi)**2)
        
        tau_vals.append(tau)
        energy_vals.append(energy)
        mu_vals.append(mu)
        max_density_vals.append(rho_max)
        
        if step % 500 == 0:
            print(f"τ = {tau:6.1f} | E = {energy:10.6f} | μ = {mu:10.6f} | ρ_max = {rho_max:8.4f}")

# Final ground state
energy_gs = compute_energy(psi)
mu_gs = compute_chemical_potential(psi)
rho_gs = np.abs(psi)**2
rho_max_gs = np.max(rho_gs)

tau_vals = np.array(tau_vals)
energy_vals = np.array(energy_vals)
mu_vals = np.array(mu_vals)
max_density_vals = np.array(max_density_vals)

print(f"\nGround state converged:")
print(f"  E_gs = {energy_gs:.6f}")
print(f"  μ_gs = {mu_gs:.6f}")
print(f"  ρ_max = {rho_max_gs:.4f}")
print(f"  Energy drop: ΔE = {energy_vals[0] - energy_gs:.6f}")

# Check convergence: dE/dτ → 0
if len(energy_vals) > 10:
    energy_drift_rate = (energy_vals[-1] - energy_vals[-10]) / (tau_vals[-1] - tau_vals[-10])
    print(f"  dE/dτ (final) = {energy_drift_rate:.6e}")

# -----------------------------------------------------------------------------
# Phase II: Verify stationarity in real time
# -----------------------------------------------------------------------------
print("\nPhase II: Stationarity verification in real time...")

psi_verify = psi.copy()
kinetic_half_real = np.exp(-1j * K2 / (2 * m) * dt_real / 2)

times_verify = []
rho_max_verify = []
norm_verify = []
phase_verify = []

for step in range(N_verify):
    t = step * dt_real
    
    # Real time evolution (Strang splitting)
    psi_k = fft2(psi_verify)
    psi_k *= kinetic_half_real
    psi_verify = ifft2(psi_k)
    
    rho = np.abs(psi_verify)**2
    V_eff = -G * rho / (1.0 + S * rho)
    psi_verify *= np.exp(-1j * V_eff * dt_real)
    
    psi_k = fft2(psi_verify)
    psi_k *= kinetic_half_real
    psi_verify = ifft2(psi_k)
    
    if step % 100 == 0:
        rho_t = np.abs(psi_verify)**2
        norm_t = np.sum(rho_t) * dx * dy
        rho_max_t = np.max(rho_t)
        
        # Extract global phase
        idx_max = np.unravel_index(np.argmax(rho_t), rho_t.shape)
        phase_t = np.angle(psi_verify[idx_max])
        
        times_verify.append(t)
        rho_max_verify.append(rho_max_t)
        norm_verify.append(norm_t)
        phase_verify.append(phase_t)

times_verify = np.array(times_verify)
rho_max_verify = np.array(rho_max_verify)
norm_verify = np.array(norm_verify)
phase_verify = np.array(phase_verify)

# Unwrap phase to check linearity
phase_unwrapped = np.unwrap(phase_verify)
if len(times_verify) > 2:
    phase_fit = np.polyfit(times_verify, phase_unwrapped, 1)
    omega_measured = -phase_fit[0]  # ψ ~ e^(-iμt) → phase = -μt
    print(f"\nStationarity check:")
    print(f"  ρ_max variation: {np.std(rho_max_verify)/np.mean(rho_max_verify)*100:.4f}%")
    print(f"  Norm variation: {np.std(norm_verify)/np.mean(norm_verify)*100:.6f}%")
    print(f"  Phase evolution: φ(t) = {phase_fit[0]:.6f} t + {phase_fit[1]:.4f}")
    print(f"  Measured μ from phase: {omega_measured:.6f}")
    print(f"  Expected μ from energy: {mu_gs:.6f}")
    print(f"  Difference: {abs(omega_measured - mu_gs):.6e}")

# -----------------------------------------------------------------------------
# Plotting: Relaxation dynamics
# -----------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].plot(tau_vals, energy_vals, 'b-', linewidth=2)
axes[0, 0].set_xlabel('Imaginary time τ', fontsize=12)
axes[0, 0].set_ylabel('Energy E(τ)', fontsize=12)
axes[0, 0].set_title('Energy Relaxation (must decrease)', fontsize=13, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(tau_vals, mu_vals, 'r-', linewidth=2)
axes[0, 1].set_xlabel('Imaginary time τ', fontsize=12)
axes[0, 1].set_ylabel('Chemical potential μ(τ)', fontsize=12)
axes[0, 1].set_title('Chemical Potential Convergence', fontsize=13, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].plot(tau_vals, max_density_vals, 'g-', linewidth=2)
axes[1, 0].set_xlabel('Imaginary time τ', fontsize=12)
axes[1, 0].set_ylabel(r'$\rho_{\max}(\tau)$', fontsize=12)
axes[1, 0].set_title('Peak Density Evolution', fontsize=13, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)

# Energy derivative (convergence rate)
if len(tau_vals) > 1:
    dE_dtau = np.gradient(energy_vals, tau_vals)
    axes[1, 1].semilogy(tau_vals, np.abs(dE_dtau), 'purple', linewidth=2)
    axes[1, 1].set_xlabel('Imaginary time τ', fontsize=12)
    axes[1, 1].set_ylabel(r'$|dE/d\tau|$', fontsize=12)
    axes[1, 1].set_title('Convergence Rate (log scale)', fontsize=13, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3, which='both')

plt.tight_layout()
plt.savefig('test4_relaxation.png', dpi=150)
print("\n✓ Saved: test4_relaxation.png")

# -----------------------------------------------------------------------------
# Plotting: Ground state profile
# -----------------------------------------------------------------------------
fig2, axes2 = plt.subplots(1, 3, figsize=(16, 5))

# 2D density
im = axes2[0].contourf(X, Y, rho_gs, levels=50, cmap='hot')
axes2[0].set_xlabel('x', fontsize=12)
axes2[0].set_ylabel('y', fontsize=12)
axes2[0].set_title(r'Ground State Density $\rho_{\mathrm{gs}}(x,y)$', fontsize=13, fontweight='bold')
plt.colorbar(im, ax=axes2[0], label=r'$|\phi|^2$')
axes2[0].set_aspect('equal')

# Radial profile
r_centers, rho_radial = radial_average(rho_gs)
axes2[1].plot(r_centers, rho_radial, 'b-', linewidth=2.5, label='Ground state')
axes2[1].set_xlabel('Radius r', fontsize=12)
axes2[1].set_ylabel(r'$\langle \rho(r) \rangle$', fontsize=12)
axes2[1].set_title('Radial Density Profile', fontsize=13, fontweight='bold')
axes2[1].grid(True, alpha=0.3)
axes2[1].legend(fontsize=11)

# V_eff profile
V_eff_gs = -G * rho_gs / (1.0 + S * rho_gs)
_, V_radial = radial_average(V_eff_gs)
axes2[2].plot(r_centers, V_radial, 'darkred', linewidth=2.5)
axes2[2].set_xlabel('Radius r', fontsize=12)
axes2[2].set_ylabel(r'$\langle V_{\mathrm{eff}}(r) \rangle$', fontsize=12)
axes2[2].set_title('Effective Potential Profile', fontsize=13, fontweight='bold')
axes2[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('test4_ground_state_profile.png', dpi=150)
print("✓ Saved: test4_ground_state_profile.png")

# -----------------------------------------------------------------------------
# Plotting: Stationarity verification
# -----------------------------------------------------------------------------
fig3, axes3 = plt.subplots(2, 2, figsize=(14, 10))

axes3[0, 0].plot(times_verify, rho_max_verify, 'b-', linewidth=2)
axes3[0, 0].axhline(rho_max_gs, color='red', linestyle='--', linewidth=1.5, label=f'ρ_max (gs) = {rho_max_gs:.4f}')
axes3[0, 0].set_xlabel('Time t', fontsize=12)
axes3[0, 0].set_ylabel(r'$\rho_{\max}(t)$', fontsize=12)
axes3[0, 0].set_title('Peak Density in Real Time (must be constant)', fontsize=13, fontweight='bold')
axes3[0, 0].legend(fontsize=10)
axes3[0, 0].grid(True, alpha=0.3)

axes3[0, 1].plot(times_verify, norm_verify / norm_target, 'g-', linewidth=2)
axes3[0, 1].axhline(1.0, color='k', linestyle='--', alpha=0.5)
axes3[0, 1].set_xlabel('Time t', fontsize=12)
axes3[0, 1].set_ylabel(r'$N(t)/N_0$', fontsize=12)
axes3[0, 1].set_title('Norm Conservation', fontsize=13, fontweight='bold')
axes3[0, 1].grid(True, alpha=0.3)

axes3[1, 0].plot(times_verify, phase_unwrapped, 'purple', linewidth=2, label='Measured phase')
if len(times_verify) > 2:
    phase_linear = phase_fit[0] * times_verify + phase_fit[1]
    axes3[1, 0].plot(times_verify, phase_linear, 'r--', linewidth=1.5, label=f'Linear fit: ω={-phase_fit[0]:.4f}')
axes3[1, 0].set_xlabel('Time t', fontsize=12)
axes3[1, 0].set_ylabel('Phase φ(t) [rad]', fontsize=12)
axes3[1, 0].set_title('Phase Evolution (must be linear)', fontsize=13, fontweight='bold')
axes3[1, 0].legend(fontsize=10)
axes3[1, 0].grid(True, alpha=0.3)

# Phase residuals
if len(times_verify) > 2:
    phase_residuals = phase_unwrapped - phase_linear
    axes3[1, 1].plot(times_verify, phase_residuals, 'darkgreen', linewidth=1.5)
    axes3[1, 1].axhline(0, color='k', linestyle='--', alpha=0.5)
    axes3[1, 1].set_xlabel('Time t', fontsize=12)
    axes3[1, 1].set_ylabel('Phase residuals [rad]', fontsize=12)
    axes3[1, 1].set_title('Phase Linearity Check', fontsize=13, fontweight='bold')
    axes3[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('test4_stationarity_check.png', dpi=150)
print("✓ Saved: test4_stationarity_check.png")

plt.show()

# -----------------------------------------------------------------------------
# Final summary
# -----------------------------------------------------------------------------
print("\n" + "="*70)
print("TEST 4 COMPLETE: GROUND STATE EXISTENCE")
print("="*70)
print(f"Target norm:           N = {norm_target}")
print(f"Ground state energy:   E_gs = {energy_gs:.6f}")
print(f"Chemical potential:    μ_gs = {mu_gs:.6f}")
print(f"Peak density:          ρ_max = {rho_max_gs:.4f}")
print(f"Energy convergence:    {energy_vals[0]:.6f} → {energy_gs:.6f}")
print(f"\nStationarity in real time:")
print(f"  ρ_max drift: {np.std(rho_max_verify)/np.mean(rho_max_verify)*100:.4f}%")
print(f"  μ (energy): {mu_gs:.6f}")
print(f"  μ (phase):  {omega_measured:.6f}")
print(f"  Agreement:  {abs(omega_measured - mu_gs)/mu_gs*100:.4f}%")

if abs(omega_measured - mu_gs)/abs(mu_gs) < 0.01 and np.std(rho_max_verify)/np.mean(rho_max_verify) < 0.001:
    print("\n✓ PASSED: Ground state is stationary")
    print("✓ Profile φ(r) evolves as φ(r)e^(-iμt) with constant density")
elif np.std(rho_max_verify)/np.mean(rho_max_verify) < 0.01:
    print("\n✓ IMPROVE: Ground state is quasi-stationary (small oscillations)")
else:
    print("\n⚠ FIX: Significant time-dependence detected")

print("="*70)
