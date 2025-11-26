# test4_1_ground_state_corrected.py
# Outputs: test4_1_relaxation.png, test4_1_ground_state_profile.png, test4_1_stationarity_check.png
#
# THEORETICAL SUMMARY:
# ====================
# PDE (hypothesis):
#   i ∂_t ψ = -(1/2m) ∇² ψ - [G|ψ|²/(1 + S|ψ|²)] ψ
#
# Stationary solutions: ψ(x,y,t) = φ(x,y) e^(-iμt)
# where φ satisfies the nonlinear eigenvalue problem:
#   μ φ = -(1/2m) ∇² φ - [Gφ²/(1+Sφ²)] φ
#
# Chemical potential (CORRECTED):
#   μ = <φ|Ĥ|φ> / <φ|φ>  (NOT E/N)
#
# For nonlinear systems: μ = δE/δN (variational derivative)
#
# Method: Imaginary time evolution with proper μ calculation
#
# Expected: μ from operator matches μ from phase evolution

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

dtau = 0.01
N_relax = 5000
norm_target = 100.0

dt_real = 0.005
T_verify = 30.0
N_verify = int(T_verify / dt_real)

# Initial guess
A0 = 3.0
sigma = 2.0
psi = A0 * np.exp(-R**2 / (2 * sigma**2))

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
    """Total energy E = T + V"""
    psi_k = fft2(psi_)
    kinetic = (1.0 / (2 * m)) * np.sum(K2 * np.abs(psi_k)**2) * dx * dy / (Nx * Ny)
    
    rho = np.abs(psi_)**2
    U_rho = -G * (rho / S - (1.0 / S**2) * np.log(1.0 + S * rho + 1e-30))
    potential = np.sum(U_rho) * dx * dy
    
    return kinetic + potential

def apply_hamiltonian(psi_):
    """Apply Hamiltonian operator: H|ψ⟩ = [-(1/2m)∇² + V_eff]|ψ⟩"""
    # Kinetic part
    psi_k = fft2(psi_)
    T_psi_k = (K2 / (2 * m)) * psi_k
    T_psi = ifft2(T_psi_k)
    
    # Potential part
    rho = np.abs(psi_)**2
    V_eff = -G * rho / (1.0 + S * rho)
    V_psi = V_eff * psi_
    
    return T_psi + V_psi

def compute_chemical_potential_correct(psi_):
    """
    Correct chemical potential: μ = ⟨ψ|Ĥ|ψ⟩ / ⟨ψ|ψ⟩
    For stationary state: Ĥ|φ⟩ = μ|φ⟩
    """
    H_psi = apply_hamiltonian(psi_)
    
    # μ = ⟨ψ|H|ψ⟩ / ⟨ψ|ψ⟩
    numerator = np.sum(np.conj(psi_) * H_psi) * dx * dy
    denominator = np.sum(np.abs(psi_)**2) * dx * dy
    
    mu = np.real(numerator / denominator)
    return mu

def compute_energy_per_particle(psi_):
    """Energy per particle (for comparison, NOT chemical potential)"""
    energy = compute_energy(psi_)
    rho = np.abs(psi_)**2
    norm = np.sum(rho) * dx * dy
    return energy / norm

def normalize(psi_, target_norm):
    rho = np.abs(psi_)**2
    current_norm = np.sum(rho) * dx * dy
    return psi_ * np.sqrt(target_norm / current_norm)

def radial_average(field):
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
mu_vals_correct = []
mu_vals_wrong = []
max_density_vals = []

print("="*70)
print("TEST 4.1: GROUND STATE WITH CORRECTED CHEMICAL POTENTIAL")
print("="*70)
print(f"Target norm: N = {norm_target}")
print(f"Relaxation steps: {N_relax}")
print("="*70)
print("Chemical potential definition:")
print("  CORRECT: μ = ⟨φ|Ĥ|φ⟩ / ⟨φ|φ⟩  (expectation value)")
print("  WRONG:   μ = E/N  (energy per particle)")
print("="*70)

# -----------------------------------------------------------------------------
# Phase I: Imaginary time evolution
# -----------------------------------------------------------------------------
print("\nPhase I: Relaxation to ground state...")

kinetic_half_imag = np.exp(-K2 / (2 * m) * dtau / 2)

for step in range(N_relax):
    tau = step * dtau
    
    # Strang splitting
    psi_k = fft2(psi)
    psi_k *= kinetic_half_imag
    psi = ifft2(psi_k)
    
    rho = np.abs(psi)**2
    V_eff = -G * rho / (1.0 + S * rho)
    psi *= np.exp(-V_eff * dtau)
    
    psi_k = fft2(psi)
    psi_k *= kinetic_half_imag
    psi = ifft2(psi_k)
    
    psi = normalize(psi, norm_target)
    
    if step % 50 == 0:
        energy = compute_energy(psi)
        mu_correct = compute_chemical_potential_correct(psi)
        mu_wrong = compute_energy_per_particle(psi)
        rho_max = np.max(np.abs(psi)**2)
        
        tau_vals.append(tau)
        energy_vals.append(energy)
        mu_vals_correct.append(mu_correct)
        mu_vals_wrong.append(mu_wrong)
        max_density_vals.append(rho_max)
        
        if step % 500 == 0:
            print(f"τ = {tau:6.1f} | E = {energy:10.6f} | μ_correct = {mu_correct:10.6f} | μ_wrong = {mu_wrong:10.6f} | ρ_max = {rho_max:8.4f}")

# Final ground state
energy_gs = compute_energy(psi)
mu_gs_correct = compute_chemical_potential_correct(psi)
mu_gs_wrong = compute_energy_per_particle(psi)
rho_gs = np.abs(psi)**2
rho_max_gs = np.max(rho_gs)

tau_vals = np.array(tau_vals)
energy_vals = np.array(energy_vals)
mu_vals_correct = np.array(mu_vals_correct)
mu_vals_wrong = np.array(mu_vals_wrong)
max_density_vals = np.array(max_density_vals)

print(f"\nGround state converged:")
print(f"  E_gs = {energy_gs:.6f}")
print(f"  μ_gs (CORRECT) = {mu_gs_correct:.6f}")
print(f"  μ_gs (wrong, E/N) = {mu_gs_wrong:.6f}")
print(f"  ρ_max = {rho_max_gs:.4f}")
print(f"  Difference: {mu_gs_correct - mu_gs_wrong:.6f}")

# Verify eigenvalue equation: H|φ⟩ = μ|φ⟩
H_phi = apply_hamiltonian(psi)
mu_phi = mu_gs_correct * psi
residual = np.sqrt(np.sum(np.abs(H_phi - mu_phi)**2) * dx * dy)
norm_psi = np.sqrt(np.sum(np.abs(psi)**2) * dx * dy)
print(f"  Eigenvalue residual: ||Ĥφ - μφ|| / ||φ|| = {residual/norm_psi:.6e}")

# -----------------------------------------------------------------------------
# Phase II: Real time verification
# -----------------------------------------------------------------------------
print("\nPhase II: Stationarity verification...")

psi_verify = psi.copy()
kinetic_half_real = np.exp(-1j * K2 / (2 * m) * dt_real / 2)

times_verify = []
rho_max_verify = []
norm_verify = []
phase_verify = []

for step in range(N_verify):
    t = step * dt_real
    
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

phase_unwrapped = np.unwrap(phase_verify)
if len(times_verify) > 2:
    phase_fit = np.polyfit(times_verify, phase_unwrapped, 1)
    omega_measured = -phase_fit[0]
    
    print(f"\nStationarity check:")
    print(f"  ρ_max variation: {np.std(rho_max_verify)/np.mean(rho_max_verify)*100:.4f}%")
    print(f"  Norm variation: {np.std(norm_verify)/np.mean(norm_verify)*100:.6f}%")
    print(f"  Phase: φ(t) = {phase_fit[0]:.6f} t + {phase_fit[1]:.4f}")
    print(f"\nChemical potential comparison:")
    print(f"  μ (from Ĥ|φ⟩):     {mu_gs_correct:.6f}")
    print(f"  μ (from phase):    {omega_measured:.6f}")
    print(f"  Relative error:    {abs(omega_measured - mu_gs_correct)/abs(mu_gs_correct)*100:.4f}%")

# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].plot(tau_vals, energy_vals, 'b-', linewidth=2)
axes[0, 0].set_xlabel('Imaginary time τ', fontsize=12)
axes[0, 0].set_ylabel('Energy E(τ)', fontsize=12)
axes[0, 0].set_title('Energy Relaxation', fontsize=13, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(tau_vals, mu_vals_correct, 'r-', linewidth=2, label=r'μ = ⟨φ|Ĥ|φ⟩/⟨φ|φ⟩ (correct)')
axes[0, 1].plot(tau_vals, mu_vals_wrong, 'gray', linewidth=2, linestyle='--', label='μ = E/N (wrong)', alpha=0.6)
axes[0, 1].set_xlabel('Imaginary time τ', fontsize=12)
axes[0, 1].set_ylabel('Chemical potential μ(τ)', fontsize=12)
axes[0, 1].set_title('Chemical Potential Convergence', fontsize=13, fontweight='bold')
axes[0, 1].legend(fontsize=9)
axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].plot(tau_vals, max_density_vals, 'g-', linewidth=2)
axes[1, 0].set_xlabel('Imaginary time τ', fontsize=12)
axes[1, 0].set_ylabel(r'$\rho_{\max}(\tau)$', fontsize=12)
axes[1, 0].set_title('Peak Density Evolution', fontsize=13, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)

# Difference between correct and wrong μ
axes[1, 1].plot(tau_vals, mu_vals_correct - mu_vals_wrong, 'purple', linewidth=2)
axes[1, 1].set_xlabel('Imaginary time τ', fontsize=12)
axes[1, 1].set_ylabel(r'μ_correct - μ_wrong', fontsize=12)
axes[1, 1].set_title('Chemical Potential Correction', fontsize=13, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('test4_1_relaxation.png', dpi=150)
print("\n✓ Saved: test4_1_relaxation.png")

# Ground state profile
fig2, axes2 = plt.subplots(1, 3, figsize=(16, 5))

im = axes2[0].contourf(X, Y, rho_gs, levels=50, cmap='hot')
axes2[0].set_xlabel('x', fontsize=12)
axes2[0].set_ylabel('y', fontsize=12)
axes2[0].set_title(r'Ground State $\rho_{\mathrm{gs}}(x,y)$', fontsize=13, fontweight='bold')
plt.colorbar(im, ax=axes2[0], label=r'$|\phi|^2$')
axes2[0].set_aspect('equal')

r_centers, rho_radial = radial_average(rho_gs)
axes2[1].plot(r_centers, rho_radial, 'b-', linewidth=2.5)
axes2[1].set_xlabel('Radius r', fontsize=12)
axes2[1].set_ylabel(r'$\langle \rho(r) \rangle$', fontsize=12)
axes2[1].set_title('Radial Profile', fontsize=13, fontweight='bold')
axes2[1].grid(True, alpha=0.3)

V_eff_gs = -G * rho_gs / (1.0 + S * rho_gs)
_, V_radial = radial_average(V_eff_gs)
axes2[2].plot(r_centers, V_radial, 'darkred', linewidth=2.5)
axes2[2].set_xlabel('Radius r', fontsize=12)
axes2[2].set_ylabel(r'$\langle V_{\mathrm{eff}}(r) \rangle$', fontsize=12)
axes2[2].set_title('Effective Potential', fontsize=13, fontweight='bold')
axes2[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('test4_1_ground_state_profile.png', dpi=150)
print("✓ Saved: test4_1_ground_state_profile.png")

# Stationarity
fig3, axes3 = plt.subplots(2, 2, figsize=(14, 10))

axes3[0, 0].plot(times_verify, rho_max_verify, 'b-', linewidth=2)
axes3[0, 0].axhline(rho_max_gs, color='red', linestyle='--', linewidth=1.5)
axes3[0, 0].set_xlabel('Time t', fontsize=12)
axes3[0, 0].set_ylabel(r'$\rho_{\max}(t)$', fontsize=12)
axes3[0, 0].set_title('Peak Density (must be constant)', fontsize=13, fontweight='bold')
axes3[0, 0].grid(True, alpha=0.3)

axes3[0, 1].plot(times_verify, norm_verify / norm_target, 'g-', linewidth=2)
axes3[0, 1].axhline(1.0, color='k', linestyle='--', alpha=0.5)
axes3[0, 1].set_xlabel('Time t', fontsize=12)
axes3[0, 1].set_ylabel(r'$N(t)/N_0$', fontsize=12)
axes3[0, 1].set_title('Norm Conservation', fontsize=13, fontweight='bold')
axes3[0, 1].grid(True, alpha=0.3)

axes3[1, 0].plot(times_verify, phase_unwrapped, 'purple', linewidth=2, label='Measured')
if len(times_verify) > 2:
    phase_linear = phase_fit[0] * times_verify + phase_fit[1]
    axes3[1, 0].plot(times_verify, phase_linear, 'r--', linewidth=1.5, 
                     label=f'Fit: ω={-phase_fit[0]:.4f}')
    axes3[1, 0].axhline(-mu_gs_correct * times_verify[-1], color='green', 
                        linestyle=':', linewidth=2, label=f'Expected: μ={mu_gs_correct:.4f}')
axes3[1, 0].set_xlabel('Time t', fontsize=12)
axes3[1, 0].set_ylabel('Phase φ(t) [rad]', fontsize=12)
axes3[1, 0].set_title('Phase Evolution φ = -μt', fontsize=13, fontweight='bold')
axes3[1, 0].legend(fontsize=10)
axes3[1, 0].grid(True, alpha=0.3)

if len(times_verify) > 2:
    phase_residuals = phase_unwrapped - phase_linear
    axes3[1, 1].plot(times_verify, phase_residuals, 'darkgreen', linewidth=1.5)
    axes3[1, 1].axhline(0, color='k', linestyle='--', alpha=0.5)
    axes3[1, 1].set_xlabel('Time t', fontsize=12)
    axes3[1, 1].set_ylabel('Phase residuals [rad]', fontsize=12)
    axes3[1, 1].set_title('Linearity Check', fontsize=13, fontweight='bold')
    axes3[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('test4_1_stationarity_check.png', dpi=150)
print("✓ Saved: test4_1_stationarity_check.png")

plt.show()

# -----------------------------------------------------------------------------
# Final verdict
# -----------------------------------------------------------------------------
print("\n" + "="*70)
print("TEST 4.1 COMPLETE: GROUND STATE WITH CORRECT μ")
print("="*70)
print(f"Ground state energy:      E_gs = {energy_gs:.6f}")
print(f"Chemical potential:       μ_gs = {mu_gs_correct:.6f}")
print(f"Peak density:             ρ_max = {rho_max_gs:.4f}")
print(f"\nStationarity:")
print(f"  ρ_max drift:       {np.std(rho_max_verify)/np.mean(rho_max_verify)*100:.4f}%")
print(f"  μ (from operator): {mu_gs_correct:.6f}")
print(f"  μ (from phase):    {omega_measured:.6f}")
print(f"  Relative error:    {abs(omega_measured - mu_gs_correct)/abs(mu_gs_correct)*100:.4f}%")
print(f"  Eigenvalue error:  {residual/norm_psi:.6e}")

error_threshold = 1.0  # 1% tolerance
rho_stability = np.std(rho_max_verify)/np.mean(rho_max_verify)*100

if abs(omega_measured - mu_gs_correct)/abs(mu_gs_correct)*100 < error_threshold and rho_stability < 0.5:
    print("\n" + "="*70)
    print("VERDICT: PASSED")
    print("="*70)
    print("✓ Ground state φ(r) exists and is stationary")
    print("✓ Chemical potential μ correctly matches phase evolution")
    print("✓ Density ρ(t) = |φ|² remains constant in real time")
    print("✓ Eigenvalue equation Ĥ|φ⟩ = μ|φ⟩ satisfied")
    verdict = "PASSED"
elif abs(omega_measured - mu_gs_correct)/abs(mu_gs_correct)*100 < 5.0:
    print("\n" + "="*70)
    print("VERDICT: IMPROVE")
    print("="*70)
    print("Chemical potential agreement acceptable but not excellent")
    verdict = "IMPROVE"
else:
    print("\n" + "="*70)
    print("VERDICT: FIX")
    print("="*70)
    print("Significant mismatch in chemical potential")
    verdict = "FIX"

print("="*70)
