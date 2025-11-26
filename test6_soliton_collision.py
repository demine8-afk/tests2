# test6_soliton_collision.py
# Outputs: test6_collision_evolution.png, test6_observables.png, test6_trajectory.png
#
# THEORETICAL SUMMARY:
# ====================
# PDE (hypothesis):
#   i ∂_t ψ = -(1/2m) ∇² ψ - [G|ψ|²/(1 + S|ψ|²)] ψ
#
# We test the collision of two localized breathers with opposite velocities.
#
# Control model (integrable 1D NLS):
#   i ∂_t ψ = -(1/2m) ∂_x² ψ - g|ψ|² ψ
#   Literature (Zakharov-Shabat): elastic collisions, phase shift only
#
# For 2D cubic NLS:
#   Literature (Malomed): inelastic collisions, possible collapse for supercritical masses
#
# Expected behavior (saturating NLS):
#   Quasi-elastic collision: solitons emerge with preserved shape
#   Small energy loss to radiation
#   No collapse due to saturation
#
# Observables:
#   N(t) — total norm
#   E(t) — total energy
#   ρ_max(t) — peak density (spike during collision)
#   x_c(t) — center of mass
#   Amplitudes before/after collision

import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2, fftfreq
from scipy.signal import find_peaks

# -----------------------------------------------------------------------------
# Parameters
# -----------------------------------------------------------------------------
Nx, Ny = 384, 256  # elongated grid for trajectory observation
Lx, Ly = 60.0, 40.0
dx, dy = Lx / Nx, Ly / Ny
x = np.linspace(-Lx/2, Lx/2, Nx, endpoint=False)
y = np.linspace(-Ly/2, Ly/2, Ny, endpoint=False)
X, Y = np.meshgrid(x, y, indexing='ij')

m = 1.0
G = 1.0
S = 0.5
dt = 0.005
T_max = 50.0
N_steps = int(T_max / dt)
save_every = 20  # save every 0.1 time units

# Initial conditions: two Gaussian wavepackets
A0 = 2.5
sigma = 2.0
d = 8.0      # half-distance between centers
v1 = 0.4     # velocity (rightward)
v2 = -0.4    # velocity (leftward)

# Wavepacket 1: left, moving right
x1_init = -d
k1 = m * v1
psi1 = A0 * np.exp(-((X - x1_init)**2 + Y**2) / (2 * sigma**2)) * np.exp(1j * k1 * X)

# Wavepacket 2: right, moving left
x2_init = d
k2 = m * v2
psi2 = A0 * np.exp(-((X - x2_init)**2 + Y**2) / (2 * sigma**2)) * np.exp(1j * k2 * X)

# Total initial state (linear superposition)
psi = psi1 + psi2

# Fourier space operators
kx = fftfreq(Nx, d=dx) * 2 * np.pi
ky = fftfreq(Ny, d=dy) * 2 * np.pi
KX, KY = np.meshgrid(kx, ky, indexing='ij')
K2 = KX**2 + KY**2
kinetic_half = np.exp(-1j * K2 / (2 * m) * dt / 2)

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------
def compute_observables(psi_):
    rho = np.abs(psi_)**2
    norm = np.sum(rho) * dx * dy
    rho_max = np.max(rho)
    
    # Center of mass
    x_c = np.sum(X * rho) * dx * dy / norm if norm > 1e-12 else 0.0
    
    # Energy
    psi_k = fft2(psi_)
    kinetic = (1.0 / (2 * m)) * np.sum(K2 * np.abs(psi_k)**2) * dx * dy / (Nx * Ny)
    
    U_rho = -G * (rho / S - (1.0 / S**2) * np.log(1.0 + S * rho + 1e-30))
    potential = np.sum(U_rho) * dx * dy
    energy = kinetic + potential
    
    return norm, rho_max, x_c, energy

def detect_peaks_2d(rho, threshold_factor=0.3):
    """Detect local maxima in 2D density field"""
    threshold = threshold_factor * np.max(rho)
    from scipy.ndimage import maximum_filter
    local_max = maximum_filter(rho, size=10)
    peaks = (rho == local_max) & (rho > threshold)
    coords = np.argwhere(peaks)
    
    if len(coords) > 0:
        # Return x-coordinates of peaks
        x_coords = x[coords[:, 0]]
        amplitudes = rho[coords[:, 0], coords[:, 1]]
        # Sort by x-coordinate
        sorted_idx = np.argsort(x_coords)
        return x_coords[sorted_idx], amplitudes[sorted_idx]
    else:
        return np.array([]), np.array([])

# Storage
times = []
norms = []
rho_maxs = []
x_centers = []
energies = []
peak_positions = []  # list of arrays
peak_amplitudes = []

# Snapshots
snapshots = []
snapshot_times = []

norm0, rho0, x_c0, E0 = compute_observables(psi)

print("="*70)
print("TEST 6: SOLITON COLLISION IN 2+1D")
print("="*70)
print(f"Initial configuration:")
print(f"  Soliton 1: x = {x1_init:.1f}, v = {v1:.2f}, A = {A0:.2f}")
print(f"  Soliton 2: x = {x2_init:.1f}, v = {v2:.2f}, A = {A0:.2f}")
print(f"  Separation: {2*d:.1f}")
print(f"  Total norm: N = {norm0:.4f}")
print(f"  Total energy: E = {E0:.6f}")
print(f"\nExpected collision time: t ≈ {d / abs(v1):.1f}")
print("="*70)

# -----------------------------------------------------------------------------
# Time evolution
# -----------------------------------------------------------------------------
for step in range(N_steps):
    t = step * dt
    
    # Strang splitting
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
        norm, rho_max, x_c, energy = compute_observables(psi)
        times.append(t)
        norms.append(norm / norm0)
        rho_maxs.append(rho_max)
        x_centers.append(x_c)
        energies.append(energy / E0)
        
        # Detect individual peaks
        rho_current = np.abs(psi)**2
        x_peaks, amps_peaks = detect_peaks_2d(rho_current, threshold_factor=0.3)
        peak_positions.append(x_peaks)
        peak_amplitudes.append(amps_peaks)
        
        # Store snapshots at key moments
        if t in [0, 10, 15, 20, 25, 30, 40, 50] or abs(t - d/abs(v1)) < 0.5:
            snapshots.append(rho_current.copy())
            snapshot_times.append(t)
        
        if step % (save_every * 25) == 0:
            print(f"t = {t:6.2f} | ρ_max = {rho_max:8.4f} | x_c = {x_c:+7.3f} | "
                  f"N/N₀ = {norm/norm0:.6f} | E/E₀ = {energy/E0:.6f}")

times = np.array(times)
norms = np.array(norms)
rho_maxs = np.array(rho_maxs)
x_centers = np.array(x_centers)
energies = np.array(energies)

# Final state
norm_f, rho_f, x_c_f, E_f = compute_observables(psi)

# -----------------------------------------------------------------------------
# Analysis
# -----------------------------------------------------------------------------
print("\n" + "="*70)
print("COLLISION ANALYSIS")
print("="*70)

# Conservation
norm_drift = abs(norm_f / norm0 - 1.0) * 100
energy_drift = abs(E_f / E0 - 1.0) * 100

print(f"Conservation:")
print(f"  Norm drift:   {norm_drift:.4f}%")
print(f"  Energy drift: {energy_drift:.4f}%")

# Peak density analysis
rho_max_initial = rho0
rho_max_collision = np.max(rho_maxs)
t_collision_idx = np.argmax(rho_maxs)
t_collision = times[t_collision_idx]

print(f"\nDensity peaks:")
print(f"  Initial ρ_max:     {rho_max_initial:.4f}")
print(f"  Collision ρ_max:   {rho_max_collision:.4f} (at t = {t_collision:.2f})")
print(f"  Final ρ_max:       {rho_f:.4f}")
print(f"  Enhancement factor: {rho_max_collision / rho_max_initial:.2f}x")

# Check if two peaks emerge after collision
final_rho = np.abs(psi)**2
x_peaks_final, amps_final = detect_peaks_2d(final_rho, threshold_factor=0.3)

if len(x_peaks_final) >= 2:
    print(f"\n✓ Two localized objects detected after collision:")
    print(f"  Peak 1: x = {x_peaks_final[0]:+7.3f}, ρ = {amps_final[0]:.4f}")
    print(f"  Peak 2: x = {x_peaks_final[-1]:+7.3f}, ρ = {amps_final[-1]:.4f}")
    
    # Compare with initial amplitudes
    amp_loss_1 = (amps_final[0] - rho_max_initial) / rho_max_initial * 100
    amp_loss_2 = (amps_final[-1] - rho_max_initial) / rho_max_initial * 100
    print(f"  Amplitude change: {amp_loss_1:.2f}%, {amp_loss_2:.2f}%")
    
    collision_type = "elastic" if abs(amp_loss_1) < 10 and abs(amp_loss_2) < 10 else "inelastic"
elif len(x_peaks_final) == 1:
    print(f"\n⚠ Only one peak detected (possible merger)")
    collision_type = "merger"
else:
    print(f"\n⚠ No clear peaks (possible dispersion)")
    collision_type = "dispersive"

print("="*70)

# -----------------------------------------------------------------------------
# Plotting: Observables
# -----------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].plot(times, rho_maxs, 'b-', linewidth=2)
axes[0, 0].axvline(t_collision, color='red', linestyle='--', linewidth=1.5, 
                   label=f'Collision (t={t_collision:.1f})')
axes[0, 0].set_xlabel('Time', fontsize=12)
axes[0, 0].set_ylabel(r'$\rho_{\max}(t)$', fontsize=12)
axes[0, 0].set_title('Peak Density — Collision Spike', fontsize=13, fontweight='bold')
axes[0, 0].legend(fontsize=10)
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(times, x_centers, 'g-', linewidth=2)
axes[0, 1].axhline(0, color='k', linestyle=':', alpha=0.5)
axes[0, 1].axvline(t_collision, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
axes[0, 1].set_xlabel('Time', fontsize=12)
axes[0, 1].set_ylabel(r'$x_{\mathrm{center}}(t)$', fontsize=12)
axes[0, 1].set_title('Center of Mass', fontsize=13, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].plot(times, norms, 'r-', linewidth=2)
axes[1, 0].axhline(1.0, color='k', linestyle='--', alpha=0.5)
axes[1, 0].set_xlabel('Time', fontsize=12)
axes[1, 0].set_ylabel(r'$N(t)/N_0$', fontsize=12)
axes[1, 0].set_title('Norm Conservation', fontsize=13, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].plot(times, energies, 'm-', linewidth=2)
axes[1, 1].axhline(1.0, color='k', linestyle='--', alpha=0.5)
axes[1, 1].set_xlabel('Time', fontsize=12)
axes[1, 1].set_ylabel(r'$E(t)/E_0$', fontsize=12)
axes[1, 1].set_title('Energy Conservation', fontsize=13, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('test6_observables.png', dpi=150)
print("\n✓ Saved: test6_observables.png")

# -----------------------------------------------------------------------------
# Plotting: Trajectories
# -----------------------------------------------------------------------------
fig2, ax = plt.subplots(1, 1, figsize=(14, 6))

# Plot individual peak positions over time
for i, t in enumerate(times):
    if len(peak_positions[i]) > 0:
        ax.scatter([t]*len(peak_positions[i]), peak_positions[i], 
                   c=peak_amplitudes[i], cmap='hot', s=30, vmin=0, vmax=rho_max_collision,
                   edgecolors='k', linewidths=0.3)

# Expected trajectories (dashed lines)
t_array = np.linspace(0, T_max, 100)
x1_expected = x1_init + v1 * t_array
x2_expected = x2_init + v2 * t_array
ax.plot(t_array, x1_expected, 'b--', linewidth=1.5, alpha=0.6, label='Expected (v₁)')
ax.plot(t_array, x2_expected, 'r--', linewidth=1.5, alpha=0.6, label='Expected (v₂)')

ax.axvline(t_collision, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Collision')
ax.set_xlabel('Time', fontsize=13)
ax.set_ylabel('x-coordinate of peaks', fontsize=13)
ax.set_title('Soliton Trajectories — Phase Shift Analysis', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('test6_trajectory.png', dpi=150)
print("✓ Saved: test6_trajectory.png")

# -----------------------------------------------------------------------------
# Plotting: Density evolution snapshots
# -----------------------------------------------------------------------------
n_snapshots = len(snapshots)
n_cols = 4
n_rows = int(np.ceil(n_snapshots / n_cols))

fig3, axes3 = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows))
axes3 = axes3.flatten() if n_snapshots > 1 else [axes3]

for i in range(n_snapshots):
    im = axes3[i].contourf(X, Y, snapshots[i], levels=40, cmap='hot')
    axes3[i].set_xlabel('x', fontsize=10)
    axes3[i].set_ylabel('y', fontsize=10)
    axes3[i].set_title(f't = {snapshot_times[i]:.1f}', fontsize=11, fontweight='bold')
    axes3[i].set_aspect('equal')
    plt.colorbar(im, ax=axes3[i], label=r'$\rho$')

for i in range(n_snapshots, len(axes3)):
    axes3[i].axis('off')

plt.tight_layout()
plt.savefig('test6_collision_evolution.png', dpi=150)
print("✓ Saved: test6_collision_evolution.png")

plt.show()

# -----------------------------------------------------------------------------
# Verdict
# -----------------------------------------------------------------------------
print("\n" + "="*70)
print("TEST 6 COMPLETE: SOLITON COLLISION")
print("="*70)
print(f"Collision time: t ≈ {t_collision:.2f}")
print(f"Peak density enhancement: {rho_max_collision / rho_max_initial:.2f}x")
print(f"Collision type: {collision_type.upper()}")
print(f"\nConservation:")
print(f"  Norm:   {norm_drift:.4e}%")
print(f"  Energy: {energy_drift:.4e}%")

if len(x_peaks_final) >= 2:
    print(f"\n✓ Two solitons emerge after collision")
    if abs(amp_loss_1) < 15 and abs(amp_loss_2) < 15:
        print(f"✓ Amplitudes preserved within 15%")
        elasticity = "quasi-elastic"
    else:
        print(f"⚠ Significant amplitude change")
        elasticity = "inelastic"
else:
    print(f"\n⚠ Collision outcome unclear")
    elasticity = "unknown"

print("="*70)

# Final verdict
threshold_norm = 0.1
threshold_energy = 1.0
threshold_amplitude = 20.0

if (norm_drift < threshold_norm and 
    energy_drift < threshold_energy and 
    len(x_peaks_final) >= 2 and
    max(abs(amp_loss_1), abs(amp_loss_2)) < threshold_amplitude):
    print("VERDICT: PASSED")
    print("="*70)
    print("✓ Collision is stable and quasi-elastic")
    print("✓ Two localized objects emerge")
    print("✓ Norm and energy conserved")
    print("✓ No collapse or uncontrolled dispersion")
    print("✓ Saturating nonlinearity stabilizes collision dynamics")
elif len(x_peaks_final) >= 2:
    print("VERDICT: IMPROVE")
    print("="*70)
    print("Collision occurs but quantitative analysis needs refinement")
    print("Consider longer evolution time or parameter adjustment")
else:
    print("VERDICT: FIX")
    print("="*70)
    print("Collision outcome unclear — check numerical parameters")

print("="*70)
