# test2_mass_frequency.py
#
# MODEL: Unified NLSE (Saturation + Repulsion)
# PURPOSE: Determine the relationship between Breather Norm (Mass) and Oscillation Frequency.
#
# We scan initial amplitudes A0 -> different Norms N.
# We measure the period T of density oscillations.
# Existence of T(N) confirms stable breather families exist.

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
K = 0.2          # <--- UNIVERSAL REPULSION
dt = 0.005
T_max = 60.0
N_steps = int(T_max / dt)
save_every = 20

# Fourier Operators
kx = fftfreq(Nx, d=dx) * 2 * np.pi
ky = fftfreq(Ny, d=dy) * 2 * np.pi
KX, KY = np.meshgrid(kx, ky, indexing='ij')
K2 = KX**2 + KY**2
kinetic_half = np.exp(-1j * K2 / (2 * m) * dt / 2)

# Amplitudes to scan (creating different masses)
amplitudes = np.array([1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0])
sigma = 2.0

# Data storage
results_N = []
results_T = []
results_omega = []

# -----------------------------------------------------------------------------
# 2. Simulation Functions
# -----------------------------------------------------------------------------
def get_norm_and_max(psi_):
    rho = np.abs(psi_)**2
    norm = np.sum(rho) * dx * dy
    rho_max = np.max(rho)
    return norm, rho_max

def run_simulation(A0):
    # Init
    psi = A0 * np.exp(-(X**2 + Y**2) / (2 * sigma**2))
    
    times_local = []
    rho_maxs_local = []
    
    # Evolution
    for step in range(N_steps):
        t = step * dt
        
        # Split-step
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
            _, r_max = get_norm_and_max(psi)
            times_local.append(t)
            rho_maxs_local.append(r_max)
            
    # Final norm check
    norm_final, _ = get_norm_and_max(psi)
    
    # Analysis
    times_local = np.array(times_local)
    rho_maxs_local = np.array(rho_maxs_local)
    
    # Ignore transient (first 10%)
    start = len(times_local) // 10
    peaks_idx, _ = find_peaks(rho_maxs_local[start:], distance=25)
    peaks_idx += start
    
    if len(peaks_idx) > 1:
        periods = np.diff(times_local[peaks_idx])
        avg_T = np.mean(periods)
    else:
        avg_T = np.nan
        
    return norm_final, avg_T, times_local, rho_maxs_local

# -----------------------------------------------------------------------------
# 3. Main Scan Loop
# -----------------------------------------------------------------------------
print(f"STARTING TEST 2 (Mass-Frequency Scan). K={K}")
print("="*60)

all_series = []

for i, A0 in enumerate(amplitudes):
    print(f"Running A0={A0:.1f}... ", end="", flush=True)
    norm, period, ts, rs = run_simulation(A0)
    
    results_N.append(norm)
    results_T.append(period)
    
    if not np.isnan(period):
        omega = 2 * np.pi / period
        results_omega.append(omega)
        print(f"N={norm:.1f}, T={period:.3f}, w={omega:.3f}")
    else:
        results_omega.append(np.nan)
        print(f"N={norm:.1f}, T=NaN")
    
    all_series.append((ts, rs, A0, norm, period))

results_N = np.array(results_N)
results_T = np.array(results_T)
results_omega = np.array(results_omega)

# Filter valid
valid_mask = ~np.isnan(results_T)
N_clean = results_N[valid_mask]
w_clean = results_omega[valid_mask]
T_clean = results_T[valid_mask]

# -----------------------------------------------------------------------------
# 4. Visualization
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

# Frequency vs Mass
ax[0].plot(N_clean, w_clean, 'o-', color='navy', lw=2)
ax[0].set_xlabel('Norm N (Mass)')
ax[0].set_ylabel('Frequency $\omega$')
ax[0].set_title(f'Frequency vs Mass (K={K})')
ax[0].grid(True, alpha=0.3)

# Period vs Mass
ax[1].plot(N_clean, T_clean, 's-', color='darkred', lw=2)
ax[1].set_xlabel('Norm N (Mass)')
ax[1].set_ylabel('Period T')
ax[1].set_title(f'Period vs Mass (K={K})')
ax[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('test2_frequency_vs_mass.png', dpi=150)
print("Saved: test2_frequency_vs_mass.png")

# Time series grid
fig2, axes2 = plt.subplots(2, 4, figsize=(16, 8))
axes2 = axes2.flatten()
for idx, (ts, rs, A0, n_val, t_val) in enumerate(all_series):
    if idx < 8:
        axes2[idx].plot(ts, rs, 'b-', alpha=0.7)
        axes2[idx].set_title(f'A0={A0}, N={n_val:.0f}\nT={t_val:.2f}')
        axes2[idx].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('test2_periods_summary.png', dpi=150)
print("Saved: test2_periods_summary.png")

# -----------------------------------------------------------------------------
# 5. SELF-CHECK
# -----------------------------------------------------------------------------
print("\n" + "="*60)
print("SELF-CHECK")
print("="*60)
if len(N_clean) >= 4:
    print(f"[ SUCCESS ] - Obtained {len(N_clean)} valid data points.")
    print(f"            Min Norm: {np.min(N_clean):.1f}, Max Norm: {np.max(N_clean):.1f}")
    print(f"            Frequency range: {np.min(w_clean):.3f} - {np.max(w_clean):.3f}")
else:
    print(f"[ FAILURE ] - Too few valid points ({len(N_clean)}). Check parameters.")
print("="*60)
