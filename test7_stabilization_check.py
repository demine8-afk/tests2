# test7_stabilization_check.py
# Outputs: test7_stabilization_result.png
#
# THEORETICAL SUMMARY:
# ====================
# PDE (Extended Model):
#   i ∂_t ψ = -(1/2m) ∇² ψ + [ V_sat(ρ) + gΦ + Kρ ] ψ
#   ∇²Φ = 4π G ρ
#
# Components:
#   V_sat(ρ) = -Gρ / (1+Sρ)  (Saturating attraction, micro-scale)
#   gΦ                       (Gravitational potential, macro-scale)
#   Kρ                       (Hard-core repulsion, short-range regularization)
#
# Objective:
#   Test the stability of a high-mass state (N=1000) under self-gravity
#   with the perturbative repulsion term K=0.2.
#
# Expected behavior:
#   1. Arrest of gravitational collapse (density remains finite).
#   2. Formation of a dynamical breathing state (stable oscillations).
#   3. Bounded RMS radius.

import os
import subprocess
import sys
import matplotlib.pyplot as plt

# --- GPU / CuPy Auto-Configuration ---
try:
    import cupy as np
    print("✅ GPU Detected. Using CuPy for simulation.")
except ImportError:
    print("⚠️ CuPy not found. Attempting auto-installation...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "cupy-cuda12x"])
    import cupy as np
    print("✅ Installation complete. Using GPU.")

# -----------------------------------------------------------------------------
# Parameters
# -----------------------------------------------------------------------------
Nx, Ny, Nz = 128, 128, 128
Lx, Ly, Lz = 40.0, 40.0, 40.0
dx = Lx / Nx

# Physics Constants (Unified Set)
m = 1.0
G = 1.0
S = 0.5
g = 1.0     # Gravity ON (Macro regime)
K = 0.2     # Repulsion parameter (Stabilizer)

dt = 0.005
T_max = 40.0
save_every = 100

target_N = 1000.0  # High mass regime

# -----------------------------------------------------------------------------
# Initialization
# -----------------------------------------------------------------------------
print(f"Initializing simulation: N={target_N}, K={K}...")

x = np.linspace(-Lx/2, Lx/2, Nx, endpoint=False)
X, Y, Z = np.meshgrid(x, x, x, indexing='ij')

# Initial Condition: Broad Gaussian Cloud
sigma = 4.0
psi = np.exp(-(X**2 + Y**2 + Z**2)/(2*sigma**2))

# Normalization
rho = np.abs(psi)**2
current_N = np.sum(rho) * dx**3
psi *= np.sqrt(target_N / current_N)

# Fourier Space Operators
kx = np.fft.fftfreq(Nx, dx) * 2*np.pi
KX, KY, KZ = np.meshgrid(kx, kx, kx, indexing='ij')
K2 = KX**2 + KY**2 + KZ**2
kinetic_half = np.exp(-1j * K2 * dt / 2)

# Storage lists (CPU)
t_list = []
rho_max_list = []
rms_list = []

print("Starting time evolution...")

# -----------------------------------------------------------------------------
# Time Evolution
# -----------------------------------------------------------------------------
steps = int(T_max/dt)

for step in range(steps):
    t = (step+1)*dt
    
    # 1. Kinetic Step (Half)
    psi = np.fft.ifftn( np.fft.fftn(psi) * kinetic_half )
    
    # 2. Potential Step (Full)
    rho = np.abs(psi)**2
    
    # 2a. Solve Poisson Equation (Gravity)
    rho_k = np.fft.fftn(rho)
    # Regularization 1e-20 prevents division by zero at k=0
    Phi_k = -4 * np.pi * G * rho_k / (K2 + 1e-20)
    Phi_k[0,0,0] = 0 
    Phi = np.real(np.fft.ifftn(Phi_k))
    
    # 2b. Calculate Total Potential
    V_sat = -G * rho / (1.0 + S*rho)
    V_grav = g * Phi
    V_rep = K * rho   # Hard-core term
    
    psi *= np.exp(-1j * (V_sat + V_grav + V_rep) * dt)
    
    # 3. Kinetic Step (Half)
    psi = np.fft.ifftn( np.fft.fftn(psi) * kinetic_half )
    
    # 4. Observables & Logging
    if step % save_every == 0:
        # Transfer data to CPU safely
        curr_rho_max = float(np.max(np.abs(psi)**2).get())
        
        # RMS Calculation
        mass = float(np.sum(rho) * dx**3)
        r2 = X**2 + Y**2 + Z**2
        mean_r2 = float(np.sum(r2 * rho) * dx**3) / mass
        curr_rms = np.sqrt(mean_r2)
        
        t_list.append(t)
        rho_max_list.append(curr_rho_max)
        rms_list.append(curr_rms)
        
        print(f"t = {t:6.2f} | ρ_max = {curr_rho_max:10.2f} | RMS = {curr_rms:6.3f}")

# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------
print("Simulation complete. Generating plots...")

# Safety: Ensure lists are standard Python lists/numpy arrays on CPU
import numpy as cpu_np
t_arr = cpu_np.array(t_list)
rho_arr = cpu_np.array(rho_max_list)
rms_arr = cpu_np.array(rms_list)

plt.figure(figsize=(14, 6))

# Left Plot: Density
plt.subplot(1, 2, 1)
plt.plot(t_arr, rho_arr, 'k-', linewidth=1.5)
plt.title(f"Peak Density Evolution (N={target_N}, K={K})")
plt.xlabel("Time")
plt.ylabel(r"$\rho_{\max}(t)$")
plt.grid(True, linestyle=':', alpha=0.6)

# Right Plot: Radius
plt.subplot(1, 2, 2)
plt.plot(t_arr, rms_arr, 'k-', linewidth=1.5)
plt.title("RMS Radius Evolution")
plt.xlabel("Time")
plt.ylabel(r"$R_{\text{rms}}(t)$")
plt.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.savefig("test7_stabilization_result.png", dpi=300)
plt.show()

print("✓ Saved: test7_stabilization_result.png")
