# test8_collision_robustness.py
# Outputs: test8_MERGER_result.png
#
# THEORETICAL SUMMARY:
# ====================
# Objective:
#   Assess the robustness of the unified model (K=0.2) under extreme conditions:
#   a high-velocity head-on collision of two massive stars (N=1000 each).
#
# Scenario:
#   Total Mass: N_tot = 2000 (2x stability threshold).
#   Velocity: v = +/- 1.5.
#   Physics: Full Extended Model (Saturating + Gravity + Hard-Core Repulsion).
#
# Expected Result:
#   The density should increase due to impact but remain bounded (no singularity).
#   The objects should not dissipate immediately.

import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fftn, ifftn, fftfreq
import time

# -----------------------------------------------------------------------------
# Parameters (Low-Res CPU version for quick check)
# -----------------------------------------------------------------------------
Nx = 64
L = 60.0
dx = L / Nx

# Unified Physics Constants
m = 1.0
G = 1.0
S = 0.5
g = 1.0
K = 0.2  # The stabilization parameter being tested

dt = 0.005
T_max = 20.0
save_every = 50

# Collision Setup
N_each = 1000.0
Distance = 16.0
Velocity = 1.5

# -----------------------------------------------------------------------------
# Initialization
# -----------------------------------------------------------------------------
print(f"Initializing Binary Collision: N_tot={2*N_each}, K={K}...")

x = np.linspace(-L/2, L/2, Nx, endpoint=False)
X, Y, Z = np.meshgrid(x, x, x, indexing='ij')

# Star 1 (Left, moving Right)
psi1 = np.exp(-((X + Distance/2)**2 + Y**2 + Z**2)/8.0).astype(np.complex128)
norm1 = np.sum(np.abs(psi1)**2) * dx**3
psi1 *= np.sqrt(N_each / norm1)
psi1 *= np.exp(1j * Velocity * X)

# Star 2 (Right, moving Left)
psi2 = np.exp(-((X - Distance/2)**2 + Y**2 + Z**2)/8.0).astype(np.complex128)
norm2 = np.sum(np.abs(psi2)**2) * dx**3
psi2 *= np.sqrt(N_each / norm2)
psi2 *= np.exp(-1j * Velocity * X)

# Total Field
psi = psi1 + psi2

# Fourier Operators
kx = fftfreq(Nx, dx) * 2*np.pi
KX, KY, KZ = np.meshgrid(kx, kx, kx, indexing='ij')
K2 = KX**2 + KY**2 + KZ**2
kinetic_half = np.exp(-1j * K2 * dt / 2)

t_list = []
rho_max_list = []

print("Starting simulation...")

# -----------------------------------------------------------------------------
# Time Evolution
# -----------------------------------------------------------------------------
steps = int(T_max/dt)
start_time = time.time()

for step in range(steps):
    t = (step+1)*dt
    
    # 1. Kinetic
    psi = ifftn( fftn(psi) * kinetic_half )
    
    # 2. Potential
    rho = np.abs(psi)**2
    
    # Gravity (Poisson Solve)
    rho_k = fftn(rho)
    Phi_k = -4 * np.pi * G * rho_k / (K2 + 1e-20)
    Phi_k[0,0,0] = 0 
    Phi = np.real(ifftn(Phi_k))
    
    # Total Potential
    V = -G * rho / (1.0 + S*rho) + g * Phi + K * rho
    
    psi *= np.exp(-1j * V * dt)
    
    # 3. Kinetic
    psi = ifftn( fftn(psi) * kinetic_half )
    
    # Logging
    if step % save_every == 0:
        curr_max = np.max(np.abs(psi)**2)
        t_list.append(t)
        rho_max_list.append(curr_max)
        
        # Minimal console output
        if step % (save_every*4) == 0:
            print(f"t = {t:5.2f} | rho_max = {curr_max:8.2f}")

elapsed = time.time() - start_time
print(f"Simulation finished in {elapsed:.1f}s")

# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------
mid_z = Nx // 2
rho_final = np.abs(psi)**2
slice_xy = rho_final[:, :, mid_z]

plt.figure(figsize=(14, 6))

# Density Evolution
plt.subplot(1, 2, 1)
plt.plot(t_list, rho_max_list, 'r-', linewidth=2)
plt.title("Collision Density Evolution")
plt.xlabel("Time")
plt.ylabel(r"$\rho_{\max}$")
plt.grid(True, alpha=0.3)

# Spatial Slice
plt.subplot(1, 2, 2)
im = plt.imshow(slice_xy, cmap='inferno', origin='lower', extent=[-L/2, L/2, -L/2, L/2])
plt.colorbar(im, label="Density")
plt.title(f"Spatial Profile at t={T_max}")
plt.xlabel("X")
plt.ylabel("Y")

plt.tight_layout()
plt.savefig("test8_MERGER_result.png", dpi=150)
plt.show()

print("✓ Saved: test8_MERGER_result.png")
