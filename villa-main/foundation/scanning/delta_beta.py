import numpy as np
import matplotlib.pyplot as plt
import xraylib

# Energies from 40 to 140 keV in 5 keV steps
E = np.arange(40, 145, 5).astype(float)

# Allocate arrays
delta = np.empty_like(E, dtype=float)
beta  = np.empty_like(E, dtype=float)
ratio = np.empty_like(E, dtype=float)

# Compute refractive indices
for i, e in enumerate(E):
    d = 1 - xraylib.Refractive_Index_Re("C", e, 0.8)
    b =     xraylib.Refractive_Index_Im("C", e, 0.8)
    delta[i] = d
    beta[i]  = b
    ratio[i] = d / b

# Determine y-axis tick range at 100-unit intervals
ymin = np.floor(np.min(ratio) / 100) * 100
ymax = np.ceil(np.max(ratio) / 100) * 100
# Plot
plt.figure(figsize=(8, 6))
plt.plot(E, ratio, marker='o', linestyle='-', linewidth=1.8, markersize=6)
plt.title(r'Ratio $\delta / \beta$ for Carbon', fontsize=14) #($\rho=0.8\,$g/cm$^3$)
plt.xlabel('Energy (keV)', fontsize=12)
plt.ylabel(r'$\delta / \beta$', fontsize=12)
plt.xlim(40, 140)
plt.ylim(ymin, ymax)
plt.xticks(np.arange(40, 145, 5))
plt.yticks(np.arange(ymin, ymax + 1, 100))
plt.grid(which='major', linestyle='--', linewidth=0.5, alpha=0.7)
plt.tight_layout()
plt.show()
