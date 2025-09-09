import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.ticker as ticker
from matplotlib.ticker import LogLocator
import seaborn as sns

# Professional scientific style
sns.set_context('paper', font_scale=1.3)
sns.set_style('white')

# Parameters
MinPixum, MaxPixum = 1, 15  # in micrometers after scaling
MinKeV, MaxKeV = 40, 150

# Detector pixel size mesh (m), then convert to micrometers
DELTA_m = np.logspace(np.log10(MinPixum * 1e-6), np.log10(MaxPixum * 1e-6), num=1000)
DELTA_um = DELTA_m * 1e6  # convert to micrometers

# Photon energy mesh in keV, for axis and computation
EV_keV = np.logspace(np.log10(MinKeV), np.log10(MaxKeV), num=1000)
EV_eV = EV_keV * 1e3

# Create meshgrids
DS_um, EV_keV_mesh = np.meshgrid(DELTA_um, EV_keV)
# For physics calculation convert back to eV and m
DS_m_mesh = DS_um * 1e-6
EV_eV_mesh = EV_keV_mesh * 1e3

# Physics: compute Lp
h = 6.62607004e-34
c = 299792458
eJ = 1.602176634e-19
LAMBDA = (h * c) / (EV_eV_mesh * eJ)
Lp = ((2*DS_m_mesh) ** 2) / LAMBDA

# Contour levels and labels (original)
levels = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 0.4, 1, 1.2, 2.2, 2.8, 5, 10, 15, 20, 30, 100, 300]
labels = ['1 mm', '3 mm', '1 cm', '3 cm', '10 cm', '30 cm', '40 cm',
          '1 m', '1.2 m', '2.2 m' , '2.8 m', '5 m', '10 m', '15 m', '20 m', '30 m', '100 m', '300 m']
fmt = {lev: lab for lev, lab in zip(levels, labels)}

# Plot
fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xscale('log')
ax.set_yscale('log')

# Draw contours
cs = ax.contour(DS_um, EV_keV_mesh, Lp,
                levels=levels, cmap='viridis', norm=LogNorm(), linewidths=1.5)
ax.clabel(cs, fmt=fmt, inline=True, fontsize=10, colors='black', inline_spacing=3)

# Annotate that these contours mark the near-field (Fraunhofer) limit
ax.text(0.05, 0.95,
        'Contour lines: near-field limit',
        transform=ax.transAxes,
        va='top', ha='left',
        fontsize=10,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

# X-axis ticks: more ticks in micrometers
ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1,2.2,2.7,3,3.7,4.3,5,6,7,8,9,10,15), numticks=13))
ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(1,11), numticks=20))
ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%g'))
ax.xaxis.set_minor_formatter(ticker.NullFormatter())

# Y-axis ticks in keV
y_major = [40, 53, 66, 71, 80, 88, 100, 110, 120, 133, 140]
y_minor = [50, 63, 76, 81, 90, 98, 110, 120, 130, 143, 150]
ax.set_yticks(y_major)
ax.set_yticks(y_minor, minor=True)
ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
ax.yaxis.set_minor_formatter(ticker.NullFormatter())

# Grids
ax.grid(which='major', linestyle='--', linewidth=0.6, alpha=0.8)
ax.grid(which='minor', linestyle=':', linewidth=0.4, alpha=0.6)

# Labels
ax.set_xlabel('Detector pixel size [µm]', fontsize=12)
ax.set_ylabel('Photon Energy [keV]', fontsize=12)
ax.tick_params(axis='both', which='major', labelsize=10)

# Finalize
plt.tight_layout()
fig.savefig('plot_keV_um.png', dpi=300)
plt.show()
