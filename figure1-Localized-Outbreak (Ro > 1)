import numpy as np
import matplotlib.pyplot as plt

L = 50.0           
J = 300            
h = L / J           
x = np.linspace(0, L, J + 1)

alpha = 3.0
delta = 2.67
epsilon = 0.67
dh = 0.05
dm = 0.15

k = 5e-4            
total_time = 40.0
n_steps = int(total_time / k)

rh = dh * k / h**2
rm_eff = (dm / epsilon) * k / h**2

def run_sim(t_end, record_times):
    
    x0 = L / 4
    width = 2.0
    ih = 0.5 * np.exp(-((x - x0)**2) / (2 * width**2))
    im = 0.5 * np.exp(-((x - x0)**2) / (2 * width**2))
    
    snaps = {}
    
    for n in range(n_steps + 1):
        t_curr = n * k
        
        for target in record_times:
            if abs(t_curr - target) < k / 2:
                snaps[target] = (ih.copy(), im.copy())    
        new_ih = np.copy(ih)
        new_im = np.copy(im)

        new_ih[1:-1] = ih[1:-1] + rh * (ih[:-2] - 2*ih[1:-1] + ih[2:]) + \
                       k * (alpha * im[1:-1] - ih[1:-1])
        new_im[1:-1] = im[1:-1] + rm_eff * (im[:-2] - 2*im[1:-1] + im[2:]) + \
                       (k / epsilon) * (delta * ih[1:-1] - im[1:-1])

        new_ih[0], new_ih[-1] = new_ih[1], new_ih[-2]
        new_im[0], new_im[-1] = new_im[1], new_im[-2]

        ih, im = new_ih, new_im

    return x, None, snaps, ih, im

snap_targets = {0.0, 10.0, 25.0, 40.0}
x, _, snaps, _, _ = run_sim(t_end=40.0, record_times=snap_targets)

fig, axes = plt.subplots(2, 2, figsize=(11, 7))
axes = axes.flatten()
col_h, col_m = '#1a6faf', '#c0392b'

for idx, tau in enumerate(sorted(snaps.keys())):
    ax = axes[idx]
    Ih_s, Im_s = snaps[tau]
    ax.plot(x, Ih_s, color=col_h, lw=2, label=r'$i_h$')
    ax.plot(x, Im_s, color=col_m, lw=2, ls='--', label=r'$i_m$')
    ax.set_title(fr'$\tau = {int(tau)}$', fontsize=13)
    ax.set_xlabel(r'$x$')
    ax.set_ylabel('Infected fraction')
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.1) # Set limit for visibility

fig.suptitle(r'Spatial profiles of $I_h$ and $I_m$; $\mathcal{R}_0\approx2.83$, $c=0$', fontsize=13)
plt.tight_layout()
plt.savefig('fig_snapshots.pdf', bbox_inches='tight')
plt.show()
