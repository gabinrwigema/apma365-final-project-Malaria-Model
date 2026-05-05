import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def run_3d_sim(D_m_val=0.15, t_end=20.0):
    # 1. Setup
    L, J = 50.0, 100
    h = L / J
    x = np.linspace(0, L, J + 1)
    dh, k = 0.05, 5e-4
    n_steps = int(t_end / k)
    
    skip = 100 
    recorded_steps = n_steps // skip
    U_history = np.zeros((recorded_steps + 1, J + 1))
    time_axis = np.linspace(0, t_end, recorded_steps + 1)

    alpha, delta, epsilon = 3.0, 2.67, 0.67
    rh = dh * k / h**2
    rm_eff = (D_m_val / epsilon) * k / h**2
    ih = 0.5 * np.exp(-((x - L/4)**2) / (2 * 2.0**2))
    im = ih.copy()
    
    rec_idx = 0
    for n in range(n_steps + 1):
        
        if n % skip == 0 and rec_idx <= recorded_steps:
            U_history[rec_idx, :] = ih
            rec_idx += 1
            
        new_ih = np.copy(ih)
        new_im = np.copy(im)
        
        new_ih[1:-1] = ih[1:-1] + rh * (ih[:-2] - 2*ih[1:-1] + ih[2:]) + k*(alpha*im[1:-1] - ih[1:-1])
        new_im[1:-1] = im[1:-1] + rm_eff * (im[:-2] - 2*im[1:-1] + im[2:]) + (k/epsilon)*(delta*ih[1:-1] - im[1:-1])
       
        new_ih[0], new_ih[-1] = new_ih[1], new_ih[-2]
        new_im[0], new_im[-1] = new_im[1], new_im[-2]
        ih, im = new_ih, new_im

    return x, time_axis, U_history

x_vals, t_vals, Z = run_3d_sim()
X, T = np.meshgrid(x_vals, t_vals)

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(X, T, Z, cmap=cm.viridis, edgecolor='none', alpha=0.9)
cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)

# 2. Label it so we know what the numbers mean
cbar.set_label('Infected Fraction ($i_h$)', rotation=270, labelpad=15)

ax.set_xlabel('Space (x)')
ax.set_ylabel('Time (t)')
ax.set_zlabel('Infected Human Fraction $I_h$')
ax.set_title('Evolution of Malaria Outbreak in Human Fraction, I_h, in Time and Space')
plt.show()
