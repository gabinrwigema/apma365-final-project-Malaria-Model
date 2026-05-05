import numpy as np
import matplotlib.pyplot as plt

def run_sim(D_m_val=0.15, t_end=40.0):
    
    L, J = 50.0, 300
    h = L / J
    x = np.linspace(0, L, J + 1)
    alpha, delta, epsilon = 3.0, 2.67, 0.67
    dh, k = 0.05, 5e-4
    n_steps = int(t_end / k)
    
    rh = dh * k / h**2
    rm_eff = (D_m_val / epsilon) * k / h**2

    x0, width = L/4, 2.0
    ih = 0.5 * np.exp(-((x - x0)**2) / (2 * width**2))
    im = 0.5 * np.exp(-((x - x0)**2) / (2 * width**2))
    
    for _ in range(n_steps):
        new_ih = np.copy(ih)
        new_im = np.copy(im)
        
        new_ih[1:-1] = ih[1:-1] + rh * (ih[:-2] - 2*ih[1:-1] + ih[2:]) + \
                       k * (alpha * im[1:-1] - ih[1:-1])
        new_im[1:-1] = im[1:-1] + rm_eff * (im[:-2] - 2*im[1:-1] + im[2:]) + \
                       (k / epsilon) * (delta * ih[1:-1] - im[1:-1])
        
        new_ih[0], new_ih[-1] = new_ih[1], new_ih[-2]
        new_im[0], new_im[-1] = new_im[1], new_im[-2]
        ih, im = new_ih, new_im
        
    return ih, x


low_data, x = run_sim(0.05)
high_data, _ = run_sim(0.50)


plt.figure(figsize=(9, 4.5))

dm_values = [0.01, 0.05, 0.20, 0.50]
colors = ['#2ecc71', '#3498db', '#f1c40f', '#e74c3c']
plt.figure(figsize=(10, 6))

for val, col in zip(dm_values, colors):
    data, x = run_sim(D_m_val=val)
    plt.plot(x, data, color=col, label=f'$D_m = {val}$')

plt.legend()
plt.title("Mapping the Outbreak of Malaria with Different Mosquito Diffusion Rates")
plt.savefig('test_plot.png')
plt.show()
