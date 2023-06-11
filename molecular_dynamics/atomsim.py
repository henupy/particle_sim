# Plot the velocity distribution to the other axis
ax = axes[1]
vels = [b.vels[n] for b in balls]
vel_mag = np.sqrt(np.sum(np.power(vels, 2), axis=1))
vel_max = np.max(vel_mag)
vel_avg = np.mean(vel_mag)
bins = np.linspace(0, vel_max * 1.1, 50)
ax.hist(vel_mag, bins=bins, density=True)
v = np.linspace(0, vel_max * 1.1, 1000)
a = 2 / (vel_avg * vel_avg)
fv = a * v * np.exp(-a * np.power(v, 2) / 2)
ax.plot(v, fv)
ax.set_xlabel('Speed [m/s]')
ax.set_ylabel('Number of particles')
ax.set_xlim(0, vel_max * 1.1)
ax.set_ylim(0, 0.025)
