import numpy as np
import matplotlib.pyplot as plt

# Parameters
time = np.linspace(0, 10, 1000)  # Time in seconds (10 seconds duration)
peak_volume = 500  # Peak tidal volume in mL
frequency = 0.2  # Breathing rate in Hz (12 breaths per minute)
angular_frequency = 2 * np.pi * frequency  # Angular frequency

# Tidal volume function
tidal_volume = (peak_volume / 2) * (1 - np.cos(angular_frequency * time))

# Plot
plt.figure(figsize=(10, 5))
plt.plot(time, tidal_volume, label="Tidal Volume (Inhale/Exhale)", color='blue')
plt.title("Tidal Volume Simulation")
plt.xlabel("Time (s)")
plt.ylabel("Tidal Volume (mL)")
plt.ylim(0, 600)  # Ensure the plot starts at 0 and peaks correctly
plt.axhline(y=0, color="gray", linestyle="--", label="Baseline")
plt.legend()
plt.grid(True)
plt.show()