import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# Number of samples
num_samples = 1000

# Equipment IDs
equipment_ids = [i for i in range(1, 11)]  # 10 different equipment IDs

# Generate random timestamps within the last 3 years
end_date = datetime.now()
start_date = end_date - timedelta(days=3*365)
timestamps = [start_date + timedelta(days=random.randint(0, 3*365)) for _ in range(num_samples)]

# Generate random data for each parameter
temperature = np.random.uniform(50, 150, num_samples)  # Temperature in degrees Fahrenheit
pressure = np.random.uniform(50, 100, num_samples)     # Pressure in PSI
vibration = np.random.uniform(0.5, 5.0, num_samples)   # Vibration in mm/s
maintenance_hours = np.random.uniform(0, 2000, num_samples)  # Maintenance hours

# Randomly assign equipment IDs
equipment_ids_random = [random.choice(equipment_ids) for _ in range(num_samples)]

# Generate random binary outcomes for failure (0 or 1)
failure = np.random.choice([0, 1], num_samples, p=[0.7, 0.3])

# Create a DataFrame
data = {
    'EquipmentID': equipment_ids_random,
    'Timestamp': timestamps,
    'Temperature': temperature,
    'Pressure': pressure,
    'Vibration': vibration,
    'MaintenanceHours': maintenance_hours,
    'Failure': failure
}
df = pd.DataFrame(data)

# Save the DataFrame to a CSV file in the current directory
df.to_csv('synthetic_equipment_data.csv', index=False)

print("Dataset created and saved as 'synthetic_equipment_data.csv'")
