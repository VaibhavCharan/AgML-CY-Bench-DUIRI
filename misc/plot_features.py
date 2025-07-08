import pandas as pd
import matplotlib.pyplot as plt
import os

csv_path = r'cybench\data\maize\NL\meteo_lai_bgr.csv'
df = pd.read_csv(csv_path, parse_dates=['date'])

# Plot LAI over time
plt.figure(figsize=(12, 6))
for (adm_id, year), group in df.groupby(['adm_id', 'year']):
    plt.plot(group['date'], group['lai'], label='LAI', color='green')
plt.xlabel('Date')
plt.ylabel('LAI')
plt.title('LAI Growth Over Time')
plt.tight_layout()
plt.savefig(os.path.join("model_results", 'lai_growth.png'))
plt.close()

# Plot BGR over time
plt.figure(figsize=(12, 6))
for (adm_id, year), group in df.groupby(['adm_id', 'year']):
    plt.plot(group['date'], group['bgr'], label='BGR', color='blue')
plt.xlabel('Date')
plt.ylabel('BGR')
plt.title('BGR Over Time')
plt.tight_layout()
plt.savefig(os.path.join("model_results", 'bgr.png'))
plt.close()