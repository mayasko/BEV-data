#This script is prepared for processing the data in paper:

#Unveiling Energy Dynamics of Battery Electric Vehicle Using High-Resolution Data

#By: M. Yasko, A. Moussa, F. Tian, H. Kazmi, J. Driesen, and W. Martinez 
#From: KU Leuven/EnergyVille Thor Park 8310, Genk, Belgium.
#Last update: 2025-06-12

#Email: mohamed.yasko@kuleuven.be

#Notes: Figure 1 and 2 are not included in this script.
'''
####################################################################################
####################################################################################

#FIGURE 3: slow charging dataset distribution
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np

# === Folder path ===
folder_path = r'D:\Research vision\Papers\Paper 1\Submission\Dataset_repo\charging sessions\slow charging sessions'

# === Init containers ===
bev1_power, bev2_power = [], []
bev1_soc, bev2_soc = [], []
bev1_temp, bev2_temp = [], []
bev1_front_temp, bev2_front_temp = [], []
bev1_energy, bev2_energy = [], []

# === Loop through files ===
for filename in sorted(os.listdir(folder_path)):
    if filename.endswith('.csv'):
        file_path = os.path.join(folder_path, filename)
        df = pd.read_csv(file_path)

        required = ['ChargeLinePower264', 'Timestamp', 'SOCave292',
                    'BMSmaxPackTemperature', 'VCFRONT_tempAmbient', 'TotalChargeKWh3D2']
        if not all(col in df.columns for col in required):
            continue

        df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
        df = df.sort_values('Timestamp')

        # Total energy is the difference in TotalChargeKWh3D2
        df = df.dropna(subset=['TotalChargeKWh3D2'])
        if df.empty:
            continue
        total_energy = df['TotalChargeKWh3D2'].iloc[-1] - df['TotalChargeKWh3D2'].iloc[0]

        if 'BEV1' in filename.upper():
            bev1_power.extend(df['ChargeLinePower264'].dropna())
            bev1_soc.extend(df['SOCave292'].dropna())
            bev1_temp.extend(df['BMSmaxPackTemperature'].dropna())
            bev1_front_temp.extend(df['VCFRONT_tempAmbient'].dropna())
            bev1_energy.append(total_energy)
        elif 'BEV2' in filename.upper():
            bev2_power.extend(df['ChargeLinePower264'].dropna())
            bev2_soc.extend(df['SOCave292'].dropna())
            bev2_temp.extend(df['BMSmaxPackTemperature'].dropna())
            bev2_front_temp.extend(df['VCFRONT_tempAmbient'].dropna())
            bev2_energy.append(total_energy)

# === Temperature bins ===
all_temps = bev1_temp + bev2_temp + bev1_front_temp + bev2_front_temp
temp_min = min(all_temps, default=0)
temp_max = max(all_temps, default=0)
temp_bins = np.linspace(temp_min, temp_max, 30)

# === Formatter ===
def format_thousands(x, _):
    return f'{int(x/1000)}k' if x >= 1000 else int(x)

formatter = FuncFormatter(format_thousands)

# === Plot style ===
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 8,
    'axes.linewidth': 0.6,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'figure.dpi': 600
})

fig, axes = plt.subplots(2, 2, figsize=(7, 5))  
blue = '#1f77b4'
orange = '#ff7f0e'
gray1 = '#666666'
gray2 = '#aaaaaa'

# 1. Power
axes[0, 0].hist(bev1_power, bins=30, alpha=0.8, color=blue, edgecolor='black', label='BEV1')
axes[0, 0].hist(bev2_power, bins=30, alpha=0.6, color=orange, edgecolor='black', label='BEV2')
axes[0, 0].set_xlabel('Power (kW)')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].yaxis.set_major_formatter(formatter)
axes[0, 0].legend(frameon=False, loc='upper center')
axes[0, 0].grid(True, linestyle='--', alpha=0.3)

# 2. Total Charge (Energy)
axes[0, 1].hist(bev1_energy, bins=30, alpha=0.8, color=blue, edgecolor='black', label='BEV1')
axes[0, 1].hist(bev2_energy, bins=30, alpha=0.6, color=orange, edgecolor='black', label='BEV2')
axes[0, 1].set_xlabel('Energy (kWh)')
axes[0, 1].legend(frameon=False, loc='upper right')
axes[0, 1].grid(True, linestyle='--', alpha=0.3)

# 3. SoC
axes[1, 0].hist(bev1_soc, bins=30, alpha=0.8, color=blue, edgecolor='black', label='BEV1')
axes[1, 0].hist(bev2_soc, bins=30, alpha=0.6, color=orange, edgecolor='black', label='BEV2')
axes[1, 0].set_xlabel('SoC (%)')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].yaxis.set_major_formatter(formatter)
axes[1, 0].legend(frameon=False, loc='upper left')
axes[1, 0].grid(True, linestyle='--', alpha=0.3)

# 4. Temperature
axes[1, 1].hist(bev1_temp, bins=temp_bins, alpha=0.8, color=blue, edgecolor='black', label='BEV1 Batt')
axes[1, 1].hist(bev2_temp, bins=temp_bins, alpha=0.6, color=orange, edgecolor='black', label='BEV2 Batt')
axes[1, 1].hist(bev1_front_temp, bins=temp_bins, alpha=0.5, color=gray1, edgecolor='black', label='BEV1 Amb')
axes[1, 1].hist(bev2_front_temp, bins=temp_bins, alpha=0.4, color=gray2, edgecolor='black', label='BEV2 Amb')
axes[1, 1].set_xlabel('Temperature (°C)')
axes[1, 1].yaxis.set_major_formatter(formatter)
axes[1, 1].legend(frameon=False, loc='upper left')
axes[1, 1].grid(True, linestyle='--', alpha=0.3)

plt.tight_layout(pad=1.2)
plt.savefig('fig3.png', dpi=600, bbox_inches='tight', transparent=True)
plt.show()

####################################################################################
####################################################################################
'''


# Figure 4 — Energy breakdown of charging sessions
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# === Folder path ===
#folder_path = r'D:\Research vision\Papers\Paper 1\datasets\BEV_data\AllAC'

folder_path = r'D:\Research vision\Papers\Paper 1\Paper 1\datasets\BEV_data\AllAC'
bev1_data, bev2_data = [], []
bev1_labels, bev2_labels = [], []

# === Session processor ===
def process_session(file_path):
    df = pd.read_csv(file_path)
    required = ['ChargeLinePower264', 'Timestamp', 'TotalChargeKWh3D2', 'TotalDischargeKWh3D2']
    if not all(col in df.columns for col in required):
        return None
    df = df.dropna(subset=required)
    if df.empty:
        return None

    duration_hr = len(df) / 3600.0
    avg_power_kw = df['ChargeLinePower264'].mean()
    grid_energy = avg_power_kw * duration_hr if pd.notna(avg_power_kw) else 0.0

    total_charge = df['TotalChargeKWh3D2'].iloc[-1] - df['TotalChargeKWh3D2'].iloc[0]
    total_discharge = df['TotalDischargeKWh3D2'].iloc[-1] - df['TotalDischargeKWh3D2'].iloc[0]

    battery_energy = max(total_charge - total_discharge, 0)
    aux_loss_kwh = total_discharge
    charger_loss = grid_energy - total_charge

    return battery_energy, charger_loss, aux_loss_kwh * 1000  # aux in Wh

# === Load all sessions ===
for filename in sorted(os.listdir(folder_path)):
    if filename.endswith('.csv'):
        file_path = os.path.join(folder_path, filename)
        result = process_session(file_path)
        if result:
            if 'BEV1' in filename.upper():
                bev1_data.append(result)
                bev1_labels.append(filename)
            elif 'BEV2' in filename.upper():
                bev2_data.append(result)
                bev2_labels.append(filename)

# === Sort sessions by total energy ===
def compute_total_energy_and_sort(data, labels):
    data_with_energy = [(b, l, a, b + l + a / 1000, lbl) for (b, l, a), lbl in zip(data, labels)]
    data_sorted = sorted(data_with_energy, key=lambda x: x[3])
    return [(b, l, a) for b, l, a, _, _ in data_sorted], [lbl for _, _, _, _, lbl in data_sorted]

bev1_data_sorted, bev1_labels_sorted = compute_total_energy_and_sort(bev1_data, bev1_labels)
bev2_data_sorted, bev2_labels_sorted = compute_total_energy_and_sort(bev2_data, bev2_labels)

# === Extract components ===
def extract(data):
    return list(zip(*data)) if data else ([], [], [])

bev1_b, bev1_l, bev1_a = extract(bev1_data_sorted)
bev2_b, bev2_l, bev2_a = extract(bev2_data_sorted)

# === Legend text formatter ===
def stat_txt(name, data, scale=1.0, unit="kWh"):
    data_scaled = [x * scale for x in data if x * scale >= 0.01]
    if not data_scaled:
        return f"{name} (n/a)"
    return f"{name} min={min(data_scaled):.1f}, max={max(data_scaled):.1f}, avg={sum(data_scaled)/len(data_scaled):.1f} {unit}"

# === Plot style ===
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 8,
    'axes.linewidth': 0.6,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'legend.frameon': False,
    'figure.dpi': 600
})

fig, axes = plt.subplots(2, 2, figsize=(7, 5))  # we'll use only axes[0, 0] and [0, 1]
colors1 = {'batt': '#1976d2', 'loss': '#888888', 'aux': '#4caf50'}
colors2 = {'batt': '#f57c00', 'loss': '#888888', 'aux': '#4caf50'}

# === Bar plot function ===
def plot_stacked(ax, b, l, a, colorset, labels):
    idx = list(range(len(b)))
    aux_kwh = [x / 1000 for x in a]
    bottom2 = b
    bottom3 = [b[i] + l[i] for i in range(len(b))]

    ax.bar(idx, b, color=colorset['batt'], edgecolor='black', label='Battery')
    ax.bar(idx, l, bottom=bottom2, color=colorset['loss'], edgecolor='black', label='Charger Loss')
    ax.bar(idx, aux_kwh, bottom=bottom3, color=colorset['aux'], edgecolor='black', label='Aux Loss')

    # Annotate peak charger loss
    if l:
        max_l_idx = l.index(max(l))
        ax.annotate(f"Peak charger loss: {l[max_l_idx]:.1f} kWh",
                    xy=(max_l_idx, b[max_l_idx] + l[max_l_idx] / 2),
                    xytext=(max_l_idx, b[max_l_idx] + l[max_l_idx] + 5),
                    ha='right', fontsize=7,
                    arrowprops=dict(arrowstyle="->", lw=0.5),
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))

    # Annotate peak aux loss
    if aux_kwh:
        max_a_idx = aux_kwh.index(max(aux_kwh))
        ax.annotate(f"Peak aux: {aux_kwh[max_a_idx]:.1f} kWh",
                    xy=(max_a_idx, bottom3[max_a_idx] + aux_kwh[max_a_idx] / 2),
                    xytext=(max_a_idx, bottom3[max_a_idx] + aux_kwh[max_a_idx] + 5),
                    ha='center', fontsize=7,
                    arrowprops=dict(arrowstyle="->", lw=0.5),
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))

    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_ylim(0, 60)
    ax.set_xlabel("Relevant charging sessions")
    ax.grid(True, linestyle='--', alpha=0.3)

# === Plot both BEVs ===
plot_stacked(axes[0, 0], bev1_b, bev1_l, bev1_a, colors1, bev1_labels_sorted)
axes[0, 0].set_title("BEV1", fontsize=10)
axes[0, 0].set_ylabel("Energy (kWh)")
axes[0, 0].legend(loc="upper left", fontsize=7, labels=[
    stat_txt("Batt: ", bev1_b),
    stat_txt("Charger:", bev1_l),
    stat_txt("Aux:", bev1_a, scale=1 / 1000, unit="kWh"),
])

plot_stacked(axes[0, 1], bev2_b, bev2_l, bev2_a, colors2, bev2_labels_sorted)
axes[0, 1].set_title("BEV2", fontsize=10)
axes[0, 1].legend(loc="upper left", fontsize=7, labels=[
    stat_txt("Batt:", bev2_b),
    stat_txt("Charger:", bev2_l),
    stat_txt("Aux:", bev2_a, scale=1 / 1000, unit="kWh"),
])

# === Clean unused axes ===
fig.delaxes(axes[1, 0])
fig.delaxes(axes[1, 1])

plt.tight_layout()
plt.savefig("fig4.png", dpi=600, bbox_inches="tight", transparent=True)
plt.show()



'''
####################################################################################
####################################################################################

# Figure 5 and 6: Sessions with highest losses

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# === Folder path ===
folder_path = r'D:\Research vision\Papers\Paper 1\datasets\BEV_data\MaxMin AC losses'

# === Filter charger loss CSVs ===
csv_files = [f for f in sorted(os.listdir(folder_path))
             if f.endswith('.csv') and 'aux loss' in f.lower()]
csv_files = csv_files[:2]

if len(csv_files) < 2:
    print("⚠️ Less than two 'charger loss' files found.")
    exit()

# === Plot style settings ===
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 8,
    'axes.linewidth': 0.6,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'legend.frameon': False,
    'figure.dpi': 600
})

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# === Plot each session ===
for i, filename in enumerate(csv_files):
    file_path = os.path.join(folder_path, filename)
    df = pd.read_csv(file_path)

    required = ['Timestamp', 'ChargeLinePower264', 'SOCave292', 'BMSmaxPackTemperature', 'VCRIGHT_tempAmbientRaw']
    if not all(col in df.columns for col in required):
        continue

    df = df[required].dropna()
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
    df = df.dropna(subset=['Timestamp']).set_index('Timestamp')

    if len(df) < 2:
        continue

    # Fill to 1s resolution
    full_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq='1S')
    df = df.reindex(full_index)
    df['ChargeLinePower264'] = df['ChargeLinePower264'].fillna(0)
    df[['SOCave292', 'BMSmaxPackTemperature', 'VCRIGHT_tempAmbientRaw']] = df[
        ['SOCave292', 'BMSmaxPackTemperature', 'VCRIGHT_tempAmbientRaw']
    ].ffill()

    # Resample to 30s
    df_resampled = df.resample('30S').mean()

    # === Stats ===
    max_batt = df_resampled['BMSmaxPackTemperature'].max()
    min_batt = df_resampled['BMSmaxPackTemperature'].min()
    max_amb = df_resampled['VCRIGHT_tempAmbientRaw'].max()
    min_amb = df_resampled['VCRIGHT_tempAmbientRaw'].min()

    # === Format min–max in legend strings ===
    batt_label = f'Batt Temp ({min_batt:.1f}–{max_batt:.1f}°C)'
    amb_label = f'Amb Temp ({min_amb:.1f}–{max_amb:.1f}°C)'

    # === Primary Axis: Charge Power ===
    ax1 = axes[i]
    ax1.plot(df_resampled.index, df_resampled['ChargeLinePower264'],
             color='tab:blue', linewidth=1.4, label='Charger Power')
    ax1.set_ylabel('Charger Power (kW)', color='tab:blue', fontsize=10)
    ax1.set_ylim(0, 12)
    ax1.tick_params(axis='y', labelcolor='tab:blue', labelsize=8)
    ax1.tick_params(axis='x', labelsize=8)
    ax1.grid(True, linestyle='--', alpha=0.4)

    # === Secondary Axis: SoC & Temps ===
    ax2 = ax1.twinx()
    ax2.plot(df_resampled.index, df_resampled['SOCave292'],
             color='tab:green', linewidth=1.3, label='SoC')
    line_batt, = ax2.plot(df_resampled.index, df_resampled['BMSmaxPackTemperature'],
                          color='red', linestyle='--', linewidth=1.3, label=batt_label)
    line_amb, = ax2.plot(df_resampled.index, df_resampled['VCRIGHT_tempAmbientRaw'],
                         color='black', linestyle='--', linewidth=1.2, label=amb_label)
    ax2.set_ylabel('SoC / Temperature (°C)', color='tab:green', fontsize=10)
    ax2.set_ylim(0, 101)
    ax2.tick_params(axis='y', labelcolor='tab:green', labelsize=8)

    # === X-axis formatting ===
    ax1.set_xlabel('Time (HH:MM)', fontsize=12)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    

    # === Legend: Only Temp Curves, SoC excluded ===
    ax2.legend(handles=[line_batt, line_amb], loc='center', fontsize=12)

    # === Title with cleaned filename ===
    ax1.set_title(f"{filename.replace('_', ' ').replace('.csv','')}",
                  fontsize=12, pad=2)

# === Final layout and save ===
plt.tight_layout()
plt.savefig("fig6.png", dpi=600, bbox_inches="tight")
plt.show()

####################################################################################
####################################################################################

# FIGURE 7: charger losses colorbars  

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# === Plot Style for Publication ===
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 8,
    'axes.linewidth': 0.6,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'legend.frameon': False,
    'figure.dpi': 600
})

# === Folder path ===
folder_path = r'D:\Research vision\Papers\Paper 1\datasets\BEV_data\AllAC'

# === Containers for scatter data ===
ambient_temps, avg_powers = [], []
charger_losses, plug_times_hr = [], []
point_sizes = []
session_stats = []

# === Function to process each CSV file ===
def process_scatter_data(file_path):
    df = pd.read_csv(file_path)
    required = ['ChargeLinePower264', 'Timestamp',
                'TotalChargeKWh3D2', 'VCRIGHT_tempAmbientRaw']
    if not all(col in df.columns for col in required):
        return None

    df = df.dropna(subset=required)
    if df.empty:
        return None

    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
    if df['Timestamp'].isna().any():
        return None

    plug_time_hr = (df['Timestamp'].iloc[-1] - df['Timestamp'].iloc[0]).total_seconds() / 3600.0

    df_power = df[df['ChargeLinePower264'] > 0]
    if df_power.empty:
        return None

    avg_power = df_power['ChargeLinePower264'].mean()
    duration_sec = len(df_power)
    grid_energy = avg_power * (duration_sec / 3600.0)

    total_charge = df['TotalChargeKWh3D2'].iloc[-1] - df['TotalChargeKWh3D2'].iloc[0]
    charger_loss = max(grid_energy - total_charge, 0)

    avg_temp = df_power['VCRIGHT_tempAmbientRaw'].mean()

    return avg_temp, avg_power, charger_loss, plug_time_hr, total_charge

# === Process files ===
for filename in sorted(os.listdir(folder_path)):
    if filename.endswith('.csv'):
        file_path = os.path.join(folder_path, filename)
        result = process_scatter_data(file_path)
        if result:
            temp, power, loss, plug, energy = result
            ambient_temps.append(temp)
            avg_powers.append(power)
            charger_losses.append(loss)
            plug_times_hr.append(plug)
            point_sizes.append(20 + energy * 2)

            session_stats.append({
                "filename": filename,
                "duration_hr": plug,
                "avg_temp": temp
            })

# === Print summary ===
if session_stats:
    longest = max(session_stats, key=lambda x: x['duration_hr'])
    coldest = min(session_stats, key=lambda x: x['avg_temp'])
    hottest = max(session_stats, key=lambda x: x['avg_temp'])

    print("\n📊 Session Summary:")
    print(f"• Longest duration: {longest['filename']} ({longest['duration_hr']:.2f} hr)")
    print(f"• Lowest avg ambient temp: {coldest['filename']} ({coldest['avg_temp']:.2f} °C)")
    print(f"• Highest avg ambient temp: {hottest['filename']} ({hottest['avg_temp']:.2f} °C)")

# === Plotting layout ===
fig = plt.figure(figsize=(5, 2.2))
gs = gridspec.GridSpec(3, 2, height_ratios=[0.05, 0.01, 0.94])  # colorbar, padding, plot

# === Plot 1: Power vs Ambient Temp ===
ax0 = fig.add_subplot(gs[2, 0])
sc1 = ax0.scatter(ambient_temps, avg_powers, c=charger_losses, s=point_sizes,
                  cmap='plasma', edgecolors='black')
ax0.set_xlabel("Avg Ambient Temp (°C)", fontsize=6)
ax0.set_ylabel("Avg Power (kW)", fontsize=6)
ax0.grid(True)

# === Plot 2: Power vs Plug Time ===
ax1 = fig.add_subplot(gs[2, 1])
sc2 = ax1.scatter(plug_times_hr, avg_powers, c=charger_losses, s=point_sizes,
                  cmap='plasma', edgecolors='black')
ax1.set_xlabel("Duration (hr)", fontsize=6)
ax1.grid(True)

# === Top colorbar for subplot 1 ===
cbar_ax1 = fig.add_subplot(gs[0, 0]) 
cb1 = plt.colorbar(sc1, cax=cbar_ax1, orientation='horizontal')
cb1.ax.xaxis.set_ticks_position('bottom')
cb1.ax.xaxis.set_label_position('top')
cb1.set_label("Charger losses (kWh)", fontsize=6)

# === Top colorbar for subplot 2 ===
cbar_ax2 = fig.add_subplot(gs[0, 1])
cb2 = plt.colorbar(sc2, cax=cbar_ax2, orientation='horizontal')
cb2.ax.xaxis.set_ticks_position('bottom')
cb2.ax.xaxis.set_label_position('top')
cb2.set_label("Charger Losses (kWh)", fontsize=6)

# === Final layout ===
fig.subplots_adjust(hspace=0.2, top=0.93, bottom=0.12)
plt.savefig("fig7.png", dpi=600, bbox_inches='tight')
plt.show()

####################################################################################
####################################################################################


# FIGURE 8: Fast charging dataset distribution
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# === Plot style ===
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 8,
    'axes.linewidth': 0.6,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'figure.dpi': 600
})

# Folder path
folder_path = 'D:\\Research vision\\Papers\\Paper 1\\datasets\\BEV_data\\AllDC'

# Colors
blue = '#1f77b4'
orange = '#ff7f0e'
gray1 = '#666666'
gray2 = '#aaaaaa'

# Init containers
bev1_power, bev2_power = [], []
bev1_energy, bev2_energy = [], []
bev1_soc, bev2_soc = [], []
bev1_temp, bev2_temp = [], []
bev1_temp_amb, bev2_temp_amb = [], []

# Process each file
for filename in os.listdir(folder_path):
    if filename.endswith('.csv'):
        file_path = os.path.join(folder_path, filename)
        df = pd.read_csv(file_path)

        required_columns = [
            'FC_dcCurrent', 'FC_dcVoltage', 'Timestamp',
            'SOCave292', 'BMSmaxPackTemperature', 'VCRIGHT_tempAmbientRaw',
            'BattVoltage132', 'RawBattCurrent132', 'TotalChargeKWh3D2'
        ]
        if not all(col in df.columns for col in required_columns):
            continue

        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df = df.sort_values('Timestamp')
        df['time_diff'] = df['Timestamp'].diff().dt.total_seconds().fillna(0)
        df['power_kw'] = (df['FC_dcCurrent'] * df['FC_dcVoltage']) / 1000.0

        # Total energy using TotalChargeKWh3D2
        total_energy = df['TotalChargeKWh3D2'].iloc[-1] - df['TotalChargeKWh3D2'].iloc[0]

        if 'BEV1' in filename.upper():
            bev1_power.extend(df['power_kw'].dropna())
            bev1_energy.append(total_energy)
            bev1_soc.extend(df['SOCave292'].dropna())
            bev1_temp.extend(df['BMSmaxPackTemperature'].dropna())
            bev1_temp_amb.extend(df['VCRIGHT_tempAmbientRaw'].dropna())
        elif 'BEV2' in filename.upper():
            bev2_power.extend(df['power_kw'].dropna())
            bev2_energy.append(total_energy)
            bev2_soc.extend(df['SOCave292'].dropna())
            bev2_temp.extend(df['BMSmaxPackTemperature'].dropna())
            bev2_temp_amb.extend(df['VCRIGHT_tempAmbientRaw'].dropna())

# === Plot setup
fig, axes = plt.subplots(2, 2, figsize=(7, 5))  # optimal for Nature journal

# 1. Power Histogram
axes[0, 0].hist(bev1_power, bins=30, edgecolor='black', alpha=0.8, label='BEV1', color=blue)
axes[0, 0].hist(bev2_power, bins=30, edgecolor='black', alpha=0.6, label='BEV2', color=orange)
axes[0, 0].set_xlabel('Power (kW)')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].legend(frameon=False, loc='upper right')
axes[0, 0].grid(True, linestyle='--', alpha=0.3)

# 2. Total Energy Histogram
axes[0, 1].hist(bev1_energy, bins=5, edgecolor='black', alpha=0.8, label='BEV1', color=blue)
axes[0, 1].hist(bev2_energy, bins=5, edgecolor='black', alpha=0.6, label='BEV2', color=orange)
axes[0, 1].set_xlabel('Energy (kWh)')
axes[0, 1].legend(frameon=False, loc='upper right')
axes[0, 1].grid(True, linestyle='--', alpha=0.3)

# 3. SoC Histogram
axes[1, 0].hist(bev1_soc, bins=30, edgecolor='black', alpha=0.8, label='BEV1', color=blue)
axes[1, 0].hist(bev2_soc, bins=30, edgecolor='black', alpha=0.6, label='BEV2', color=orange)
axes[1, 0].set_xlabel('SoC (%)')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].legend(frameon=False, loc='upper left')
axes[1, 0].grid(True, linestyle='--', alpha=0.3)

# 4. Temperature Histogram
axes[1, 1].hist(bev1_temp, bins=30, edgecolor='black', alpha=0.8, label='BEV1 Batt', color=blue)
axes[1, 1].hist(bev2_temp, bins=30, edgecolor='black', alpha=0.6, label='BEV2 Batt', color=orange)
axes[1, 1].hist(bev1_temp_amb, bins=30, edgecolor='black', alpha=0.3, label='BEV1 Amb', color=gray1)
axes[1, 1].hist(bev2_temp_amb, bins=30, edgecolor='black', alpha=0.3, label='BEV2 Amb', color=gray2)
axes[1, 1].set_xlabel('Temperature (°C)')
axes[1, 1].legend(frameon=False, loc='upper left')
axes[1, 1].grid(True, linestyle='--', alpha=0.3)

# Format y-axis in thousands
def format_thousands(x, _):
    return f'{int(x/1000)}k' if x >= 1000 else str(int(x))

for ax in axes.flat:
    if ax != axes[0, 1]:  # exclude energy subplot
        ax.yaxis.set_major_formatter(FuncFormatter(format_thousands))

plt.tight_layout()
output_path = "fig8.png"
plt.savefig(output_path, dpi=1000, bbox_inches='tight')
plt.show()

####################################################################################
####################################################################################


# FIGURE 9: temperature and power scatter plots with colorbars
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import pearsonr

# === Plot style ===
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 8,
    'axes.linewidth': 0.6,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'figure.dpi': 600
})

# === Folder path ===
folder_path = r'D:\Research vision\Papers\Paper 1\datasets\BEV_data\AllDC'

# === Data collection ===
rows = []
for filename in sorted(os.listdir(folder_path)):
    if filename.endswith('.csv'):
        df = pd.read_csv(os.path.join(folder_path, filename))
        required = ['SOCave292', 'BMSmaxPackTemperature', 'VCRIGHT_tempAmbientRaw',
                    'FC_dcCurrent', 'FC_dcVoltage', 'BMS_preconditionAllowed']
        if not all(col in df.columns for col in required):
            continue
        df = df[required].dropna()
        if df.empty:
            continue

        soc = df['SOCave292'].iloc[0]
        batt_temp = df['BMSmaxPackTemperature'].iloc[0]
        amb_temp = df['VCRIGHT_tempAmbientRaw'].iloc[0]
        delta_temp = batt_temp - amb_temp
        max_power = (df['FC_dcCurrent'] * df['FC_dcVoltage']).max() / 1000.0
        precond = int(df['BMS_preconditionAllowed'].iloc[0])  # only first row

        rows.append({
            'SoC': soc,
            'MaxPower_kW': max_power,
            'BatteryTemp': batt_temp,
            'AmbientTemp': amb_temp,
            'DeltaTemp': delta_temp,
            'Highlight': (precond == 0)
        })

df_all = pd.DataFrame(rows)


# === Gridspec layout ===
fig = plt.figure(figsize=(7, 3.5))
gs = gridspec.GridSpec(2, 3, height_ratios=[0.08, 1], hspace=0.2, wspace=0.3)

# === Axes setup ===
axes = [fig.add_subplot(gs[1, i]) for i in range(3)]
cbar_axes = [fig.add_subplot(gs[0, i]) for i in range(3)]

# === Scatter with highlighting ===
def plot_conditional_scatter(ax, x, y, c, cmap, highlight_mask):
    ax.scatter(x[~highlight_mask], y[~highlight_mask], c=c[~highlight_mask], cmap=cmap,
               s=120, edgecolor='black', alpha=0.9, label='precond=1')
    ax.scatter(x[highlight_mask], y[highlight_mask], c=c[highlight_mask], cmap=cmap,
               s=70, alpha=0.9, label='precond=0')

highlight_mask = df_all['Highlight']

# === Apply to each subplot ===
plot_conditional_scatter(axes[0], df_all['SoC'], df_all['MaxPower_kW'],
                         df_all['BatteryTemp'], 'plasma', highlight_mask)
plot_conditional_scatter(axes[1], df_all['SoC'], df_all['MaxPower_kW'],
                         df_all['AmbientTemp'], 'plasma', highlight_mask)
plot_conditional_scatter(axes[2], df_all['SoC'], df_all['MaxPower_kW'],
                         df_all['DeltaTemp'], 'coolwarm', highlight_mask)

# === Axis labels and grid ===
for ax in axes:
    ax.set_xlabel('Initial SoC (%)')
    ax.grid(True, linestyle='--', alpha=0.5)
axes[0].set_ylabel('Max charger power (kW)')
axes[0].legend(loc='upper right', fontsize=7)

# === Colorbars + labels ===
def add_top_colorbar(fig, ax, scatter, label):
    cbar = plt.colorbar(scatter, cax=ax, orientation='horizontal')
    cbar.ax.tick_params(labelsize=7, pad=1)
    ax.set_title(label, fontsize=8, pad=3)

dummy1 = axes[0].scatter(df_all['SoC'], df_all['MaxPower_kW'],
                         c=df_all['BatteryTemp'], cmap='plasma')
dummy2 = axes[1].scatter(df_all['SoC'], df_all['MaxPower_kW'],
                         c=df_all['AmbientTemp'], cmap='plasma')
dummy3 = axes[2].scatter(df_all['SoC'], df_all['MaxPower_kW'],
                         c=df_all['DeltaTemp'], cmap='coolwarm')

add_top_colorbar(fig, cbar_axes[0], dummy1, 'Batt Temp (°C)')
add_top_colorbar(fig, cbar_axes[1], dummy2, 'Ambient Temp (°C)')
add_top_colorbar(fig, cbar_axes[2], dummy3, 'ΔT (Batt - Amb, °C)')

# === Pearson correlation coefficients ===
corr1, _ = pearsonr(df_all['MaxPower_kW'], df_all['SoC'])
corr2, _ = pearsonr(df_all['MaxPower_kW'], df_all['BatteryTemp'])   
corr3, _ = pearsonr(df_all['MaxPower_kW'], df_all['AmbientTemp'])
corr4, _ = pearsonr(df_all['MaxPower_kW'], df_all['DeltaTemp'])

# === Annotate correlation on plots ===
axes[0].text(0.55, 0.8, f"ρsoc = {corr1:.2f}", transform=axes[0].transAxes,
             fontsize=7, va='top', ha='left', fontweight='bold')
axes[0].text(0.55, 0.7, f"ρbatt = {corr2:.2f}", transform=axes[0].transAxes,
             fontsize=7, va='top', ha='left', fontweight='bold')
axes[1].text(0.65, 0.9, f"ρ = {corr3:.2f}", transform=axes[1].transAxes,
             fontsize=7, va='top', ha='left', fontweight='bold')
axes[2].text(0.65, 0.9, f"ρ = {corr4:.2f}", transform=axes[2].transAxes,
             fontsize=7, va='top', ha='left', fontweight='bold')
# === Final save ===
plt.savefig('fig9.png', dpi=600, bbox_inches='tight')
plt.show()

####################################################################################
####################################################################################


#FIGURE 10: Fast charging energy breakdown
import os
import pandas as pd
import matplotlib.pyplot as plt

# === Load your dataset ===
file_path = "DCcharging.csv"  # Update path if needed
df = pd.read_csv(file_path)

# === Sort data by total energy ===
df['TotalEnergy'] = df['BatteryEnergy_kWh'] + df['AuxLoss_kWh'] + df['ChargerLoss_kWh']
df_sorted = df.sort_values(by='TotalEnergy').reset_index(drop=True)

# === Extract data ===
sessions = df_sorted['Session']
batt = df_sorted['BatteryEnergy_kWh']
aux = df_sorted['AuxLoss_kWh']
charger = df_sorted['ChargerLoss_kWh']
diss = df_sorted['BMS_Dissipation_kWh']
x = range(len(sessions))

# === Stats ===
def stats(series):
    return series.min(), series.max(), series.mean()

bmin, bmax, bavg = stats(batt)
dmin, dmax, davg = stats(diss)
amin, amax, aavg = stats(aux)
cmin, cmax, cavg = stats(charger)

# === Plot setup ===
fig, ax = plt.subplots(figsize=((4.5, 3.5)), dpi=300)
colors = {'battery': '#fdae61', 'diss': '#f46d43', 'aux': '#888888', 'charger': '#d62728'}

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 8,
    'axes.linewidth': 0.6,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'legend.frameon': False
})

# === Plot bars with annotation ===
for i in x:
    battery_only = batt[i] - diss[i]
    ax.bar(i, battery_only, color=colors['battery'], edgecolor='black')
    ax.bar(i, diss[i], bottom=battery_only, color=colors['diss'], edgecolor='black')
    ax.bar(i, aux[i], bottom=batt[i], color=colors['aux'], edgecolor='black')
    ax.bar(i, charger[i], bottom=batt[i] + aux[i], color=colors['charger'], edgecolor='black')

    # Annotate segments
    if battery_only > 1:
        ax.text(i, battery_only / 2, f"{battery_only:.1f}", ha='center', va='center', fontsize=8, color='black')
    if diss[i] > 0.5:
        ax.text(i, battery_only + diss[i] / 2, f"{diss[i]:.1f}", ha='center', va='center', fontsize=6.5, color='white')
    if aux[i] > 0.5:
        ax.text(i, batt[i] + aux[i] / 2, f"{aux[i]:.1f}", ha='center', va='center', fontsize=6.5, color='white')
    if charger[i] > 0.5:
        ax.text(i, batt[i] + aux[i] + charger[i] / 2, f"{charger[i]:.1f}", ha='center', va='center', fontsize=6.5, color='white')

# === Legend (top right) ===
ax.legend([
    plt.Rectangle((0, 0), 1, 1, color=colors['battery']),
    plt.Rectangle((0, 0), 1, 1, color=colors['diss']),
    plt.Rectangle((0, 0), 1, 1, color=colors['aux']),
    plt.Rectangle((0, 0), 1, 1, color=colors['charger']),
], ['Batt Energy', 'Dispt', 'Aux', 'Charger'],
    fontsize=8, loc='upper right')

# === Stats (top left) ===
stats_text = "\n".join([
    f"Batt: min={bmin:.1f} max={bmax:.1f} avg={bavg:.1f}",
    f"Dispt: min={dmin:.1f} max={dmax:.1f} avg={davg:.1f}",
    f"Aux: min={amin:.1f} max={amax:.1f} avg={aavg:.1f}",
    f"Charger: min={cmin:.1f} max={cmax:.1f} avg={cavg:.1f}"
])
ax.text(0.01, 0.98, stats_text, transform=ax.transAxes,
        fontsize=8, va='top', ha='left', family='monospace',
        bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))

# === Final formatting ===
ax.set_ylabel("Energy (kWh)")
ax.set_ylim(0, 90)
ax.set_xticks(x)
ax.set_xticklabels([])  # remove session names
ax.grid(True, linestyle='--', alpha=0.4)

plt.tight_layout()
plt.savefig("fig10.png", bbox_inches='tight', dpi=300)
plt.show()

####################################################################################
####################################################################################

#FIGURE 11: Main driving routes with GPS tracks

import pandas as pd
from pathlib import Path
import folium
from collections import Counter
import math, os, webbrowser

data_folder  = Path("Driving_sessions")           # folder with CSVs
var_lat, var_lon = "GPSLatitude04F", "GPSLongitude04F"
ROUND   = 4           # decimals for grouping identical start/end (~11 m)
MIN_KM  = 10          # ignore trips shorter than this
OUT_HTML = "route_map.html"
OUT_PNG  = "fig11.png,"
OUT_DIR  = Path("route_coords")                   # GPS-track CSVs
OUT_DIR.mkdir(exist_ok=True)

COLORS = {"BEV1": "blue", "BEV2": "orange"}
STYLES = {1: [], 2: [10, 20]}   # solid for #1, dashed for #2


def haversine(p1, p2):
    """Great-circle distance (km) between two [lat,lon] pts."""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(math.radians, [*p1, *p2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    return 2 * R * math.asin(math.sqrt(a))

def route_key(s, e):
    """Rounded start/end so near-identical rides group together."""
    return (
        round(s[0], ROUND), round(s[1], ROUND),
        round(e[0], ROUND), round(e[1], ROUND),
    )

def path_distance_km(path):
    """Sum of segment distances along the track."""
    return sum(haversine(path[i], path[i+1]) for i in range(len(path)-1))


freq = {"BEV1": Counter(), "BEV2": Counter()}
key_path, key_raw_km = {}, {}

for csv in data_folder.glob("*.csv"):
    try:
        df = pd.read_csv(csv, usecols=[var_lat, var_lon]).dropna()
        if df.empty:
            continue

        path = df[[var_lat, var_lon]].values.tolist()
        start, end = path[3], path[-1]
        raw_km = haversine(start, end)
        if raw_km < MIN_KM:
            continue

        tag = ("BEV1" if "BEV1" in csv.name
               else "BEV2" if "BEV2" in csv.name else None)
        if not tag:
            continue

        k = route_key(start, end)
        freq[tag][k] += 1
        key_path.setdefault(k, path)   # keep first full track
        key_raw_km[k] = raw_km
    except Exception as e:
        print(f"⚠️  Skipping {csv.name}: {e}")


def top2(counter):
    ranked = [(c, key_raw_km[k], k) for k, c in counter.items()]
    ranked.sort(key=lambda t: (-t[0], -t[1]))   # freq desc, then longer
    return ranked[:2]

sel = {bev: top2(cnt) for bev, cnt in freq.items()}
if not any(sel.values()):
    raise ValueError("No qualifying trips (≥10 km) found.")


for bev, infos in sel.items():
    for idx, (cnt, raw_km, k) in enumerate(infos, 1):
        track = key_path[k]
        track_km = path_distance_km(track)
        out = pd.DataFrame(track, columns=["latitude", "longitude"])
        out["point_index"] = out.index
        out["trips_matched"] = cnt
        out["raw_haversine_km"] = raw_km
        out["track_distance_km"] = track_km
        fname = OUT_DIR / f"{bev}_route{idx}.csv"
        out.to_csv(fname, index=False)
        print(f"✓ Saved {fname}")


all_pts = [pt for infos in sel.values() for _,_,k in infos for pt in key_path[k]]
c_lat = sum(p[0] for p in all_pts) / len(all_pts)
c_lon = sum(p[1] for p in all_pts) / len(all_pts)
m = folium.Map([c_lat, c_lon], tiles="OpenStreetMap", zoom_start=9, control_scale=True)

legend_items = []
for bev, infos in sel.items():
    for idx, (cnt, raw_km, k) in enumerate(infos, 1):
        track = key_path[k]
        track_km = path_distance_km(track)
        label = f"{bev}_{track_km:.1f} km"
        folium.PolyLine(
            track,
            color=COLORS[bev],
            weight=6,
            dash_array=STYLES[idx],
            tooltip=label
        ).add_to(m)
        folium.Marker(track[0], icon=folium.Icon(color=COLORS[bev], icon="play"),
                      popup=f"{label} start").add_to(m)
        folium.Marker(track[-1], icon=folium.Icon(color=COLORS[bev], icon="stop"),
                      popup=f"{label} end").add_to(m)
        legend_items.append((label, COLORS[bev]))

# fit view tightly
m.fit_bounds([[min(p[0] for p in all_pts), min(p[1] for p in all_pts)],
              [max(p[0] for p in all_pts), max(p[1] for p in all_pts)]])

# legend (upper-right)
legend = ('<div style="position:fixed; top:40px; right:40px; width:140px;'
          'z-index:9999; font-size:14px; background:white; border:2px solid grey;'
          'border-radius:6px; padding:8px;"><b></b><br>')
for txt, col in legend_items:
    legend += (f'<i style="background:{col};width:12px;height:12px;'
               'display:inline-block;"></i>&nbsp;'+txt+'<br>')
legend += '</div>'
m.get_root().html.add_child(folium.Element(legend))

# ── 8. Show inline (if notebook) and save HTML & PNG ───────────────────────
try:
    from IPython.display import display
    display(m)
except ModuleNotFoundError:
    pass

m.save(OUT_HTML)
print(f"✓ Map saved to {OUT_HTML}")
try:
    webbrowser.open('file://' + os.path.abspath(OUT_HTML))
except Exception:
    print("Open route_map.html manually to view the map.")


try:
    from io import BytesIO
    from PIL import Image          # pip install pillow
    # 1) render the default 800×600 map
    raw_png = m._to_png(5)         # 5-second delay for tiles to load
    img = Image.open(BytesIO(raw_png))

    # 2) upscale ×2 with high-quality Lanczos resampling → 1600×1200
    scale = 2
    big = img.resize((img.width * scale, img.height * scale), Image.LANCZOS)

    # 3) save with 300 dpi metadata
    big.save(OUT_PNG, dpi=(300, 300))
    print("✓ high-res PNG saved to", OUT_PNG)

except Exception as e:
    print("⚠️  PNG snapshot skipped (needs selenium/pyppeteer + pillow):", e)

####################################################################################
####################################################################################

# FiGURE 12: Driving dataset histograms

import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np


folder = Path("D:\Research vision\Papers\Paper 1\Submission\Dataset_repo\driving sessions")

REQ = [
    "DI_uiSpeed",
    "Odometer3B6",
    "TotalDischargeKWh3D2",
    "BMS_kwhRegenChargeTotal",
    "BMSmaxPackTemperature",
    "VCRIGHT_tempAmbientRaw"
]

MAX_OD_DIFF = 1000          # km
MAX_EN_DIFF = 200            # kWh
MIN_TRIP_KM = 1              # ignore short trips

blue, orange = "#1976d2", "#f57c00"

bev1 = dict(speed=[], temp=[], temp_amb=[], km=[], dch=[], reg=[])
bev2 = dict(speed=[], temp=[], temp_amb=[], km=[], dch=[], reg=[])

def robust_edge(series, first=True, lim=MAX_EN_DIFF):
    if series.empty:
        return None
    a = series.iloc[0] if first else series.iloc[-1]
    b = series.iloc[1] if first and len(series) > 1 else (
        series.iloc[-2] if not first and len(series) > 1 else a
    )
    return a if abs(b - a) < lim else b

def delta(series, unit="km"):
    if len(series) < 2:
        return None
    first = robust_edge(series, True, MAX_EN_DIFF)
    last  = robust_edge(series, False, MAX_EN_DIFF)
    if first is None or last is None:
        return None
    diff = last - first
    if diff < 0:
        return None
    if unit == "km"  and diff > MAX_OD_DIFF:
        return None
    if unit == "kWh" and diff > MAX_EN_DIFF:
        return None
    return diff

def dedup_zeros(col):
    out, prev0 = [], False
    for v in col.dropna():
        if v == 0:
            if not prev0:
                out.append(v)
            prev0 = True
        else:
            out.append(v); prev0 = False
    return out

k_fmt = mtick.FuncFormatter(lambda x, _: "0" if x == 0 else f"{x/1000:g}k")

for csv in sorted(folder.glob("*.csv")):
    df = pd.read_csv(csv)
    if not all(c in df.columns for c in REQ):
        continue

    tag = bev1 if "BEV1" in csv.name else bev2

    # distance Δkm
    km = delta(df["Odometer3B6"].astype(float).dropna(), "km")
    if km is None or km < MIN_TRIP_KM:
        continue
    tag["km"].append(km)

    # discharge and regen
    dch = delta(df["TotalDischargeKWh3D2"].astype(float).dropna(), "kWh")
    reg = delta(df["BMS_kwhRegenChargeTotal"].astype(float).dropna(), "kWh")
    if dch is not None: tag["dch"].append(dch)
    if reg is not None: tag["reg"].append(-reg)

    # speed, internal temp, ambient temp
    tag["speed"].extend(dedup_zeros(df["DI_uiSpeed"]))
    tag["temp"].extend(dedup_zeros(df["BMSmaxPackTemperature"]))
    tag["temp_amb"].extend(dedup_zeros(df["VCRIGHT_tempAmbientRaw"]))


fig, ax = plt.subplots(2, 2, figsize=(6.5, 4.5), dpi=300)

# 1 Speed
ax[0,0].hist(bev1["speed"], bins=30, alpha=.7, label="BEV1", color=blue, edgecolor="black")
ax[0,0].hist(bev2["speed"], bins=30, alpha=.7, label="BEV2", color=orange, edgecolor="black")
ax[0,0].set(xlabel="Speed (km/h)", ylabel="Frequency")
ax[0,0].yaxis.set_major_formatter(k_fmt)
ax[0,0].legend(fontsize=6); ax[0,0].grid(ls="--", alpha=.6)

# 2 Energy (discharge and regen)
for data, col, lbl in [
    (bev1["dch"], blue,      "BEV1"),
    (bev2["dch"], orange,    "BEV2"),
    (bev1["reg"], "#CFF0FD", "BEV1 reg"),
    (bev2["reg"], "#f84c4c", "BEV2 reg")]:
    ax[0,1].hist(data, bins=30, alpha=.6,
                 label=lbl, color=col, edgecolor="black",
                 hatch="//" if "regen" in lbl else None)

ax[0,1].axvline(0, color="black", lw=0.8, ls="--")
ax[0,1].set(xlabel="Energy (kWh)", ylabel="Frequency")
ax[0,1].yaxis.set_major_formatter(k_fmt)
ax[0,1].legend(fontsize=6); ax[0,1].grid(ls="--", alpha=.6)
ax[0,1].set_xlim(-10, 40)

# 3 Distance
ax[1,0].hist(bev1["km"], bins=30, alpha=.7, label="BEV1", color=blue, edgecolor="black")
ax[1,0].hist(bev2["km"], bins=30, alpha=.7, label="BEV2", color=orange, edgecolor="black")
ax[1,0].set(xlabel="Distance (km)", ylabel="Frequency")
ax[1,0].yaxis.set_major_formatter(k_fmt)
ax[1,0].legend(fontsize=6); ax[1,0].grid(ls="--", alpha=.6)
ax[1,0].set_xlim(0, 200)

# 4 Battery internal + ambient temperature
for data, col, lbl in [
    (bev1["temp"], blue,         "BEV1 batt"),
    (bev2["temp"], orange,       "BEV2 batt"),
    (bev1["temp_amb"], "#E7E4E4", "BEV1 amb"),
    (bev2["temp_amb"], "#aaaaaa", "BEV2 amb")]:
    ax[1,1].hist(data, bins=30, alpha=.6,
                 label=lbl, color=col, edgecolor="black",
                 hatch=".." if "ambient" in lbl else None)

ax[1,1].set(xlabel="Temperature (°C)", ylabel="Frequency")
ax[1,1].yaxis.set_major_formatter(k_fmt)
ax[1,1].legend(fontsize=6); ax[1,1].grid(ls="--", alpha=.6)

plt.tight_layout()
plt.savefig("fig12.png", dpi=300, bbox_inches="tight")
plt.show()

####################################################################################
####################################################################################
# FIGURE 13: Driving sessions efficiency vs ambient temperature scatter plot
import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# ── paths & constants ────────────────────────────────────────────────────
FOLDER = Path(r"D:\Research vision\Papers\Paper 1\Submission\Dataset_repo\driving sessions")
REQ = {"TotalDischargeKWh3D2", "BMS_kwhRegenChargeTotal", "Odometer3B6", "VCFRONT_tempAmbient"}

MIN_TRIP_KM = 1
MAX_TRIP_KM = 1000
MAX_EN_KWH = 700

clr = {"BEV1": "#1976d2", "BEV2": "#f57c00"}

# ── helpers ───────────────────────────────────────────────────────────────
def median_edges(s):
    return np.median(s.head(10)), np.median(s.tail(10))

def robust_delta(series, unit):
    a, b = median_edges(series)
    d = b - a
    if unit == "km":
        return d if MIN_TRIP_KM <= d <= MAX_TRIP_KM else None
    if unit == "kWh":
        return d if 0 < d <= MAX_EN_KWH else None
    return None

def iqr_keep(arr):
    q1, q3 = np.percentile(arr, [25, 75])
    iqr = q3 - q1
    lo, hi = q1 - 1.5*iqr, q3 + 1.5*iqr
    return (arr >= lo) & (arr <= hi)

# ── collect sessions ─────────────────────────────────────────────────────
eff, temp, tag = [], [], []

for csv in FOLDER.glob("*.csv"):
    if "_d" not in csv.stem.lower():
        continue
    try:
        df = pd.read_csv(csv, usecols=lambda c: c in REQ).dropna()
    except Exception:
        continue
    if len(df) < 20:
        continue

    km = robust_delta(df["Odometer3B6"].astype(float), "km")
    disc_a, disc_b = median_edges(df["TotalDischargeKWh3D2"].astype(float))
    regen_a, regen_b = median_edges(df["BMS_kwhRegenChargeTotal"].astype(float))
    adjusted_kwh = (disc_b - disc_a) - (regen_b - regen_a)

    if km is None or not (0 < adjusted_kwh <= MAX_EN_KWH):
        continue

    eff.append(adjusted_kwh * 1000 / km)
    temp.append(df["VCFRONT_tempAmbient"].astype(float).mean())
    tag.append("BEV1" if "bev1" in csv.stem.lower() else "BEV2")

# ── cleaning and correlation ─────────────────────────────────────────────
eff, temp, tag = map(np.array, (eff, temp, tag))
mask = iqr_keep(eff) & iqr_keep(temp)
eff, temp, tag = eff[mask], temp[mask], tag[mask]

if len(eff) == 0:
    print("No qualifying data.")
    raise SystemExit

r, p = pearsonr(temp, eff)
idx_sorted = np.argsort(eff)
low_idx, high_idx = idx_sorted[:25], idx_sorted[-25:]

# ── plotting ─────────────────────────────────────────────────────────────
plt.figure(figsize=(8, 5))

for lab in ("BEV1", "BEV2"):
    pts = tag == lab
    plt.scatter(temp[pts], eff[pts], s=36, color=clr[lab], alpha=0.7, label=lab)

plt.scatter(temp[low_idx], eff[low_idx], s=140, color="#2ecc71", alpha=0.9, label="_nolegend_")
plt.scatter(temp[high_idx], eff[high_idx], s=140, color="#ef240e", alpha=0.9, label="_nolegend_")

plt.xlabel("Average ambient temperature (°C)")
plt.ylabel("Efficiency (Wh/km)")
plt.grid(ls="--", alpha=0.6)
plt.legend()

#plt.gca().text(
    #0.95, 0.1,
    #f"r = {r:.2f}\np-value   = {p:.3g}",
    #transform=plt.gca().transAxes,
    #ha="right", va="bottom",
    #bbox=dict(boxstyle="round", fc="white", alpha=.85, lw=0.5)
#)

plt.tight_layout()
plt.savefig("fig13.png", dpi=300, bbox_inches="tight")
plt.show()

####################################################################################
####################################################################################
# FIGURE 14: Discharge vs Ambient Temperature scatter plot with clustering
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from datetime import time

# === CONFIGURATION ===
FOLDER = Path("D:/Research vision/Papers/Paper 1/Submission/Dataset_repo/parking sessions_new")
DISCHARGE_COL = "TotalDischargeKWh3D2"
TIME_COL = "Timestamp"
TEMP_COL = "VCFRONT_tempAmbient"
PRECOND_COLS = ["UI_batteryPreconditioningRequest", "BMS_preconditionAllowed"]

# === Time parsing ===
def parse_time(col):
    if pd.api.types.is_numeric_dtype(col):
        factor = 1 if col.iloc[0] > 1e9 else 1_000
        return pd.to_datetime(col * factor, unit='ms', errors='coerce')
    return pd.to_datetime(col, errors='coerce')

# === Updated Duration Clustering Function ===
def cluster_by_duration(duration_hr):
    if duration_hr <= 12:
        return "≤ 12h"
    elif duration_hr <= 24:
        return "12–24h"
    elif duration_hr <= 48:
        return "24–48h"
    else:
        return "> 48h"

# === Collect data ===
records = []
export_records = []
cluster_files = {"≤ 12h": [], "12–24h": [], "24–48h": [], "> 48h": []}

for file in FOLDER.glob("*.csv"):
    try:
        df = pd.read_csv(file)

        # Preconditioning flags
        has_ui = "UI_batteryPreconditioningRequest" in df.columns and (df["UI_batteryPreconditioningRequest"] == 1).any()
        has_bms = "BMS_preconditionAllowed" in df.columns and (df["BMS_preconditionAllowed"] == 1).any()
        both_precond = has_ui and has_bms

        if any(col not in df.columns for col in [DISCHARGE_COL, TIME_COL, TEMP_COL]):
            continue

        dis_vals = df[DISCHARGE_COL].dropna()
        time_vals = parse_time(df[TIME_COL].dropna())
        temp_vals = df[TEMP_COL].dropna()
        if len(dis_vals) < 2 or time_vals.empty or temp_vals.empty:
            continue

        start_time = time_vals.iloc[0]
        end_time = time_vals.iloc[-1]
        discharge = dis_vals.iloc[-1] - dis_vals.iloc[0]
        duration_hr = (end_time - start_time).total_seconds() / 3600
        avg_temp = temp_vals.mean()

        if discharge >= 0 and duration_hr >= 0.25:
            cluster = cluster_by_duration(duration_hr)
            cluster_files[cluster].append(file.name)
            export_records.append({
                "Cluster": cluster,
                "Filename": file.name,
                "TotalDischarge_kWh": round(discharge, 4)
            })
            records.append({
                "avg_temp": avg_temp,
                "discharge": discharge,
                "cluster": cluster,
                "highlight": both_precond
            })

    except Exception as e:
        print(f"Skipping {file.name}: {e}")

# === Convert to DataFrame ===
df = pd.DataFrame(records)

# === Marker and color styles ===
cluster_styles = {
    "≤ 12h":    {"color": "#1976d2", "marker": "o"},
    "12–24h":   {"color": "#2ca02c", "marker": "v"},
    "24–48h":   {"color": "#f57c00", "marker": "s"},
    "> 48h":    {"color": "#d62728", "marker": "D"},
}

# === Plot ===
plt.figure(figsize=(7, 5))

for cluster, style in cluster_styles.items():
    subset = df[df["cluster"] == cluster]
    regular = subset[subset["highlight"] == False]
    highlighted = subset[subset["highlight"] == True]
    avg_dis = subset["discharge"].mean()

    label_base = f"{cluster} : avg = {avg_dis:.2f} kWh"

    # Plot regular points
    plt.scatter(
        regular["avg_temp"], regular["discharge"],
        color=style["color"], marker=style["marker"],
        label=label_base, alpha=0.75, s=40
    )

    # Plot preconditioned sessions with black edge
    plt.scatter(
        highlighted["avg_temp"], highlighted["discharge"],
        facecolors=style["color"], edgecolors="black", linewidths=1.2,
        marker=style["marker"], s=60
    )

plt.xlabel("Avg Ambient Temperature [°C]", fontsize=10)
plt.ylabel("Energy [kWh]", fontsize=10)
#plt.title("Discharge vs Ambient Temp\nClusters by Duration")
plt.grid(True, linestyle="--", alpha=0.3)
plt.ylim(0, 12)
plt.legend(title="Black edge = user precond", loc="upper left")
plt.tight_layout()

# === Save High-Quality Figure ===
plt.savefig("fig14.png", dpi=1200)
plt.show()

# === Export CSV Summary ===
export_df = pd.DataFrame(export_records)
export_df.to_csv("discharge_per_cluster_final.csv", index=False)

# === Print Filenames Per Cluster ===
for cluster, files in cluster_files.items():
    print(f"\n{cluster} ({len(files)} files):")
    for name in files:
        print(f" - {name}")

####################################################################################
####################################################################################

#FIGURE 15: Total energy consumption breakdown for two BEVs
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from datetime import time

# === File Paths (adjust filenames as needed) ===
FOLDER = Path("D:/Research vision/Papers/Paper 1/Submission/Dataset_repo/TotalEnergy_2years")
bev1_file = FOLDER / "FebdriveBEV1.csv"
bev2_file = FOLDER / "OctdriveBEV2.csv"

# === Columns to Use ===
cols = [
    'BMS_kwhAcChargeTotalModule1', 'BMS_kwhAcChargeTotalModule2',
    'BMS_kwhAcChargeTotalModule3', 'BMS_kwhAcChargeTotalModule4',
    'BMS_kwhDcChargeTotalModule1', 'BMS_kwhDcChargeTotalModule2',
    'BMS_kwhDcChargeTotalModule3', 'BMS_kwhDcChargeTotalModule4',
    'BMS_kwhRegenChargeTotal', 'BMS_kwhDriveDischargeTotal',
    'TotalDischargeKWh3D2', 'Odometer3B6'
]

# === Load Data ===
df1 = pd.read_csv(bev1_file, skiprows=[1])
df2 = pd.read_csv(bev2_file, skiprows=[1])
df1[cols] = df1[cols].apply(pd.to_numeric, errors='coerce')
df2[cols] = df2[cols].apply(pd.to_numeric, errors='coerce')

# === Component Extraction ===
def extract_components(df):
    ac = df.iloc[-1][[f'BMS_kwhAcChargeTotalModule{i}' for i in range(1, 5)]].sum()
    dc = df.iloc[-1][[f'BMS_kwhDcChargeTotalModule{i}' for i in range(1, 5)]].sum()
    regen = df['BMS_kwhRegenChargeTotal'].iloc[-1]
    drive = df['BMS_kwhDriveDischargeTotal'].iloc[-1]
    total_dis = df['TotalDischargeKWh3D2'].iloc[-1]
    odo = df['Odometer3B6'].iloc[-1]
    return ac/1000, dc/1000, regen/1000, drive/1000, (total_dis - drive)/1000, total_dis/1000, odo/1000

ac1, dc1, regen1, drive1, base1, dis1, odo1 = extract_components(df1)
ac2, dc2, regen2, drive2, base2, dis2, odo2 = extract_components(df2)

# === Plot ===
fig, axs = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
labels = ['Tot. Charge (MWh)', 'Tot. Discharge (MWh)', 'Distance (1000 km)']
x = [0, 1, 2]
width = 0.6

data = [
    {"ax": axs[0], "ac": ac1, "dc": dc1, "regen": regen1, "base": base1, "drive": drive1, "odo": odo1, "total_dis": dis1, "title": "BEV1"},
    {"ax": axs[1], "ac": ac2, "dc": dc2, "regen": regen2, "base": base2, "drive": drive2, "odo": odo2, "total_dis": dis2, "title": "BEV2"}
]

for d in data:
    ax = d["ax"]
    # Stacked Charging
    ac_b = ax.bar(x[0], d["ac"], width, color='orange', label="Slow")
    dc_b = ax.bar(x[0], d["dc"], width, bottom=d["ac"], color='lightcoral', label="Fast")
    re_b = ax.bar(x[0], d["regen"], width, bottom=d["ac"] + d["dc"], color='firebrick', label="Regen")

    # Stacked Discharge
    aux_b = ax.bar(x[1], d["base"], width, color='green', label="Aux")
    dr_b = ax.bar(x[1], d["drive"], width, bottom=d["base"], color='mediumseagreen', label="Drive")

    # Odometer
    odo_b = ax.bar(x[2], d["odo"], width, color='navy', label="Distance")

    # Annotate bars
    def annotate(bar, val, base=0):
        ax.text(bar[0].get_x() + bar[0].get_width() / 2, base + val / 2, f'{val:.1f}',
                ha='center', va='center', color='white', fontsize=9)

    annotate(ac_b, d["ac"])
    annotate(dc_b, d["dc"], base=d["ac"])
    annotate(re_b, d["regen"], base=d["ac"] + d["dc"])
    annotate(aux_b, d["base"])
    annotate(dr_b, d["drive"], base=d["base"])
    annotate(odo_b, d["odo"])

    # Totals
    ax.text(x[0], d["ac"] + d["dc"] + d["regen"] + 1.5, f"{d['ac'] + d['dc'] + d['regen']:.1f}", ha='center')
    ax.text(x[1], d["total_dis"] + 1.5, f"{d['total_dis']:.1f}", ha='center')
    ax.text(x[2], d["odo"] + 1.5, f"{d['odo']:.1f}", ha='center')
    ax.set_title(d["title"])
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 90)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

axs[0].legend(loc='upper left', bbox_to_anchor=(0, 1), ncol=2, fontsize=9)
axs[0].set_ylim(0, 70)
plt.tight_layout()
plt.savefig("fig15.png", dpi=600)
plt.show()

####################################################################################
####################################################################################
# FIGURE 16: Battery temperature vs ambient temperature scatter plot for charging and driving sessions

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import pearsonr

# === Config ===
var1 = "BMSmaxPackTemperature"
var2 = "VCFRONT_tempAmbient"
base_path = Path(r"D:\Research vision\Papers\Paper 1\Submission\Dataset_repo")

# Define which folders go into which subplot
subplot_data = [
    {
        "sessions": {
            "slow charging sessions": {"label": "Slow Charging", "color": "black"},
            "fast charging sessions": {"label": "Fast Charging", "color": "red"},
        }
    },
    {
        "sessions": {
            "driving sessions": {"label": "Driving", "color": "red"},
            "parking sessions": {"label": "Parking", "color": "black"},
        }
    }
]

# === Plot style ===
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.linewidth': 0.8,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'xtick.major.width': 0.7,
    'ytick.major.width': 0.7,
    'legend.frameon': False,
    'figure.dpi': 1200
})
sns.set_style("whitegrid")

# === Plot layout ===
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

# === Loop through subplots ===
for ax, subplot in zip(axes, subplot_data):
    for folder_name, props in subplot["sessions"].items():
        folder = base_path / folder_name
        data_frames = []

        for file in folder.glob("*.csv"):
            try:
                df = pd.read_csv(file, usecols=[var1, var2]).dropna()
                if not df.empty:
                    data_frames.append(df)
            except Exception:
                continue

        if data_frames:
            df_all = pd.concat(data_frames, ignore_index=True)
            corr, _ = pearsonr(df_all[var1], df_all[var2])
            ax.scatter(df_all[var1], df_all[var2],
                       s=8, color=props["color"], alpha=0.5,
                       label=f'{props["label"]} (r={corr:.2f})')

    # Axis labels
    ax.set_xlabel("Battery Temp (°C)",fontsize=12)
    if ax == axes[0]:
        ax.set_ylabel("Ambient Temp (°C)", fontsize=12)

    ax.legend(loc="upper left", fontsize=12)
    ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1] + 2)

# === Save figure ===
plt.tight_layout()
fig.savefig("fig16.png", dpi=1200, bbox_inches="tight", transparent=True)
plt.close()
####################################################################################
####################################################################################
'''


