#This script is prepared for processing the data in paper:

#Unveiling Energy Dynamics of Battery Electric Vehicle Using High-Resolution Data

#By: M. Yasko, A. Moussa, F. Tian, H. Kazmi, J. Driesen, and W. Martinez 
#From: KU Leuven/EnergyVille Thor Park 8310, Genk, Belgium.
#Last update: 2025-06-12

#Email: mohamed.yasko@kuleuven.be

'''
####################################################################################
####################################################################################
#1 RAW DATASET PROCESSING 

import os
import glob
import pandas as pd
import cantools
from asammdf import MDF
from datetime import timedelta

root_folder = r'D:\BEV\Rawdataset'
dbc_path = r'Model3CAN.dbc'
output_root = r'D:\BEV\ProcessedData'

signals_to_extract = [
    'RawBattCurrent132', 'BattVoltage132', 'BMSdissipation312',
    'BMSmaxPackTemperature', 'BMSminPackTemperature', 'ChargeLinePower264',
    'ChargeLineVoltage264', 'ChargeLineCurrent264', 'SOCave292',
    'TotalDischargeKWh3D2', 'TotalChargeKWh3D2', 'UI_Range',
    'VCFRONT_tempAmbient', 'VCRIGHT_tempAmbientRaw',
    'FCMinVlimit244', 'FCMaxVlimit244', 'FCCurrentLimit244', 'FCPowerLimit244',
    'FCMaxPowerLimit541', 'FCMaxCurrentLimit541', 'FC_dcCurrent', 'FC_dcVoltage',
    'BMS_kwhDriveDischargeTotal', 'BMS_kwhRegenChargeTotal', 'BMS_maxDischargePower',
    'BMS_maxRegenPower', 'DI_uiSpeed', 'Odometer3B6', 'UI_batteryPreconditioningRequest',
    'BMS_preconditionAllowed',
]

print("Loading DBC...")
db = cantools.database.load_file(dbc_path)

mf4_files = sorted(glob.glob(os.path.join(root_folder, '**', '*.MF4'), recursive=True))
if not mf4_files:
    print("⚠ No MF4 files found!")
    exit()

date_file_counter = {}
skipped_files = []

for mf4_path in mf4_files:
    try:
        print(f"\nProcessing file: {mf4_path}")
        mdf = MDF(mf4_path)
        mdf_start_time = mdf.header.start_time
        date_str = mdf_start_time.date().isoformat()
        month_str = mdf_start_time.strftime('%Y-%m')

        month_folder = os.path.join(output_root, month_str)
        date_folder = os.path.join(month_folder, date_str)
        os.makedirs(date_folder, exist_ok=True)

        file_count = date_file_counter.get(date_str, 0) + 1
        output_csv = os.path.join(date_folder, f'file{file_count}.csv')

        if os.path.exists(output_csv):
            print(f"⏭ Skipping {mf4_path} (already processed)")
            date_file_counter[date_str] = file_count
            continue

        signal_dfs = {}

        for group_index, group in enumerate(mdf.groups):
            channel_names = [ch.name for ch in group.channels]
            if 'CAN_DataFrame' not in channel_names:
                continue

            timestamps = mdf.get('CAN_DataFrame', group=group_index).timestamps
            ids = mdf.get('CAN_DataFrame.ID', group=group_index).samples.astype(int)
            data_bytes = mdf.get('CAN_DataFrame.DataBytes', group=group_index).samples

            for ts, can_id, data in zip(timestamps, ids, data_bytes):
                try:
                    frame_data = bytes(data)
                    msg = db.get_message_by_frame_id(can_id)
                    decoded = msg.decode(frame_data)
                    for signal_name, signal_value in decoded.items():
                        if signal_name not in signals_to_extract:
                            continue
                        if signal_name not in signal_dfs:
                            signal_dfs[signal_name] = []
                        signal_dfs[signal_name].append((ts, signal_value))
                except Exception:
                    continue

        if not signal_dfs:
            print(f"⚠ No target signals decoded in {mf4_path}")
            date_file_counter[date_str] = file_count
            continue

        combined_df = pd.DataFrame()

        for signal_name, samples in signal_dfs.items():
            temp_df = pd.DataFrame(samples, columns=['RelativeSeconds', signal_name])
            temp_df['Timestamp'] = [
                mdf_start_time + timedelta(seconds=float(s))
                for s in temp_df['RelativeSeconds']
            ]
            temp_df = temp_df.set_index('Timestamp').resample('1S').mean()
            temp_df = temp_df[[signal_name]]
            if combined_df.empty:
                combined_df = temp_df
            else:
                combined_df = combined_df.join(temp_df, how='outer')

        combined_df.reset_index(inplace=True)
        date_file_counter[date_str] = file_count

        combined_df.to_csv(output_csv, index=False)
        print(f"Saved: {output_csv}")

    except Exception as e:
        print(f"Error processing {mf4_path}: {e}")
        skipped_files.append(mf4_path)

if skipped_files:
    print("The following files were skipped due to errors:")
    for f in skipped_files:
        print(f" - {f}")
else:
    print("All files processed without errors.")
####################################################################################
####################################################################################
#2 SLOW CHARGING SESSIONS EXTRACTION

import os
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")


input_folder = Path("Newdataset_BEV2")
output_folder = Path("charging_sessions_BEV2")
output_folder.mkdir(parents=True, exist_ok=True)


def is_consecutive(date1, date2): # Check if two dates are consecutive
    return date2 - date1 == timedelta(days=1)


def group_consecutive_dates(date_list):# Group dates into consecutive sequences
    sorted_dates = sorted(set(date_list))
    grouped, temp = [], [sorted_dates[0]]
    for i in range(1, len(sorted_dates)):
        if is_consecutive(sorted_dates[i - 1], sorted_dates[i]):
            temp.append(sorted_dates[i])
        else:
            grouped.append(temp)
            temp = [sorted_dates[i]]
    grouped.append(temp)
    return grouped


def plot_charging_data(df, time_col, variables, label, save_dir):
    plt.figure(figsize=(14, 6))
    for var in variables:
        if var not in df.columns:
            print(f"⚠️ Column '{var}' not found in {label}")
            continue
        plt.scatter(df[time_col], df[var], marker='o', label=var)
    if 'DI_uiSpeed' in df.columns:
        plt.scatter(df[time_col], df['DI_uiSpeed'], linestyle='--', label='DI_uiSpeed', alpha=0.7)
    plt.xlabel(time_col)
    plt.ylabel('Value')
    plt.title(f"Charging Data Plot: {label}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{label}_plot.png"))
    plt.close()


def extract_all_charging_sessions(df, power_col='ChargeLinePower264', speed_col='DI_uiSpeed', threshold=0.5):
    df = df.copy()
    df[power_col] = df[power_col].fillna(0)
    df[speed_col] = df[speed_col].fillna(0)
    sessions = []
    df['segment_id'] = (df[speed_col] > 0).cumsum()
    print(f"🔍 Found {df['segment_id'].nunique()} non-driving segments")
    for seg_id, group in df.groupby('segment_id'):
        group = group.reset_index(drop=True)
        power = group[power_col].values
        rising_edges = (power[:-1] < threshold) & (power[1:] >= threshold)
        if not rising_edges.any():
            continue
        start_idx = rising_edges.argmax() + 1
        falling_region = power[start_idx:]
        falling_edges = (falling_region[1:] < threshold) & (falling_region[:-1] >= threshold)
        end_idx = start_idx + falling_edges.nonzero()[0][-1] + 1 if falling_edges.any() else len(group)
        session_df = group.iloc[start_idx:end_idx + 1].copy()
        if not session_df.empty:
            sessions.append(session_df)
    if not sessions:
        print("⚠️ No valid charging sessions found")
    return sessions


all_files = list(input_folder.rglob("*.csv"))
file_info = []

for file in all_files:
    try:
        df = pd.read_csv(file, usecols=['Timestamp'], parse_dates=['Timestamp'])
        df = df.dropna(subset=['Timestamp'])
        if df.empty:
            continue
        earliest_ts = df['Timestamp'].min()
        file_info.append((file, earliest_ts))
    except Exception as e:
        print(f"⚠️ Skipping {file.name} due to error: {e}")


monthly_files = defaultdict(list)
for file_path, timestamp in file_info:
    month_key = timestamp.strftime("%Y-%m")
    monthly_files[month_key].append(file_path)


for month_key, file_paths in monthly_files.items():
    print(f"\n📦 Processing month: {month_key}")
    month_output_folder = output_folder / month_key
    month_output_folder.mkdir(parents=True, exist_ok=True)

   
    dfs = []
    for file in file_paths:
        try:
            df = pd.read_csv(file, parse_dates=['Timestamp'])
            if 'DI_uiSpeed' not in df.columns:
                df['DI_uiSpeed'] = 0.0
            dfs.append(df)
        except Exception as e:
            print(f"⚠️ Could not read {file.name}: {e}")
    if not dfs:
        continue

    full_df = pd.concat(dfs, ignore_index=True)
    full_df = full_df.dropna(subset=['Timestamp'])
    full_df = full_df.sort_values('Timestamp').reset_index(drop=True)
    full_df['Date'] = full_df['Timestamp'].dt.date

    
    grouped_dates = group_consecutive_dates(full_df['Date'].unique())

    
    for date_group in grouped_dates:
        mask = full_df['Date'].isin(date_group)
        group_df = full_df[mask].copy()
        group_df.drop(columns=['Date'], inplace=True)
        group_label = f"{month_key}_{date_group[0]}-{date_group[-1]}"
        combined_file = month_output_folder / f"{group_label}_combined.csv"
        group_df.to_csv(combined_file, index=False)
        print(f"✅ Saved combined data: {combined_file}")
        plot_charging_data(group_df, 'Timestamp', ['ChargeLinePower264', 'SOCave292', 'DI_uiSpeed'], f"{group_label}_combined", month_output_folder)

        if 'ChargeLinePower264' not in group_df.columns:
            print("❌ No ChargeLinePower264 column. Skipping session extraction.")
            continue

        sessions = extract_all_charging_sessions(group_df)
        for i, session_df in enumerate(sessions, 1):
            start_time = session_df['Timestamp'].iloc[0]
            end_time = session_df['Timestamp'].iloc[-1]
            session_date = start_time.strftime("%Y-%m-%d")
            suffix = "night" if end_time.date() > start_time.date() else "day"
            session_label = f"{session_date}-{suffix}"
            session_file = month_output_folder / f"{session_label}.csv"
            session_df.to_csv(session_file, index=False)
            print(f"✅ Saved session {i}: {session_file}")
            vars_to_plot = [v for v in ['ChargeLinePower264', 'SOCave292'] if v in session_df.columns]
            plot_charging_data(session_df, 'Timestamp', vars_to_plot, session_label, month_output_folder)
####################################################################################
####################################################################################
#3 FAST CHARGING SESSIONS EXTRACTION
import os
import pandas as pd

root_folder = r"D:\BEV_data\BEV1"  # Update this to your root folder
save_folder = r"D:\BEV_data\BEV1\BEV1_DC"

os.makedirs(save_folder, exist_ok=True)
session_counter = 1

for dirpath, _, filenames in os.walk(root_folder):
    for file in filenames:
        if 'combined' in file.lower() and file.lower().endswith('.csv'):
            try:
                file_path = os.path.join(dirpath, file)
                df = pd.read_csv(file_path)

                if 'FC_dcCurrent' not in df.columns:
                    print(f"⚠️ FC_dcCurrent not in {file}, skipping.")
                    continue

                # Filter where FC_dcCurrent > 0
                df_filtered = df[df['FC_dcCurrent'] > 0]

                if not df_filtered.empty:
                    save_name = f"BEV1_DC{session_counter}.csv"
                    df_filtered.to_csv(os.path.join(save_folder, save_name), index=False)
                    print(f"✅ Saved: {save_name}")
                    session_counter += 1

            except Exception as e:
                print(f"❌ Error processing {file}: {e}")
####################################################################################
####################################################################################
#4 DRIVING SESSIONS EXTRACTION
import os
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict


input_folder = Path("Newdataset_BEV2")
output_folder = Path("driving_sessions_BEV2")
output_folder.mkdir(exist_ok=True)


def plot_driving_data(df, time_col, label, save_dir):
    plt.figure(figsize=(14, 6))
    if 'DI_uiSpeed' in df.columns:
        plt.plot(df[time_col], df['DI_uiSpeed'], linestyle='-', marker='o', label='DI_uiSpeed')
    plt.xlabel(time_col)
    plt.ylabel('Speed')
    plt.title(f"Driving Session: {label}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    filename = save_dir / f"{label}_plot.png"
    plt.savefig(filename)
    plt.close()
    print(f"📈 Plot saved: {filename}")


def extract_driving_sessions_from_speed_files(csv_files, time_col='Timestamp', max_gap_seconds=60):
    dfs = []
    for f in csv_files:
        try:
            df = pd.read_csv(f)
            if 'DI_uiSpeed' in df.columns and 'Timestamp' in df.columns:
                df['source_file'] = f.name
                dfs.append(df)
        except Exception as e:
            print(f"⚠️ Failed to load {f.name}: {e}")
    if not dfs:
        print("❌ No driving-related files found.")
        return []

    df = pd.concat(dfs, ignore_index=True)
    df[time_col] = pd.to_datetime(df[time_col], utc=True, errors='coerce')
    df = df.dropna(subset=[time_col])
    df = df.sort_values(by=time_col).reset_index(drop=True)

    time_diff = df[time_col].diff().dt.total_seconds().fillna(0)
    df['segment_id'] = (time_diff > max_gap_seconds).cumsum()
    sessions = [group.drop(columns='segment_id') for _, group in df.groupby('segment_id') if not group.empty]
    print(f"✅ Extracted {len(sessions)} driving session(s).")
    return sessions

def group_consecutive_dates(date_list):
    sorted_dates = sorted(set(date_list))
    grouped, temp = [], [sorted_dates[0]]
    for i in range(1, len(sorted_dates)):
        if sorted_dates[i] - sorted_dates[i - 1] == timedelta(days=1):
            temp.append(sorted_dates[i])
        else:
            grouped.append(temp)
            temp = [sorted_dates[i]]
    grouped.append(temp)
    return grouped


all_csvs = list(input_folder.rglob("*.csv"))
file_dates = []


for csv in all_csvs:
    try:
        df = pd.read_csv(csv, usecols=['Timestamp'])
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], utc=True, errors='coerce')
        df = df.dropna(subset=['Timestamp'])
        if not df.empty:
            file_dates.append((csv, df['Timestamp'].min().date()))
    except Exception as e:
        print(f"⚠️ Skipping {csv.name}: {e}")


monthly_files = defaultdict(list)
for fpath, ts_date in file_dates:
    month_key = ts_date.strftime("%Y-%m")
    monthly_files[month_key].append((fpath, ts_date))


for month_key, entries in monthly_files.items():
    print(f"\n🔍 Processing month: {month_key}")
    month_output_folder = output_folder / month_key
    month_output_folder.mkdir(parents=True, exist_ok=True)

    
    files_by_date = defaultdict(list)
    for fpath, ts_date in entries:
        files_by_date[ts_date].append(fpath)

    
    date_groups = group_consecutive_dates(list(files_by_date.keys()))

    for date_group in date_groups:
        group_label = f"{month_key}_{date_group[0]}-{date_group[-1]}"
        group_files = []
        for date in date_group:
            group_files.extend(files_by_date[date])
        group_files = sorted(group_files)

       
        driving_sessions = extract_driving_sessions_from_speed_files(group_files)

        summary_rows = []
        session_counts = {}

        for drive_df in driving_sessions:
            start_time = drive_df['Timestamp'].iloc[0]
            end_time = drive_df['Timestamp'].iloc[-1]
            session_date = start_time.strftime("%Y-%m-%d")
            label_suffix = "night" if end_time.date() > start_time.date() else "day"
            base_label = f"{session_date}-{label_suffix}"

            session_counts.setdefault(base_label, 0)
            session_counts[base_label] += 1
            drive_label = f"{base_label}_drive{session_counts[base_label]}"

            drive_file = month_output_folder / f"{drive_label}.csv"
            drive_df.to_csv(drive_file, index=False)
            print(f"🚗 Saved: {drive_file}")

            plot_driving_data(drive_df, 'Timestamp', drive_label, month_output_folder)

            summary_rows.append({
                "session_file": drive_file.name,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "duration_seconds": (end_time - start_time).total_seconds(),
                "num_rows": len(drive_df),
                "max_speed": drive_df['DI_uiSpeed'].max(),
                "avg_speed": drive_df['DI_uiSpeed'].mean()
            })

        if summary_rows:
            summary_df = pd.DataFrame(summary_rows)
            summary_file = month_output_folder / f"{group_label}_summary.csv"
            summary_df.to_csv(summary_file, index=False)
            print(f"📝 Summary saved: {summary_file}")
####################################################################################
####################################################################################
#5 PARKING SESSIONS EXTRACTION
import os
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict

input_folder = Path("Newdataset_BEV2")
output_folder = Path("parking_sessions_BEV2")
charging_folder = Path("charging_sessions_BEV2")
output_folder.mkdir(parents=True, exist_ok=True)

def is_consecutive(date1, date2):
    return date2 - date1 == timedelta(days=1)

def group_consecutive_dates(date_list):
    sorted_dates = sorted(set(date_list))
    grouped, temp = [], [sorted_dates[0]]
    for i in range(1, len(sorted_dates)):
        if is_consecutive(sorted_dates[i - 1], sorted_dates[i]):
            temp.append(sorted_dates[i])
        else:
            grouped.append(temp)
            temp = [sorted_dates[i]]
    grouped.append(temp)
    return grouped

def load_session_intervals(session_folder: Path):
    intervals = []
    for f in session_folder.glob("*.csv"):
        name = f.name.lower()
        if name.endswith("combined.csv") or name.endswith("drive.csv"):
            continue
        try:
            df = pd.read_csv(f, usecols=['Timestamp'])
            df['Timestamp'] = pd.to_datetime(df['Timestamp'], utc=True)
            if not df.empty:
                intervals.append((df['Timestamp'].min(), df['Timestamp'].max()))
        except:
            continue
    return intervals

def extract_parking_sessions(full_df, used_intervals, time_col='Timestamp', max_gap_seconds=60):
    full_df = full_df.copy()
    full_df[time_col] = pd.to_datetime(full_df[time_col], utc=True)
    used_mask = pd.Series(False, index=full_df.index)
    for start, end in used_intervals:
        used_mask |= (full_df[time_col] >= start) & (full_df[time_col] <= end)
    parking_df = full_df[~used_mask].sort_values(by=time_col).reset_index(drop=True)
    if parking_df.empty:
        return []
    time_diff = parking_df[time_col].diff().dt.total_seconds().fillna(0)
    segment_id = (time_diff > max_gap_seconds).cumsum()
    parking_df['segment_id'] = segment_id
    sessions = [g.drop(columns='segment_id') for _, g in parking_df.groupby('segment_id') if not g.empty]
    print(f"🅿️ Found {len(sessions)} parking session(s).")
    return sessions

def plot_parking_data(df, time_col, label, save_dir):
    plt.figure(figsize=(14, 6))
    if 'ChargeLinePower264' in df.columns:
        plt.plot(df[time_col], df['ChargeLinePower264'], linestyle='--', marker='o', label='ChargeLinePower264', alpha=0.6)
    plt.xlabel(time_col)
    plt.ylabel('Value')
    plt.title(f"Parking Session: {label}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    file_path = save_dir / f"{label}_plot.png"
    plt.savefig(file_path)
    plt.close()
    print(f"📊 Plot saved: {file_path}")


def process_parking_only():
    # Step 1: Find all non-driving CSVs
    all_csvs = list(input_folder.rglob("*.csv"))
    non_drive_files = []
    for f in all_csvs:
        try:
            df_sample = pd.read_csv(f, nrows=1)
            if 'DI_uiSpeed' not in df_sample.columns:
                non_drive_files.append(f)
        except:
            continue

    
    date_file_map = defaultdict(list)
    for f in non_drive_files:
        try:
            df = pd.read_csv(f, usecols=['Timestamp'])
            df['Timestamp'] = pd.to_datetime(df['Timestamp'], utc=True)
            if not df.empty:
                date = df['Timestamp'].dt.date.min()
                date_file_map[date].append(f)
        except:
            continue

    
    month_dates_map = defaultdict(list)
    for date in date_file_map.keys():
        month = date.strftime("%Y-%m")
        month_dates_map[month].append(date)

    for month, dates in month_dates_map.items():
        print(f"\n🔍 Processing month: {month}")
        output_month_folder = output_folder / month
        output_month_folder.mkdir(parents=True, exist_ok=True)

        date_groups = group_consecutive_dates(dates)

        for group in date_groups:
            files = []
            for d in group:
                files.extend(date_file_map[d])

            dfs = []
            for f in files:
                try:
                    df = pd.read_csv(f)
                    dfs.append(df)
                except:
                    continue
            if not dfs:
                continue

            full_df = pd.concat(dfs, ignore_index=True)
            if 'Timestamp' not in full_df.columns:
                continue
            full_df['Timestamp'] = pd.to_datetime(full_df['Timestamp'], utc=True)
            full_df = full_df.sort_values('Timestamp').reset_index(drop=True)

            
            used_intervals = load_session_intervals(charging_folder / month)

           
            parking_sessions = extract_parking_sessions(full_df, used_intervals)

        
            session_counts = {}
            for session_df in parking_sessions:
                start_time = session_df['Timestamp'].iloc[0]
                end_time = session_df['Timestamp'].iloc[-1]
                session_date = start_time.strftime("%Y-%m-%d")
                suffix = "night" if end_time.date() > start_time.date() else "day"
                base_label = f"{session_date}-{suffix}_park"
                session_counts.setdefault(base_label, 0)
                session_counts[base_label] += 1
                session_label = f"{base_label}{session_counts[base_label]}"
                session_file = output_month_folder / f"{session_label}.csv"
                session_df.to_csv(session_file, index=False)
                print(f"✅ Saved: {session_file}")
                plot_parking_data(session_df, 'Timestamp', session_label, output_month_folder)

if __name__ == "__main__":
    process_parking_only()
####################################################################################
####################################################################################
 
'''