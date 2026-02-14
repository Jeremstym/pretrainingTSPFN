# import pandas as pd
# import csv
# import gzip
# import numpy as np
# import os


# def sepsis_maker():
#     path = "/data/stympopper/BenchmarkTSPFN/EICU-CRD"
#     path_dest = "/data/stympopper/BenchmarkTSPFN/processed/EICU_preprocessed"

#     # 1. Load the full list of patient IDs (The "Universe" of patients)
#     # We only need the ID column to keep memory low
#     df_patient = pd.read_csv(f'{path}/patient.csv.gz', usecols=['patientunitstayid'], compression='gzip')
#     all_ids = df_patient['patientunitstayid'].unique()

#     # 2. Extract Sepsis IDs from the diagnosis table
#     # We use case=False and na=False for safety
#     df_dx = pd.read_csv(f'{path}/diagnosis.csv.gz', usecols=['patientunitstayid', 'diagnosisstring'], compression='gzip')
#     sepsis_ids = df_dx[df_dx["diagnosisstring"].str.contains("sepsis", case=False, na=False)]['patientunitstayid'].unique()

#     # 3. Create the Label DataFrame
#     labels_df = pd.DataFrame({'patientunitstayid': all_ids})

#     # 4. Assign the Label: 1 if in sepsis_ids, 0 otherwise
#     labels_df['sepsis_label'] = labels_df['patientunitstayid'].isin(sepsis_ids).astype(int)

#     # 5. Save to a new CSV
#     labels_df.to_csv(f'{path_dest}/patient_sepsis_labels.csv', index=False)

#     # Verification
#     print(f"Total patients processed: {len(labels_df)}")
#     print(f"Sepsis cases (Label 1): {labels_df['sepsis_label'].sum()}")
#     print(f"Non-Sepsis cases (Label 0): {len(labels_df) - labels_df['sepsis_label'].sum()}")

# def status_maker():
#     path = "/data/stympopper/BenchmarkTSPFN/EICU-CRD"
#     path_dest = "/data/stympopper/BenchmarkTSPFN/processed/EICU_preprocessed"
#     # 1. Load the patient table 
#     # We need patientunitstayid (the unique stay) and hospitaldischargestatus (the outcome)
#     df_patient = pd.read_csv(f'{path}/patient.csv.gz', usecols=['patientunitstayid', 'hospitaldischargestatus'], compression='gzip')

#     # 2. Create the label
#     # We look specifically for the string 'Expired'
#     # Using .fillna('') to handle any missing values before the comparison
#     df_patient['mortality_label'] = (df_patient['hospitaldischargestatus'].fillna('') == 'Expired').astype(int)

#     # 3. Create a clean DataFrame with just the ID and the Label
#     mortality_labels = df_patient[['patientunitstayid', 'mortality_label']]

#     # 4. Save to CSV
#     mortality_labels.to_csv('patient_mortality_labels.csv', index=False)

#     # Verification Output
#     counts = mortality_labels['mortality_label'].value_counts()
#     print(f"Total Stays: {len(mortality_labels)}")
#     print(f"Class 0 (Survived Hospital): {counts[0]}")
#     print(f"Class 1 (Expired in Hospital): {counts[1]}")
#     print(f"Mortality Rate: {round((counts[1]/len(mortality_labels))*100, 2)}%")

# def respiratory_makers():
#     path = "/data/stympopper/BenchmarkTSPFN/EICU-CRD"
#     path_dest = "/data/stympopper/BenchmarkTSPFN/processed/EICU_preprocessed"
    
#     # 1. Load the full list of patient IDs (The "Universe" of patients)
#     # We only need the ID column to keep memory low
#     df_patient = pd.read_csv(f'{path}/patient.csv.gz', usecols=['patientunitstayid'], compression='gzip')
#     all_ids = df_patient['patientunitstayid'].unique()

#     # 2. Extract Sepsis IDs from the diagnosis table
#     # We use case=False and na=False for safety
#     df_dx = pd.read_csv(f'{path}/diagnosis.csv.gz', usecols=['patientunitstayid', 'diagnosisstring'], compression='gzip')
#     respiratory_failure_ids = df_dx[df_dx["diagnosisstring"].str.contains("respiratory failure", case=False, na=False)]['patientunitstayid'].unique()

#     # 3. Create the Label DataFrame
#     labels_df = pd.DataFrame({'patientunitstayid': all_ids})

#     # 4. Assign the Label: 1 if in respiratory_failure_ids, 0 otherwise
#     labels_df['respiratory_failure_label'] = labels_df['patientunitstayid'].isin(respiratory_failure_ids).astype(int)

#     # 5. Save to a new CSV
#     labels_df.to_csv(f'{path_dest}/patient_respiratory_failure_labels.csv', index=False)

#     # Verification
#     print(f"Total patients processed: {len(labels_df)}")
#     print(f"Respiratory Failure cases (Label 1): {labels_df['respiratory_failure_label'].sum()}")
#     print(f"Non-Respiratory Failure cases (Label 0): {len(labels_df) - labels_df['respiratory_failure_label'].sum()}")

# def hr_makers():

#     # --- SET YOUR PATHS HERE ---
#     PATH_VITAL_PERIODIC = '/data/stympopper/BenchmarkTSPFN/EICU-CRD/vitalPeriodic.csv.gz'
#     DESTINATION_FOLDER = '/data/stympopper/BenchmarkTSPFN/processed/EICU_preprocessed/heart_rate_ts'
#     # ---------------------------

#     if not os.path.exists(DESTINATION_FOLDER):
#         os.makedirs(DESTINATION_FOLDER)

#     def save_patient_data(pid, data_list, obs_offset_list):
#         """Helper to save the accumulated list as a numpy file."""
#         if not pid or not data_list:
#             return
        
#         # Convert list to numpy array (handling potential empty strings/missing values)
#         # We use float32 to save space
#         arr = np.array(data_list, dtype=np.float32)
#         obs = np.array(obs_offset_list, dtype=np.float32)
        
#         file_path = os.path.join(DESTINATION_FOLDER, f"{pid}.npz")
#         np.savez_compressed(file_path, heart_rate=arr, observation_offset=obs)

#     print("Starting extraction... This may take a while.")

#     with gzip.open(PATH_VITAL_PERIODIC, 'rt') as f:
#         reader = csv.DictReader(f)
        
#         current_patient_id = None
#         current_patient_data = []
        
#         for row in reader:
#             row_id = row['patientunitstayid']
#             observation_offset = row['observationoffset']
#             assert observation_offset.isdigit(), f"observationoffset is not a digit: {observation_offset}"
#             hr_value = row['heartrate']
            
#             # Clean the data: handle empty strings or nulls
#             try:
#                 val = float(hr_value) if hr_value else np.nan
#             except ValueError:
#                 val = np.nan

#             # Logic: If ID changes, save the accumulated data and start fresh
#             if row_id != current_patient_id:
#                 if current_patient_id is not None:
#                     save_patient_data(current_patient_id, current_patient_data, current_patient_obs_offsets)
                
#                 print(f"Processing patient ID: {row_id}")
#                 # Reset for the new patient
#                 current_patient_id = row_id
#                 current_patient_data = [val]
#                 current_patient_obs_offsets = [float(observation_offset)]
#             else:
#                 # Same patient, just append
#                 current_patient_data.append(val)
#                 current_patient_obs_offsets.append(float(observation_offset))
                
#         # Save the very last patient in the file
#         save_patient_data(current_patient_id, current_patient_data, current_patient_obs_offsets)

#     print(f"Extraction complete. Files saved in {DESTINATION_FOLDER}")

# def oxygen_makers():

#     # --- SET YOUR PATHS HERE ---
#     PATH_VITAL_PERIODIC = '/data/stympopper/BenchmarkTSPFN/EICU-CRD/vitalPeriodic.csv.gz'
#     DESTINATION_FOLDER = '/data/stympopper/BenchmarkTSPFN/processed/EICU_preprocessed/oxygen_saturation_ts'
#     # ---------------------------

#     if not os.path.exists(DESTINATION_FOLDER):
#         os.makedirs(DESTINATION_FOLDER)

#     def save_patient_data(pid, data_list, obs_offset_list):
#         """Helper to save the accumulated list as a numpy file."""
#         if not pid or not data_list:
#             return
        
#         # Convert list to numpy array (handling potential empty strings/missing values)
#         # We use float32 to save space
#         arr = np.array(data_list, dtype=np.float32)
#         obs = np.array(obs_offset_list, dtype=np.float32)
        
#         file_path = os.path.join(DESTINATION_FOLDER, f"{pid}.npz")
#         np.savez_compressed(file_path, heart_rate=arr, observation_offset=obs)

#     print("Starting extraction... This may take a while.")

#     with gzip.open(PATH_VITAL_PERIODIC, 'rt') as f:
#         reader = csv.DictReader(f)
        
#         current_patient_id = None
#         current_patient_data = []
#         current_patient_obs_offsets = []
        
#         for row in reader:
#             row_id = row['patientunitstayid']
#             observation_offset = row['observationoffset']
#             assert observation_offset.isdigit(), f"observationoffset is not a digit: {observation_offset}"
#             sao2_value = row['sao2']
            
#             # Clean the data: handle empty strings or nulls
#             try:
#                 val = float(sao2_value) if sao2_value else np.nan
#             except ValueError:
#                 val = np.nan

#             # Logic: If ID changes, save the accumulated data and start fresh
#             if row_id != current_patient_id:
#                 if current_patient_id is not None:
#                     save_patient_data(current_patient_id, current_patient_data, current_patient_obs_offsets)
                
#                 print(f"Processing patient ID: {row_id}")
#                 # Reset for the new patient
#                 current_patient_id = row_id
#                 current_patient_data = [val]
#                 current_patient_obs_offsets = [float(observation_offset)]
#             else:
#                 # Same patient, just append
#                 current_patient_data.append(val)
#                 current_patient_obs_offsets.append(float(observation_offset))
                
#         # Save the very last patient in the file
#         save_patient_data(current_patient_id, current_patient_data, current_patient_obs_offsets)

#     print(f"Extraction complete. Files saved in {DESTINATION_FOLDER}")

# def respiration_rate_makers():

#     # --- SET YOUR PATHS HERE ---
#     PATH_VITAL_PERIODIC = '/data/stympopper/BenchmarkTSPFN/EICU-CRD/vitalPeriodic.csv.gz'
#     DESTINATION_FOLDER = '/data/stympopper/BenchmarkTSPFN/processed/EICU_preprocessed/respiration_rate_ts'
#     # ---------------------------

#     if not os.path.exists(DESTINATION_FOLDER):
#         os.makedirs(DESTINATION_FOLDER)

#     def save_patient_data(pid, data_list, obs_offset_list):
#         """Helper to save the accumulated list as a numpy file."""
#         if not pid or not data_list:
#             return
        
#         # Convert list to numpy array (handling potential empty strings/missing values)
#         # We use float32 to save space
#         arr = np.array(data_list, dtype=np.float32)
#         obs = np.array(obs_offset_list, dtype=np.float32)
        
#         file_path = os.path.join(DESTINATION_FOLDER, f"{pid}.npz")
#         np.savez_compressed(file_path, heart_rate=arr, observation_offset=obs)

#     print("Starting extraction... This may take a while.")

#     with gzip.open(PATH_VITAL_PERIODIC, 'rt') as f:
#         reader = csv.DictReader(f)
        
#         current_patient_id = None
#         current_patient_data = []
#         current_patient_obs_offsets = []
        
#         for row in reader:
#             row_id = row['patientunitstayid']
#             respiration_value = row['respiration']
#             observation_offset = row['observationoffset']
            
#             assert observation_offset.isdigit(), f"observationoffset is not a digit: {observation_offset}"
#             current_patient_obs_offsets.append(float(observation_offset))
#             # Clean the data: handle empty strings or nulls
#             try:
#                 val = float(respiration_value) if respiration_value else np.nan
#             except ValueError:
#                 val = np.nan

#             # Logic: If ID changes, save the accumulated data and start fresh
#             if row_id != current_patient_id:
#                 if current_patient_id is not None:
#                     save_patient_data(current_patient_id, current_patient_data, current_patient_obs_offsets)
                
#                 print(f"Processing patient ID: {row_id}")
#                 # Reset for the new patient
#                 current_patient_id = row_id
#                 current_patient_data = [val]
#                 current_patient_obs_offsets = [float(observation_offset)]
#             else:
#                 # Same patient, just append
#                 current_patient_data.append(val)
#                 current_patient_obs_offsets.append(float(observation_offset))
                
#         # Save the very last patient in the file
#         save_patient_data(current_patient_id, current_patient_data, current_patient_obs_offsets)

#     print(f"Extraction complete. Files saved in {DESTINATION_FOLDER}")

# def bp_rate_makers():

#     # --- SET YOUR PATHS HERE ---
#     PATH_VITAL_PERIODIC = '/data/stympopper/BenchmarkTSPFN/EICU-CRD/vitalPeriodic.csv.gz'
#     DESTINATION_FOLDER = '/data/stympopper/BenchmarkTSPFN/processed/EICU_preprocessed/blood_pressure_ts'
#     # ---------------------------

#     if not os.path.exists(DESTINATION_FOLDER):
#         os.makedirs(DESTINATION_FOLDER)

#     def save_patient_data(pid, data_list, obs_offset_list):
#         """Helper to save the accumulated list as a numpy file."""
#         if not pid or not data_list:
#             return
        
#         # Convert list to numpy array (handling potential empty strings/missing values)
#         # We use float32 to save space
#         arr = np.array(data_list, dtype=np.float32)
#         obs = np.array(obs_offset_list, dtype=np.float32)
        
#         file_path = os.path.join(DESTINATION_FOLDER, f"{pid}.npz")
#         np.savez_compressed(file_path, heart_rate=arr, observation_offset=obs)

#     print("Starting extraction... This may take a while.")

#     with gzip.open(PATH_VITAL_PERIODIC, 'rt') as f:
#         reader = csv.DictReader(f)
        
#         current_patient_id = None
#         current_patient_data = []
#         current_patient_obs_offsets = []
        
#         for row in reader:
#             row_id = row['patientunitstayid']
#             observation_offset = row['observationoffset']
#             assert observation_offset.isdigit(), f"observationoffset is not a digit: {observation_offset}"
#             current_patient_obs_offsets.append(float(observation_offset))
#             bp_value = row['systemicmean']
            
#             # Clean the data: handle empty strings or nulls
#             try:
#                 val = float(bp_value) if bp_value else np.nan
#             except ValueError:
#                 val = np.nan

#             # Logic: If ID changes, save the accumulated data and start fresh
#             if row_id != current_patient_id:
#                 if current_patient_id is not None:
#                     save_patient_data(current_patient_id, current_patient_data, current_patient_obs_offsets)
                
#                 print(f"Processing patient ID: {row_id}")
#                 # Reset for the new patient
#                 current_patient_id = row_id
#                 current_patient_data = [val]
#                 current_patient_obs_offsets = [float(observation_offset)]
#             else:
#                 # Same patient, just append
#                 current_patient_data.append(val)
#                 current_patient_obs_offsets.append(float(observation_offset))
                
#         # Save the very last patient in the file
#         save_patient_data(current_patient_id, current_patient_data, current_patient_obs_offsets)

#     print(f"Extraction complete. Files saved in {DESTINATION_FOLDER}")

# def temperature_makers():

#     # --- SET YOUR PATHS HERE ---
#     PATH_VITAL_PERIODIC = '/data/stympopper/BenchmarkTSPFN/EICU-CRD/vitalPeriodic.csv.gz'
#     DESTINATION_FOLDER = '/data/stympopper/BenchmarkTSPFN/processed/EICU_preprocessed/temperature_ts'
#     # ---------------------------

#     if not os.path.exists(DESTINATION_FOLDER):
#         os.makedirs(DESTINATION_FOLDER)

#     def save_patient_data(pid, data_list, obs_offset_list):
#         """Helper to save the accumulated list as a numpy file."""
#         if not pid or not data_list:
#             return
        
#         # Convert list to numpy array (handling potential empty strings/missing values)
#         # We use float32 to save space
#         arr = np.array(data_list, dtype=np.float32)
#         obs = np.array(obs_offset_list, dtype=np.float32)
        
#         file_path = os.path.join(DESTINATION_FOLDER, f"{pid}.npz")
#         np.savez_compressed(file_path, temperature=arr, observation_offset=obs)

#     print("Starting extraction... This may take a while.")

#     with gzip.open(PATH_VITAL_PERIODIC, 'rt') as f:
#         reader = csv.DictReader(f)
        
#         current_patient_id = None
#         current_patient_data = []
#         current_patient_obs_offsets = []
        
#         for row in reader:
#             row_id = row['patientunitstayid']
#             temperature_value = row['temperature']
#             observation_offset = row['observationoffset']
            
#             # Clean the data: handle empty strings or nulls
#             try:
#                 val = float(temperature_value) if temperature_value else np.nan
#             except ValueError:
#                 val = np.nan

#             # Logic: If ID changes, save the accumulated data and start fresh
#             if row_id != current_patient_id:
#                 if current_patient_id is not None:
#                     save_patient_data(current_patient_id, current_patient_data, current_patient_obs_offsets)
                
#                 print(f"Processing patient ID: {row_id}")
#                 # Reset for the new patient
#                 current_patient_id = row_id
#                 current_patient_data = [val]
#                 current_patient_obs_offsets = [float(observation_offset)]
#             else:
#                 # Same patient, just append
#                 current_patient_data.append(val)
#                 current_patient_obs_offsets.append(float(observation_offset))
                
#         # Save the very last patient in the file
#         save_patient_data(current_patient_id, current_patient_data, current_patient_obs_offsets)

#     print(f"Extraction complete. Files saved in {DESTINATION_FOLDER}")

# if __name__ == "__main__":
#     # sepsis_maker()
#     # status_maker()
#     # respiratory_makers()
#     # hr_makers()
#     # oxygen_makers()
#     # respiration_rate_makers()
#     # bp_rate_makers()
#     temperature_makers()


# import os
# import numpy as np
# import pandas as pd
# from tqdm import tqdm

# def find_truly_valid_patients(folders_dict, patient_list):
#     """
#     folders_dict: {'hr': 'path/hr_folder', 'temp': 'path/temp_folder', ...}
#     patient_list: List or Series of IDs to check
#     """
#     valid_ids = []

#     print("Scanning .npz files for non-NaN data...")
#     for pid in tqdm(patient_list):
#         is_patient_valid = True
        
#         # Check every vital folder for this patient
#         for var_name, folder_path in folders_dict.items():
#             file_path = os.path.join(folder_path, f"{pid}.npz")
            
#             # 1. Check if file exists
#             if not os.path.exists(file_path):
#                 is_patient_valid = False
#                 break
            
#             # 2. Load the file (Lazy loading)
#             try:
#                 with np.load(file_path) as data:
#                     # Replace 'data.files[0]' with your actual key if it's constant, e.g., 'heart_rate'
#                     # np.all(np.isnan(...)) returns True if the whole array is NaN
#                     array_key = data.files[0] 
#                     if np.all(np.isnan(data[array_key])) or data[array_key].size == 0:
#                         is_patient_valid = False
#                         break
#             except Exception:
#                 is_patient_valid = False
#                 break
        
#         if is_patient_valid:
#             valid_ids.append(pid)

#     return valid_ids

# if __name__ == "__main__":
#     # --- EXECUTION ---
#     folders = {
#         'heart_rate': '/data/stympopper/BenchmarkTSPFN/processed/EICU_preprocessed/heart_rate_ts',
#         'respiration_rate': '/data/stympopper/BenchmarkTSPFN/processed/EICU_preprocessed/respiration_rate_ts',
#         'spo2': '/data/stympopper/BenchmarkTSPFN/processed/EICU_preprocessed/oxygen_saturation_ts',
#         'blood_pressure': '/data/stympopper/BenchmarkTSPFN/processed/EICU_preprocessed/blood_pressure_ts',
#         'temperature': '/data/stympopper/BenchmarkTSPFN/processed/EICU_preprocessed/temperature_ts'
#     }

#     # Load the current index you have
#     df_all_registered = pd.read_csv('/data/stympopper/BenchmarkTSPFN/processed/EICU_preprocessed/patient_mortality_labels.csv')
#     all_pids = df_all_registered['patientunitstayid'].tolist()

#     # Run the deep check
#     final_valid_pids = find_truly_valid_patients(folders, all_pids)

#     # Save the real index
#     df_final = pd.DataFrame(final_valid_pids, columns=['patientunitstayid'])
#     df_final.to_csv('/data/stympopper/BenchmarkTSPFN/processed/EICU_preprocessed/final_non_empty_patients.csv', index=False)

#     print(f"\nFinal count: {len(df_final)} patients have actual data for all variables.")


import pandas as pd
import csv
import gzip
import numpy as np
import os
from tqdm import tqdm

import pandas as pd
import csv
import gzip
import numpy as np
import os
from tqdm import tqdm

def extract_multi_channel_vitals(min_valid_points=100):
    """
    Processes vitalPeriodic once, aligns 5 channels, and skips low-quality stays.
    min_valid_points: Minimum non-NaN values required PER CHANNEL to save the patient.
    """
    PATH_VITAL_PERIODIC = '/data/stympopper/BenchmarkTSPFN/EICU-CRD/vitalPeriodic.csv.gz'
    DESTINATION_FOLDER = '/data/stympopper/BenchmarkTSPFN/processed/EICU_preprocessed/multi_channel_ts'
    
    CHANNELS = {
        'heartrate': 0,
        'respiration': 1,
        'sao2': 2,
        'systemicmean': 3,
        'temperature': 4
    }
    CHANNEL_NAMES = ['heart_rate', 'respiration', 'spo2', 'blood_pressure', 'temperature']
    
    if not os.listdir(DESTINATION_FOLDER): # Only create if empty/missing
        os.makedirs(DESTINATION_FOLDER, exist_ok=True)

    def save_patient_bundle(pid, records):
        if not pid or not records:
            return
        
        records.sort(key=lambda x: x['offset'])
        offsets = np.array([r['offset'] for r in records], dtype=np.float32)
        values = np.full((len(records), len(CHANNELS)), np.nan, dtype=np.float32)
        
        for i, rec in enumerate(records):
            for col_name, ch_idx in CHANNELS.items():
                values[i, ch_idx] = rec[col_name]
        
        # --- QUALITY CONTROL CHECK ---
        # Count non-NaN values for each channel
        valid_counts = np.count_nonzero(~np.isnan(values), axis=0)
        
        # If ANY channel has fewer than min_valid_points, we skip this patient
        if np.any(valid_counts < min_valid_points):
            # Optimization: could log skipped IDs to a file if needed
            return 

        file_path = os.path.join(DESTINATION_FOLDER, f"{pid}.npz")
        np.savez_compressed(
            file_path, 
            data=values, 
            offsets=offsets, 
            columns=CHANNEL_NAMES
        )

    print(f"Starting Extraction (Threshold: {min_valid_points} points per channel)...")

    with gzip.open(PATH_VITAL_PERIODIC, 'rt') as f:
        reader = csv.DictReader(f)
        current_pid = None
        current_records = []
        
        # Use a status bar that updates based on line count if possible, 
        # or just a periodic print
        for row in reader:
            pid = row['patientunitstayid']
            try:
                record = {
                    'offset': float(row['observationoffset']),
                    'heartrate': float(row['heartrate']) if row['heartrate'] else np.nan,
                    'respiration': float(row['respiration']) if row['respiration'] else np.nan,
                    'sao2': float(row['sao2']) if row['sao2'] else np.nan,
                    'systemicmean': float(row['systemicmean']) if row['systemicmean'] else np.nan,
                    'temperature': float(row['temperature']) if row['temperature'] else np.nan
                }
            except ValueError:
                continue

            if pid != current_pid:
                if current_pid is not None:
                    save_patient_bundle(current_pid, current_records)
                current_pid = pid
                current_records = [record]
            else:
                current_records.append(record)
                
        save_patient_bundle(current_pid, current_records)

    print(f"Done! Multi-channel files saved in {DESTINATION_FOLDER}")

if __name__ == "__main__":
    extract_multi_channel_vitals()