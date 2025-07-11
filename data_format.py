# import os
# import glob
# import pandas as pd
# from datetime import datetime

# # 1) Get all .wav files in the directory
# audio_dir = 'audio_2021_chunk_1'
# wav_files = glob.glob(os.path.join(audio_dir, '*.wav')) + glob.glob(os.path.join(audio_dir, '*.WAV'))

# # 2) Build a DataFrame of raw Python datetimes & hive IDs
# records = []
# for p in wav_files:
#     fname = os.path.basename(p)
#     base = fname.rsplit('.',1)[0]
#     date_str, time_str, hive_str = base.split('_')
#     dt = datetime.strptime(f"{date_str}_{time_str}", "%d-%m-%Y_%Hh%M")
#     hive_id = int(hive_str.split('-')[-1])
#     records.append({'audio_path': p, 'audio_datetime': pd.to_datetime(dt), 'hive_id': hive_id})

# audio_df = pd.DataFrame(records)

# # 3) Load and prepare inspections
# ins = pd.read_csv('inspections_2021.csv', parse_dates=['Date'])
# ins = ins.rename(columns={'Tag number':'hive_id'})
# ins = ins.sort_values(['hive_id','Date']).reset_index(drop=True)

# # 4) Perform groupwise asof merge
# merged_list = []
# for hive, group in audio_df.groupby('hive_id'):
#     grp = group.sort_values('audio_datetime').reset_index(drop=True)
#     ins_h = ins[ins.hive_id == hive][['hive_id','Date','Colony Size','Fob 1st','Fob 2nd','Fob 3rd','Queen status','Frames of Honey']]
#     if ins_h.empty:
#         # no inspections for this hive, skip or fill with NaNs
#         grp[['Colony Size','Fob 1st','Fob 2nd','Fob 3rd','Queen status','Frames of Honey']] = pd.NA
#         merged_list.append(grp)
#     else:
#         merged_h = pd.merge_asof(
#             grp, 
#             ins_h.rename(columns={'Date':'inspection_datetime'}),
#             left_on='audio_datetime',
#             right_on='inspection_datetime',
#             direction='backward',
#             by='hive_id'
#         )   
#         merged_list.append(merged_h)

# merged = pd.concat(merged_list, ignore_index=True)

# # 5) Final column selection & renaming
# labels = merged[[
#     'audio_path','audio_datetime','hive_id',
#     'Colony Size','Fob 1st','Fob 2nd','Fob 3rd',
#     'Queen status','Frames of Honey'
# ]].rename(columns={
#     'audio_datetime':'datetime',
#     'Colony Size':'colony_size',
#     'Fob 1st':'frames_box1',
#     'Fob 2nd':'frames_box2',
#     'Fob 3rd':'frames_box3',
#     'Queen status':'queen_status',
#     'Frames of Honey':'frames_honey'
# })

# # 6) Inspect & save
# print(labels.head(10))
# labels.to_csv('audio_file_labels_asof.csv', index=False)

import os
import glob
import pandas as pd
from datetime import datetime

# 1) Find all chunk folders and all .wav/.WAV files in them
chunk_dirs = [d for d in os.listdir('.') if d.startswith('audio_2021_chunk')]
wav_files = []
for d in chunk_dirs:
    wav_files += glob.glob(os.path.join(d, '*.wav')) + glob.glob(os.path.join(d, '*.WAV'))

# 2) Build a DataFrame of raw Python datetimes & hive IDs
records = []
for p in wav_files:
    fname = os.path.basename(p)
    base = fname.rsplit('.', 1)[0]
    try:
        date_str, time_str, hive_str = base.split('_')
        dt = datetime.strptime(f"{date_str}_{time_str}", "%d-%m-%Y_%Hh%M")
        hive_id = int(hive_str.split('-')[-1])
        records.append({'audio_path': p, 'audio_datetime': pd.to_datetime(dt), 'hive_id': hive_id})
    except Exception as e:
        print(f"Skipping file {p}: {e}")

audio_df = pd.DataFrame(records)

# 3) Load and prepare inspections
ins = pd.read_csv('inspections_2021.csv', parse_dates=['Date'])
ins = ins.rename(columns={'Tag number': 'hive_id'})
ins = ins.sort_values(['hive_id', 'Date']).reset_index(drop=True)

# 4) Perform groupwise asof merge
merged_list = []
for hive, group in audio_df.groupby('hive_id'):
    grp = group.sort_values('audio_datetime').reset_index(drop=True)
    ins_h = ins[ins.hive_id == hive][['hive_id', 'Date', 'Colony Size', 'Fob 1st', 'Fob 2nd', 'Fob 3rd', 'Queen status', 'Frames of Honey']]
    if ins_h.empty:
        grp[['Colony Size', 'Fob 1st', 'Fob 2nd', 'Fob 3rd', 'Queen status', 'Frames of Honey']] = pd.NA
        merged_list.append(grp)
    else:
        merged_h = pd.merge_asof(
            grp,
            ins_h.rename(columns={'Date': 'inspection_datetime'}),
            left_on='audio_datetime',
            right_on='inspection_datetime',
            direction='backward',
            by='hive_id'
        )
        merged_list.append(merged_h)

merged = pd.concat(merged_list, ignore_index=True)

# 5) Final column selection & renaming
labels = merged[[
    'audio_path', 'audio_datetime', 'hive_id',
    'Colony Size', 'Fob 1st', 'Fob 2nd', 'Fob 3rd',
    'Queen status', 'Frames of Honey'
]].rename(columns={
    'audio_datetime': 'datetime',
    'Colony Size': 'colony_size',
    'Fob 1st': 'frames_box1',
    'Fob 2nd': 'frames_box2',
    'Fob 3rd': 'frames_box3',
    'Queen status': 'queen_status',
    'Frames of Honey': 'frames_honey'
})

# 6) Inspect & save
print(labels.head(10))
labels.to_csv('audio_file_labels_asof.csv', index=False)

