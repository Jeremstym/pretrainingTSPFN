# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# By Wei-Bang Jiang
# Based on BIOT code base
# https://github.com/ycq091044/BIOT
# --------------------------------------------------------
import mne
import numpy as np
import pandas as pd
import os
import pickle
from tqdm import tqdm

"""
https://github.com/Abhishaike/EEG_Event_Classification
"""

drop_channels = [
    "PHOTIC-REF",
    "IBI",
    "BURSTS",
    "SUPPR",
    "EEG ROC-REF",
    "EEG LOC-REF",
    "EEG EKG1-REF",
    "EMG-REF",
    "EEG C3P-REF",
    "EEG C4P-REF",
    "EEG SP1-REF",
    "EEG SP2-REF",
    "EEG LUC-REF",
    "EEG RLC-REF",
    "EEG RESP1-REF",
    "EEG RESP2-REF",
    "EEG EKG-REF",
    "RESP ABDOMEN-REF",
    "ECG EKG-REF",
    "PULSE RATE",
    "EEG PG2-REF",
    "EEG PG1-REF",
]
drop_channels.extend([f"EEG {i}-REF" for i in range(20, 129)])
chOrder_standard = [
    "EEG FP1-REF",
    "EEG FP2-REF",
    "EEG F3-REF",
    "EEG F4-REF",
    "EEG C3-REF",
    "EEG C4-REF",
    "EEG P3-REF",
    "EEG P4-REF",
    "EEG O1-REF",
    "EEG O2-REF",
    "EEG F7-REF",
    "EEG F8-REF",
    "EEG T3-REF",
    "EEG T4-REF",
    "EEG T5-REF",
    "EEG T6-REF",
    "EEG A1-REF",
    "EEG A2-REF",
    "EEG FZ-REF",
    "EEG CZ-REF",
    "EEG PZ-REF",
    "EEG T1-REF",
    "EEG T2-REF",
]

# Map your label IDs to a priority (Higher number = Higher priority)
# Note: Check your specific mapping, but usually:
# SPSW/GPED/PLED are the most important.
priority_map = {
    1: 10,  # GPED (High priority)
    2: 10,  # PLED
    3: 10,  # SPSW
    4: 10,  # TRIP
    5: 5,  # ARTF / EYEM (Medium priority)
    0: 1,  # BCKG (Low priority)
}


def convert_signals(signals, Rawdata):
    signal_names = {k: v for (k, v) in zip(Rawdata.info["ch_names"], list(range(len(Rawdata.info["ch_names"]))))}
    new_signals = np.vstack(
        (
            signals[signal_names["EEG FP1-REF"]] - signals[signal_names["EEG F7-REF"]],  # 0
            (signals[signal_names["EEG F7-REF"]] - signals[signal_names["EEG T3-REF"]]),  # 1
            (signals[signal_names["EEG T3-REF"]] - signals[signal_names["EEG T5-REF"]]),  # 2
            (signals[signal_names["EEG T5-REF"]] - signals[signal_names["EEG O1-REF"]]),  # 3
            (signals[signal_names["EEG FP2-REF"]] - signals[signal_names["EEG F8-REF"]]),  # 4
            (signals[signal_names["EEG F8-REF"]] - signals[signal_names["EEG T4-REF"]]),  # 5
            (signals[signal_names["EEG T4-REF"]] - signals[signal_names["EEG T6-REF"]]),  # 6
            (signals[signal_names["EEG T6-REF"]] - signals[signal_names["EEG O2-REF"]]),  # 7
            (signals[signal_names["EEG FP1-REF"]] - signals[signal_names["EEG F3-REF"]]),  # 14
            (signals[signal_names["EEG F3-REF"]] - signals[signal_names["EEG C3-REF"]]),  # 15
            (signals[signal_names["EEG C3-REF"]] - signals[signal_names["EEG P3-REF"]]),  # 16
            (signals[signal_names["EEG P3-REF"]] - signals[signal_names["EEG O1-REF"]]),  # 17
            (signals[signal_names["EEG FP2-REF"]] - signals[signal_names["EEG F4-REF"]]),  # 18
            (signals[signal_names["EEG F4-REF"]] - signals[signal_names["EEG C4-REF"]]),  # 19
            (signals[signal_names["EEG C4-REF"]] - signals[signal_names["EEG P4-REF"]]),  # 20
            (signals[signal_names["EEG P4-REF"]] - signals[signal_names["EEG O2-REF"]]),
        )
    )  # 21

    keep_channels = [0, 1, 2, 3, 4, 5, 6, 7, 14, 15, 16, 17, 18, 19, 20, 21]
    return new_signals, keep_channels


def BuildEvents(signals, times, EventData, keep_channels):
    """
    signals: (22, timestamps) or (23, timestamps)
    EventData: [.rec file contents]
    keep_channels: [0, 1, 2, 3, 4, 5, 6, 7, 14, 15, 16, 17, 18, 19, 20, 21]
    """
    fs = 200.0
    window_samples = int(fs * 5)

    # 1. Filter EventData to only include rows from the montages we care about
    mask = np.isin(EventData[:, 0], keep_channels)
    filtered_EventData = EventData[mask]

    # 2. Deduplicate based on Start and End times
    df = pd.DataFrame(filtered_EventData, columns=["chan", "start", "end", "label_id"])
    df["priority"] = df["label_id"].map(priority_map).fillna(0)
    df_sorted = df.sort_values(by=["start", "end", "priority"], ascending=[True, True, False])
    # df_unique = df.drop_duplicates(subset=["start", "end", "label_id"])
    df_unique = df_sorted.drop_duplicates(
        subset=["start", "end"], keep="first"
    )  # We remove label_id to avoid multiple labels for same event
    unique_event_data = df_unique.drop(columns=["priority"]).to_numpy()

    numEvents = len(unique_event_data)

    # We only take the rows of the signal that are in keep_channels
    if signals.shape[0] > len(keep_channels):
        raise ValueError("Signals shape does not match keep_channels length")
    else:
        selected_signals = signals
    numChan = len(keep_channels)

    features = np.zeros([numEvents, numChan, window_samples])
    offending_channel = np.zeros([numEvents, 1])
    labels = np.zeros([numEvents, 1])

    # EDGE PADDING (Safer context than triple buffer)
    pad_width = int(fs * 5)
    signals_padded = np.pad(selected_signals, ((0, 0), (pad_width, pad_width)), mode="edge")

    for i in range(numEvents):
        t_start = unique_event_data[i, 1]
        t_end = unique_event_data[i, 2]

        # Get indices from the times array
        idx_start = np.searchsorted(times, t_start)
        idx_end = np.searchsorted(times, t_end)

        # Center the 5s window on the actual event duration
        center_idx = (idx_start + idx_end) // 2

        # Calculate slice boundaries relative to padded signal
        # start_slice = (center_idx + pad_width) - (window_samples // 2)
        # end_slice = start_slice + window_samples
        start_slice = pad_width + idx_start - 2 * int(fs)
        end_slice = pad_width + idx_end + 2 * int(fs)
        assert end_slice - start_slice == window_samples, "Slice length mismatch"

        features[i, :, :] = signals_padded[:, start_slice:end_slice]

        # Save metadata
        offending_channel[i, :] = int(unique_event_data[i, 0])
        labels[i, :] = int(unique_event_data[i, 3])

    return [features, offending_channel, labels]


def readEDF(fileName):
    Rawdata = mne.io.read_raw_edf(fileName, preload=True)
    if drop_channels is not None:
        useless_chs = []
        for ch in drop_channels:
            if ch in Rawdata.ch_names:
                useless_chs.append(ch)
        Rawdata.drop_channels(useless_chs)
    if chOrder_standard is not None and len(chOrder_standard) == len(Rawdata.ch_names):
        Rawdata.reorder_channels(chOrder_standard)
    if Rawdata.ch_names != chOrder_standard:
        raise ValueError

    Rawdata.filter(l_freq=0.1, h_freq=75.0)
    Rawdata.notch_filter(50.0)
    Rawdata.resample(200, n_jobs=5)

    _, times = Rawdata[:]
    signals = Rawdata.get_data(units="uV")
    signals /= 100.0  # Normalize to 0.1mV units
    RecFile = fileName[0:-3] + "rec"
    eventData = np.genfromtxt(RecFile, delimiter=",")
    Rawdata.close()
    return [signals, times, eventData, Rawdata]


def load_up_objects(BaseDir, Features, OffendingChannels, Labels, OutDir):
    for dirName, subdirList, fileList in tqdm(os.walk(BaseDir)):
        print("Found directory: %s" % dirName)
        for fname in fileList:
            if fname[-4:] == ".edf":
                print("\t%s" % fname)
                try:
                    [signals, times, event, Rawdata] = readEDF(
                        dirName + "/" + fname
                    )  # event is the .rec file in the form of an array
                    signals, keep_channels = convert_signals(signals, Rawdata)
                except (ValueError, KeyError):
                    print("something funky happened in " + dirName + "/" + fname)
                    continue
                signals, offending_channels, labels = BuildEvents(signals, times, event, keep_channels)
                for idx, (signal, offending_channel, label) in enumerate(zip(signals, offending_channels, labels)):
                    sample = {
                        "signal": signal,
                        "offending_channel": offending_channel,
                        "label": label,
                    }
                    save_pickle(
                        sample,
                        os.path.join(OutDir, fname.split(".")[0] + "-" + str(idx) + ".pkl"),
                    )

    return Features, Labels, OffendingChannels


def save_pickle(object, filename):
    with open(filename, "wb") as f:
        pickle.dump(object, f)


if __name__ == "__main__":

    """
    TUEV dataset is downloaded from https://isip.piconepress.com/projects/tuh_eeg/html/downloads.shtml
    """

    root = "/data/stympopper/filteredTUEV/edf"
    train_out_dir = os.path.join(root, "processed_train")
    eval_out_dir = os.path.join(root, "processed_eval")
    if not os.path.exists(train_out_dir):
        os.makedirs(train_out_dir)
    if not os.path.exists(eval_out_dir):
        os.makedirs(eval_out_dir)

    BaseDirTrain = os.path.join(root, "train")
    fs = 200
    TrainFeatures = np.empty((0, 16, fs))  # 0 for lack of intialization, 22 for channels, fs for num of points
    TrainLabels = np.empty([0, 1])
    TrainOffendingChannel = np.empty([0, 1])
    load_up_objects(BaseDirTrain, TrainFeatures, TrainLabels, TrainOffendingChannel, train_out_dir)

    BaseDirEval = os.path.join(root, "eval")
    fs = 200
    EvalFeatures = np.empty((0, 16, fs))  # 0 for lack of intialization, 22 for channels, fs for num of points
    EvalLabels = np.empty([0, 1])
    EvalOffendingChannel = np.empty([0, 1])
    load_up_objects(BaseDirEval, EvalFeatures, EvalLabels, EvalOffendingChannel, eval_out_dir)

    # transfer to train, eval, and test
    root = "/data/stympopper/filteredTUEV/edf"
    seed = 4523
    np.random.seed(seed)

    train_files = os.listdir(os.path.join(root, "processed_train"))
    train_sub = list(set([f.split("_")[0] for f in train_files]))
    print("train sub", len(train_sub))
    test_files = os.listdir(os.path.join(root, "processed_eval"))

    val_sub = np.random.choice(train_sub, size=int(len(train_sub) * 0.2), replace=False)
    train_sub = list(set(train_sub) - set(val_sub))
    val_files = [f for f in train_files if f.split("_")[0] in val_sub]
    train_files = [f for f in train_files if f.split("_")[0] in train_sub]

    for file in train_files:
        os.system(
            f"mv {os.path.join(root, 'processed_train', file)} {os.path.join(root, 'processed', 'processed_train')}"
        )
    for file in val_files:
        os.system(
            f"mv {os.path.join(root, 'processed_train', file)} {os.path.join(root, 'processed', 'processed_eval')}"
        )
    for file in test_files:
        os.system(
            f"mv {os.path.join(root, 'processed_eval', file)} {os.path.join(root, 'processed', 'processed_test')}"
        )
