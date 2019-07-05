import numpy as np
import pyxdf as xdf

import matplotlib.pyplot as plt
import re
import resampy
import mne


def xdf_loader(xdf_file):
    """ Loads an appropriate EEG file into a mne raw object.
    
    Args:
        xdf_file: The path to an .xdf file from which the data is extracted.
        
    Returns:
        raw: The mne raw object.
    
    """
    
    # Load the xdf file
    stream = xdf.load_xdf(xdf_file)
    
    # Extract the necessary event and eeg information
    stream_names = np.array([item['info']['name'][0] for item in stream[0]])
    game_state = list(stream_names).index('Game State')
    eeg_data = list(stream_names).index('NeuroneStream')

    sfreq = int(stream[0][eeg_data]['info']['nominal_srate'][0])
    game_state_series = np.array([item[0] for item in stream[0][game_state]['time_series']])
    game_state_times = np.array(stream[0][game_state]['time_stamps'])
    
    times = stream[0][eeg_data]['time_stamps']
    data = stream[0][eeg_data]['time_series'].T
    
    ch_types = ['eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg',
                'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg',
                'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg',
                'eeg', 'eeg', 'emg', 'emg', 'emg', 'emg', 'eog', 'eog', 'eog', 'eog',
                'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg',
                'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg',
                'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg',
                'eeg', 'eeg', 'ecg', 'bio', 'bio']

    ch_names = ['Fp1', 'Fpz', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1',
                'FC2', 'FC6', 'M1', 'T7', 'C3', 'Cz', 'C4', 'T8', 'M2', 'CP5',
                'CP1', 'CP2', 'CP6', 'P7', 'P3', 'Pz', 'P4', 'P8', 'POz', 'O1',
                'Oz', 'O2', 'EMG_RH', 'EMG_LH', 'EMG_RF', 'EMG_LF', 'EOG_R', 'EOG_L', 'EOG_U', 'EOG_D',
                'AF7', 'AF3', 'AF4', 'AF8', 'F5', 'F1', 'F2', 'F6', 'FC3', 'FCz',
                'FC4', 'C5', 'C1', 'C2', 'C6', 'CP3', 'CPz', 'CP4', 'P5', 'P1',
                'P2', 'P6', 'PO5', 'PO3', 'PO4', 'PO6', 'FT7', 'FT8', 'TP7', 'TP8',
                'PO7', 'PO8', 'ECG', 'Respiratory', 'GSR']

    if len(data)<len(ch_types):
        extra = len(data) - len(ch_types)
        ch_types = ch_types[:extra]
        ch_names = ch_names[:extra]

    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    
    raw = mne.io.RawArray(data, info)
    
    events = np.zeros([game_state_series.size, 3], 'int')
    # Calculate the closest frame in the EEG data for each event time
    events[:, 0] = [np.argmin(np.abs(times-event_time)) for event_time in game_state_times]
    
    legend = np.unique(game_state_series)
    class_vector = np.zeros(game_state_series.size, dtype='int')
    event_id = {}
    for ii in np.arange(legend.size):
        class_vector += (game_state_series == legend[ii])*ii
        event_id[legend[ii]] = ii
        
    events[:,2] = class_vector
    
    # This will not get saved to .fif and has to be worked around
    raw.events = events
    raw.event_id = event_id

    # Set eeg sensor locations
    raw.set_montage(mne.channels.read_montage("standard_1005", ch_names=raw.info["ch_names"]))
    # Reference to the common average
    raw.set_eeg_reference()
    
    return(raw)


def fif_save(raw, fif):
    
    evt = mne.Annotations(onset=raw.times[raw.events[:, 0]], duration=np.zeros(len(raw.events)), description=["Event/" + list(raw.event_id.keys())[here] for here in raw.events[:, 2]])
    
    raw.set_annotations(evt)
    
    raw.save(fif)


def fif_loader(fif):
    raw = mne.io.Raw(fif)
    stims = raw.pick_types(meg=False, stim=True)


def quickplot(raw):
    raw.drop_channels(["EOG_U"])
    raw.notch_filter(np.arange(50,2500,50))
    raw.notch_filter(np.arange(90,2500,90))
    raw.notch_filter(np.arange(52.1,2500,52.1))
    raw.set_eeg_reference()
    raw.plot(remove_dc = True, scalings='auto', events=raw.events, event_id=raw.event_id)


def pds_topo(raw):
    montage = mne.channels.read_montage("standard_1005",ch_names=raw.info["ch_names"])
    raw.set_montage(montage)
    raw.plot_psd_topo(fmax=250)


def pool_epochs(path, files, regexp, timeframe_start, timeframe_end, target_fps):
    """ Extracts and accumulates timeframes around events from different files.
    Args:
        path: If the files share a single path, you can specify it here.
        
        files: A list of .xdf files to extract data from.
        
        regexp: A regular expression that defines the format of the extracted events.
        
        timeframe_start: The time in seconds before the event, in which the EEG Data is extracted.
        
        timeframe_end: The time in seconds after the event, in which the EEG Data is extracted.
        
        target_fps: Downsample the EEG-data to this value.         
    
    Returns:
        epoch: An mne Epochs object
    """
    
    master_id = {}
    epoch = []
    master_legend = []
    for file in files:
        current_raw = xdf_loader(path+file)
        current_events = current_raw.events
        current_id = current_raw.event_id
        
        # Compute which actions are available in the current file
        here = np.array([bool(re.search(regexp, element)) for element in list(current_id.keys())])
        legend = np.array(list(current_id.keys()))[here]
        # Update Master legend and ID if the current file includes new actions
        for event in legend[[item not in master_legend for item in legend]]:
            master_id[event] = len(master_id)
            master_legend = np.append(master_legend, event)

        picked_events = np.empty([0, 3], dtype=int)
        picked_id = {}
        
        for this in legend:
            # Get all appropriate events
            picked_events = np.append(picked_events, current_events[current_events[:, 2] == current_id[this]], axis=0)
            # Update the ID according to master
            picked_events[:, 2][picked_events[:, 2] == current_id[this]] = master_id[this]
            # Build up a temp ID dict for the current Epochs
            picked_id[this] = master_id[this]
        
        # Building empty Epochs will throw errors 
        if not picked_id:    
            continue
        current_epoch = mne.Epochs(current_raw, picked_events, picked_id, tmin=-timeframe_start, tmax=timeframe_end)
        current_epoch.load_data()
        current_epoch.resample(target_fps)
        
        # Append the current epochs if there are epochs to append to
        if not epoch:
            epoch = current_epoch.copy()
        else:
            epoch = mne.EpochsArray(np.append(epoch[:].get_data(), current_epoch[:].get_data(), axis=0), info=epoch.info, events=np.append(epoch.events, current_epoch.events, axis=0), event_id=master_id)
        
    return epoch


def plot_relspec(epochs, events, fmax=np.inf):
    epochs.pick_types(meg=False, eeg=True)
    
    cond_1 = epochs[events[0]]
    cond_2 = epochs[events[1]]

    psds_1, freqs = mne.time_frequency.psd_welch(cond_1, n_fft=int(epochs.info['sfreq']), fmax=fmax)
    psds_2,_ = mne.time_frequency.psd_welch(cond_2, n_fft=int(epochs.info['sfreq']), fmax=fmax)

    # psds_1 = 20 * np.log10(psds_1)
    # psds_2 = 20 * np.log10(psds_2)
    
    mean_1 = np.median(psds_1, axis=0)
    mean_2 = np.median(psds_2, axis=0)
    
    div = 20 * np.log10(mean_1/mean_2)
    
    def my_callback(ax, ch_idx):
        """
        This block of code is executed once you click on one of the channel axes
        in the plot. To work with the viz internals, this function should only take
        two parameters, the axis and the channel or data index.
        """
        ax.plot(freqs, div[ch_idx], color='red')
        ax.set_xlabel = 'Frequency (Hz)'
        ax.set_ylabel = 'Power (dB)'

    f = plt.figure()
    for ax, idx in mne.viz.iter_topography(epochs.info,
                               fig_facecolor='white',
                               axis_facecolor='white',
                               axis_spinecolor='white',
                               on_pick=my_callback,
                                           fig=f):
        ax.plot(div[idx], color='red')
        ax.axhline(color='black')
        ax.axvline(color='black')

    plt.gcf().suptitle('Power spectral densities')
    plt.show()


def plot_dualspec(epochs, events, fmax=np.inf):
    epochs.pick_types(meg=False, eeg=True)

    cond_1 = epochs[events[0]]
    cond_2 = epochs[events[1]]

    psds_1, freqs = mne.time_frequency.psd_welch(cond_1, n_fft=int(epochs.info['sfreq']), fmax=fmax)
    psds_2,_ = mne.time_frequency.psd_welch(cond_2, n_fft=int(epochs.info['sfreq']), fmax=fmax)

    psds_1 = 20 * np.log10(psds_1)
    psds_2 = 20 * np.log10(psds_2)

    mean_1 = np.median(psds_1, axis=0)
    mean_2 = np.median(psds_2, axis=0)

    def my_callback(ax, ch_idx):
        """
        This block of code is executed once you click on one of the channel axes
        in the plot. To work with the viz internals, this function should only take
        two parameters, the axis and the channel or data index.
        """
        ax.plot(freqs, mean_1[ch_idx], color='red')
        ax.plot(freqs, mean_2[ch_idx], color='blue')
        ax.set_xlabel = 'Frequency (Hz)'
        ax.set_ylabel = 'Power (dB)'

    f = plt.figure()
    for ax, idx in mne.viz.iter_topography(epochs.info,
                               fig_facecolor='white',
                               axis_facecolor='white',
                               axis_spinecolor='white',
                               on_pick=my_callback,
                                           fig=f):
        ax.plot(mean_1[idx], color='red')

    for ax, idx in mne.viz.iter_topography(epochs.info,
                               fig_facecolor='white',
                               axis_facecolor='white',
                               axis_spinecolor='white',
                                           fig=f):
        ax.plot(mean_2[idx], color='blue')

    plt.gcf().suptitle('Power spectral densities')
    plt.show()


def plot_reltfr(epochs, events, timeframe_start, timeframe_stop):
    #epochs.pick_types(meg=False, eeg=True)
    
    cond_1 = epochs[events[0]]
    cond_2 = epochs[events[1]]
    
    wsize = int(epochs.info['sfreq'] + epochs.info['sfreq'] % 4)
    tstep = int(wsize/10)
    freqs = mne.time_frequency.stftfreq(wsize=wsize, sfreq=epochs.info['sfreq'])
    
    img_1 = [mne.time_frequency.stft(cond_1._data[x], wsize, tstep) for x in np.arange(cond_1._data.shape[0])]
    img_2 = [mne.time_frequency.stft(cond_2._data[x], wsize, tstep) for x in np.arange(cond_2._data.shape[0])]
    
    data_1 = np.abs(img_1)**2
    data_2 = np.abs(img_2)**2
    
    mean_1 = np.median(data_1, axis=0)
    mean_2 = np.median(data_2, axis=0)
    
    div = mean_1/mean_2
    
    tfr = mne.time_frequency.AverageTFR(info=epochs.info, data=div, times=np.arange(timeframe_start, timeframe_stop, .1), freqs=freqs, nave=len(epochs[events[0]])+len(epochs[events[1]]))
    
    tfr.plot_topo(fmax=55, dB=True, title = ('TFR for "' + events[0] + '"/"' + events[1] + '"'))


def plot_relsw(epochs, events, fmax= np.inf):
    #epochs.pick_types(meg=False, eeg=True)

    cond_1 = epochs[events[0]]
    cond_2 = epochs[events[1]]

    wsize = int(epochs.info['sfreq'] + epochs.info['sfreq'] % 4)
    tstep = int(wsize/10)
    freqs = mne.time_frequency.stftfreq(wsize=wsize, sfreq=epochs.info['sfreq'])

    img_1 = [mne.time_frequency.stft(cond_1._data[x], wsize, tstep) for x in np.arange(cond_1._data.shape[0])]
    img_2 = [mne.time_frequency.stft(cond_2._data[x], wsize, tstep) for x in np.arange(cond_2._data.shape[0])]

    data_1 = np.abs(img_1)**2
    data_2 = np.abs(img_2)**2

    mean_1 = np.median(data_1, axis=3)
    mean_2 = np.median(data_2, axis=3)

    mean_1 = np.median(mean_1, axis=0)
    mean_2 = np.median(mean_2, axis=0)

    div = 20 * np.log10(mean_1/mean_2)

    def my_callback(ax, ch_idx):
        """
        This block of code is executed once you click on one of the channel axes
        in the plot. To work with the viz internals, this function should only take
        two parameters, the axis and the channel or data index.
        """
        ax.plot(freqs[freqs <= fmax], div[ch_idx][freqs <= fmax], color='red')
        ax.set_xlabel = 'Frequency (Hz)'
        ax.set_ylabel = 'Power (dB)'

    f = plt.figure()
    for ax, idx in mne.viz.iter_topography(epochs.info,
                               fig_facecolor='white',
                               axis_facecolor='white',
                               axis_spinecolor='white',
                               on_pick=my_callback,
                                           fig=f):
        ax.plot(freqs[freqs <= fmax], div[idx][freqs <= fmax], color='red')
        ax.axhline(color='black')
        ax.axvline(color='black')

    plt.gcf().suptitle('Power spectral densities')
    plt.show()


def plot_dualsw(epochs, events, fmax=np.inf):
    #epochs.pick_types(meg=False, eeg=True)

    cond_1 = epochs[events[0]]
    cond_2 = epochs[events[1]]

    wsize = int(epochs.info['sfreq']+ epochs.info['sfreq'] % 4)
    tstep = int(wsize/10)
    freqs = mne.time_frequency.stftfreq(wsize=wsize, sfreq=epochs.info['sfreq'])

    img_1 = [mne.time_frequency.stft(cond_1._data[x], wsize, tstep) for x in np.arange(cond_1._data.shape[0])]
    img_2 = [mne.time_frequency.stft(cond_2._data[x], wsize, tstep) for x in np.arange(cond_2._data.shape[0])]

    data_1 = np.abs(img_1)**2
    data_2 = np.abs(img_2)**2

    mean_1 = np.median(data_1, axis=3)
    mean_2 = np.median(data_2, axis=3)

    mean_1 = np.median(mean_1, axis=0)
    mean_2 = np.median(mean_2, axis=0)

    div = 20 * np.log10(mean_1/mean_2)

    def my_callback(ax, ch_idx):
        """
        This block of code is executed once you click on one of the channel axes
        in the plot. To work with the viz internals, this function should only take
        two parameters, the axis and the channel or data index.
        """
        ax.plot(freqs[freqs <= fmax], mean_1[ch_idx][freqs <= fmax], color='red')
        ax.plot(freqs[freqs <= fmax], mean_2[ch_idx][freqs <= fmax], color='blue')
        ax.set_xlabel = 'Frequency (Hz)'
        ax.set_ylabel = 'Power (dB)'

    f = plt.figure()
    for ax, idx in mne.viz.iter_topography(epochs.info,
                               fig_facecolor='white',
                               axis_facecolor='white',
                               axis_spinecolor='white',
                               on_pick=my_callback,
                                           fig=f):
        ax.plot(freqs[freqs <= fmax], mean_1[idx][freqs <= fmax], color='red')

    for ax, idx in mne.viz.iter_topography(epochs.info,
                               fig_facecolor='white',
                               axis_facecolor='white',
                               axis_spinecolor='white',
                                           fig=f):
        ax.plot(freqs[freqs <= fmax], mean_2[idx][freqs <= fmax], color='blue')

    plt.gcf().suptitle('Power spectral densities')
    plt.show()


def plot_relsw_topomap(epochs, events, fmin=0, fmax=np.inf, reject=None):
    epochs.pick_types(meg=False, eeg=True)

    cond_1 = epochs[events[0]]
    cond_2 = epochs[events[1]]

    wsize = int(epochs.info['sfreq']+ epochs.info['sfreq']%4)
    tstep = int(wsize/10)
    freqs = mne.time_frequency.stftfreq(wsize=wsize, sfreq=epochs.info['sfreq'])

    img_1 = [mne.time_frequency.stft(cond_1._data[x], wsize, tstep) for x in np.arange(cond_1._data.shape[0])]
    img_2 = [mne.time_frequency.stft(cond_2._data[x], wsize, tstep) for x in np.arange(cond_2._data.shape[0])]

    data_1 = np.abs(img_1)**2
    data_2 = np.abs(img_2)**2

    avg_1 = np.median(data_1, axis=3)
    avg_2 = np.median(data_2, axis=3)
    
    if reject:
        split = avg_1.shape[0]
        val = np.append(avg_1, avg_2, axis=0)[:, :, 201:].sum(axis=(1, 2))

        # plt.figure()
        # plt.hist(val, bins=50)
        # plt.show()
        # plt.gcf().suptitle("Trial/Power Histogram")
        # plt.xlabel("Power")
        # plt.ylabel("# Trials")
        
        ind = val.argsort()[-reject:]
        
        avg_1 = np.delete(avg_1, ind[ind<split], axis=0)
        avg_2 = np.delete(avg_2, ind[ind>=split]-split, axis=0)

    mean_1 = np.median(avg_1, axis=0)
    mean_2 = np.median(avg_2, axis=0)

    div = 20 * np.log10(mean_1/mean_2)

    plt.figure()
    mne.viz.plot_topomap(np.median(div[:, (freqs >= fmin) & (freqs <= fmax)], axis=1), pos=epochs.info)
    plt.gcf().suptitle(('"' + events[0] + '"/"' + events[1] + '" for '+str(fmin) + ' - ' + str(fmax) + 'Hz'))

