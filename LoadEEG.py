import mne
import mne_interface
import os
import numpy as np
from scipy import signal
from braindecode.datautil.signalproc import exponential_running_standardize
from braindecode.datautil.signal_target import SignalAndTarget


def process_epochs(epochs, filters, channels, target_fs):

    r''' function for epoch preprocesing: filtering, channels extraction and resampling

    Arguments:
        epochs (mne.Epochs object, required): mne.Epochs object that contains epoched data
        filters (bool, required): if True - apply filters to data:
            highpass (butter) filter with fc=1Hz
            notch filter with fc=50Hz
            lowpass (butter) filter with fc=30Hz
            NOTE: currently filtering is  implemented with filfilt function
        channels (str, required): channels to extract from data
            'eeg': only eeg channels,
            'eeg_eog': eog channels averaged with respect to eeg channels,
            'MI': only motor-imagery relevant channels (C-)
        target_fs (int,required): resampling frequency

    Output:
        epochs (numpy array, (number of trials, number of channels, signal length)): processed epochs
    '''

    if channels == 'eeg_eog':
        epochs.pick_types(eeg=True, eog=True)
        print('Only eog averaged with eeg')
        ch_names = np.array(epochs.ch_names)
        epochs = epochs.get_data()

        epochs = eog_preprocess(epochs, ch_names)

    elif channels == 'eeg':
        epochs.pick_types(eeg=True)
        print('Only eeg')
        epochs = epochs.get_data()

    elif channels == 'MI':

        channels_motor = ['FC5', 'FC1', 'FC2', 'FC6', 'C3', 'C4', 'CP5', 'CP1', 'CP2',
                          'CP6', 'FC3', 'FCz', 'FC4', 'C5', 'C1', 'C2', 'C6', 'CP3', 'CPz',
                          'CP4']
        chan_nr = np.array([])
        for channel in channels_motor:
            chan_nr = np.append(chan_nr,
                                epochs.ch_names.index(channel)).astype(int)

        epochs = epochs.get_data()

        epochs = epochs[:, chan_nr, :]
        print('Only MI paradigm relevant channels')

    if filters:
        ### filters ###

        # Butter filter (highpass) for 1 Hz
        b_1, a_1 = signal.butter(6, 1/target_fs, btype='high', output='ba')

        # Butter filter (lowpass) for 30 Hz
        b_30, a_30 = signal.butter(6, 30/target_fs, btype='low', output='ba')

        # Notch filter with 50 HZ
        f0 = 50.0
        Q = 30.0  # Quality factor
        # Design notch filter
        b_50, a_50 = signal.iirnotch(f0, Q, target_fs)

        #####

        epochs = signal.filtfilt(b_50, a_50, epochs)
        epochs = signal.filtfilt(b_1, a_1, epochs)
        epochs = signal.filtfilt(b_30, a_30, epochs)
        print('Filtered: highpass with 1Hz and lowpass with 30Hz')

    # exp_mean_standardize
    for ep in range(epochs.shape[0]):
        processed = exponential_running_standardize(epochs[ep, :, :].swapaxes(0, 1), factor_new=0.001,
                                                    init_block_size=None, eps=0.0001)
        epochs[ep, :, :] = np.swapaxes(processed, 0, 1)

    return epochs


def eog_preprocess(epochs, ch_names):

    r""" Function that averages eog channels with respect to eeg frontal channels and each other:
                result_channel1 = left_eog - right_eog
                result_channel2 = Fp1 - down_eog
                result_channel3 = Fp2 - up_eog
        Used to check whether there are eog artifacts that help classifier (if you have
        good accuracy with only this channels - your classifier learns artifacts

    Arguments:
        epochs (numpy array, required): epochs from one run
        ch_names (list, required): channel names

    Outputs:
        epochs_eog (numpy array, (number of trials, 3, signal length)): data with averaged eog

    """
    left_eog = np.where(ch_names == 'EOG_L')[0]
    right_eog = np.where(ch_names == 'EOG_R')[0]

    down_eog = np.where(ch_names == 'EOG_D')[0]
    up_eog = np.where(ch_names == 'EOG_U')[0]

    fp1 = np.where(ch_names == 'Fp1')[0]
    fp2 = np.where(ch_names == 'Fp2')[0]

    epochs_eog = np.empty((epochs.shape[0], 3, epochs.shape[2]))
    epochs_eog[:, 0, :] = np.squeeze(np.subtract(epochs[:, left_eog, :], epochs[:, right_eog, :]))
    epochs_eog[:, 1, :] = np.squeeze(np.subtract(epochs[:, fp1, :], epochs[:, down_eog, :]))
    epochs_eog[:, 2, :] = np.squeeze(np.subtract(epochs[:, fp2, :], epochs[:, up_eog, :]))

    return epochs_eog


def epochsData(file_name, target_fs, path=None, tmin=0, tmax=3, filters=False, channels='MI'):

    r""" Function for data loading from xdf file and epoching

    Arguments:
        file_name (str, required): name of the folder with .xdf files from one subject.
            Note: there is no need to specify a name of the xdf file itself,
            the function will load all files from the subject folder
        target_fs (int, required): resampling frequency
        path (str, optional): path with data files, default - None, which means '/home/tanja/DecodingEEG/Data'
        tmin (int, optional): starting time of the trial with respect to the trial onset, default=0
        tmax (int, optional): ending time of the trial with respect to the trial onset, default=3
        filters (bool, optional): applying filters to the signal, default=True
        channels (str, optional)L type of channels to extract from data, default='eeg':
            'eeg': only eeg channels,
            'eeg_eog': eog channels averaged with respect to eeg channels,
            'MI': only motor-imagery relevant channels (C-)

    Outputs:
        data (numpy array (number of runs, number of trials in a run, number of channels, signal length))
        labels (numpy array (number of runs, number of trials in a run, 1)


    """

    # Load raw data
    if path is None:
        path = os.path.join('/home/tanja', 'DecodingEEG', 'Data', file_name)

    files_in_dir = os.listdir(path)
    files = [path + '/' + file for file in files_in_dir]
    runs = []
    for file in files:
        runs = runs + [mne_interface.xdf_loader(file)]

    # Extract epochs:
    label_dict = {
        'Monster left': 0,
        'Monster right': 1
    }

    conditions = ['Monster left', 'Monster right']
    runs_epochs = []
    labels = []
    for i, run in enumerate(runs):
        condition_dict = {}
        event_dict = {}  # use this for decoding labels

        for cond in conditions:
            if cond in run.event_id.keys():
                condition_dict[cond] = run.event_id[cond]
                event_dict[run.event_id[cond]] = cond

        trials = mne.Epochs(run, run.events, event_id=condition_dict,
                            tmin=tmin, tmax=tmax, preload=True, baseline=None)
        runs_epochs = runs_epochs + [trials]

        labels = labels + [np.array([label_dict[event_dict[label]] for label in trials.events[:, 2]])]

    data = []
    for run in runs_epochs:
        run = run.resample(target_fs)
        data = data + [process_epochs(run, filters=filters, channels=channels,
                                      target_fs=target_fs)]

    return np.array(data), np.array(labels)


def load_subjects(subject_files, training_design, target_fs=256,
                  tmin=0, tmax=3, filters=True, channels='eeg', return_only_test=False):

    r""" load subject data of one or multiple subjects

    Arguments:
        subject_files (list, required): list of file names with data from subjects
        training_design (str, required): mode of training either 'leave_one_out' or 'mix'
            'leave_one_out': first (n-1) subjects are used for training data and their last runs
                            are used for validation, one subject's data is left for testing
            'mix': for all n subjects take first r-2 runs for training set, one to last r-1 for
                    validation and last r for testing
        target_fs (int, optional): resampling frequency, default=256 Hz
        tmin (int, optional): starting time of the trial with respect to the trial onset, default=0
        tmax (int, optional): ending time of the trial with respect to the trial onset, default=3
        filters (bool, optional): applying filters to the signal, default=True
        channels (str, optional)L type of channels to extract from data, default='eeg':
            'eeg': only eeg channels,
            'eeg_eog': eog channels averaged with respect to eeg channels,
            'MI': only motor-imagery relevant channels (C-)
        return_only test (bool, optional): if True do not split the data and return just test set
            (used for pseudo-online training), default=False

    Outputs:
        if return_only_test=True:
            test_set (SignalAndTarget object): test data from one subject
        if return_only_test=False:
            train (numpy array) - train data
            validation (numpy array) - validation data
            test (numpy array) - test data
            train_labels (numpy array) - train labels
            validation_labels (numpy array) - validation labels
            test_labels (numpy array) - test labels

        """

    for nr, file_name in enumerate(subject_files):

        if return_only_test:
            data, labels = epochsData(file_name, target_fs=target_fs, tmin=tmin, tmax=tmax,
                                      filters=filters, channels=channels)

            test = np.concatenate(data, axis=0)
            test_labels = np.concatenate(labels, axis=0)
            test_set = SignalAndTarget(test.copy(), test_labels)

        else:
            data, labels = epochsData(file_name, target_fs=target_fs, tmin=tmin, tmax=tmax,
                       filters=filters, channels=channels)

            if training_design == 'leave_one_out':
                if nr == 0:
                    train = np.concatenate(data[0:-1], axis=0)
                    validation = data[-1]
                    train_labels = np.concatenate(labels[0:-1], axis=0)
                    validation_labels = labels[-1]
                elif nr == len(subject_files)-1:
                    test = np.concatenate(data, axis=0)
                    test_labels = np.concatenate(labels, axis=0)
                else:
                    train = np.concatenate((train, np.concatenate(data[0:-1], axis=0)), axis=0)
                    validation = np.concatenate((validation, data[-1]), axis=0)
                    train_labels = np.concatenate((train_labels, np.concatenate(labels[0:-1], axis=0)), axis=0)
                    validation_labels = np.concatenate((validation_labels, labels[-1]), axis=0)

            if training_design == 'mix':

                if nr == 0:
                    train = np.concatenate(data[0:-2], axis=0)
                    validation = data[-2]
                    test = data[-1]
                    train_labels = np.concatenate(labels[0:-2], axis=0)
                    validation_labels = labels[-2]
                    test_labels = labels[-1]
                else:
                    train = np.concatenate((train, np.concatenate(data[0:-3], axis=0)), axis=0)
                    validation = np.concatenate((validation, data[-3]), axis=0)
                    test = np.concatenate((test, data[-1]), axis=0)
                    train_labels = np.concatenate((train_labels, np.concatenate(labels[0:-3], axis=0)), axis=0)
                    validation_labels = np.concatenate((validation_labels, labels[-3]), axis=0)
                    test_labels = np.concatenate((test_labels, labels[-1]), axis=0)

    if return_only_test:
        return test_set
    else:
        return train, validation, test, train_labels, validation_labels, test_labels





import torchcontrib.optim

