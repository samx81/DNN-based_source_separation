import os

import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from algorithm.frequency_mask import ideal_binary_mask, ideal_ratio_mask, wiener_filter_mask

EPS = 1e-12

class WSJ0Dataset(torch.utils.data.Dataset):
    def __init__(self, wav_root, list_path):
        super().__init__()
        
        self.wav_root = os.path.abspath(wav_root)
        self.list_path = os.path.abspath(list_path)

class WaveDataset(WSJ0Dataset):
    def __init__(self, wav_root, list_path, samples=32000, least_sample=None, overlap=None, n_sources=2, chunk=True):
        super().__init__(wav_root, list_path)

        wav_root = os.path.abspath(wav_root)
        clean_list_path = os.path.join(wav_root, 'clean.scp')
        noisy_list_path = os.path.join(wav_root, 'noisy.scp')
        length_list_path = os.path.join(wav_root, 'length.list')
        
        self.chunk = chunk
        self.samples = samples if samples else 16000 * 10
        self.overlap = overlap if overlap else self.samples // 2
        self.least_sample = least_sample if least_sample else self.samples // 4
        
        self.json_data = []

        with open(noisy_list_path) as f:
            ff = f.readlines()
            noisy_dict = {line.split()[0]: line.split()[1] for line in ff}

        with open(clean_list_path) as f:
            ff = f.readlines()
            clean_dict = {line.split()[0]: line.split()[1] for line in ff}

        if os.path.exists(length_list_path):
            with open(length_list_path, 'r') as f:
                ff = f.readlines()
                len_dict = {line.split()[0]: int(line.split()[1]) for line in ff}
        else:
            len_dict = {}
            print_str = ''
            for id in tqdm(sorted(clean_dict.keys()),"Preparing Data Info..."):
                len_dict[id] = torchaudio.info(clean_dict[id]).num_frames
                print_str += f'{id} {len_dict[id]}\n'
            with open(length_list_path, 'w') as f:
                f.write(print_str)


        for id in tqdm(clean_dict.keys(),"Loading Dataset..."):
            # wave, _ = torchaudio.load(clean_dict[id])
            
            # _, T_total = wave.size()
            T_total = len_dict[id]
            if chunk:
                for start_idx in range(0, T_total, samples - self.overlap):
                
                    end_idx = start_idx + samples
                    if end_idx > T_total:
                        end_idx = T_total
                    if end_idx - start_idx < self.least_sample:
                        break
                    data = {
                        'sources': {},
                        'mixture': {}
                    }
                    
                    source_data = {
                        'path': clean_dict[id],
                        'start': start_idx,
                        'end': end_idx
                    }
                    data['sources'] = source_data
                    
                    mixture_data = {
                        'path': noisy_dict[id],
                        'start': start_idx,
                        'end': end_idx
                    }
                    data['mixture'] = mixture_data
                    data['ID'] = id
                
                    self.json_data.append(data)
            else:
                data = {
                    'sources': {},
                    'mixture': {}
                }
                    
                source_data = {
                    'path': clean_dict[id],
                    'start': 0,
                    'end': T_total
                }
                data['sources'] = source_data
                
                mixture_data = {
                    'path': noisy_dict[id],
                    'start': 0,
                    'end': T_total
                }
                data['mixture'] = mixture_data
                data['ID'] = id
            
                self.json_data.append(data)
        print(len(self.json_data), flush=True)
        
    def __getitem__(self, idx):
        """
        Returns:
            mixture (1, T) <torch.Tensor>
            sources (n_sources, T) <torch.Tensor>
            segment_IDs (n_sources,) <list<str>>
        """
        data = self.json_data[idx]
        sources = []

        mixture_data = data['mixture']
        start, end = mixture_data['start'], mixture_data['end']
        wav_path = os.path.join(self.wav_root, mixture_data['path'])
        wave, _ = torchaudio.load(wav_path)
        mixture = wave[:, start: end]
        
        source_data = data['sources']
        start, end = source_data['start'], source_data['end']
        wav_path = source_data['path']
        wave, _ = torchaudio.load(wav_path)
        wave = wave[:, start: end]

        wav_len = end - start
        if self.chunk and wav_len < self.samples:
            P = self.samples - wav_len
            wave = F.pad(wave, (0, P), "constant")
            mixture = F.pad(mixture, (0, P), "constant")
        
        sources.append(wave)
        
        sources = torch.cat(sources, dim=0)
            
        segment_ID = self.json_data[idx]['ID'] + '_{}-{}'.format(start, end)
        
        return mixture, sources, segment_ID
        
    def __len__(self):
        return len(self.json_data)

class WaveTrainDataset(WaveDataset):
    def __init__(self, wav_root, list_path, samples=32000, overlap=None, n_sources=2):
        super().__init__(wav_root, list_path, samples=samples, overlap=overlap, n_sources=n_sources)
    
    def __getitem__(self, idx):
        mixture, sources, _ = super().__getitem__(idx)
        
        return mixture, sources

class WaveEvalDataset(WaveDataset):
    def __init__(self, wav_root, list_path, max_samples=None, n_sources=2, chunk=False):
        super().__init__(wav_root, list_path, samples=max_samples, least_sample=0, n_sources=n_sources, chunk=chunk)

        # wav_root = os.path.abspath(wav_root)
        # list_path = os.path.abspath(list_path)

        # self.json_data = []
        
        # with open(list_path) as f:
        #     for line in f:
        #         ID = line.strip()
        #         wav_path = os.path.join(wav_root, 'mix', '{}.wav'.format(ID))

        #         wave, _ = torchaudio.load(wav_path)
                
        #         _, T_total = wave.size()

    # TODO: 這邊的 max_sample 要注意        
        #         if max_samples is None:
        #             samples = T_total
        #         else:
        #             if T_total < max_samples:
        #                 samples = T_total
        #             else:
        #                 samples = max_samples
                
        #         data = {
        #             'sources': {},
        #             'mixture': {}
        #         }
                
        #         for source_idx in range(n_sources):
        #             source_data = {
        #                 'path': os.path.join('s{}'.format(source_idx + 1), '{}.wav'.format(ID)),
        #                 'start': 0,
        #                 'end': samples
        #             }
        #             data['sources']['s{}'.format(source_idx + 1)] = source_data
                
        #         mixture_data = {
        #             'path': os.path.join('mix', '{}.wav'.format(ID)),
        #             'start': 0,
        #             'end': samples
        #         }
        #         data['mixture'] = mixture_data
        #         data['ID'] = ID
            
        #         self.json_data.append(data)
    
    def __getitem__(self, idx):
        mixture, sources, _ = super().__getitem__(idx)
        segment_ID = self.json_data[idx]['ID']
    
        return mixture, sources, segment_ID

class WaveTestDataset(WaveEvalDataset):
    def __init__(self, wav_root, list_path, max_samples=None, n_sources=2):
        super().__init__(wav_root, list_path, max_samples=max_samples, n_sources=n_sources,chunk=False)
        
    def __getitem__(self, idx):
        """
        Returns:
            mixture (1, T) <torch.Tensor>
            sources (n_sources, T) <torch.Tensor>
            segment_ID <str>
        """
        mixture, sources, segment_ID = super().__getitem__(idx)
        
        return mixture, sources, segment_ID

class SpectrogramDataset(WaveDataset):
    def __init__(self, wav_root, list_path, fft_size, hop_size=None, window_fn='hann', normalize=False, samples=32000, overlap=None, n_sources=2):
        super().__init__(wav_root, list_path, samples=samples, overlap=overlap, n_sources=n_sources)
        
        if hop_size is None:
            hop_size = fft_size//2
        
        self.fft_size, self.hop_size = fft_size, hop_size
        self.n_bins = fft_size//2 + 1

        if window_fn:
            if window_fn == 'hann':
                self.window = torch.hann_window(fft_size, periodic=True)
            elif window_fn == 'hamming':
                self.window = torch.hamming_window(fft_size, periodic=True)
            else:
                raise ValueError("Invalid argument.")
        else:
            self.window = None
        
        self.normalize = normalize
        
    def __getitem__(self, idx):
        """
        Returns:
            mixture (1, n_bins, n_frames) <torch.Tensor>
            sources (n_sources, n_bins, n_frames) <torch.Tensor>
            T (), <int>: Number of samples in time-domain
            segment_IDs (n_sources,) <list<str>>
        """
        mixture, sources, segment_IDs = super().__getitem__(idx)

        n_dims = mixture.dim()
        T = mixture.size(-1)

        if n_dims > 2:
            mixture_channels = mixture.size()[:-1]
            sources_channels = sources.size()[:-1]
            mixture = mixture.reshape(-1, mixture.size(-1))
            sources = sources.reshape(-1, sources.size(-1))

        mixture = torch.stft(mixture, n_fft=self.fft_size, hop_length=self.hop_size, window=self.window, normalized=self.normalize, return_complex=True) # (1, n_bins, n_frames)
        sources = torch.stft(sources, n_fft=self.fft_size, hop_length=self.hop_size, window=self.window, normalized=self.normalize, return_complex=True) # (n_sources, n_bins, n_frames)

        if n_dims > 2:
            mixture = mixture.reshape(*mixture_channels, *mixture.size()[-2:])
            sources = sources.reshape(*sources_channels, *sources.size()[-2:])
        
        return mixture, sources, T, segment_IDs

class IdealMaskSpectrogramDataset(SpectrogramDataset):
    def __init__(self, wav_root, list_path, fft_size, hop_size=None, window_fn='hann', normalize=False, mask_type='ibm', threshold=40, samples=32000, overlap=None, n_sources=2, eps=EPS):
        super().__init__(wav_root, list_path, fft_size, hop_size=hop_size, window_fn=window_fn, normalize=normalize, samples=samples, overlap=overlap, n_sources=n_sources)
        
        if mask_type == 'ibm':
            self.generate_mask = ideal_binary_mask
        elif mask_type == 'irm':
            self.generate_mask = ideal_ratio_mask
        elif mask_type == 'wfm':
            self.generate_mask = wiener_filter_mask
        else:
            raise NotImplementedError("Not support mask {}".format(mask_type))
        
        self.threshold = threshold
        self.eps = eps
    
    def __getitem__(self, idx):
        """
        Returns:
            mixture (1, n_bins, n_frames) <torch.Tensor>
            sources (n_sources, n_bins, n_frames) <torch.Tensor>
            ideal_mask (n_sources, n_bins, n_frames) <torch.Tensor>
            threshold_weight (1, n_bins, n_frames) <torch.Tensor>
            T (), <int>: Number of samples in time-domain
            segment_IDs (n_sources,) <list<str>>
        """
        threshold = self.threshold
        eps = self.eps
        
        mixture, sources, T, segment_IDs = super().__getitem__(idx) # (1, n_bins, n_frames), (n_sources, n_bins, n_frames)
        sources_amplitude = torch.abs(sources)
        ideal_mask = self.generate_mask(sources_amplitude)
        
        mixture_amplitude = torch.abs(mixture)
        log_amplitude = 20 * torch.log10(mixture_amplitude + eps)
        max_log_amplitude = torch.max(log_amplitude)
        threshold = 10**((max_log_amplitude - threshold) / 20)
        threshold_weight = torch.where(mixture_amplitude > 0, torch.ones_like(mixture_amplitude), torch.zeros_like(mixture_amplitude))
        
        return mixture, sources, ideal_mask, threshold_weight, T, segment_IDs

class IdealMaskSpectrogramTrainDataset(IdealMaskSpectrogramDataset):
    def __init__(self, wav_root, list_path, fft_size, hop_size=None, window_fn='hann', normalize=False, mask_type='ibm', threshold=40, samples=32000, overlap=None, n_sources=2, eps=EPS):
        super().__init__(wav_root, list_path, fft_size, hop_size=hop_size, window_fn=window_fn, normalize=normalize, mask_type=mask_type, threshold=threshold, samples=samples, overlap=overlap, n_sources=n_sources, eps=eps)
    
    def __getitem__(self, idx):
        """
        Returns:
            mixture (1, n_bins, n_frames) <torch.Tensor>
            sources (n_sources, n_bins, n_frames) <torch.Tensor>
            ideal_mask (n_sources, n_bins, n_frames) <torch.Tensor>
            threshold_weight (1, n_bins, n_frames) <torch.Tensor>
        """
        mixture, sources, ideal_mask, threshold_weight, _, _ = super().__getitem__(idx)
        
        return mixture, sources, ideal_mask, threshold_weight

class IdealMaskSpectrogramEvalDataset(IdealMaskSpectrogramDataset):
    def __init__(self, wav_root, list_path, fft_size, hop_size=None, window_fn='hann', normalize=False, mask_type='ibm', threshold=40, max_samples=None, n_sources=2, eps=EPS):
        super().__init__(wav_root, list_path, fft_size, hop_size=hop_size, window_fn=window_fn, normalize=normalize, mask_type=mask_type, threshold=threshold, n_sources=n_sources, eps=eps)

        wav_root = os.path.abspath(wav_root)
        list_path = os.path.abspath(list_path)

        self.json_data = []
        
        with open(list_path) as f:
            for line in f:
                ID = line.strip()
                wav_path = os.path.join(wav_root, 'mix', '{}.wav'.format(ID))
                
                wave, _ = torchaudio.load(wav_path)
                
                _, T_total = wave.size()
                
                if max_samples is None:
                    samples = T_total
                else:
                    if T_total < max_samples:
                        samples = T_total
                    else:
                        samples = max_samples
                
                data = {
                    'sources': {},
                    'mixture': {}
                }
                
                for source_idx in range(n_sources):
                    source_data = {
                        'path': os.path.join('s{}'.format(source_idx + 1), '{}.wav'.format(ID)),
                        'start': 0,
                        'end': samples
                    }
                    data['sources']['s{}'.format(source_idx+1)] = source_data
                
                mixture_data = {
                    'path': os.path.join('mix', '{}.wav'.format(ID)),
                    'start': 0,
                    'end': samples
                }
                data['mixture'] = mixture_data
                data['ID'] = ID
            
                self.json_data.append(data)

    def __getitem__(self, idx):
        """
        Returns:
            mixture (1, n_bins, n_frames) <torch.Tensor>
            sources (n_sources, n_bins, n_frames) <torch.Tensor>
            ideal_mask (n_sources, n_bins, n_frames) <torch.Tensor>
            threshold_weight (1, n_bins, n_frames) <torch.Tensor>
        """
        mixture, sources, ideal_mask, threshold_weight, _, _ = super().__getitem__(idx)
    
        return mixture, sources, ideal_mask, threshold_weight

class IdealMaskSpectrogramTestDataset(IdealMaskSpectrogramDataset):
    def __init__(self, wav_root, list_path, fft_size, hop_size=None, window_fn='hann', normalize=False, mask_type='ibm', threshold=40, n_sources=2, eps=EPS):
        super().__init__(wav_root, list_path, fft_size, hop_size=hop_size, window_fn=window_fn, normalize=normalize, mask_type=mask_type, threshold=threshold, n_sources=n_sources, eps=eps)

    def __getitem__(self, idx):
        """
        Returns:
            mixture (1, n_bins, n_frames) <torch.Tensor>
            sources (n_sources, n_bins, n_frames) <torch.Tensor>
            ideal_mask (n_sources, n_bins, n_frames) <torch.Tensor>
            threshold_weight (1, n_bins, n_frames) <torch.Tensor>
            T () <int>
            segment_IDs (n_sources,) <list<str>>
        """
        mixture, sources, ideal_mask, threshold_weight, T, segment_IDs = super().__getitem__(idx)

        return mixture, sources, ideal_mask, threshold_weight, T, segment_IDs

"""
    Data loader
"""

class TrainDataLoader(torch.utils.data.DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def __len__(self):
        return len(self.dataset)

class EvalDataLoader(torch.utils.data.DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        assert self.batch_size == 1, "batch_size is expected 1, but given {}".format(self.batch_size)

class TestDataLoader(torch.utils.data.DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        assert self.batch_size == 1, "batch_size is expected 1, but given {}".format(self.batch_size)
        
        self.collate_fn = test_collate_fn

def test_collate_fn(batch):
    batched_mixture, batched_sources = None, None
    batched_segment_ID = []
    
    for mixture, sources, segmend_ID in batch:
        mixture = mixture.unsqueeze(dim=0)
        sources = sources.unsqueeze(dim=0)
        
        if batched_mixture is None:
            batched_mixture = mixture
            batched_sources = sources
        else:
            batched_mixture = torch.cat([batched_mixture, mixture], dim=0)
            batched_sources = torch.cat([batched_sources, sources], dim=0)
        
        batched_segment_ID.append(segmend_ID)
    
    return batched_mixture, batched_sources, batched_segment_ID

class AttractorTestDataLoader(torch.utils.data.DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        assert self.batch_size == 1, "batch_size is expected 1, but given {}".format(self.batch_size)
        
        self.collate_fn = attractor_test_collate_fn

def attractor_test_collate_fn(batch):
    batched_mixture, batched_sources, batched_assignment, batched_weight_threshold = [], [], [], []
    batched_T = []
    batched_segment_ID = []
    
    for mixture, sources, assignment, weight_threshold, T, segmend_ID in batch:
        mixture = mixture.unsqueeze(dim=0)
        sources = sources.unsqueeze(dim=0)
        assignment = assignment.unsqueeze(dim=0)
        weight_threshold = weight_threshold.unsqueeze(dim=0)
        
        batched_mixture.append(mixture)
        batched_sources.append(sources)
        batched_assignment.append(assignment)
        batched_weight_threshold.append(weight_threshold)

        batched_T.append(T)
        batched_segment_ID.append(segmend_ID)
    
    batched_mixture = torch.cat(batched_mixture, dim=0)
    batched_sources = torch.cat(batched_sources, dim=0)
    batched_assignment = torch.cat(batched_assignment, dim=0)
    batched_weight_threshold = torch.cat(batched_weight_threshold, dim=0)
    
    return batched_mixture, batched_sources, batched_assignment, batched_weight_threshold, batched_T, batched_segment_ID

"""
Dataset for unknown number of sources.
"""

class MixedNumberSourcesWaveDataset(WSJ0Dataset):
    def __init__(self, wav_root, list_path, samples=32000, overlap=None, max_n_sources=3):
        super().__init__(wav_root, list_path)

        wav_root = os.path.abspath(wav_root)
        list_path = os.path.abspath(list_path)
        
        if overlap is None:
            overlap = samples//2
        
        self.json_data = []
        
        with open(list_path) as f:
            for line in f:
                ID = line.strip()
                wav_path = os.path.join(wav_root, 'mix', '{}.wav'.format(ID))
                
                wave, _ = torchaudio.load(wav_path)
                _, T_total = wave.size()

                n_sources = 0

                for source_idx in range(max_n_sources):
                    wav_path = os.path.join(wav_root, 's{}'.format(source_idx+1), '{}.wav'.format(ID))
                    if not os.path.exists(wav_path):
                        break
                    n_sources += 1
                
                for start_idx in range(0, T_total, samples - overlap):
                    end_idx = start_idx + samples
                    if end_idx > T_total:
                        break
                    data = {
                        'sources': {},
                        'mixture': {}
                    }
                    
                    for source_idx in range(n_sources):
                        source_data = {
                            'path': os.path.join('s{}'.format(source_idx+1), '{}.wav'.format(ID)),
                            'start': start_idx,
                            'end': end_idx
                        }
                        data['sources']['s{}'.format(source_idx+1)] = source_data
                    
                    mixture_data = {
                        'path': os.path.join('mix', '{}.wav'.format(ID)),
                        'start': start_idx,
                        'end': end_idx
                    }
                    data['mixture'] = mixture_data
                    data['ID'] = ID
                
                    self.json_data.append(data)
        
    def __getitem__(self, idx):
        """
        Returns:
            mixture (1, T) <torch.Tensor>
            sources (n_sources, T) <torch.Tensor>
            segment_IDs (n_sources,) <list<str>>
        """
        data = self.json_data[idx]
        sources = []
        
        for key in data['sources'].keys():
            source_data = data['sources'][key]
            start, end = source_data['start'], source_data['end']
            wav_path = os.path.join(self.wav_root, source_data['path'])
            wave, _ = torchaudio.load(wav_path)
            sources.append(wave)
        
        sources = torch.cat(sources, dim=0)
        
        mixture_data = data['mixture']
        start, end = mixture_data['start'], mixture_data['end']
        wav_path = os.path.join(self.wav_root, mixture_data['path'])
        mixture, _ = torchaudio.load(wav_path)
            
        segment_ID = self.json_data[idx]['ID'] + '_{}-{}'.format(start, end)
        
        mixture = torch.Tensor(mixture).float()
        sources = torch.Tensor(sources).float()
        
        return mixture, sources, segment_ID
        
    def __len__(self):
        return len(self.json_data)

class MixedNumberSourcesWaveTrainDataset(MixedNumberSourcesWaveDataset):
    def __init__(self, wav_root, list_path, samples=32000, overlap=None, max_n_sources=2):
        super().__init__(wav_root, list_path, samples=samples, overlap=overlap, max_n_sources=max_n_sources)
    
    def __getitem__(self, idx):
        mixture, sources, _ = super().__getitem__(idx)
        
        return mixture, sources

class MixedNumberSourcesWaveEvalDataset(MixedNumberSourcesWaveDataset):
    def __init__(self, wav_root, list_path, max_samples=None, max_n_sources=3):
        super().__init__(wav_root, list_path, max_n_sources=max_n_sources)

        wav_root = os.path.abspath(wav_root)
        list_path = os.path.abspath(list_path)

        self.json_data = []
        
        with open(list_path) as f:
            for line in f:
                ID = line.strip()
                wav_path = os.path.join(wav_root, 'mix', '{}.wav'.format(ID))
                
                wave, _ = torchaudio.load(wav_path)
                _, T_total = wave.size()
                
                if max_samples is None:
                    samples = T_total
                else:
                    if T_total < max_samples:
                        samples = T_total
                    else:
                        samples = max_samples
                
                n_sources = 0

                for source_idx in range(max_n_sources):
                    wav_path = os.path.join(wav_root, 's{}'.format(source_idx+1), '{}.wav'.format(ID))
                    if not os.path.exists(wav_path):
                        break
                    n_sources += 1
                
                data = {
                    'sources': {},
                    'mixture': {}
                }
                
                for source_idx in range(n_sources):
                    source_data = {
                        'path': os.path.join('s{}'.format(source_idx+1), '{}.wav'.format(ID)),
                        'start': 0,
                        'end': samples
                    }
                    data['sources']['s{}'.format(source_idx+1)] = source_data
                
                mixture_data = {
                    'path': os.path.join('mix', '{}.wav'.format(ID)),
                    'start': 0,
                    'end': samples
                }
                data['mixture'] = mixture_data
                data['ID'] = ID
            
                self.json_data.append(data)
    
    def __getitem__(self, idx):
        mixture, sources, _ = super().__getitem__(idx)
        segment_ID = self.json_data[idx]['ID']
    
        return mixture, sources, segment_ID

class MixedNumberSourcesTrainDataLoader(TrainDataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.collate_fn = mixed_number_sources_train_collate_fn

class MixedNumberSourcesEvalDataLoader(EvalDataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.collate_fn = mixed_number_sources_eval_collate_fn

def mixed_number_sources_train_collate_fn(batch):
    batched_mixture, batched_sources = [], []

    for mixture, sources in batch:
        batched_mixture.append(mixture)
        batched_sources.append(sources)

    batched_mixture = nn.utils.rnn.pad_sequence(batched_mixture, batch_first=True)
    batched_sources = nn.utils.rnn.pack_sequence(batched_sources, enforce_sorted=False) # n_sources is different from data to data
    
    return batched_mixture, batched_sources

def mixed_number_sources_eval_collate_fn(batch):
    batched_mixture, batched_sources, segment_ID = [], [], []
    batched_segment_ID = []

    for mixture, sources, segment_ID in batch:
        batched_mixture.append(mixture)
        batched_sources.append(sources)
        batched_segment_ID.append(segment_ID)

    batched_mixture = nn.utils.rnn.pad_sequence(batched_mixture, batch_first=True)
    batched_sources = nn.utils.rnn.pack_sequence(batched_sources, enforce_sorted=False) # n_sources is different from data to data
    
    return batched_mixture, batched_sources, batched_segment_ID

if __name__ == '__main__':
    torch.manual_seed(111)
    
    n_sources = 2
    data_type = 'tt'
    min_max = 'max'
    wav_root = "../../../../../db/wsj0-mix/{}speakers/wav8k/{}/{}".format(n_sources, min_max, data_type)
    list_path = "../../../../dataset/wsj0-mix/{}speakers/mix_{}_spk_{}_{}_mix".format(n_sources, n_sources, min_max, data_type)
    
    dataset = WaveTrainDataset(wav_root, list_path, n_sources=n_sources)
    loader = TrainDataLoader(dataset, batch_size=4, shuffle=True)
    
    for mixture, sources in loader:
        print(mixture.size(), sources.size())
        break
    
    dataset = WaveTestDataset(wav_root, list_path, n_sources=n_sources)
    loader = EvalDataLoader(dataset, batch_size=1, shuffle=False)
    
    for mixture, sources, segment_ID in loader:
        print(mixture.size(), sources.size())
        print(segment_ID)
        break
