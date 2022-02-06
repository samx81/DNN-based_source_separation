import os

import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import h5py
import random

from algorithm.frequency_mask import ideal_binary_mask, ideal_ratio_mask, wiener_filter_mask
from pysndfx import AudioEffectsChain

EPS = 1e-12

def mask_samples(samps, mask_length=10, max_mask=100, masking_type='zero'):
    T = len(samps)
    #num_mask = max_mask
    num_mask = np.random.randint(0, max_mask)
    for i in range(0, num_mask):
        #mask_length = np.random.uniform(low=0.0, high=mask_length)
        #mask_length = int(mask_length)
        t0 = np.random.randint(0, T - mask_length)
        if masking_type == 'zero':
            samps[t0:t0+mask_length] = 0
        elif masking_type == 'mean':
            samps[t0:t0+mask_length] = np.mean(samps)
    return samps

def spliceout(spec_noisy, spec_clean, interval_num, max_interval, noise=None ):
    # interval_num = N, max_interval = T
    spec_len = spec_noisy.shape[-1] # tau
    mask = np.ones(spec_len, dtype=bool)
    for i in range(interval_num):
        remove_length = np.random.randint(max_interval)
        start = np.random.randint(spec_len - remove_length)
        mask[start : start + remove_length] = False
    noisy, clean = spec_noisy[mask], spec_clean[mask]
    if noise is not None:
        return noisy, clean, noise[mask] 
    return noisy, clean
    # return noisy

def speed_perturb(signal, c_speed):
        # audio effect
        AE = AudioEffectsChain()
        AE = AE.speed(c_speed)
        fx = (AE)
        signal = fx(signal)
        return signal

def masktwice_collatefn(data):
    """
    Grab a list of tuple ( mixture, source )
    Where mixture and tuple already have batch dim, so we only need to cat them
    """
    # print(len(data))
    mix_lst, src_lst = [], []
    for mix, src in data:
        mix_lst.append(mix)
        src_lst.append(src)

    mixture = torch.cat(mix_lst, dim=0)
    source  = torch.cat(src_lst, dim=0)
    # print(mixture.shape, flush=True)
    return mixture, source

class WSJ0Dataset(torch.utils.data.Dataset):
    def __init__(self, wav_root, list_path):
        super().__init__()
        
        self.wav_root = os.path.abspath(wav_root)
        self.list_path = os.path.abspath(list_path)

class WaveDataset(WSJ0Dataset):
    def __init__(self, wav_root, list_path, samples=32000, least_sample=None, overlap=None, 
                n_sources=2, chunk=True, noise_loss=False, use_h5py=False,
                mask=None, shift=False, speed_perturb=False):
        super().__init__(wav_root, list_path)

        wav_root = os.path.abspath(wav_root)

        self.noise_loss = noise_loss
        self.use_h5py = use_h5py
        
        self.chunk = chunk
        self.samples = samples if samples else 16000 * 10
        self.overlap = overlap if overlap else self.samples // 2
        self.least_sample = least_sample if least_sample else self.samples // 4

        self.speed_perturb = speed_perturb
        self.shift = shift
        self.mask = mask
        self.mask_len = 10   #10
        self.max_mask = 150  #160

        print(self.least_sample)
        
        self.json_data = []

        if use_h5py:
            noisy_list_path = os.path.join(wav_root, 'noisy.scp')
            num_lines = sum(1 for line in open(noisy_list_path))
            h5_name = f'dataset_16k_noise.h5'if noise_loss else f'dataset_16k.h5'
            h5_file = os.path.join(wav_root, h5_name)
            print(h5_file, num_lines)
            if not os.path.isfile(h5_file):
                raise Exception()

            h5_file = h5py.File(h5_file, 'r')

            self.clean_dset = h5_file.get('clean', None)
            self.noisy_dset = h5_file.get('noisy', None)
            self.id_dset = h5_file.get('id', None)
            self.len_dset = h5_file.get('len', None)

            if self.noise_loss:
                self.noise_dset = h5_file.get('noise', None)

            # print(type(self.id_dset))
            self.json_data = []
            total= []
            for i,v in enumerate(list(self.id_dset)[:num_lines]):
                if self.len_dset[i] > self.least_sample:
                    self.json_data.append(i)
                else:
                    # print(v, flush=True)
                    total.append(self.len_dset[i])

            # print(np.mean(total))
        else:
            clean_list_path = os.path.join(wav_root, 'clean.scp')
            noisy_list_path = os.path.join(wav_root, 'noisy.scp')
            noise_list_path = os.path.join(wav_root, 'noise.scp')
            length_list_path = os.path.join(wav_root, 'length.list')

            with open(noisy_list_path) as f:
                ff = f.readlines()
                noisy_dict = {line.split()[0]: line.split()[1] for line in ff}
            
            if self.noise_loss:
                with open(noise_list_path) as f:
                    ff = f.readlines()
                    noise_dict = {line.split()[0]: line.split()[1] for line in ff}

            if not os.path.exists(clean_list_path):
                print("No Clean Ref. files, make sure you're running eval stage.")
                print("Will use noisy data as clean ref.")
                clean_dict = noisy_dict
            else:
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
                for id in tqdm(sorted(noisy_dict.keys()),"Preparing Data Info..."):
                    len_dict[id] = torchaudio.info(noisy_dict[id]).num_frames
                    print_str += f'{id} {len_dict[id]}\n'
                with open(length_list_path, 'w') as f:
                    f.write(print_str)


            for id in tqdm(noisy_dict.keys(),"Loading Dataset..."):
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

                        data = {}
                        
                        data['sources'] = {
                            'path': clean_dict[id],
                            'start': start_idx, 'end': end_idx
                        }

                        if self.noise_loss:
                            data['noise'] = {
                                'path': noise_dict[id],
                                'start': start_idx, 'end': end_idx
                            }
                        
                        data['mixture'] = {
                            'path': noisy_dict[id],
                            'start': start_idx, 'end': end_idx
                        }
                        data['ID'] = f'{id}-{start_idx}-{start_idx}'
                    
                        self.json_data.append(data)
                else:
                    data = {}
                        
                    data['sources'] = {
                        'path': clean_dict[id],
                        'start': 0, 'end': T_total
                    }
                    
                    data['mixture'] = {
                        'path': noisy_dict[id],
                        'start': 0, 'end': T_total
                    }
                    data['ID'] = f'{id}'
                
                    self.json_data.append(data)

        print('Use h5py', use_h5py)
        
        print('Training samples num:', len(self.json_data), flush=True)
        self.spliceout_len = 64 #150
        self.spliceout_num = 2  #10

    def random_start_end(self, target_len, least_len, sample_len):
        random_start = random.randint(0, sample_len - least_len)
        random_end = target_len + random_start
        random_end = sample_len if random_end > sample_len else random_end
        
        return random_start, random_end
        
    def __getitem__(self, idx):
        """
        Returns:
            mixture (1, T) <torch.Tensor>
            sources (n_sources, T) <torch.Tensor>
            segment_IDs (n_sources,) <list<str>>
        """
        sources = []

        if self.use_h5py:
            idx = self.json_data[idx]
            clean = np.array(self.clean_dset[idx])
            noisy = np.array(self.noisy_dset[idx])

            if self.noise_loss:
                noise = np.array(self.noise_dset[idx])
            
            sample_length = self.len_dset[idx]
            ###
            # few situation,
            # 1. audio > chuck, random choose
            # 2. audio < chuck, padding ( but do it later )
            # 3. audio < least_sample, drop 
            if self.chunk:
                if sample_length > self.samples:
                    # random_start = random.randint(0, sample_length - self.least_sample)
                    # random_end = self.samples + random_start
                    # random_end = sample_length if random_end > sample_length else random_end
                    random_start, random_end= self.random_start_end(self.samples, self.least_sample, sample_length)

                    clean = clean[random_start:random_end]
                    noisy = noisy[random_start:random_end]

                    if self.noise_loss:
                        noise = noise[random_start:random_end]
                else:
                    clean = clean[:sample_length]
                    noisy = noisy[:sample_length]

                    if self.noise_loss:
                        noise = noise[:sample_length]

            # Augmentation
            self.shift_len = 10000
            self.speed = (0.95, 1.05)
            if self.speed_perturb:
                c_speed = random.uniform(*self.speed)
                noisy, clean = speed_perturb(noisy, c_speed), speed_perturb(clean, c_speed)
                if len(noisy) > self.samples:
                    noisy, clean = noisy[:self.samples], clean[:self.samples]         

            if self.mask == None:
                pass
            elif self.mask == 'spliceout':
                if self.noise_loss:
                    noisy, clean, noise = spliceout(noisy, clean, self.spliceout_num, self.spliceout_len,noise=noise)
                else:
                    noisy, clean = spliceout(noisy, clean, self.spliceout_num, self.spliceout_len)
                # noisy = spliceout(noisy, clean, self.spliceout_num, self.spliceout_len)
            elif self.mask=='zero':
                noisy = mask_samples(
                    noisy, mask_length=self.mask_len,
                    max_mask=self.max_mask, masking_type=self.mask
                )
            elif self.mask=='zerotwice':
                n1 = mask_samples(
                    noisy, mask_length=self.mask_len,
                    max_mask=self.max_mask, masking_type=self.mask
                )
                n2 = mask_samples(
                    noisy, mask_length=self.mask_len,
                    max_mask=self.max_mask, masking_type=self.mask
                )
            elif self.mask=='TENET':
                n1 = mask_samples(
                    noisy, mask_length=self.mask_len,
                    max_mask=self.max_mask, masking_type=self.mask
                )
                n2, c2 = np.flip(noisy, axis=0), np.flip(clean, axis=0)
                n2 = mask_samples(
                    n2, mask_length=self.mask_len,
                    max_mask=self.max_mask, masking_type=self.mask
                )
                
            elif self.mask == 'both':
                noisy, clean = spliceout(noisy, clean, self.spliceout_num, self.spliceout_len)
                noisy = mask_samples(
                    noisy, mask_length=self.mask_len,
                    max_mask=self.max_mask, masking_type='zero'
                )
            else:
                raise ValueError("Not support mask {}".format(self.mask))


            if self.shift:
                if self.mask == 'zerotwice':
                    shift_offset = np.random.randint(0, self.shift_len, size=1)
                    n1, n2, clean = np.roll(n1, shift_offset), np.roll(n2, shift_offset), np.roll(clean, shift_offset)
                if self.mask == 'TENET':
                    shift_offset = np.random.randint(0, self.shift_len, size=1)
                    n1, n2, clean, c2 = np.roll(n1, shift_offset), np.roll(n2, shift_offset), np.roll(clean, shift_offset), np.roll(c2, shift_offset)
                else:
                    shift_offset = np.random.randint(0, self.shift_len, size=1)
                    noisy, clean = np.roll(noisy, shift_offset), np.roll(clean, shift_offset)
            
            # Another masking variant is in galrnet.py
            if self.mask=='zerotwice':
                wave = torch.tensor(clean).unsqueeze(0)
                wave = torch.stack([wave, wave], dim=0)
                n1 = torch.tensor(n1).unsqueeze(0)
                n2 = torch.tensor(n2).unsqueeze(0)
                mixture = torch.stack([n1, n2], dim=0)
                # Leave the unpack and stacking to driver?
            elif self.mask=='TENET':
                wave = torch.tensor(clean).unsqueeze(0)
                c2 = torch.tensor(c2).unsqueeze(0)
                wave = torch.stack([wave, c2], dim=0)
                n1 = torch.tensor(n1).unsqueeze(0)
                n2 = torch.tensor(n2).unsqueeze(0)
                mixture = torch.stack([n1, n2], dim=0)
                # Leave the unpack and stacking to driver?
            else:
                wave = torch.tensor(clean).unsqueeze(0)
                mixture = torch.tensor(noisy).unsqueeze(0)

            if self.noise_loss:
                if self.mask=='zerotwice':
                    noise = torch.tensor(noise).unsqueeze(0)
                    noise = torch.stack([noise, noise], dim=0)
                else:
                    noise = torch.tensor(noise).unsqueeze(0)

            if len(clean) < self.samples:
                P = self.samples - len(clean)
                wave = F.pad(wave, (0, P), "constant")
                if self.noise_loss:
                    noise = F.pad(noise, (0, P), "constant")

            if len(noisy) < self.samples:
                if self.mask == 'spliceout':
                    MP = self.samples - len(noisy)
                    # MP_L = MP//2
                    # ML_R = MP//2 + 1 if MP % 2 != 0 else MP_L
                    # assert MP == P, f'沒做好, {MP}, {P}'
                    mixture = F.pad(mixture, (0, MP), "constant")
                    # mixture = F.pad(mixture, (MP_L, ML_R), "constant")
                else:
                    mixture = F.pad(mixture, (0, P), "constant")
            
            segment_ID = str(self.id_dset[idx])

        else:
            data = self.json_data[idx]

            mixture_data = data['mixture']
            start, end = mixture_data['start'], mixture_data['end']
                
            segment_ID = self.json_data[idx]['ID'] + '_{}-{}'.format(start, end)

            wav_path = os.path.join(self.wav_root, mixture_data['path'])
            wave, _ = torchaudio.load(wav_path)
            mixture = wave[:, start: end]

            if self.noise_loss:
                wav_path = data['noise']['path']
                wave, _ = torchaudio.load(wav_path)
                noise = wave[:, data['noise']['start']: data['noise']['end']]
            
            wav_path = data['sources']['path']
            wave, _ = torchaudio.load(wav_path)
            wave = wave[:, data['sources']['start'] : data['sources']['end']]
            
            wav_len = end - start
            if self.chunk and wav_len < self.samples:
                P = self.samples - wav_len
                wave = F.pad(wave, (0, P), "constant")
                mixture = F.pad(mixture, (0, P), "constant")
                if self.noise_loss:
                    noise = F.pad(noise, (0, P), "constant")

            if self.use_h5py:
                self.exist_dset[idx] = True
                self.clean_dset[idx] = wave[0]
                self.noisy_dset[idx] = mixture[0]
        
        sources.append(wave)

        if self.noise_loss:
            sources.append(noise)
        if self.mask in ['zerotwice', 'TENET']:
            sources = torch.cat(sources, dim=1)
        else:
            sources = torch.cat(sources, dim=0)
        return mixture, sources, segment_ID
        
    def __len__(self):
        return len(self.json_data)

class WaveTrainDataset(WaveDataset):
    def __init__(self, wav_root, list_path, samples=32000, least_sample=None,overlap=None, n_sources=2, noise_loss=False, use_h5py=False, mask=None, shift=False, speed_perturb=False):
        super().__init__(wav_root, list_path, samples=samples, overlap=overlap, least_sample=least_sample,
                        n_sources=n_sources, noise_loss=noise_loss, use_h5py=use_h5py, mask=mask, speed_perturb=speed_perturb, shift=shift)
    
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
        s
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
