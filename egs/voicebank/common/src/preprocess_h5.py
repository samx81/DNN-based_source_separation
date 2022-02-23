import os

import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import h5py

# from algorithm.frequency_mask import ideal_binary_mask, ideal_ratio_mask, wiener_filter_mask

EPS = 1e-12

def package_into_h5(wav_root='data/voicebank/tr', sr=16000, noise=False):

    if sr != 16000:
        print('Not support resampling for now')
        return

    # Gets Path
    wav_root = os.path.abspath(wav_root)

    clean_list_path = os.path.join(wav_root, 'clean.scp')
    noisy_list_path = os.path.join(wav_root, 'noisy.scp')
    noise_list_path = os.path.join(wav_root, 'noise.scp')
    length_list_path = os.path.join(wav_root, 'length.list')

    # Open list
    with open(noisy_list_path) as f:
        ff = f.readlines()
        noisy_dict = {line.split()[0]: line.split()[1] for line in ff}
        
    with open(clean_list_path) as f:
        ff = f.readlines()
        clean_dict = {line.split()[0]: line.split()[1] for line in ff}

    if noise:
        print('Will include noise.')
        with open(noise_list_path) as f:
            ff = f.readlines()
            noise_dict = {line.split()[0]: line.split()[1] for line in ff}

    # Extract Audio Length
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

    # Open h5py
    h5_name = f'dataset_{sr//1000}k.h5' if not noise else f'dataset_{sr//1000}k_noise.h5'
    h5_file = os.path.join(wav_root, h5_name)
    if os.path.isfile(h5_file):
        print('Dataset for h5 is already exists!')

    h5_file = h5py.File(h5_file, 'a')

    num_audios = len(len_dict)
    max_audio_length = max(len_dict.values())
    dt = h5py.special_dtype(vlen=str)
    
    # Create dataset
    id_dset = h5_file.create_dataset("id", (num_audios,), dtype=dt)
    len_dset = h5_file.create_dataset("len", (num_audios, ), dtype='i')
    clean_dset = h5_file.create_dataset("clean", (num_audios, max_audio_length), dtype='f', compression="gzip", fillvalue=0)
    noisy_dset = h5_file.create_dataset("noisy", (num_audios, max_audio_length), dtype='f', compression="gzip", fillvalue=0)
    if noise:
        noise_dset = h5_file.create_dataset("noise", (num_audios, max_audio_length), dtype='f', compression="gzip", fillvalue=0)

    for idx, id in tqdm(enumerate(noisy_dict.keys()),"Loading Dataset...", total=num_audios):
        T_total = len_dict[id]
        id_dset[idx] = id
        len_dset[idx] = T_total

        wav_path = os.path.join(wav_root, clean_dict[id])
        wave, _ = torchaudio.load(wav_path)
        clean_dset[idx, :T_total] = wave[0]

        wav_path = os.path.join(wav_root, noisy_dict[id])
        wave, _ = torchaudio.load(wav_path)
        noisy_dset[idx, :T_total] = wave[0]

        if noise:
            wav_path = os.path.join(wav_root, noise_dict[id])
            wave, _ = torchaudio.load(wav_path)
            noise_dset[idx, :T_total] = wave[0]


# class WaveDataset(WSJ0Dataset):
#     def __init__(self, wav_root, list_path, samples=32000, least_sample=None, overlap=None, n_sources=2, chunk=True, noise_loss=False, use_h5py=False):
#         super().__init__(wav_root, list_path)

#         wav_root = os.path.abspath(wav_root)
#         clean_list_path = os.path.join(wav_root, 'clean.scp')
#         noisy_list_path = os.path.join(wav_root, 'noisy.scp')
#         noise_list_path = os.path.join(wav_root, 'noise.scp')
#         length_list_path = os.path.join(wav_root, 'length.list')

#         # noisy_h5 = os.path.join(wav_root, 'noisy.h5')
#         # noisy_h5 = h5py.File(noisy_h5, 'a')
#         # noise_h5 = os.path.join(wav_root, 'clean.scp')

#         self.noise_loss = noise_loss
#         self.use_h5py = use_h5py
        
#         self.chunk = chunk
#         self.samples = samples if samples else 16000 * 10
#         self.overlap = overlap if overlap else self.samples // 2
#         self.least_sample = least_sample if least_sample else self.samples // 4
        
#         self.json_data = []

#         with open(noisy_list_path) as f:
#             ff = f.readlines()
#             noisy_dict = {line.split()[0]: line.split()[1] for line in ff}
        
#         if self.noise_loss:
#             with open(noise_list_path) as f:
#                 ff = f.readlines()
#                 noise_dict = {line.split()[0]: line.split()[1] for line in ff}

#         if not os.path.exists(clean_list_path):
#             print("No Clean Ref. files, make sure you're running eval stage.")
#             print("Will use noisy data as clean ref.")
#             clean_dict = noisy_dict
#         else:
#             with open(clean_list_path) as f:
#                 ff = f.readlines()
#                 clean_dict = {line.split()[0]: line.split()[1] for line in ff}

#         if os.path.exists(length_list_path):
#             with open(length_list_path, 'r') as f:
#                 ff = f.readlines()
#                 len_dict = {line.split()[0]: int(line.split()[1]) for line in ff}
#         else:
#             len_dict = {}
#             print_str = ''
#             for id in tqdm(sorted(noisy_dict.keys()),"Preparing Data Info..."):
#                 len_dict[id] = torchaudio.info(noisy_dict[id]).num_frames
#                 print_str += f'{id} {len_dict[id]}\n'
#             with open(length_list_path, 'w') as f:
#                 f.write(print_str)


#         for id in tqdm(noisy_dict.keys(),"Loading Dataset..."):
#             # wave, _ = torchaudio.load(clean_dict[id])
            
#             # _, T_total = wave.size()
#             T_total = len_dict[id]
#             if chunk:
#                 for start_idx in range(0, T_total, samples - self.overlap):
                
#                     end_idx = start_idx + samples
#                     if end_idx > T_total:
#                         end_idx = T_total
#                     if end_idx - start_idx < self.least_sample:
#                         break
#                     data = {
#                         'sources': {},
#                         'mixture': {}
#                     }
                    
#                     source_data = {
#                         'path': clean_dict[id],
#                         'start': start_idx,
#                         'end': end_idx
#                     }
#                     data['sources'] = source_data

#                     if self.noise_loss:
#                         noise_data = {
#                             'path': noise_dict[id],
#                             'start': start_idx,
#                             'end': end_idx
#                         }
#                         data['noise'] = noise_data
                    
#                     mixture_data = {
#                         'path': noisy_dict[id],
#                         'start': start_idx,
#                         'end': end_idx
#                     }
#                     data['mixture'] = mixture_data
#                     data['ID'] = f'{id}-{start_idx}-{start_idx}'
                
#                     self.json_data.append(data)
#             else:
#                 data = {
#                     'sources': {},
#                     'mixture': {}
#                 }
                    
#                 source_data = {
#                     'path': clean_dict[id],
#                     'start': 0,
#                     'end': T_total
#                 }
#                 data['sources'] = source_data
                
#                 mixture_data = {
#                     'path': noisy_dict[id],
#                     'start': 0,
#                     'end': T_total
#                 }
#                 data['mixture'] = mixture_data
#                 data['ID'] = f'{id}'
            
#                 self.json_data.append(data)

#         print('Use h5py', use_h5py)

#         if use_h5py:
#             self.id2idx = {v['ID']:i for i, v in enumerate(self.json_data)}
#             clean_h5 = os.path.join(wav_root, f'clean_{samples}.h5')
#             if os.path.isfile(clean_h5):
#                 clean_h5 = h5py.File(clean_h5, 'r')
#             else:  
#                 clean_h5 = h5py.File(clean_h5, 'a')

#             if 'clean' in clean_h5:
#                 self.clean_dset = clean_h5.get('clean', None)
#                 self.noisy_dset = clean_h5.get('noisy', None)
#                 self.exist_dset = clean_h5.get('exist', None)
#                 # self.id_dset = clean_h5.get('id', None)
                
#             else:
#                 self.clean_dset = clean_h5.create_dataset("clean", (len(self.json_data), samples), dtype='f')
#                 self.noisy_dset = clean_h5.create_dataset("noisy", (len(self.json_data), samples), dtype='f')
#                 self.exist_dset = clean_h5.create_dataset("exist", (len(self.json_data), ), dtype='?', fillvalue=False)
#                 # dt = h5py.special_dtype(vlen=str)
#                 # self.id_dset = clean_h5.create_dataset("id", (len(self.json_data),), dtype=dt)
#                 # self.id2idx = {v:i for i,v in enumerate(self.id_dset)}
        
#         print(len(self.json_data), flush=True)
        
#     def __getitem__(self, idx):
#         """
#         Returns:
#             mixture (1, T) <torch.Tensor>
#             sources (n_sources, T) <torch.Tensor>
#             segment_IDs (n_sources,) <list<str>>
#         """
#         data = self.json_data[idx]
#         sources = []

#         mixture_data = data['mixture']
#         start, end = mixture_data['start'], mixture_data['end']
            
#         segment_ID = self.json_data[idx]['ID'] + '_{}-{}'.format(start, end)

#         if self.use_h5py and self.exist_dset[idx]:
#             wave = torch.tensor(np.array(self.clean_dset[idx])).unsqueeze(0)
#             mixture = torch.tensor(np.array(self.noisy_dset[idx])).unsqueeze(0)
#         else:
#             wav_path = os.path.join(self.wav_root, mixture_data['path'])
#             wave, _ = torchaudio.load(wav_path)
#             mixture = wave[:, start: end]

#             if self.noise_loss:
#                 noise_data = data['noise']
#                 start, end = noise_data['start'], noise_data['end']
#                 wav_path = noise_data['path']
#                 wave, _ = torchaudio.load(wav_path)
#                 noise = wave[:, start: end]
            
#             source_data = data['sources']
#             start, end = source_data['start'], source_data['end']
#             wav_path = source_data['path']
#             wave, _ = torchaudio.load(wav_path)
#             wave = wave[:, start: end]
            
#             wav_len = end - start
#             if self.chunk and wav_len < self.samples:
#                 P = self.samples - wav_len
#                 wave = F.pad(wave, (0, P), "constant")
#                 mixture = F.pad(mixture, (0, P), "constant")
#                 if self.noise_loss:
#                     noise = F.pad(noise, (0, P), "constant")

#             if self.use_h5py:
#                 self.exist_dset[idx] = True
#                 self.clean_dset[idx] = wave[0]
#                 self.noisy_dset[idx] = mixture[0]
        
#         sources.append(wave)
#         if self.noise_loss:
#             sources.append(noise)
        
#         sources = torch.cat(sources, dim=0)
        
#         return mixture, sources, segment_ID
        
#     def __len__(self):
#         return len(self.json_data)

if __name__ == '__main__':
    package_into_h5(wav_root='data/4sec_aidns/tr', noise=False)
