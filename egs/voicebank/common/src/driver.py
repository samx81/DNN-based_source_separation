import os, sys
import shutil
import subprocess
import time
import uuid

import numpy as np
from mir_eval.separation import bss_eval_sources
import torch
import torchaudio
import torch.nn as nn
from pystoi import stoi as pystoi
from tqdm import tqdm

from utils.utils import draw_loss_curve
from criterion.pit import pit

BITS_PER_SAMPLE_WSJ0 = 16
MIN_PESQ = -0.5

class MyNegSISNR(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()

        self.eps = eps
        
    def forward(self, input, target):
        """
        Args:
            input (batch_size, T) or (batch_size, n_sources, T), or (batch_size, n_sources, n_mics, T)
            target (batch_size, T) or (batch_size, n_sources, T) or (batch_size, n_sources, n_mics, T)
        Returns:
            loss (batch_size,) or (batch_size, n_sources) or (batch_size, n_sources, n_mics)
        """
        
        return -torch.mean(cal_sisnr(input, target, self.eps))
    

def cal_sisnr(x, s, eps=1e-8):
    """
    Arguments:
    x: separated signal, N x S tensor
    s: reference signal, N x S tensor
    Return:
    sisnr: N tensor
    """

    def l2norm(mat, keepdim=False):
        return torch.norm(mat, dim=-1, keepdim=keepdim)

    if x.shape != s.shape:
        raise RuntimeError(
            "Dimention mismatch when calculate si-snr, {} vs {}".format(
                x.shape, s.shape))

    x_zm = x - torch.mean(x, dim=-1, keepdim=True)
    s_zm = s - torch.mean(s, dim=-1, keepdim=True)
    t = torch.sum(
        x_zm * s_zm, dim=-1,keepdim=True) * s_zm / (l2norm(s_zm, keepdim=True)**2 + eps)
    return 20 * torch.log10(eps + l2norm(t) / (l2norm(x_zm - t) + eps))

def si_snr(x, s, eps=1e-8, remove_dc=True):
    """
    Compute Si-SNR
    Arguments:
        x: vector, enhanced/separated signal
        s: vector, reference signal(ground truth)
    """

    def vec_l2norm(x):
        return np.linalg.norm(x, 2)

    # zero mean, seems do not hurt results
    if remove_dc:
        x_zm = x - np.mean(x)
        s_zm = s - np.mean(s)
        t = np.inner(x_zm, s_zm) * s_zm / (vec_l2norm(s_zm)**2 + eps)
        n = x_zm - t
    else:
        t = np.inner(x, s) * s / (vec_l2norm(s)**2 + eps)
        n = x - t
    return 20 * np.log10(vec_l2norm(t) / (vec_l2norm(n) + eps) + eps)

class TrainerBase:
    def __init__(self, model, loader, pit_criterion, optimizer, args):
        self.train_loader, self.valid_loader = loader['train'], loader['valid']
        
        self.model = model
        
        self.criterion = pit_criterion
        self.optimizer = optimizer
        
        self._reset(args)
    
    def _reset(self, args):
        self.sr = args.sr
        self.n_sources = args.n_sources
        self.max_norm = args.max_norm
        
        self.model_dir = args.model_dir
        self.loss_dir = args.loss_dir
        self.sample_dir = args.sample_dir
        
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.loss_dir, exist_ok=True)
        os.makedirs(self.sample_dir, exist_ok=True)
        
        self.epochs = args.epochs

        self.noise_loss = args.noise_loss
        print(f'train with noise:{self.noise_loss}')
        
        self.train_loss = torch.empty(self.epochs)
        self.valid_loss = torch.empty(self.epochs)
        
        self.use_cuda = args.use_cuda
        
        if args.continue_from:
            package = torch.load(args.continue_from, map_location=lambda storage, loc: storage)
            
            self.start_epoch = package['epoch']
            
            self.train_loss[:self.start_epoch] = package['train_loss'][:self.start_epoch]
            self.valid_loss[:self.start_epoch] = package['valid_loss'][:self.start_epoch]
            
            self.best_loss = package['best_loss']
            self.prev_loss = self.valid_loss[self.start_epoch-1]
            self.no_improvement = package['no_improvement']
            
            if isinstance(self.model, nn.DataParallel):
                self.model.module.load_state_dict(package['state_dict'])
            else:
                self.model.load_state_dict(package['state_dict'])
            
            self.optimizer.load_state_dict(package['optim_dict'])
        else:
            model_path = os.path.join(self.model_dir, "best.pth")
            
            if os.path.exists(model_path):
                if args.overwrite:
                    print("Overwrite models.")
                else:
                    raise ValueError("{} already exists. If you continue to run, set --overwrite to be True.".format(model_path))
            
            self.start_epoch = 0
            
            self.best_loss = float('infinity')
            self.prev_loss = float('infinity')
            self.no_improvement = 0
    
    def run(self):
        for epoch in range(self.start_epoch, self.epochs):
            start = time.time()
            train_loss, valid_loss = self.run_one_epoch(epoch)
            end = time.time()
            
            print("[Epoch {}/{}] loss (train): {:.5f}, loss (valid): {:.5f}, {:.3f} [sec], best_loss:{:.5f}".format(
                epoch+1, self.epochs, train_loss, valid_loss, end - start, self.best_loss), flush=True)
            
            self.train_loss[epoch] = train_loss
            self.valid_loss[epoch] = valid_loss
            
            if valid_loss < self.best_loss:
                self.best_loss = valid_loss
                self.no_improvement = 0
                model_path = os.path.join(self.model_dir, "best.pth")
                self.save_model(epoch, model_path)
            else:
                if valid_loss >= self.prev_loss:
                    self.no_improvement += 1
                    if self.no_improvement >= 10:
                        print("Stop training")
                        break
                    if self.no_improvement >= 3:
                        for param_group in self.optimizer.param_groups:
                            prev_lr = param_group['lr']
                            lr = 0.5 * prev_lr
                            print("Learning rate: {} -> {}".format(prev_lr, lr))
                            param_group['lr'] = lr
                else:
                    self.no_improvement = 0
            
            self.prev_loss = valid_loss
            
            model_path = os.path.join(self.model_dir, "last.pth")
            self.save_model(epoch, model_path)
            
            save_path = os.path.join(self.loss_dir, "loss.png")
            draw_loss_curve(train_loss=self.train_loss[:epoch+1], valid_loss=self.valid_loss[:epoch+1], save_path=save_path)
    
    def run_one_epoch(self, epoch):
        """
        Training
        """
        train_loss = self.run_one_epoch_train(epoch)
        valid_loss = self.run_one_epoch_eval(epoch)

        return train_loss, valid_loss
    
    def run_one_epoch_train(self, epoch):
        """
        Training
        """
        self.model.train()
        
        train_loss = 0
        n_train_batch = len(self.train_loader)
        t = tqdm(enumerate(self.train_loader), leave=False,
                    total=(len(self.train_loader)//self.train_loader.batch_size))
        for idx, (mixture, sources) in t:
            if self.use_cuda:
                mixture = mixture.cuda()
                sources = sources.cuda()
            
            estimated_sources, _ = self.model(mixture)
            # loss = self.pit_criterion(estimated_sources[:,0], sources)
            # print(estimated_sources.shape, sources.shape)
            loss = self.criterion(estimated_sources[:,0], sources[:,0])
            if self.noise_loss:
                loss += self.criterion(estimated_sources[:,1], sources[:,1])
            
            self.optimizer.zero_grad()
            loss.backward()
            
            if self.max_norm:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_norm)
            
            self.optimizer.step()
            
            train_loss += loss.item()
            t.set_postfix_str(f'loss: {loss.item():.5f}')
            
            if (idx + 1) % 500 == 0:
                t.write("[Epoch {}/{}] iter {}/{} loss: {:.5f}".format(epoch+1, self.epochs, idx+1, n_train_batch, loss.item()))
        
        train_loss /= n_train_batch
        
        return train_loss
    
    def run_one_epoch_eval(self, epoch):
        """
        Validation
        """
        self.model.eval()
        
        valid_loss = 0
        n_valid = len(self.valid_loader.dataset)
        
        with torch.no_grad():
            t = tqdm(enumerate(self.valid_loader), leave=False,
                    total=(len(self.valid_loader)//self.valid_loader.batch_size))
            for idx, (mixture, sources, segment_IDs) in t:
                if self.use_cuda:
                    mixture = mixture.cuda()
                    sources = sources.cuda()
                output, _ = self.model(mixture)
                # loss, _ = self.pit_criterion(output[:,0], sources, batch_mean=False)
                
                #  loss = self.criterion(output[:,0], torch.squeeze(sources,0))
                print(output[:,0].shape, sources[:,0].shape)
                loss = self.criterion(output[:,0], sources[:,0])
                # loss = loss.sum(dim=0)
                valid_loss += loss.item()
                
                if idx < 5:
                    mixture = mixture[0].squeeze(dim=0).cpu()
                    estimated_sources = output[0].cpu()
                    
                    save_dir = os.path.join(self.sample_dir, segment_IDs[0])
                    os.makedirs(save_dir, exist_ok=True)
                    save_path = os.path.join(save_dir, "mixture.wav")
                    norm = torch.abs(mixture).max()
                    mixture = mixture / norm
                    signal = mixture.unsqueeze(dim=0) if mixture.dim() == 1 else mixture
                    torchaudio.save(save_path, signal, sample_rate=self.sr, bits_per_sample=BITS_PER_SAMPLE_WSJ0)
                    
                    for source_idx, estimated_source in enumerate(estimated_sources):
                        save_path = os.path.join(save_dir, "epoch{}-{}.wav".format(epoch+1, source_idx+1))
                        norm = torch.abs(estimated_source).max()
                        estimated_source = estimated_source / norm
                        signal = estimated_source.unsqueeze(dim=0) if estimated_source.dim() == 1 else estimated_source
                        torchaudio.save(save_path, signal, sample_rate=self.sr, bits_per_sample=BITS_PER_SAMPLE_WSJ0)
        
        valid_loss /= n_valid
        
        return valid_loss
    
    def save_model(self, epoch, model_path='./tmp.pth'):
        if isinstance(self.model, nn.DataParallel):
            package = self.model.module.get_config()
            package['state_dict'] = self.model.module.state_dict()
        else:
            package = self.model.get_package()
            package['state_dict'] = self.model.state_dict()
            
        package['optim_dict'] = self.optimizer.state_dict()
        
        package['best_loss'] = self.best_loss
        package['no_improvement'] = self.no_improvement
        
        package['train_loss'] = self.train_loss
        package['valid_loss'] = self.valid_loss
        
        package['epoch'] = epoch + 1
        
        torch.save(package, model_path)

class TesterBase:
    def __init__(self, model, loader, pit_criterion, args):
        self.loader = loader
        
        self.model = model
        
        self.pit_criterion = pit_criterion
        
        self._reset(args)
        
    def _reset(self, args):
        self.sr = args.sr
        self.n_sources = args.n_sources
        
        self.out_dir = args.out_dir
        
        if self.out_dir is not None:
            self.out_dir = os.path.abspath(args.out_dir)
            os.makedirs(self.out_dir, exist_ok=True)
        
        self.use_cuda = args.use_cuda
        self.metric = args.no_metric
        
        package = torch.load(args.model_path, map_location=lambda storage, loc: storage)
        
        if isinstance(self.model, nn.DataParallel):
            self.model.module.load_state_dict(package['state_dict'])
        else:
            self.model.load_state_dict(package['state_dict'])
    
    def run(self):
        self.model.eval()
        
        test_loss = 0
        test_loss_improvement = 0
        test_sdr_improvement = 0
        test_si_snr_score = 0
        test_sar = 0
        test_pesq = 0
        test_stoi = 0
        n_pesq_error = 0
        n_test = len(self.loader.dataset)

        print("ID, Loss, Loss improvement, SDR improvement, SI-SNR, SAR, PESQ",flush=True)
        if self.metric:
            tmp_dir = os.path.join(os.getcwd(), 'tmp')
            os.makedirs(tmp_dir, exist_ok=True)
            shutil.copy('./PESQ', os.path.join(tmp_dir, 'PESQ'))
            os.chdir(tmp_dir)
        
        with torch.no_grad():
            # for idx, (mixture, sources, segment_IDs) in enumerate(self.loader): # tqdm(enumerate(self.loader)):
            for idx, (mixture, sources, segment_IDs) in tqdm(enumerate(self.loader), total=len(self.loader),file=sys.stdout):

                if self.use_cuda:
                    mixture = mixture.cuda()
                    sources = sources.cuda()
                
                loss_mixture = self.pit_criterion(mixture, sources, batch_mean=False)
                loss_mixture = loss_mixture.sum(dim=0)
                
                output, _ = self.model(mixture)
                output = output[:,0] # -> only need 1st output
                
                loss = self.pit_criterion(output, sources.squeeze(dim=1), batch_mean=False)
                loss = loss.sum(dim=0)
                loss_improvement = loss_mixture.item() - loss.item()
                
                mixture = mixture[0].squeeze(dim=0).cpu() # -> (T,)
                sources = sources[0].cpu() # -> (n_sources, T)
                estimated_sources = output.cpu() # -> (n_sources, T)
                # perm_idx = perm_idx[0] # -> (n_sources,)
                segment_IDs = segment_IDs[0] # -> <str>

                si_snr_score, sdr_improvement, sar, stoi = 0,0,0,0

                if self.metric:
                # repeated_mixture = torch.tile(mixture, (self.n_sources, 1))
                    repeated_mixture = mixture
                    result_estimated = bss_eval_sources(
                        reference_sources=sources.numpy(),
                        estimated_sources=estimated_sources.numpy()
                    )
                    result_mixed = bss_eval_sources(
                        reference_sources=sources.numpy(),
                        estimated_sources=repeated_mixture.numpy()
                    )
                    si_snr_score = si_snr(estimated_sources.numpy(), sources.numpy())
                    sdr_improvement = np.mean(result_estimated[0] - result_mixed[0])
                    # sir_improvement = np.mean(result_estimated[1] - result_mixed[1])
                    sar = np.mean(result_estimated[2])
                    stoi = pystoi(np.squeeze(sources.numpy()), 
                                np.squeeze(estimated_sources.numpy()), self.sr) 

                norm = torch.abs(mixture).max()
                mixture /= norm
                mixture_ID = segment_IDs
                
                # Generate random number temporary wav file.
                random_ID = str(uuid.uuid4())

                # if self.out_dir is not None:
                #     mixture_path = os.path.join(self.out_dir, "{}.wav".format(mixture_ID))
                #     signal = mixture.unsqueeze(dim=0) if mixture.dim() == 1 else mixture
                #     torchaudio.save(mixture_path, signal, sample_rate=self.sr, bits_per_sample=BITS_PER_SAMPLE_WSJ0)

                source, estimated_source = sources, estimated_sources

                # Estimated source
                norm = torch.abs(estimated_source).max()
                estimated_source /= norm
                if self.out_dir is not None:
                    estimated_path = os.path.join(self.out_dir, "{}-estimated.wav".format(mixture_ID))
                else:
                    estimated_path = "tmp-estimated_{}.wav".format(random_ID)
                signal = estimated_source.unsqueeze(dim=0) if estimated_source.dim() == 1 else estimated_source
                torchaudio.save(estimated_path, signal, sample_rate=self.sr, bits_per_sample=BITS_PER_SAMPLE_WSJ0)
                
                pesq = 0
                if self.metric:
                    # Target
                    norm = torch.abs(source).max()
                    source /= norm
                    if self.out_dir is not None:
                        source_path = os.path.join(self.out_dir, "{}-target.wav".format(mixture_ID))
                    else:
                        source_path = "tmp-target_{}.wav".format(random_ID)
                    signal = source.unsqueeze(dim=0) if source.dim() == 1 else source
                    torchaudio.save(source_path, signal, sample_rate=self.sr, bits_per_sample=BITS_PER_SAMPLE_WSJ0)
                    
                    command = "./PESQ +{} {} {}".format(self.sr, source_path, estimated_path)
                    command += " | grep Prediction | awk '{print $5}'"
                    pesq_output = subprocess.check_output(command, shell=True)
                    pesq_output = pesq_output.decode().strip()
                    
                    if pesq_output == '':
                        # If processing error occurs in PESQ software, it is regarded as PESQ score is -0.5. (minimum of PESQ)
                        n_pesq_error += 1
                        pesq += MIN_PESQ
                    else:
                        pesq += float(pesq_output)
                    
                    subprocess.call("rm {}".format(source_path), shell=True)
                    # subprocess.call("rm {}".format(estimated_path), shell=True)
                    
                    pesq /= self.n_sources
                # print("{}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}".format(mixture_ID, loss.item(), loss_improvement, sdr_improvement, sir_improvement, sar, pesq),flush=True)
                # print("{}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}".format(mixture_ID, loss.item(), loss_improvement, sdr_improvement, si_snr_score, sar, pesq),flush=True)
                tqdm.write("{}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f} {:.3f}".format(mixture_ID, loss.item(), loss_improvement, sdr_improvement, si_snr_score, sar, pesq, stoi))


                test_loss += loss.item()
                test_loss_improvement += loss_improvement
                test_sdr_improvement += sdr_improvement
                test_si_snr_score += si_snr_score
                # test_sir_improvement += sir_improvement
                test_sar += sar
                test_pesq += pesq
                test_stoi += stoi
        
        os.chdir("../") # back to the original directory

        test_loss /= n_test
        test_loss_improvement /= n_test
        test_sdr_improvement /= n_test
        # test_sir_improvement /= n_test
        test_si_snr_score /= n_test
        test_sar /= n_test
        test_pesq /= n_test
        test_stoi /= n_test
            
        # print("Loss: {:.3f}, loss improvement: {:3f}, SDR improvement: {:3f}, SIR improvement: {:3f}, SAR: {:3f}, PESQ: {:.3f}".format(test_loss, test_loss_improvement, test_sdr_improvement, test_sir_improvement, test_sar, test_pesq))
        print("Loss: {:.3f}, loss-i: {:3f}, SDRi: {:3f}, SI-SNR: {:3f}, SAR: {:3f}, PESQ: {:.3f}, STOI: {:.3f}".format(
            test_loss, test_loss_improvement, test_sdr_improvement, test_si_snr_score, test_sar, test_pesq, test_stoi))
        print("Evaluation of PESQ returns error {} times.".format(n_pesq_error))

class Trainer(TrainerBase):
    def __init__(self, model, loader, pit_criterion, optimizer, args):
        super().__init__(model, loader, pit_criterion, optimizer, args)

class Tester(TesterBase):
    def __init__(self, model, loader, pit_criterion, args):
        super().__init__(model, loader, pit_criterion, args)

class AttractorTrainer(TrainerBase):
    def __init__(self, model, loader, criterion, optimizer, args):
        self.train_loader, self.valid_loader = loader['train'], loader['valid']
        
        self.model = model
        
        self.criterion = criterion
        self.optimizer = optimizer
        
        self._reset(args)
    
    def _reset(self, args):
        # Override
        super()._reset(args)

        self.fft_size, self.hop_size = args.fft_size, args.hop_size

        if args.window_fn:
            if args.window_fn == 'hann':
                self.window = torch.hann_window(self.fft_size, periodic=True)
            elif args.window_fn == 'hamming':
                self.window = torch.hamming_window(self.fft_size, periodic=True)
            else:
                raise ValueError("Invalid argument.")
        else:
            self.window = None
        
        self.normalize = self.train_loader.dataset.normalize
        assert self.normalize == self.valid_loader.dataset.normalize, "Nomalization of STFT is different between `train_loader.dataset` and `valid_loader.dataset`."
    
    def run_one_epoch_train(self, epoch):
        # Override
        """
        Training
        """
        self.model.train()
        
        train_loss = 0
        n_train_batch = len(self.train_loader)
        
        for idx, (mixture, sources, assignment, threshold_weight) in enumerate(self.train_loader):
            if self.use_cuda:
                mixture = mixture.cuda()
                sources = sources.cuda()
                assignment = assignment.cuda()
                threshold_weight = threshold_weight.cuda()
                
            mixture_amplitude = torch.abs(mixture)
            sources_amplitude = torch.abs(sources)
            
            estimated_sources_amplitude = self.model(mixture_amplitude, assignment=assignment, threshold_weight=threshold_weight, n_sources=sources.size(1))
            loss = self.criterion(estimated_sources_amplitude, sources_amplitude)
            
            self.optimizer.zero_grad()
            loss.backward()
            
            if self.max_norm:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_norm)
            
            self.optimizer.step()
            
            train_loss += loss.item()
            
            if (idx + 1)%100 == 0:
                print("[Epoch {}/{}] iter {}/{} loss: {:.5f}".format(epoch+1, self.epochs, idx+1, n_train_batch, loss.item()), flush=True)
        
        train_loss /= n_train_batch
        
        return train_loss
    
    def run_one_epoch_eval(self, epoch):
        # Override
        """
        Validation
        """
        n_sources = self.n_sources
        
        self.model.eval()
        
        valid_loss = 0
        n_valid = len(self.valid_loader.dataset)
        
        with torch.no_grad():
            for idx, (mixture, sources, assignment, threshold_weight) in enumerate(self.valid_loader):
                """
                mixture (batch_size, 1, n_bins, n_frames)
                sources (batch_size, n_sources, n_bins, n_frames)
                assignment (batch_size, n_sources, n_bins, n_frames)
                threshold_weight (batch_size, n_bins, n_frames)
                """
                if self.use_cuda:
                    mixture = mixture.cuda()
                    sources = sources.cuda()
                    threshold_weight = threshold_weight.cuda()
                    assignment = assignment.cuda()
                
                mixture_amplitude = torch.abs(mixture)
                sources_amplitude = torch.abs(sources)
                
                output = self.model(mixture_amplitude, assignment=None, threshold_weight=threshold_weight, n_sources=n_sources)
                # At the test phase, assignment may be unknown.
                loss, _ = pit(self.criterion, output, sources_amplitude, batch_mean=False)
                loss = loss.sum(dim=0)
                valid_loss += loss.item()
                
                if idx < 5:
                    mixture = mixture[0].cpu() # -> (1, n_bins, n_frames)
                    mixture_amplitude = mixture_amplitude[0].cpu() # -> (1, n_bins, n_frames)
                    estimated_sources_amplitude = output[0].cpu() # -> (n_sources, n_bins, n_frames)
                    ratio = estimated_sources_amplitude / mixture_amplitude
                    estimated_sources = ratio * mixture # -> (n_sources, n_bins, n_frames)
                    estimated_sources = torch.istft(estimated_sources, n_fft=self.fft_size, hop_length=self.hop_size, normalized=self.normalize, window=self.window) # -> (n_sources, T)
                    estimated_sources = estimated_sources.cpu()
                    
                    mixture = torch.istft(mixture, n_fft=self.fft_size, hop_length=self.hop_size, normalized=self.normalize, window=self.window) # -> (1, T)
                    mixture = mixture.squeeze(dim=0) # -> (T,)
                    
                    save_dir = os.path.join(self.sample_dir, "{}".format(idx+1))
                    os.makedirs(save_dir, exist_ok=True)
                    save_path = os.path.join(save_dir, "mixture.wav")
                    norm = torch.abs(mixture).max()
                    mixture = mixture / norm
                    signal = mixture.unsqueeze(dim=0) if mixture.dim() == 1 else mixture
                    torchaudio.save(save_path, signal, sample_rate=self.sr, bits_per_sample=BITS_PER_SAMPLE_WSJ0)
                    
                    for source_idx, estimated_source in enumerate(estimated_sources):
                        save_path = os.path.join(save_dir, "epoch{}-{}.wav".format(epoch+1,source_idx+1))
                        norm = torch.abs(estimated_source).max()
                        estimated_source = estimated_source / norm
                        signal = estimated_source.unsqueeze(dim=0) if estimated_source.dim() == 1 else estimated_source
                        torchaudio.save(save_path, signal, sample_rate=self.sr, bits_per_sample=BITS_PER_SAMPLE_WSJ0)
        
        valid_loss /= n_valid
        
        return valid_loss

class AttractorTester(TesterBase):
    def __init__(self, model, loader, pit_criterion, args):
        self.loader = loader
        
        self.model = model
        
        self.pit_criterion = pit_criterion
        
        self._reset(args)
    
    def _reset(self, args):
        # Override
        super()._reset(args)

        self.fft_size, self.hop_size = args.fft_size, args.hop_size

        if args.window_fn:
            if args.window_fn == 'hann':
                self.window = torch.hann_window(self.fft_size, periodic=True)
            elif args.window_fn == 'hamming':
                self.window = torch.hamming_window(self.fft_size, periodic=True)
            else:
                raise ValueError("Invalid argument.")
        else:
            self.window = None
        
        self.normalize = self.train_loader.dataset.normalize
    
    def run(self):
        self.model.eval()

        n_sources = self.n_sources

        test_loss = 0
        test_sdr_improvement = 0
        test_sir_improvement = 0
        test_sar = 0
        test_pesq = 0
        n_pesq_error = 0
        n_test = len(self.loader.dataset)

        tmp_dir = os.path.join(os.getcwd(), 'tmp')
        os.makedirs(tmp_dir, exist_ok=True)
        shutil.copy('./PESQ', os.path.join(tmp_dir, 'PESQ'))
        os.chdir(tmp_dir)

        with torch.no_grad():
            for idx, (mixture, sources, ideal_mask, threshold_weight, T, segment_IDs) in enumerate(self.loader):
                """
                    mixture (1, 1, n_bins, n_frames)
                    sources (1, n_sources, n_bins, n_frames)
                    assignment (1, n_sources, n_bins, n_frames)
                    threshold_weight (1, n_bins, n_frames)
                    T (1,)
                """
                if self.use_cuda:
                    mixture = mixture.cuda()
                    sources = sources.cuda()
                    ideal_mask = ideal_mask.cuda()
                    threshold_weight = threshold_weight.cuda()
                
                mixture_amplitude = torch.abs(mixture) # -> (1, 1, n_bins, n_frames)
                sources_amplitude = torch.abs(sources)
                
                output = self.model(mixture_amplitude, assignment=None, threshold_weight=threshold_weight, n_sources=n_sources)
                loss, perm_idx = self.pit_criterion(output, sources_amplitude, batch_mean=False)
                loss = loss.sum(dim=0)
                
                mixture = mixture[0].cpu()
                sources = sources[0].cpu()
    
                mixture_amplitude = mixture_amplitude[0].cpu() # -> (1, n_bins, n_frames)
                estimated_sources_amplitude = output[0].cpu() # -> (n_sources, n_bins, n_frames)
                ratio = estimated_sources_amplitude / mixture_amplitude
                estimated_sources = ratio * mixture # -> (n_sources, n_bins, n_frames)
                
                perm_idx = perm_idx[0] # -> (n_sources,)
                T = T[0]  # -> <int>
                segment_IDs = segment_IDs[0] # -> <str>
                mixture = torch.istft(mixture, n_fft=self.fft_size, hop_length=self.hop_size, normalized=self.normalize, window=self.window, length=T).squeeze(dim=0) # -> (T,)
                sources = torch.istft(sources, n_fft=self.fft_size, hop_length=self.hop_size, normalized=self.normalize, window=self.window, length=T) # -> (n_sources, T)
                estimated_sources = torch.istft(estimated_sources, n_fft=self.fft_size, hop_length=self.hop_size, normalized=self.normalize, window=self.window, length=T) # -> (n_sources, T)
                
                repeated_mixture = torch.tile(mixture, (self.n_sources, 1))
                result_estimated = bss_eval_sources(
                    reference_sources=sources.numpy(),
                    estimated_sources=estimated_sources.numpy()
                )
                result_mixed = bss_eval_sources(
                    reference_sources=sources.numpy(),
                    estimated_sources=repeated_mixture.numpy()
                )
        
                sdr_improvement = np.mean(result_estimated[0] - result_mixed[0])
                sir_improvement = np.mean(result_estimated[1] - result_mixed[1])
                sar = np.mean(result_estimated[2])

                norm = torch.abs(mixture).max()
                mixture /= norm
                mixture_ID = segment_IDs

                # Generate random number temporary wav file.
                random_ID = str(uuid.uuid4())
                    
                if idx < 10 and self.out_dir is not None:
                    mixture_path = os.path.join(self.out_dir, "{}.wav".format(mixture_ID))
                    signal = mixture.unsqueeze(dim=0) if mixture.dim() == 1 else mixture
                    torchaudio.save(mixture_path, signal, sample_rate=self.sr, bits_per_sample=BITS_PER_SAMPLE_WSJ0)
                    
                for order_idx in range(self.n_sources):
                    source, estimated_source = sources[order_idx], estimated_sources[perm_idx[order_idx]]
                    
                    # Target
                    norm = torch.abs(source).max()
                    source /= norm
                    if idx < 10 and  self.out_dir is not None:
                        source_path = os.path.join(self.out_dir, "{}_{}-target.wav".format(mixture_ID, order_idx + 1))
                        signal = source.unsqueeze(dim=0) if source.dim() == 1 else source
                        torchaudio.save(source_path, signal, sample_rate=self.sr, bits_per_sample=BITS_PER_SAMPLE_WSJ0)
                    source_path = "tmp-{}-target_{}.wav".format(order_idx + 1, random_ID)
                    signal = source.unsqueeze(dim=0) if source.dim() == 1 else source
                    torchaudio.save(source_path, signal, sample_rate=self.sr, bits_per_sample=BITS_PER_SAMPLE_WSJ0)

                    # Estimated source
                    norm = torch.abs(estimated_source).max()
                    estimated_source /= norm
                    if idx < 10 and  self.out_dir is not None:
                        estimated_path = os.path.join(self.out_dir, "{}_{}-estimated.wav".format(mixture_ID, order_idx + 1))
                        signal = estimated_source.unsqueeze(dim=0) if estimated_source.dim() == 1 else estimated_source
                        torchaudio.save(estimated_path, signal, sample_rate=self.sr, bits_per_sample=BITS_PER_SAMPLE_WSJ0)
                    estimated_path = "tmp-{}-estimated_{}.wav".format(order_idx + 1, random_ID)
                    signal = estimated_source.unsqueeze(dim=0) if estimated_source.dim() == 1 else estimated_source
                    torchaudio.save(estimated_path, signal, sample_rate=self.sr, bits_per_sample=BITS_PER_SAMPLE_WSJ0)
                
                pesq = 0
                    
                for source_idx in range(self.n_sources):
                    source_path = "tmp-{}-target_{}.wav".format(source_idx + 1, random_ID)
                    estimated_path = "tmp-{}-estimated_{}.wav".format(source_idx + 1, random_ID)
                    
                    command = "./PESQ +{} {} {}".format(self.sr, source_path, estimated_path)
                    command += " | grep Prediction | awk '{print $5}'"
                    pesq_output = subprocess.check_output(command, shell=True)
                    pesq_output = pesq_output.decode().strip()

                    if pesq_output == '':
                        # If processing error occurs in PESQ software, it is regarded as PESQ score is -0.5. (minimum of PESQ)
                        n_pesq_error += 1
                        pesq += -0.5
                    else:
                        pesq += float(pesq_output)
                    
                    subprocess.call("rm {}".format(source_path), shell=True)
                    subprocess.call("rm {}".format(estimated_path), shell=True)
                
                pesq /= self.n_sources
                print("{}, {:.3f}, {:.3f}".format(mixture_ID, loss.item(), pesq), flush=True)
                
                test_loss += loss.item()
                test_sdr_improvement += sdr_improvement
                test_sir_improvement += sir_improvement
                test_sar += sar
                test_pesq += pesq

        test_loss /= n_test
        test_sdr_improvement /= n_test
        test_sir_improvement /= n_test
        test_sar /= n_test
        test_pesq /= n_test
        
        os.chdir("../") # back to the original directory

        print("Loss: {:.3f}, SDR improvement: {:3f}, SIR improvement: {:3f}, SAR: {:3f}, PESQ: {:.3f}".format(test_loss, test_sdr_improvement, test_sir_improvement, test_sar, test_pesq))
        print("Evaluation of PESQ returns error {} times".format(n_pesq_error))

class AnchoredAttractorTrainer(AttractorTrainer):
    def __init__(self, model, loader, criterion, optimizer, args):
        super().__init__(model, loader, criterion, optimizer, args)
    
    def run_one_epoch_train(self, epoch):
        # Override
        """
        Training
        """
        self.model.train()
        
        train_loss = 0
        n_train_batch = len(self.train_loader)
        
        for idx, (mixture, sources, threshold_weight) in enumerate(self.train_loader):
            if self.use_cuda:
                mixture = mixture.cuda()
                sources = sources.cuda()
                threshold_weight = threshold_weight.cuda()
            
            mixture_amplitude = torch.abs(mixture)
            sources_amplitude = torch.abs(sources)
            
            estimated_sources_amplitude = self.model(mixture_amplitude, threshold_weight=threshold_weight, n_sources=sources.size(1))
            loss = self.criterion(estimated_sources_amplitude, sources_amplitude)
            
            self.optimizer.zero_grad()
            loss.backward()
            
            if self.max_norm:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_norm)
            
            self.optimizer.step()
        
            train_loss += loss.item()
            
            if (idx + 1)%100 == 0:
                print("[Epoch {}/{}] iter {}/{} loss: {:.5f}".format(epoch+1, self.epochs, idx+1, n_train_batch, loss.item()), flush=True)
        
        train_loss /= n_train_batch
        
        return train_loss
    
    def run_one_epoch_eval(self, epoch):
        # Override
        """
        Validation
        """
        n_sources = self.n_sources
        
        self.model.eval()
        
        valid_loss = 0
        n_valid = len(self.valid_loader.dataset)
        
        with torch.no_grad():
            for idx, (mixture, sources, threshold_weight) in enumerate(self.valid_loader):
                """
                    mixture (batch_size, 1, n_bins, n_frames)
                    sources (batch_size, n_sources, n_bins, n_frames)
                    threshold_weight (batch_size, n_bins, n_frames)
                """
                if self.use_cuda:
                    mixture = mixture.cuda()
                    sources = sources.cuda()
                    threshold_weight = threshold_weight.cuda()
                
                mixture_amplitude = torch.abs(mixture)
                sources_amplitude = torch.abs(sources)
                
                output = self.model(mixture_amplitude, threshold_weight=threshold_weight, n_sources=n_sources)
                # At the test phase, assignment may be unknown.
                loss, _ = pit(self.criterion, output, sources_amplitude, batch_mean=False)
                loss = loss.sum(dim=0)
                valid_loss += loss.item()
                
                if idx < 5:
                    mixture = mixture[0].cpu() # -> (1, n_bins, n_frames)
                    mixture_amplitude = mixture_amplitude[0].cpu() # -> (1, n_bins, n_frames)
                    estimated_sources_amplitude = output[0].cpu() # -> (n_sources, n_bins, n_frames)
                    ratio = estimated_sources_amplitude / mixture_amplitude
                    estimated_sources = ratio * mixture # (n_sources, n_bins, n_frames)
                    estimated_sources = torch.istft(estimated_sources, n_fft=self.fft_size, hop_length=self.hop_size, normalized=self.normalize, window=self.window) # -> (n_sources, T)
                    estimated_sources = estimated_sources.cpu()
                    
                    mixture = torch.istft(mixture, n_fft=self.fft_size, hop_length=self.hop_size, normalized=self.normalize, window=self.window) # -> (1, T)
                    mixture = mixture.squeeze(dim=0) # -> (T,)
                    
                    save_dir = os.path.join(self.sample_dir, "{}".format(idx + 1))
                    os.makedirs(save_dir, exist_ok=True)
                    save_path = os.path.join(save_dir, "mixture.wav")
                    norm = torch.abs(mixture).max()
                    mixture = mixture / norm
                    signal = mixture.unsqueeze(dim=0) if mixture.dim() == 1 else mixture
                    torchaudio.save(save_path, signal, sample_rate=self.sr, bits_per_sample=BITS_PER_SAMPLE_WSJ0)
                    
                    for source_idx, estimated_source in enumerate(estimated_sources):
                        save_path = os.path.join(save_dir, "epoch{}-{}.wav".format(epoch + 1, source_idx + 1))
                        norm = torch.abs(estimated_source).max()
                        estimated_source = estimated_source / norm
                        signal = estimated_source.unsqueeze(dim=0) if estimated_source.dim() == 1 else estimated_source
                        torchaudio.save(save_path, signal, sample_rate=self.sr, bits_per_sample=BITS_PER_SAMPLE_WSJ0)
        
        valid_loss /= n_valid
        
        return valid_loss
