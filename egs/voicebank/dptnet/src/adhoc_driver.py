import os
import time
from turtle import backward
import torch
import torch.nn as nn

from utils.utils import draw_loss_curve
from driver import TrainerBase, TesterBase
from tqdm import tqdm
from pesq import pesq as pypesq
import torchaudio

BITS_PER_SAMPLE_WSJ0 = 16

class AdhocTrainer(TrainerBase):
    def __init__(self, model, loader, pit_criterion, optimizer, args):
        super().__init__(model, loader, pit_criterion, optimizer, args)
        self.best_pesq = 0

    def _reset(self, args):
        super()._reset(args)

        self.d_model = args.sep_bottleneck_channels

        # Learning rate
        self.k1, self.k2 = args.k1, args.k2
        self.warmup_steps = args.warmup_steps

        if args.continue_from:
            config = torch.load(args.continue_from, map_location=lambda storage, loc: storage)
            self.step = config['step']
        else:
            self.step = 0
    
    def run(self):
        for epoch in range(self.start_epoch, self.epochs):
            start = time.time()
            train_loss, valid_loss, pesq = self.run_one_epoch(epoch)
            end = time.time()
            
            print("[Epoch {}/{}] loss (train): {:.5f}, loss (valid): {:.5f}, {:.3f} [sec], best_loss:{:.5f}, imp{}, pesq: {}".format(
                                epoch+1, self.epochs, train_loss, valid_loss, end - start, self.best_loss, self.no_improvement, pesq), flush=True)            
            self.train_loss[epoch] = train_loss
            self.valid_loss[epoch] = valid_loss

            if valid_loss < self.best_loss:
                print(f"save best model, last: {self.best_loss}, current:{valid_loss}", flush=True)
                self.best_loss = valid_loss
                self.no_improvement = 0
                model_path = os.path.join(self.model_dir, "best.pth")
                self.save_model(epoch, model_path)
            else:
                if valid_loss >= self.prev_loss:
                    self.no_improvement += 1
                else:
                    self.no_improvement = 0
            
            self.prev_loss = valid_loss

            if pesq > self.best_pesq:
                print(f"save best pesq model, last: {self.best_pesq:.4f}, current:{pesq:.4f}", flush=True)
                self.best_pesq = pesq
                model_path = os.path.join(self.model_dir, "best_pesq.pth")
                self.save_model(epoch, model_path)
            
            model_path = os.path.join(self.model_dir, "last.pth")
            self.save_model(epoch, model_path)
            
            save_path = os.path.join(self.loss_dir, "loss.png")
            draw_loss_curve(train_loss=self.train_loss[:epoch + 1], valid_loss=self.valid_loss[:epoch + 1], save_path=save_path)

            if self.no_improvement >= 10:
                print("Stop training.")
                break
    
    def run_one_epoch_train(self, epoch):
        """
        Training
        """
        self.model.train()
        
        train_loss = 0
        n_train_batch = len(self.train_loader)
        total_steps = (len(self.train_loader)//self.train_loader.batch_size)
        t = tqdm(enumerate(self.train_loader), leave=False, total=total_steps)
        
        # for idx, (mixture, sources) in enumerate(self.train_loader):
        for idx, (mixture, sources) in t:
            if self.use_cuda:
                mixture = mixture.cuda()
                sources = sources.cuda()
            
            # estimated_sources = self.model(mixture)
            # loss, _ = self.pit_criterion(estimated_sources, sources)
            estimated_sources, latent = self.model(mixture)
            loss = self.criterion(estimated_sources[:,0], sources[:,0])

            
            self.optimizer.zero_grad()
            loss.backward()
            
            if self.max_norm:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_norm)
            
            lr = self.update_lr(epoch)
            self.optimizer.step()
            
            train_loss += loss.item()
            
            t.set_postfix_str(f'loss: {loss.item():.3f}, lr: {lr:.6f}')
            
            # if (idx + 1)%100 == 0:
        t.write("[Epoch {}/{}] lr: {:.5f}".format(epoch+1, self.epochs, lr))
        
        train_loss /= n_train_batch
        
        return train_loss

    def update_lr(self, epoch):
        """
        Update learning rate for CURRENT step
        """
        step = self.step
        warmup_steps = self.warmup_steps

        if step > warmup_steps:
            k = self.k2
            lr = k * 0.98 ** ((epoch + 1) // 2)
        else:
            k = self.k1
            d_model = self.d_model
            lr = k * d_model ** (-0.5) * (step + 1) * warmup_steps ** (-1.5)

        prev_lr = None

        for param_group in self.optimizer.param_groups:
            if (step + 1) % 100 == 0 and prev_lr is None:
                prev_lr = param_group['lr']
                if lr == prev_lr:
                    break
                else:
                    print("Learning rate: {} -> {}".format(prev_lr, lr), flush=True)
            param_group['lr'] = lr

        self.step = step + 1
        return lr
    
    def save_model(self, epoch, model_path='./tmp.pth'):
        if isinstance(self.model, nn.DataParallel):
            config = self.model.module.get_config()
            config['state_dict'] = self.model.module.state_dict()
        else:
            config = self.model.get_config()
            config['state_dict'] = self.model.state_dict()
            
        config['optim_dict'] = self.optimizer.state_dict()
        
        config['best_loss'] = self.best_loss
        config['no_improvement'] = self.no_improvement
        
        config['train_loss'] = self.train_loss
        config['valid_loss'] = self.valid_loss
        
        config['epoch'] = epoch + 1
        config['step'] = self.step # self.step is already updated in `update_lr`, so you don't have to plus 1.
        
        torch.save(config, model_path)

class AdhocTrainer_AMP(AdhocTrainer):
    def __init__(self, model, loader, pit_criterion, optimizer, args):
        super().__init__(model, loader, pit_criterion, optimizer, args)
        self.scaler = torch.cuda.amp.GradScaler()
    
    def run_one_epoch_train(self, epoch):
        """
        Training
        """
        self.model.train()
        
        train_loss = 0
        n_train_batch = len(self.train_loader)
        total_steps = (len(self.train_loader)//self.train_loader.batch_size)
        t = tqdm(enumerate(self.train_loader), leave=False, total=total_steps)
        
        # for idx, (mixture, sources) in enumerate(self.train_loader):
        for idx, (mixture, sources) in t:
            if self.use_cuda:
                mixture = mixture.cuda()
                sources = sources.cuda()
            
            # estimated_sources = self.model(mixture)
            # loss, _ = self.pit_criterion(estimated_sources, sources)
            with torch.cuda.amp.autocast():
                estimated_sources, latent = self.model(mixture)
            loss = self.criterion(estimated_sources[:,0], sources[:,0])

            
            self.optimizer.zero_grad()
            # loss.backward()
            self.scaler.scale(loss).backward()
            
            if self.max_norm:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_norm)
            
            self.update_lr(epoch)
            # self.optimizer.step()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            train_loss += loss.item()
            
            t.set_postfix_str(f'loss: {loss.item():.5f}')
            
            # if (idx + 1)%100 == 0:
            #     t.write("[Epoch {}/{}] iter {}/{} loss: {:.5f}".format(epoch+1, self.epochs, idx+1, total_steps, loss.item()))
        
        train_loss /= n_train_batch
        
        return train_loss

class TENET_Trainer(TrainerBase):
    def __init__(self, model, loader, pit_criterion, optimizer, args):
        super().__init__(model, loader, pit_criterion, optimizer, args)

    def _reset(self, args):
        super()._reset(args)

        self.d_model = args.sep_bottleneck_channels

        # Learning rate
        self.k1, self.k2 = args.k1, args.k2
        self.warmup_steps = args.warmup_steps
        self.patience = args.patience

        if args.continue_from:
            config = torch.load(args.continue_from, map_location=lambda storage, loc: storage)
            self.step = config['step']
            self.best_pesq = config.get('best_pesq_loss', -0.5)
            scheduler = config.get('scheduler', None)
            if scheduler:
                self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=self.patience, factor= 0.5, verbose=True)
                self.scheduler.load_state_dict(scheduler) 
                for param_group in self.optimizer.param_groups:
                    print(f"Continue with Learning Rate with {param_group['lr']}", flush=True)
        else:
            self.step = 0
            self.best_pesq = -0.5
    
    def run(self):
        for epoch in range(self.start_epoch, self.epochs):
            start = time.time()
            train_loss, valid_loss, pesq = self.run_one_epoch(epoch)
            end = time.time()
            
            print("[Epoch {}/{}] loss (train): {:.5f}, loss (valid): {:.5f}, {:.3f} [sec], best_loss:{:.5f}, imp{}, pesq:{:.4f}".format(
                                epoch+1, self.epochs, train_loss, valid_loss, end - start, self.best_loss, self.no_improvement, pesq), flush=True)            
            self.train_loss[epoch] = train_loss
            self.valid_loss[epoch] = valid_loss

            if valid_loss < self.best_loss:
                print(f"save best model, last: {self.best_loss}, current:{valid_loss}", flush=True)
                self.best_loss = valid_loss
                self.no_improvement = 0
                model_path = os.path.join(self.model_dir, "best.pth")
                self.save_model(epoch, model_path)
            else:
                if valid_loss >= self.prev_loss:
                    self.no_improvement += 1
                else:
                    self.no_improvement = 0

            if pesq > self.best_pesq:
                print(f"save best pesq model, last: {self.best_pesq:.4f}, current:{pesq:.4f}", flush=True)
                self.best_pesq = pesq
                model_path = os.path.join(self.model_dir, "best_pesq.pth")
                self.save_model(epoch, model_path)
            
            self.prev_loss = valid_loss
            
            model_path = os.path.join(self.model_dir, "last.pth")
            self.save_model(epoch, model_path)
            
            save_path = os.path.join(self.loss_dir, "loss.png")
            draw_loss_curve(train_loss=self.train_loss[:epoch + 1], valid_loss=self.valid_loss[:epoch + 1], save_path=save_path)

            if self.no_improvement >= 10:
                print("Stop training.")
                break
        print("Epochs reached, Stop training.")
    
    def run_one_epoch_train(self, epoch):
        """
        Training
        """
        self.model.train()
        
        train_loss = 0
        n_train_batch = len(self.train_loader)
        total_steps = (len(self.train_loader)//self.train_loader.batch_size)
        t = tqdm(enumerate(self.train_loader), leave=False, total=total_steps)
        
        # for idx, (mixture, sources) in enumerate(self.train_loader):
        for idx, (mixture, sources) in t:
            if self.use_cuda:
                mixture = mixture.cuda()
                sources = sources.cuda()
            
            # estimated_sources = self.model(mixture)
            # loss, _ = self.pit_criterion(estimated_sources, sources)
            # 可以在這裡直接操作 Model 之間的互動?
            # forward_mix, backward_mix = mixture.chunk(2)
            forward_src, backward_src = sources.chunk(2)
            estimated_sources, latent = self.model(mixture)
            # fw_estimated_sources, latent = self.model(forward_mix)
            # bw_estimated_sources, latent = self.model(backward_mix)
            fw_estimated_sources, bw_estimated_sources = estimated_sources.chunk(2)
            loss = self.criterion(fw_estimated_sources[:,0], forward_src[:,0]) * 0.7
            loss += self.criterion(bw_estimated_sources[:,0], backward_src[:,0]) * 0.3

            
            self.optimizer.zero_grad()
            loss.backward()
            
            if self.max_norm:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_norm)
            
            lr = self.update_lr_reduce(epoch)
            self.optimizer.step()
            
            train_loss += loss.item()
            
            t.set_postfix_str(f'loss: {loss.item():.5f}')
            
            # if (idx + 1)%100 == 0:
            #     t.write("[Epoch {}/{}] iter {}/{} loss: {:.5f}".format(epoch+1, self.epochs, idx+1, total_steps, loss.item()))
        t.write("[Epoch {}/{}] lr: {:.5f}".format(epoch+1, self.epochs, lr))
        
        train_loss /= n_train_batch
        
        return train_loss

    def run_one_epoch_eval(self, epoch):
        """
        Validation
        """
        self.model.eval()
        
        valid_loss = 0
        pesq = 0

        valid_loss_denoise = []
        n_valid = len(self.valid_loader.dataset)
        
        with torch.no_grad():
            t = tqdm(enumerate(self.valid_loader), leave=False,
                    total=(len(self.valid_loader)//self.valid_loader.batch_size))
            for idx, (mixture, sources, segment_IDs) in t:
                if self.use_cuda:
                    mixture = mixture.cuda()
                    sources = sources.cuda()
                output, latent = self.model(mixture)#, eval=True)
                
                #  loss = self.criterion(output[:,0], torch.squeeze(sources,0))
                if self.criterion_str in ['l2_sisnr', 'ttf1', 'ttf2','ttf3'] :
                    T = sources.size()[-1]
                    if self.model.module.enc_basis in ['DCCRN', 'DCTCN', 'TorchSTFT', 'TENET', 'DCT', 'FiLM_DCT']:
                        padding = (100 - (T - 400) % 100) % 100
                    else:
                        padding = (stride - (T - kernel_size) % stride) % stride
                    padding_left = padding // 2
                    padding_right = padding - padding_left
                    gt = self.model.module.encoder(nn.functional.pad(sources, (padding_left, padding_right)))
                    loss = self.criterion(output[:,0], sources[:,0], latent[:,0], gt)
                else:
                    loss = self.criterion(output[:,0], sources[:,0])

                # loss = loss.sum(dim=0)
                valid_loss += loss.item()
                pesq += pypesq(self.sr, sources[:,0].squeeze().cpu().numpy(), output[:,0].cpu().squeeze().numpy(), 'wb' )
                
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
        pesq /= n_valid

        if hasattr(self, 'scheduler'):
            self.scheduler.step(valid_loss)
        
        if len(valid_loss_denoise) != 0:

            valid_loss_denoise = [(v / n_valid) for v in valid_loss_denoise]
            return (valid_loss, valid_loss_denoise)

        return valid_loss, pesq

    def update_lr_reduce(self, epoch):
        """
        Update learning rate for CURRENT step
        """
        step = self.step
        warmup_steps = self.warmup_steps

        if step > warmup_steps:
            k = self.k2
            lr = k
            if not hasattr(self, 'scheduler'):
                self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=self.patience, factor= 0.5, verbose=True)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
                    print(f"Change to Normal Learning Rate with {k}", flush=True)
            else:
                for param_group in self.optimizer.param_groups:
                    lr = param_group['lr']
        else:
            k = self.k1
            d_model = self.d_model
            lr = k * d_model ** (-0.5) * (step + 1) * warmup_steps ** (-1.5)

            prev_lr = None

            for param_group in self.optimizer.param_groups:
                if (step + 1) % 100 == 0 and prev_lr is None:
                    prev_lr = param_group['lr']
                    if lr == prev_lr:
                        break
                    else:
                        print("Learning rate: {} -> {}".format(prev_lr, lr), flush=True)
                param_group['lr'] = lr

        self.step = step + 1
        return lr

    def update_lr(self, epoch):
        """
        Update learning rate for CURRENT step
        """
        step = self.step
        warmup_steps = self.warmup_steps

        if step > warmup_steps:
            k = self.k2
            lr = k * 0.98 ** ((epoch + 1) // 2)
        else:
            k = self.k1
            d_model = self.d_model
            lr = k * d_model ** (-0.5) * (step + 1) * warmup_steps ** (-1.5)

        prev_lr = None

        for param_group in self.optimizer.param_groups:
            if (step + 1) % 100 == 0 and prev_lr is None:
                prev_lr = param_group['lr']
                if lr == prev_lr:
                    break
                else:
                    print("Learning rate: {} -> {}".format(prev_lr, lr), flush=True)
            param_group['lr'] = lr

        self.step = step + 1
        return lr
    
    def save_model(self, epoch, model_path='./tmp.pth'):
        if isinstance(self.model, nn.DataParallel):
            config = self.model.module.get_config()
            config['state_dict'] = self.model.module.state_dict()
        else:
            config = self.model.get_config()
            config['state_dict'] = self.model.state_dict()
        
        config['scheduler']  = self.scheduler.state_dict() if hasattr(self, 'scheduler') else None
        config['optim_dict'] = self.optimizer.state_dict()
        
        config['best_loss'] = self.best_loss
        config['best_pesq_loss'] = self.best_pesq
        config['no_improvement'] = self.no_improvement
        
        config['train_loss'] = self.train_loss
        config['valid_loss'] = self.valid_loss
        
        config['epoch'] = epoch + 1
        config['step'] = self.step # self.step is already updated in `update_lr`, so you don't have to plus 1.
        
        torch.save(config, model_path)

class TENET_Noise_Trainer(TENET_Trainer):
    def __init__(self, model, loader, pit_criterion, optimizer, args):
        super().__init__( model, loader, pit_criterion, optimizer, args)
    
    def run_one_epoch_train(self, epoch):
        """
        Training
        """
        self.model.train()
        
        train_loss = 0
        n_train_batch = len(self.train_loader)
        total_steps = (len(self.train_loader)//self.train_loader.batch_size)
        t = tqdm(enumerate(self.train_loader), leave=False, total=total_steps)
        
        # for idx, (mixture, sources) in enumerate(self.train_loader):
        for idx, (mixture, sources) in t:
            if self.use_cuda:
                mixture = mixture.cuda()
                sources = sources.cuda()
            
            forward_src, backward_src = sources.chunk(2)
           
            estimated_sources, latent = self.model(mixture)

            fw_estimated_sources, bw_estimated_sources = estimated_sources.chunk(2)
            loss = self.criterion(fw_estimated_sources[:,0], forward_src[:,0]) * 0.7
            loss += self.criterion(bw_estimated_sources[:,0], backward_src[:,0]) * 0.3

            
            self.optimizer.zero_grad()
            loss.backward()
            
            if self.max_norm:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_norm)
            
            self.update_lr(epoch)
            self.optimizer.step()
            
            train_loss += loss.item()
            
            t.set_postfix_str(f'loss: {loss.item():.5f}')
            
            # if (idx + 1)%100 == 0:
            #     t.write("[Epoch {}/{}] iter {}/{} loss: {:.5f}".format(epoch+1, self.epochs, idx+1, total_steps, loss.item()))
        
        train_loss /= n_train_batch
        
        return train_loss


class Tester(TesterBase):
    def __init__(self, model, loader, pit_criterion, args):
        super().__init__(model, loader, pit_criterion, args)
