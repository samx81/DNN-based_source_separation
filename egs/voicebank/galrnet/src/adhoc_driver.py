import os
import time

from utils.utils import draw_loss_curve
from driver import TrainerBase, TesterBase
import torch
import torch.nn as nn
from tqdm import tqdm
import torchaudio

class AdhocTrainer(TrainerBase):
    def __init__(self, model, loader, pit_criterion, optimizer, args):
        super().__init__(model, loader, pit_criterion, optimizer, args)
    
    def run(self):
        for epoch in range(self.start_epoch, self.epochs):
            start = time.time()
            train_loss, valid_loss = self.run_one_epoch(epoch)
            end = time.time()
            
            print("[Epoch {}/{}] loss (train): {:.5f}, loss (valid): {:.5f}, {:.3f} [sec], no_improve: {}".format(epoch + 1, self.epochs, train_loss, valid_loss, end - start, self.no_improvement), flush=True)
            
            self.train_loss[epoch] = train_loss
            self.valid_loss[epoch] = valid_loss

            if (epoch + 1) % 2 == 0:
                for param_group in self.optimizer.param_groups:
                    prev_lr = param_group['lr']
                    lr = 0.96 * prev_lr
                    print("Learning rate: {} -> {}".format(prev_lr, lr))
                    param_group['lr'] = lr
            
            if valid_loss < self.best_loss:
                self.best_loss = valid_loss
                self.no_improvement = 0
                model_path = os.path.join(self.model_dir, "best.pth")
                self.save_model(epoch, model_path)
                print(f"best model saved, best_loss: {self.best_loss}, valid_loss:{valid_loss}", flush=True)
            else:
                if valid_loss >= self.prev_loss:
                    self.no_improvement += 1
                else:
                    self.no_improvement = 0
            
            self.prev_loss = valid_loss
            
            model_path = os.path.join(self.model_dir, "last.pth")
            self.save_model(epoch, model_path)
            
            save_path = os.path.join(self.loss_dir, "loss.png")
            draw_loss_curve(train_loss=self.train_loss[:epoch+1], valid_loss=self.valid_loss[:epoch+1], save_path=save_path)

            if self.no_improvement >= 10:
                print("Stop training")
                break

class TwoStageTrainer(TrainerBase):
    def __init__(self, model1, model2, loader, pit_criterion, optimizer, args):
        super().__init__(model1, loader, pit_criterion, optimizer, args)
        self.model2 = model2
    
    def run(self):
        for epoch in range(self.start_epoch, self.epochs):
            start = time.time()
            train_loss, valid_loss = self.run_one_epoch(epoch)
            end = time.time()
            
            print("[Epoch {}/{}] loss (train): {:.5f}, loss (valid): {:.5f}, {:.3f} [sec], no_improve: {}".format(epoch + 1, self.epochs, train_loss, valid_loss, end - start, self.no_improvement), flush=True)
            
            self.train_loss[epoch] = train_loss
            self.valid_loss[epoch] = valid_loss

            if (epoch + 1) % 2 == 0:
                for param_group in self.optimizer.param_groups:
                    prev_lr = param_group['lr']
                    lr = 0.96 * prev_lr
                    print("Learning rate: {} -> {}".format(prev_lr, lr))
                    param_group['lr'] = lr
            
            if valid_loss < self.best_loss:
                self.best_loss = valid_loss
                self.no_improvement = 0
                model_path = os.path.join(self.model_dir, "best.pth")
                self.save_model(epoch, model_path)
                print(f"best model saved, best_loss: {self.best_loss}, valid_loss:{valid_loss}", flush=True)
            else:
                if valid_loss >= self.prev_loss:
                    self.no_improvement += 1
                else:
                    self.no_improvement = 0
            
            self.prev_loss = valid_loss
            
            model_path = os.path.join(self.model_dir, "last.pth")
            self.save_model(epoch, model_path)
            
            save_path = os.path.join(self.loss_dir, "loss.png")
            draw_loss_curve(train_loss=self.train_loss[:epoch+1], valid_loss=self.valid_loss[:epoch+1], save_path=save_path)

            if self.no_improvement >= 10:
                print("Stop training")
                break
    
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
            
            estimated_sources = self.model(mixture)
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
                output = self.model(mixture)
                # loss, _ = self.pit_criterion(output[:,0], sources, batch_mean=False)
                loss = self.criterion(output[:,0], torch.squeeze(sources,0))
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
            package = self.model.module.get_package()
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

class Tester(TesterBase):
    def __init__(self, model, loader, pit_criterion, args):
        super().__init__(model, loader, pit_criterion, args)