import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq.models.wav2vec import Wav2VecModel

class PerceptualLoss(nn.Module):
    def __init__(self,
        model_type='wav2vec',
        loss_type='lp',    # wsd or lp
        PRETRAINED_MODEL_PATH = '/path/to/model_ckpt.pt',
        device=None,
    ):
        super().__init__()
        self.model_type = model_type
        self.loss_type = loss_type

        if model_type == 'wav2vec':
            ckpt = torch.load(PRETRAINED_MODEL_PATH, map_location="cpu")
            self.model = Wav2VecModel.build_model(ckpt['args'], task=None)
            self.model.load_state_dict(ckpt['model'])
            self.model = self.model.feature_extractor
            #self.model = nn.DataParallel(self.model)
            #device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            print('Please assign a loss model')
            sys.exit()

        self.model = nn.DataParallel(self.model)
        self.model.to(device)
        self.model.eval()

    def forward(self, y_hat, y):
        y_hat, y = map(self.model, [y_hat, y])
        if self.loss_type == 'wsd':
            # WSD
            return torch.mean(y) - torch.mean(y_hat)
            #return torch.mean(y * y_hat)
        else:
            # EMD
            return torch.abs(y_hat - y).mean()
class PerceptualLoss_Hubert(nn.Module):
    pass
# class PerceptualLoss_Hubert(nn.Module):
#     def __init__(self,
#         loss_type='lp',    # wsd or lp
#         device=None,
#     ):
#         super().__init__()
#         from s3prl.hub import distilhubert
#         self.loss_type = loss_type

#         self.model = distilhubert()

#         self.model = nn.DataParallel(self.model)
#         self.model.to(device)
#         self.model.eval()

#     def forward(self, y_hat, y):
#         # print(y_hat.shape, y.shape)
#         # y_hat, y = map(self.model, [y_hat, y])
#         y_hat, y = self.model(y_hat), self.model(y)
#         y_hat, y = y_hat['last_hidden_state'], y['last_hidden_state']
#         if self.loss_type == 'wsd':
#             # WSD
#             return torch.mean(y) - torch.mean(y_hat)
#             #return torch.mean(y * y_hat)
#         else:
#             # EMD
#             return torch.abs(y_hat - y).mean()

if __name__ == "__main__":
    #per_loss = PerceptualLoss(PRETRAINED_MODEL_PATH="/share/nas167/fuann/pretrain/wav2vec_large.pt")
    per_loss = PerceptualLoss_Hubert()
    wav1 = torch.rand((2, 16000))
    wav2 = torch.rand((2, 16000))
    loss = per_loss(wav1, wav2)
    print(loss)
