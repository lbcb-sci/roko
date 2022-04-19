#train.py
from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torchmetrics
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins import DDPPlugin
#from progressbar import ProgressBar
from v3v4v5_evoformer import Evoformer
from roko_data_module import RokoDataModule

POSITIONAL_FEATURES = 5
READ_FEATURES = 12
OUTPUT_CLASSES = 5
MASK_CLASSES = 5

class Polisher(pl.LightningModule):

    def __init__(self, model_name, lr=3e-4, epochs=100, **model_parameters):
        super().__init__()
        #constructor
        self.model_name = model_name
        self.cls = getattr(Polisher,self.model_name)
        self.backbone_cls = self.cls(**model_parameters)

        # Metrics
        self.train_accuracy = torchmetrics.Accuracy()
        self.val_accuracy = torchmetrics.Accuracy()

        # Hyperparameters
        self.lr = lr
        self.epochs = epochs

    class Original_roko(nn.Module):
        def __init__(self, **params):
            super().__init__()
            # Model
            self.embedding = nn.Embedding(READ_FEATURES, 50)
            self.do = nn.Dropout(0.2)

            self.fc1 = nn.Linear(50, 100)
            self.do1 = nn.Dropout(0.2)

            self.fc2 = nn.Linear(100, 10)
            self.do2 = nn.Dropout(0.2)

            self.hidden_size = params.get('hidden_size')

            self.num_layers = params.get('num_layers')
            self.in_size = params.get('in_size')

            #self.gru = nn.GRU(self.in_size, self.hidden_size, num_layers=self.num_layers,
             #             batch_first=True, bidirectional=True, dropout=0.2)
            self.gru = nn.GRU(self.in_size + 5, self.hidden_size, num_layers=self.num_layers, batch_first=True, bidirectional=True, dropout=0.2)

            for param in self.gru.parameters():
                if len(param.shape) >= 2:
                    init.orthogonal_(param.data)
                else:
                    init.normal_(param.data)

            self.fc4 = nn.Linear(2 * self.hidden_size, 5)


        def forward(self, x, x2): # x B R S, x2 B S F=5
            x = self.do(self.embedding(x)) # B R S E
            x = x.permute((0, 2, 3, 1)) # B S E R

            x = F.relu(self.fc1(x)) # B S E 100
            x = self.do1(x)

            x = F.relu(self.fc2(x)) # B S E 10
            x = self.do2(x)

            x = x.reshape(-1, 90, self.in_size) # B S in_size=500
            x = torch.cat([x, x2], 2) # B S in_size+5=500+5=505
            x, _ = self.gru(x) # B S 2*hidden_size=256

            return self.fc4(x) # B S 5

    class Attention_roko(nn.Module):
        def __init__(self, **params):
            super().__init__()
            # Model
            self.hidden_size = 128
            self.num_layers = 3
            self.in_size = 128 # P

            self.embedding = nn.Embedding(READ_FEATURES, params.get('embedding_dim'))
            self.do = nn.Dropout(0.2)
            self.evoformer = Evoformer(msa_embedding_dim = params.get('embedding_dim'), heads = params.get('heads'), num_blocks = params.get('evoformer_blocks'), p_keep_lowest = params.get('p_keep'))
            self.fc4 = nn.Linear(params.get('embedding_dim')+2*self.hidden_size, 5)
            self.fcmask = nn.Linear(params.get('embedding_dim'),5)

            # pos_stat network
            self.fc1 = nn.Linear(5, 128) # F P
            self.gru = nn.GRU(self.in_size, self.hidden_size, num_layers=self.num_layers,
                          batch_first=True, bidirectional=True, dropout=0.2)


        def forward(self, x, pos_stat): # x B R S, pos_stat B S F
            x = self.do(self.embedding(x)) # B R S E
            x = self.evoformer(x) # B R S E
            #print("pos_stat.shape",pos_stat.shape)
            pos_stat = F.relu(self.fc1(pos_stat)) # B S P
            pos_stat, _ = self.gru(pos_stat) # B S 256=2*hidden_size
            out = torch.cat((x[:,0],pos_stat),2)

            return self.fc4(out)

        def forward_train(self, x, mask, pos_stat): # x B R S, mask B R S, pos_stat B S F
            x = self.do(self.embedding(x)) # B R S E
            x = self.evoformer(x) # B R S E

            #print("before nn: should be B S F, 8 90 5",pos_stat.shape)
            pos_stat = F.relu(self.fc1(pos_stat)) # should be B S P
            #print("after nn before gru: should be B S P, 8 90 128",pos_stat.shape)
            pos_stat, _ = self.gru(pos_stat) # B S 256=2*hidden_size
            #print("after gru: should be B S 2*H, 8 90 256",pos_stat.shape)

            # x (B R S E) -> take the first row of R dimension -> (B S E)
            out = torch.cat((x[:,0],pos_stat),2) # x B S E, pos_stat B S 2*H out B S (E+2*Hidden_size)
            #print("after concat: should be B S E+2*H, 8 90 128+256", out.shape)
            #sys.exit()

            # pass 'out' to linear layer --> self.fc4(out) = BxSx5
            # x[mask] = N_masked E --> self.fcmask(x[mask]) = N_masked 5
            return self.fc4(out), self.fcmask(x[mask]) # B S 5 and N_masked 5


    def forward(self, x,x2):
        return self.backbone_cls(x,x2)


    def cross_entropy_loss(self, logits, labels):
        return F.cross_entropy(logits, labels)


    def training_step(self, batch, batch_idx):
        x, y, x2 = batch
        x = x.long() # x.size() = B R S, y.size() = B S
        y = y.long()
        x2 = x2.transpose(1,2).float() # B F S --> transpose --> B S F

        # Masking
        # 1) MASK = we create our mask if > 0.8
        # 2) unkown -> we set elements which are unknown
        # 3) final mask -> bitwise-and (mask AND NOT unknown)
        # 4) we get values before_masking, but without strand info -> use for ground truth (target labels)
        # 5) We take random values 0-11 (with strand info) and apply it to the masked positions. after fc4 in forward it becomes 0-5 

        # generate a boolean mask
        mask = torch.rand(x.size(), device = self.device) > 0.80 # a tensor with true and false values, BxRxS

        # a boolean tensor with true at positions where the values are 5 or 11
        unknown = (x == 5) | (x == 11) # true if x at the position is 5 or 11  size = BxRxS

        # for each position, make masked positions false at places where x is 5 or 11
        mask &= ~unknown

        # save original values
        before_masking = torch.remainder(x[mask],6) # size = number of elements masked (1 dimensional)

        # apply masks, x is still B R S
        x[mask] = torch.randint(0, high = 5, size=x[mask].size(), dtype=torch.uint8, device=self.device) + torch.div(x[mask], 6, rounding_mode='floor')*6


        if self.model_name == 'Attention_roko':
            logits, attn_out = self.backbone_cls.forward_train(x, mask, x2) # logits = B C S (B x 5 x S), y = B S, attn_out = N_masked x 5, before_masking = N_masked
            masking_loss = self.cross_entropy_loss(attn_out, before_masking)
        else:
            logits = self.forward(x, x2)
            masking_loss = 0

        logits = logits.transpose(1,2)
        train_loss = self.cross_entropy_loss(logits, y)

        overall_loss = train_loss + 0.1 * masking_loss
        train_acc_batch = self.train_accuracy(logits, y)

        self.log('train_loss', train_loss)
        self.log('masking_loss', masking_loss)
        self.log('overall_loss', overall_loss)
        return overall_loss


    def training_epoch_end(self, outputs):
        self.log('Train_acc_epoch', self.train_accuracy, prog_bar=True)


    def validation_step(self, batch, batch_idx):
        x, y, x2 = batch
        x = x.long()
        y = y.long()
        x2 = x2.transpose(1,2).float()

        logits = self.forward(x,x2)
        logits = logits.transpose(1,2) # logits = B C S
        loss = self.cross_entropy_loss(logits, y)
        self.val_accuracy(logits, y)
        self.log('val_loss', loss)


    def validation_epoch_end(self, outputs):
        self.log('val_acc_epoch', self.val_accuracy, prog_bar=True)


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


def main():
    # argument parser
    parser = ArgumentParser()
    parser.add_argument('datapath', type=str) # training data directory or path if it is just one file
    parser.add_argument('out', type=str) # directory to save model checkpoints
    parser.add_argument('backbone', type=str) # choose attn 'Attention_roko' or rnn 'Original_roko'
    parser.add_argument('--valpath', type=str, default=None) # validation data directory or path if it is just one file
    parser.add_argument('--memory', action='store_true') # if not specified, default=False, if specified in command, memory=True
    parser.add_argument('--t', type=int, default=4) # number of threads
    parser.add_argument('--b', type=int, default=8) # batch size
    parser.add_argument('--p_keep', type=float, default=1) # lowest keep rate of block for stochastic depth, default=1 keeps everything


    # hyperparameters
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--epochs", type=int, default=100)
    # parser.add_argument("--patience", type=int, default=30)

    # roko_attn parameters
    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--evoformer_blocks", type=int, default=8)

    # roko_rnn parameters
    parser.add_argument("--in_size", type=int, default=500)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=3)


    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    # weights and biases
    wandb_logger = WandbLogger(project='docker_roko', log_model='all')

    # early stopping
    # early_stop_callback = EarlyStopping(monitor="val_acc_epoch", patience=args.patience, mode = 'max')

    # model checkpoint
    checkpoint_callback = ModelCheckpoint(save_top_k=-1, dirpath=args.out, filename='{epoch}-{val_loss:.5f}-{val_acc_epoch:.5f}')

    # initialize trainer
    trainer = pl.Trainer.from_argparse_args(args,
                                            gpus=[3,4,5,7],
                                            precision = 16,
                                            gradient_clip_val=1.0,
                                            logger=wandb_logger,
                                            strategy=DDPPlugin(find_unused_parameters=True),
                                            callbacks=[checkpoint_callback]) #, early_stop_callback]) #track_grad_norm=2, limit_train_batches=100, limit_val_batches=100)

    # data
    data = RokoDataModule(args.datapath, args.b, args.memory, args.valpath, args.t)

    # Instantiate model
    model = Polisher(model_name = args.backbone, 
                     lr=args.lr, 
                     epochs=args.epochs, 
                     embedding_dim = args.embedding_dim, 
                     heads = args.heads, 
                     evoformer_blocks = args.evoformer_blocks,
                     p_keep = args.p_keep,
                     hidden_size = args.hidden_size, 
                     in_size = args.in_size, 
                     num_layers = args.num_layers)

    # wandb_logger.watch(model) # Log gradients, parameters and model topology

    trainer.fit(model, datamodule=data)


if __name__ == "__main__":
    main()