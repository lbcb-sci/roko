#train.py
from cProfile import label
from typing import *

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torchmetrics
import os
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins.training_type.ddp import DDPPlugin
from pytorch_lightning.utilities.cli import LightningCLI

from v6v7v8_evoformer import Evoformer, PositionalEncoding
from roko_data_module import RokoDataModule
#import sys

POSITIONAL_FEATURES = 5
READ_FEATURES = 12
OUTPUT_CLASSES = 5
MASK_CLASSES = 5
SEQ_LEN = 90
NUM_R = 30

class GRUNetwork(nn.Module):
    def __init__(self,
                 input_dim: int, # P
                 hidden_size: int,
                 n_layers: int,
                 dropout: float = 0.0) -> None:
        super().__init__()

        # pos_stat network
        self.linear = nn.Linear(POSITIONAL_FEATURES, input_dim) # F P
        self.gru = nn.GRU(input_dim, 
                          hidden_size, 
                          num_layers = n_layers,
                          dropout=dropout,
                          batch_first=True, 
                          bidirectional=True)

    def forward(self, pos_stat: torch.Tensor) -> torch.Tensor:
        #print("before nn: should be B S F, 8 90 5", pos_stat.shape)
        pos_stat = F.relu(self.linear(pos_stat)) # should be B S P
        #print("after nn before gru: should be B S P, 8 90 128", pos_stat.shape)
        pos_stat, _ = self.gru(pos_stat) # B S 256=2*hidden_size
        #print("after gru: should be B S 2*H, 8 90 256", pos_stat.shape)
        return pos_stat # BxNxSx2H


class AttentionNetwork(nn.Module):
    def __init__(self, 
                 embed_dim: int,
                 nheads: int,
                 nblocks: int,
                 depth_prob: float = 1.0,
                 pos_dropout: float = 0.0) -> None:
        super().__init__()
        # Model

        self.embedding = nn.Embedding(READ_FEATURES, embed_dim)
        self.pe = PositionalEncoding(embed_dim, pos_dropout, SEQ_LEN, NUM_R) # positional encoding for each base in sequences and for each sequence in reads
        self.evoformer = Evoformer(msa_embedding_dim = embed_dim, 
                                   heads = nheads, 
                                   num_blocks = nblocks, 
                                   p_keep_lowest = depth_prob)

    def forward(self,
                x: torch.Tensor) -> torch.Tensor: # x B R S
        x = self.embedding(x) # B R S E
        x = self.pe(x, 'col') # B R S E
        x = self.pe(x, 'row') # B R S E
        x = self.evoformer(x) # B R S E
        return x


class Polisher(pl.LightningModule):

    def __init__(self, 
                 gru_input_dim: int = 128,
                 gru_hidden_dim: int =128,
                 gru_n_layers: int = 1,
                 gru_dropout: float = 0.0,
                 attn_embed_dim: int = 128,
                 attn_n_heads: int = 8,
                 attn_n_blocks: int = 8,
                 attn_depth_prob: float = 1.0,
                 attn_pos_dropout: float = 0.0,
                 reads_mask_prob: float = 0.2,
                 alpha: float = 0.1,
                 lr=3e-4) -> None:
        super().__init__()
        #constructor
        self.save_hyperparameters()

        self.gru = GRUNetwork(gru_input_dim, gru_hidden_dim, gru_n_layers,
                              gru_dropout)
        self.attn = AttentionNetwork(attn_embed_dim, attn_n_heads,
                                     attn_n_blocks, attn_depth_prob,
                                     attn_pos_dropout)
        self.maskfc = nn.Linear(attn_embed_dim,MASK_CLASSES)
        self.fc = nn.Linear(attn_embed_dim + 2 * gru_hidden_dim, OUTPUT_CLASSES)
        

        # Metrics
        self.train_accuracy = torchmetrics.Accuracy()
        self.val_accuracy = torchmetrics.Accuracy()

    def forward(self, 
                reads: torch.Tensor,
                pos_stat: torch.Tensor) -> torch.Tensor: # reads B R S, pos_stat B S F
        
        gru_output = self.gru(pos_stat) # B S 2H
        attn_output = self.attn(reads) # B S E
        output = torch.cat((attn_output[:,0], gru_output), 2) # B S (E+2H)

        return self.fc(output)

    def forward_train(self,
                      reads: torch.Tensor,
                      mask: torch.Tensor, 
                      pos_stat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]: # x B R S, mask B R S, pos_stat B S F
        
        gru_output = self.gru(pos_stat) # B S 2H
        attn_output = self.attn(reads) # B S E
        masked_output = self.maskfc(attn_output[mask]) # x[mask] = N_masked E --> self.fcmask(x[mask]) = N_masked 5

        # x (B R S E) -> take the first row of R dimension -> (B S E)
        output = torch.cat((attn_output[:,0], gru_output), 2) # x B S E, pos_stat B S 2*H, output B S (E+2H)
        
        # pass 'output' to linear layer --> self.fc(output) = BxSx5
        return self.fc(output), masked_output # B S 5 and N_masked 5


    def cross_entropy_loss(self, logits, labels):
        return F.cross_entropy(logits, labels)

    def read_masking(self,
                     reads: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Masking
        # 1) MASK = we create our mask if the random number generated < 0.2 (20% chance of a read being masked)
        # 2) unkown -> we set elements which are unknown
        # 3) final mask -> bitwise-and (mask AND NOT unknown)
        # 4) we get values before_masking, but without strand info -> use for ground truth (target labels)
        # 5) We take random values 0-11 (with strand info) and apply it to the masked positions. after fc4 in forward it becomes 0-5 

        # generate a boolean mask
        mask = torch.rand(reads.size(), 
                          device = self.device) < self.hparams.reads_mask_prob # a tensor with true and false values, BxRxS

        # a boolean tensor with true at positions where the values are 5 or 11
        unknown = (reads == 5) | (reads == 11) # true if x at the position is 5 or 11  size = BxRxS

        # for each position, make masked positions false at places where x is 5 or 11
        mask &= ~unknown

        # save original values before masking
        mask_target = torch.remainder(reads[mask],6) # size = number of elements masked (1 dimensional)

        # apply masks, x is still B R S
        reads[mask] = torch.randint(0, 
                                    high = 5, 
                                    size=reads[mask].size(), 
                                    dtype=torch.uint8, 
                                    device=self.device) + torch.div(reads[mask], 6, rounding_mode = 'floor')*6
        return mask_target, mask



    def training_step(self, batch, batch_idx):
        reads, labels, pos_stat = batch
        reads = reads.long() # x.size() = B R S, y.size() = B S
        labels = labels.long()
        pos_stat = pos_stat.transpose(1,2).float() # B F S --> transpose --> B S F

        mask_target, mask = self.read_masking(reads)
        seq_logits, mask_logits = self.forward_train(reads, mask, pos_stat) # logits = B C S (B x 5 x S), y = B S, attn_out = N_masked x 5, before_masking = N_masked
        
        seq_logits = seq_logits.transpose(1,2)
        prediction_loss = self.cross_entropy_loss(seq_logits, labels)
        self.log('prediction_loss', prediction_loss)

        masking_loss = self.cross_entropy_loss(mask_logits, mask_target)
        self.log('masking_loss', masking_loss)

        overall_loss = prediction_loss + self.hparams.alpha * masking_loss
        self.log('overall_loss', overall_loss)

        train_acc_batch = self.train_accuracy(seq_logits, labels)
        self.log('train_acc', train_acc_batch)
        #sys.exit()
        
        return overall_loss

    def validation_step(self, batch, batch_idx):
        reads, labels, pos_stat = batch
        reads = reads.long()
        labels = labels.long()
        pos_stat = pos_stat.transpose(1,2).float()
        
        seq_logits = self.forward(reads,pos_stat)
        seq_logits = seq_logits.transpose(1,2) # logits = B C S
        loss = self.cross_entropy_loss(seq_logits, labels)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', self.val_accuracy(seq_logits, labels))

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer

def get_trainer_defaults() -> Dict[str, Any]:
    checkpoint_callback = ModelCheckpoint(
        monitor='val_acc',
        save_top_k=-1,
        #dirpath="/scratch/model", # how do we get this dir from user argument? 
        #Leave it unset in ModelCheckpoint instantiation, it will be set to trainer's weights_save_path or if it is also None then Trainer.default_root_dir.
        filename='{epoch}-{val_loss:.5f}-{val_acc:.5f}')

    trainer_defaults = {
        'callbacks': [checkpoint_callback],
        'logger': WandbLogger(project='docker_roko', log_model='all', save_dir=os.getcwd()), # weights and biases
        'strategy': DDPPlugin(find_unused_parameters = True) # 'strategy' parameter requires pytorch lightning v1.5 and above
    }

    return trainer_defaults

def cli_main():
    LightningCLI(
        Polisher,
        RokoDataModule,
        seed_everything_default = 42, # always put seed so we can repeat the experiment
        save_config_overwrite=True,
        trainer_defaults=get_trainer_defaults())

if __name__ == "__main__":
    cli_main()
