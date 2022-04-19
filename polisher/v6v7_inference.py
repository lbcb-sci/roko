import torch
from Bio import SeqIO, SeqRecord
from Bio.Seq import Seq
import h5py
from collections import defaultdict, Counter
import argparse
from torch.utils.data import Dataset, DataLoader
import itertools
import numpy as np
from v6v7v8_polisher import Polisher
#from GPUtil import showUtilization as gpu_usage
#from numba import cuda

GPU_NUM = 1

GAP = '*'
ALPHABET = 'ACGT' + GAP + 'N'
encoding = {v: i for i, v in enumerate(ALPHABET)}
decoding = {v: k for k, v in encoding.items()}


class ToTensor:
    def __call__(self, sample):
        contig, position, x, x2 = sample

        return contig, position, torch.from_numpy(x), torch.from_numpy(x2)


class InferenceDataset(Dataset):
    def __init__(self, path, transform=None):
        self.path = path
        self.contigs = {}

        fd = None
        try:
            fd = h5py.File(path, 'r')

            self.idx = {}
            self.size = 0

            groups = list(fd.keys())
            groups.remove('contigs')

            for g in groups:
                s = fd[g].attrs['size']

                for j in range(s):
                    self.idx[self.size + j] = (g, j)
                self.size += s

            end_group = fd['contigs']
            for k in end_group:
                contig = str(k)
                seq = end_group[k].attrs['seq']
                length = end_group[k].attrs['len']

                self.contigs[contig] = (seq, length)

        finally:
            if fd is not None:
                fd.close()

        self.transform = transform
        self.fd = None

    def __getitem__(self, idx):
        if not self.fd:
            self.fd = h5py.File(self.path, 'r')

        g, p = self.idx[idx]

        group = self.fd[g]

        contig = group.attrs['contig']
        X = group['examples'][p]
        X2 = group['stats'][p]
        position = group['positions'][p]

        sample = (contig, position, X, np.nan_to_num(X2/np.sum(X2,axis=0)))
        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return self.size

    def close_fd(self):
        if self.fd:
            self.fd.close()


def infer(data, model_path, out, workers=0, batch_size=128, gpu='6'):
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda:'+gpu if use_cuda else 'cpu')
    print(device)

    
    model = Polisher.load_from_checkpoint(checkpoint_path=model_path).to(device)
    model.freeze()
    #model = RNN(IN_SIZE, HIDDEN_SIZE, NUM_LAYERS).to(device)
    #model.load_state_dict(torch.load(model_path))

    if device.type == 'cuda' and GPU_NUM > 1:
        model = nn.DataParallel(model, list(range(GPU_NUM)))

    model.eval()

    result = defaultdict(lambda: defaultdict(lambda: Counter()))
    records = []

    dataset = InferenceDataset(data, transform=ToTensor())
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers, prefetch_factor = 1)

    info = []

    print('Inference started')
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            c, pos, x, x2 = batch
            #print("device used:", device)
            #print(x.shape,x2.shape)
            #torch.cuda.memory_summary(device = 6) # why doesnt this print out anything??
            #print("Initial GPU Usage")
            #gpu_usage()
            # x, x2 = x.type(torch.cuda.LongTensor if device.type == 'cuda' else torch.LongTensor), x2.type(torch.cuda.FloatTensor if device.type == 'cuda' else torch.FloatTensor)
            x = x.to(device).long()
            x2 = x2.to(device).float()

            logits = model(x,x2.transpose(1,2))
            Y = torch.argmax(logits, dim=2).long()
            Y = Y.cpu().numpy()

            for cb, pb, yb in zip(c, pos, Y):
                for p, y in zip(pb, yb):
                    base = decoding[y]

                    curr_pos = (p[0].item(), p[1].item())
                    result[cb][curr_pos][base] += 1

            if (i + 1) % 100 == 0:
                print(f'{i + 1} batches processed')

    contigs = dataset.contigs
    with open(str.split(out,'.fasta')[0]+"_aux_file.txt", "w") as aux_file:
        for contig in result:
            values = result[contig]

            pos_sorted = sorted(values)
            pos_sorted = list(itertools.dropwhile(lambda x: x[1] != 0, pos_sorted))

            first = pos_sorted[0][0]
            contig_data = contigs[contig]
            seq = contig_data[0][:first]

            aux_file.write(f'{contig}\n')

            for i, p in enumerate(pos_sorted):
                base, _ = values[p].most_common(1)[0]
                # save the position and base to a file: 
                aux_file.write(f'{p}\t{base}\t{values[p]}\n')

                if base == GAP:
                    continue
                seq += base

            last_pos = pos_sorted[-1][0]
            seq += contig_data[0][last_pos+1:]

            seq = Seq(seq)
            record = SeqRecord.SeqRecord(seq, id=contig)
            records.append(record)

    with open(out, 'w') as f:
        SeqIO.write(records, f, 'fasta')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data', type=str)
    parser.add_argument('model', type=str)
    parser.add_argument('out', type=str)
    parser.add_argument('--t', type=int, default=0)
    parser.add_argument('--b', type=int, default=32)
    parser.add_argument('--gpu', type=str, default='6')
    args = parser.parse_args()

    infer(args.data, args.model, args.out, args.t, args.b, args.gpu)
    print("done")


if __name__ == '__main__':
    main()
