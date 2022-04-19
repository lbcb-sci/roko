import h5py
import numpy as np


class Storage:
    def __init__(self, name, infer):
        self.name = name
        self.pos = []
        self.X = []
        self.X2 = []

        if not infer:
            self.Y = []

        self.infer = infer

    def extend(self, pos, X, Y, X2):
        if self.infer:
            assert len(pos) == len(X) == len(X2)
        else:
            assert len(pos) == len(X) == len(Y) == len(X2)

        for i, p in enumerate(pos):
            self.pos.append(p)
            self.X.append(X[i])
            self.X2.append(X2[i])

            if not self.infer:
                self.Y.append(Y[i])

    def write(self, f):
        if not self.pos:
            return

        if self.infer:
            assert len(self.pos) == len(self.X) == len(self.X2)
        else:
            assert len(self.pos) == len(self.X) == len(self.Y) == len(self.X2)

        start, end = self.pos[0][0][0], self.pos[-1][-1][0]

        group = f.create_group(f'{self.name}_{start}-{end}')
        group['positions'] = self.pos
        if not self.infer:
            group['labels'] = self.Y
        group.attrs['contig'] = self.name
        group.attrs['size'] = len(self.pos)

        print(f'Writing to {group.name}')
        group.create_dataset('examples', data=self.X, chunks=(1, 30, 90))
        group.create_dataset('stats', data=self.X2, chunks=(1, 5, 90))

    def clear(self):
        del self.pos[:]
        del self.X[:]
        del self.X2[:]

        if not self.infer:
            del self.Y[:]

class DataWriter:
    def __init__(self, filename, infer):
        self.filename = filename
        self.infer = infer

        self.storages = dict()

    def __enter__(self):
        self.fd = h5py.File(self.filename, 'w', swmr=True)
        return self

    def __exit__(self, type, value, traceback):
        self.fd.close()

    def store(self, contig, positions, examples, labels, pos_stats):
        try:
            storage = self.storages[contig]
        except KeyError:
            storage = self.storages[contig] = Storage(contig, self.infer)

        storage.extend(positions, examples, labels, pos_stats)

    def write(self):
        for storage in self.storages.values():
            storage.write(self.fd)
            storage.clear()

    def write_contigs(self, refs):
        contigs_group = self.fd.create_group('contigs')

        for n, r in refs:
            contig = contigs_group.create_group(n)
            contig.attrs['name'] = n
            contig.attrs['seq'] = r
            contig.attrs['len'] = len(r)
