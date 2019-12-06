import h5py
import numpy as np


class Storage:
    def __init__(self, name, infer):
        self.name = name
        self.pos = []
        self.X = []

        if not infer:
            self.Y = []

        self.infer = infer

    def extend(self, pos, X, Y):
        if self.infer:
            assert len(pos) == len(X)
        else:
            assert len(pos) == len(X) == len(Y)

        for i, p in enumerate(pos):
            self.pos.append(p)
            self.X.append(X[i])

            if not self.infer:
                self.Y.append(Y[i])

    def write(self, f):
        if not self.pos:
            return

        if self.infer:
            assert len(self.pos) == len(self.X)
        else:
            assert len(self.pos) == len(self.X) == len(self.Y)

        start, end = self.pos[0][0][0], self.pos[-1][-1][0]

        group = f.create_group(f'{self.name}_{start}-{end}')
        group['positions'] = self.pos
        if not self.infer:
            group['labels'] = self.Y
        group.attrs['contig'] = self.name
        group.attrs['size'] = len(self.pos)

        print(f'Writing to {group.name}')
        group.create_dataset('examples', data=self.X, chunks=(1, 200, 90))

    def clear(self):
        del self.pos[:]
        del self.X[:]

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

    def store(self, contig, positions, examples, labels):
        try:
            storage = self.storages[contig]
        except KeyError:
            storage = self.storages[contig] = Storage(contig, self.infer)

        storage.extend(positions, examples, labels)

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
