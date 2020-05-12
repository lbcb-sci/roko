import torch
import torch.nn as nn
from datasets import TrainDataset, InMemoryTrainDataset, TrainToTensor
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
import argparse
from ignite.engine import Events, Engine
from ignite.metrics import RunningAverage, Accuracy, Loss
from ignite.handlers import EarlyStopping, ModelCheckpoint
from tqdm import tqdm
from rnn_model import *
from utils import LabelSmoothing

BATCH_SIZE = 128
EPOCHS = 100
LR = 1e-4
PATIENCE = 7


def train(train_path, out, val_path=None, mem=False, workers=0, batch_size=128):
    print('Dataset loading')

    if mem:
        data_class = InMemoryTrainDataset
    else:
        data_class = TrainDataset

    train_ds = data_class(train_path, transform=TrainToTensor())
    if val_path:
        val_ds = data_class(val_path, transform=TrainToTensor())

    train_dl = DataLoader(train_ds, batch_size, True, num_workers=workers)
    if val_path:
        val_dl = DataLoader(val_ds, batch_size, num_workers=workers)

    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if use_cuda else 'cpu')
    print(f'Device: {device}')

    model = RNN(IN_SIZE, HIDDEN_SIZE, NUM_LAYERS).to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    optim = torch.optim.Adam(model.parameters(), lr=LR)

    criterion = LabelSmoothing(0.1)

    def step(engine, batch):
        x, y = batch
        x, y = x.type(torch.cuda.LongTensor if device.type == 'cuda' else torch.LongTensor), y.to(device)


        model.train()
        model.zero_grad()

        output = model(x).transpose(1, 2)
        #loss = F.cross_entropy(output, y)
        loss = criterion(output, y)

        loss.backward()
        optim.step()

        return loss.item()

    def eval(engine, batch):
        model.eval()
        with torch.no_grad():
            x, y = batch
            x, y = x.type(torch.cuda.LongTensor if device.type == 'cuda' else torch.LongTensor), y.to(device)

            out = model(x).transpose(1, 2)
            return out, y

    trainer = Engine(step)
    evaluator = Engine(eval)

    RunningAverage(output_transform=lambda x: x).attach(trainer, 'train_loss')
    Accuracy().attach(evaluator, 'val_acc')
    #Loss(F.cross_entropy).attach(evaluator, 'val_loss')
    Loss(criterion).attach(evaluator, 'val_loss')

    if val_path:
        # EarlyStopping
        def score_function(engine):
            val_acc = engine.state.metrics['val_acc']
            return val_acc

        handler = EarlyStopping(PATIENCE, score_function, trainer)
        evaluator.add_event_handler(Events.COMPLETED, handler)

        # ModelCheckpoint
        mc = ModelCheckpoint(out, 'rnn', score_function=score_function, score_name='acc', require_empty=False)
        evaluator.add_event_handler(Events.COMPLETED, mc, {'model': model})

    desc = 'ITERATION - loss: {}'
    pbar = tqdm(initial=0, leave=False, total=len(train_dl), desc=desc.format(0))

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_train_loss(engine):
        i = (engine.state.iteration - 1) % len(train_dl) + 1

        if i % 100 == 0:
            train_loss = trainer.state.metrics['train_loss']
            pbar.desc = desc.format(train_loss)
            pbar.update(100)

    if val_path:
        @trainer.on(Events.EPOCH_COMPLETED)
        def log_val_results(engine):
            evaluator.run(val_dl)
            metrics = evaluator.state.metrics
            val_acc = metrics['val_acc']
            val_loss = metrics['val_loss']

            tqdm.write(f'Val epoch: {engine.state.epoch}, acc: {val_acc}, loss: {val_loss}')

            pbar.n = pbar.last_print_n = 0

    tqdm.write('Training started')
    trainer.run(train_dl, max_epochs=EPOCHS)
    pbar.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('train', type=str)
    parser.add_argument('out', type=str)
    parser.add_argument('--val', type=str, default=None)
    parser.add_argument('--memory', action='store_true', default=False)
    parser.add_argument('--t', type=int, default=0)
    parser.add_argument('--b', type=int, default=128)
    args = parser.parse_args()

    train(args.train, args.out, args.val, args.memory, args.t, args.b)


if __name__ == '__main__':
    main()
