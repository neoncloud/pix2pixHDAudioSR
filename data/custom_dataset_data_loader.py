from queue import Queue
import torch.utils.data
from data.base_data_loader import BaseDataLoader
from torch.utils.data import SubsetRandomSampler
from threading import Thread
import random
import os


def CreateDataset(opt):
    dataset = None
    if opt.phase == 'train':
        from data.audio_dataset import AudioDataset
        dataset = AudioDataset(opt)
    elif opt.phase == 'test':
        from data.audio_dataset import AudioTestDataset
        dataset = AudioTestDataset(opt)

    print("dataset [%s] was created" % (dataset.name()))
    # dataset.initialize(opt)
    return dataset

class CustomDatasetDataLoader(BaseDataLoader):
    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, opt):
        self.q_size = 16
        self.idx = 0
        self.load_stream = torch.cuda.Stream(device='cuda')
        self.queue: Queue = Queue(maxsize=self.q_size)
        BaseDataLoader.initialize(self, opt)
        self.dataset = CreateDataset(opt)
        dataset_size = len(self.dataset)
        indices = list(range(dataset_size))
        split = int(torch.floor(torch.Tensor(
            [opt.validation_split * dataset_size])))

        if opt.val_indices is not None:
            self.val_indices = torch.load(opt.val_indices)
            self.train_indices = torch.tensor(
                list(set(indices) - set(self.val_indices)))
        else:
            if not opt.serial_batches:
                random.seed(opt.seed)
                random.shuffle(indices)
            self.train_indices, self.val_indices = indices[split:], indices[:split]
            self.data_lenth = min(len(self.train_indices),
                                  self.opt.max_dataset_size)
            torch.save(self.val_indices, os.path.join(
                self.opt.checkpoints_dir, self.opt.name, 'validation_indices.pt'))

        # Creating PT data samplers and loaders:
        if opt.phase == "train":
            train_sampler = SubsetRandomSampler(self.train_indices)
            valid_sampler = SubsetRandomSampler(self.val_indices)
            self.dataloader = torch.utils.data.DataLoader(
                self.dataset,
                batch_size=opt.batchSize,
                sampler=train_sampler,
                num_workers=int(opt.nThreads),
                prefetch_factor=8,
                pin_memory=True)
            if len(self.val_indices) != 0:
                self.eval_data_lenth = len(self.val_indices)
                self.eval_dataloder = torch.utils.data.DataLoader(
                    self.dataset,
                    batch_size=4,
                    sampler=valid_sampler,
                    num_workers=1,
                    pin_memory=False)
            else:
                self.eval_dataloder = None
                self.eval_data_lenth = 0
        elif opt.phase == "test":
            self.dataloader = torch.utils.data.DataLoader(
                self.dataset,
                batch_size=opt.batchSize,
                num_workers=int(opt.nThreads),
                shuffle=False,
                pin_memory=True)
            self.eval_dataloder = None
            self.eval_data_lenth = 0
    
    def load_loop(self) -> None:  # The loop that will load into the queue in the background
        for i, sample in enumerate(self.dataloader):
            self.queue.put(self.load_instance(sample))
            if i == len(self):
                break
    
    def load_instance(self, sample:dict):
        with torch.cuda.stream(self.load_stream):
            return {k:v.cuda(non_blocking=True) for k,v in sample.items()}

    def load_data(self):
        return self.dataloader
    
    def async_load_data(self):
        return self

    def eval_data(self):
        return self.eval_dataloder

    def eval_data_len(self):
        return self.eval_data_lenth

    def __len__(self):
        return self.data_lenth
    
    def __iter__(self):
        if_worker = not hasattr(self, "worker") or not self.worker.is_alive()  # type: ignore[has-type]
        if if_worker and self.queue.empty() and self.idx == 0:
            self.worker = Thread(target=self.load_loop)
            self.worker.daemon = True
            self.worker.start()
        return self

    def __next__(self):
        # If we've reached the number of batches to return
        # or the queue is empty and the worker is dead then exit
        done = not self.worker.is_alive() and self.queue.empty()
        done = done or self.idx >= len(self)
        if done:
            self.idx = 0
            self.queue.join()
            self.worker.join()
            raise StopIteration
        # Otherwise return the next batch
        out = self.queue.get()
        self.queue.task_done()
        self.idx += 1
        return out

