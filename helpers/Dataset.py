from scipy import misc
import os
import sys
import time
import random
import numpy as np
from tqdm import tqdm
from Queue import Queue
from threading import Thread, Lock, BoundedSemaphore
from itertools import cycle
from cv2 import resize

class DataLoader(object):
    def __init__(self, file_path, data_path='./', mean=None, resize=None, crop=None, 
                 scale=1.0, batch_size=1, prefetch=0, start_prefetch=True):
        self.file_path = file_path
        self.data_path = data_path
        self.mean = mean
        self.resize = resize
        self.crop = crop
        self.scale = scale
        self.current = 0
        
        self.batch_size = batch_size
        self.prefetch = prefetch
        self.prefetch_started = False
        self.prefetch_stop = False
        self.prefetch_queue = Queue(maxsize=prefetch)
        
        if start_prefetch:
            self.start_prefetch()
        
    def start_prefetch(self):
        if self.prefetch > 0 and not self.prefetch_started:
            self.prefetch_thread = Thread(target=self.batch_prefetcher)
            self.prefetch_thread.start()
            self.prefetch_started = True
        
        return self
    
    def wait_prefetch(self):
        if self.prefetch > 0:
            last = 0
            print >> sys.stdout, "Prefetching data"
            
            with tqdm(total=self.prefetch) as pbar:
                while self.prefetch_queue.qsize() < self.prefetch:
                    current = self.prefetch_queue.qsize()
                    if current - last > 0:
                        pbar.update(current - last)
                        last = current
                        
                    time.sleep(0.200)
                    
                if last < self.prefetch_queue.qsize():
                    pbar.update(self.prefetch_queue.qsize() - last)
                        
        time.sleep(0.5)
        return self
    
    def batch_prefetcher(self):
        while not self.prefetch_stop:
            batch = self.gen_batch()
            
            while not self.prefetch_stop:
                try:
                    self.prefetch_queue.put(batch, True, 1.0)
                    break
                except:
                    pass
            
    def next_batch(self):        
        if self.prefetch > 0:
            batch = self.prefetch_queue.get()   
        else:
            batch = self.gen_batch()
    
        return batch
    
    def shuffle(self, a, b):
        if len(a) != len(b):
            print >> sys.stderr, "Unmatching sets"
            
        combined = zip(a, b)
        random.shuffle(combined)
        return zip(*combined)
    
    def process_image(self, img):
        img = img.astype(np.float32) * self.scale

        if self.resize is not None:
            img = resize(img, (self.resize[0], self.resize[1]))
            
        if self.crop is not None:
            h, w = img.shape[0], img.shape[1]
            crop_height, crop_width = ((h-self.crop[0])/2, (w-self.crop[1])/2)
            img = img[crop_height:crop_height+self.crop[0], crop_width:crop_width+self.crop[1], ...]
            
        if self.mean is not None:
            img -= self.mean
            
        return img
    
    def reset(self):
        pass
    
    def close(self):
        if self.prefetch > 0:
            self.prefetch_stop = True
            self.prefetch_thread.join()


class TextLoader(DataLoader):
    def __init__(self, shuffle=True, **kwargs):
        super(TextLoader, self).__init__(**kwargs)
        
        lines = open(self.file_path).readlines()
        pairs = [line.split() for line in lines]
        
        paths, labels = zip(*pairs)
        
        if shuffle:
            self.bare_paths, self.bare_labels = self.shuffle(paths, labels)
        else:
            self.bare_paths, self.bare_labels = paths, labels
        
        assert self.batch_size <= len(self.bare_paths), "Length Batch " + str(len(self.bare_paths)) + " > " + str(len(self.bare_paths))
        
        self.paths, self.labels = cycle(self.bare_paths), cycle(self.bare_labels)
        
    def read_image(self, path):
        img = misc.imread(self.data_path + path)
        return self.process_image(img)

    def gen_batch(self):
        images = [self.read_image(next(self.paths)) for i in range(self.batch_size)]
        labels = [next(self.labels) for i in range(self.batch_size)]
        
        return np.asarray(images), np.reshape(np.asarray(labels), (-1, 1))
    
    def reset(self):
        self.paths, self.labels = cycle(self.bare_paths), cycle(self.bare_labels)

    def __len__(self):
        return len(self.labels)


class NumpyLoader(DataLoader):
    def __init__(self, **kwargs):
        super(NumpyLoader, self).__init__(**kwargs)
        self.data = cycle(self.file_path)
        
        assert self.batch_size <= len(self.data), "Length Batch " + str(n) + " > " + str(len(self.data))
        
    def gen_batch(self):
        data = [next(self.data) for i in range(self.batch_size)]
        return data, None
    
    def reset(self):
        self.data = cycle(self.file_path)

    def __len__(self):
        return len(self.labels)


try:
    import lmdb

    caffe_root = '/home/deep/caffe'
    sys.path.insert(0, caffe_root + '/python')

    import caffe
    from caffe.proto import caffe_pb2


    class LmdbLoader(DataLoader):
        def __init__(self, **kwargs):
            self.env = lmdb.open(kwargs['file_path'])
            self.txn = lmdb.Transaction(self.env)
            self.cur = self.txn.cursor()
            self.num_items = self.env.stat()['entries']
            self.labels = None

            super(LmdbLoader, self).__init__(**kwargs)

            assert self.batch_size < self.num_items, "Length Batch " + str(n) + " > " + str(self.num_items)

        def load_labels(self, **kwargs):
            kwargs['batch_size'] = self.batch_size
            kwargs['prefetch'] = 0 # We already prefetch here!
            self.labels = LmdbLoader(**kwargs)

        def read_image(self, path):
            img = misc.imread(self.data_path + path)
            return self.process_image(img)

        def get_one(self):
            if not self.cur.next():
                self.cur.first()

            key, value = self.cur.item()

            datum = caffe_pb2.Datum()
            datum.ParseFromString(value)
            data = caffe.io.datum_to_array(datum)

            # CxHxW to HxWxC (OpenCV)
            img = np.transpose(data, (1,2,0))
            img = self.process_image(img)        
            return img, key

        def gen_batch(self):
            pairs = [self.get_one() for i in range(self.batch_size)]
            images, labels = zip(*pairs)        
            images = np.asarray(images)

            if self.labels is None:
                labels = np.reshape(np.asarray(labels), (-1, 1))
            else:
                labels, _ = self.labels.next_batch()

            return images, labels

        def close(self):
            super(LmdbLoader, self).close()
            self.cur.close()
            self.env.close()

            if self.labels is not None:
                self.labels.close()

            return self

        def reset(self):
            self.cur.first()

            if self.labels is not None:
                self.labels.reset()

        def __len__(self):
            return self.num_items
except:
    print >> sys.stderr, "LMDB Loading won't be available"

    
def merge_two_dicts(x, y):
    '''Given two dicts, merge them into a new dict as a shallow copy.'''
    z = x.copy()
    z.update(y)
    return z


class Dataset(object):
    def __init__(self, config={}, dtype=TextLoader):
        self.train = None
        self.test = None
        
        if len(config) > 0:
            common = {} if 'common' not in config else config['common']
            
            if 'train' in config:
                train = merge_two_dicts(config['train'], common)
                self.load_train(dtype=dtype, **train)
                
            if 'test' in config:
                test = merge_two_dicts(config['test'], common)
                self.load_test(dtype=dtype, **test)
        
    def load_train(self, dtype=TextLoader, **kwargs):
        self.train = dtype(**kwargs)
        
    def load_test(self, dtype=TextLoader, **kwargs):
        self.test = dtype(**kwargs)
        
    def close(self):
        if self.train is not None:
            self.train.close()
            
        if self.test is not None:
            self.test.close()
