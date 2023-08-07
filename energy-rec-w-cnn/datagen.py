import numpy as np
from tensorflow.keras.utils import Sequence
from zlib import decompress

class DataGenerator(Sequence):
    def __init__(self, files, batch_size, views, planes, cells, n_channels):
        self.files = files
        self.batch_size = batch_size
        self.views = views
        self.planes = planes
        self.cells = cells
        self.n_channels = n_channels # Should be 3, but not hard-coded just in case
        
    def __normalize__(self, data):
        data_max = np.amax(data)
        data_min = np.amin(data)
        if data_max == 0:
            return np.zeros(np.shape(data))
        return (data - data_min) / (data_max - data_min)
        
    def __get_nuenergy__(self, filename:str) -> float:
        'Returns NuEnergy of an event (units: GeV).'
        info = open(filename, "rb").readlines()
        nuenergy = float(info[1])
        return nuenergy
    
    def __get_pixels_map__(self, filename:str) -> np.ndarray:
        'Returns the pixel map of an event as a views*planes x cells NumPy array.'
        file = open(filename, 'rb').read()
        pixels_map = np.frombuffer(decompress(file), dtype=np.uint8)
        pixels_map = pixels_map.reshape(self.views, self.planes, self.cells)
        return pixels_map
    
    def __get_data_and_labels__(self, files:list) -> tuple[np.ndarray, np.ndarray]:
        'Generates lists of the data and labels associated with the input filelist.'
        data = [self.__get_pixels_map__(file + ".gz") for file in files]
        labels = [self.__get_nuenergy__(file + ".info") for file in files]
        return np.array(data), np.array(labels)
    
    def __len__(self):
        return int(np.floor(len(self.files) / self.batch_size))
        
    def __getitem__(self, index):
        indices = list(range(index*self.batch_size, (index+1)*self.batch_size))
        curr_batch = [self.files[i] for i in indices]        
        maps, labels = self.__get_data_and_labels__(curr_batch)
        
        maps_u = np.asarray(maps)[:, 0:1]
        maps_v = np.asarray(maps)[:, 1:2]
        maps_z = np.asarray(maps)[:, 2:3]
        network = []
        for i in range(len(maps_z)):
            network.append(np.dstack((self.__normalize__(maps_u[i][0]),
                                        self.__normalize__(maps_v[i][0]),
                                        self.__normalize__(maps_z[i][0]))))
        network = np.array(network).reshape([self.batch_size, self.planes, self.cells, self.n_channels])
                
        return network, labels
    
    def on_epoch_end(self):
        'Updates indices after each epoch'
        self.indices = np.arange(len(self.files))