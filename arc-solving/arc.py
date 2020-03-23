"""
Classes specific to ARC
Problem, Grid, DetectedObject
"""
import json
import numpy as np
from scipy.ndimage import measurements
import matplotlib as mpl
import matplotlib.pyplot as plt

ARC_COLORS = ['black', 'royalblue', 'red', 'limegreen', 'yellow', 'lightsteelblue', 'deeppink', 'darkorange', 'lightskyblue','maroon']


class Problem:
    def __init__(self, name):
        self.grids = []

        d = None
        with open('ARC-master/data/training/'+name+'.json') as json_data:
            d = json.load(json_data)
            json_data.close()

        for example in d['train']:
            self.grids.append({
                "input": Grid(example["input"], "Input"),
                "target": Grid(example["output"], "Target")
                })


    def get_grid(self, index, input_or_output_grid):
        return self.grids[index][input_or_output_grid]

class Grid:
    def __init__(self, data, title=""):
        self.title = title
        self.data = np.array(data)
        self.objects = self.get_objects()

    def get_objects(self):
        """
        Returns an array of DetectedObjects
        where a DetectedObject is a contiguous block of a single color
        """
        objects = []

        for color_x in np.unique(self.data):
            # if different colors...then different objects
            # return an array with 1s where that color is, 0 elsewhere
            data_with_only_color_x = np.where(self.data==color_x, 1, 0) 
            #if items of the same color are separated...then different objects
            data_with_only_color_x_and_object_labels, num_features = measurements.label(data_with_only_color_x)
            for object_i in range(1,num_features+1):
                # return an array with 1s where that object is, 0 elsewhere
                data_with_only_object_i = np.where(data_with_only_color_x_and_object_labels==object_i, 1, 0) 
                objects.append(DetectedObject(data_with_only_object_i))

        return objects

    def get_object(self, index):
        return self.objects[index]

    def change_color(self, obj, new_color):
        """ 
        Sets color value of the data where the 'obj' is to 'new_color'
        """
        self.data = np.where(obj.location==1, new_color, self.data)

    def show(self):
        cmap = mpl.colors.ListedColormap(ARC_COLORS)
        bounds = np.arange(-.5,10.5,1)#[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N+1)

        plt.imshow(self.data, cmap=cmap, norm=norm)
        
        plt.grid(True, color='dimgray')
        plt.xticks(np.arange(-.5, self.data.shape[1]), labels="")# plt.xticks([])
        plt.yticks(np.arange(-.5, self.data.shape[0]), labels="")# plt.yticks([])

        plt.title(self.title)
        # plt.colorbar()

        # plt.show()

class DetectedObject:
    def __init__(self, location):
        """ 
        'location' is a matrix with 1s where object is, 0s elsewhere
        """
        self.location = location
