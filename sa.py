from pycocotools import mask as maskUtils
import os
from PIL import Image
import numpy as np
import os, json

def load_sample(root, id, annotation_dir='annotations',image_dir='images'):
    img = os.path.join(root,image_dir,id+".jpg") 
    annotation  = os.path.join(root,annotation_dir,id+".json") 
    return (img,annotation)

class SA1BDataset:
    """A class to load SA-1B: https://segment-anything.com/dataset/index.html
    
    Attributes:
        min_object (int): The minimum number of pixels required for an object to be considered valid.
        samples (list): A list of loaded samples from the dataset.
        image_info (list): A list containing image information for each sample.

    """
    def __init__(self, dataset_dir,ids=None,annotation_dir='annotations',image_dir='images',min_object=0):
        """Initializes SA1BDataset class.
        
        Args:
            dataset_dir (str): The directory containing the dataset.
            ids (list, optional): A list of sample IDs to load. If not provided, the class
                                  will load all samples found in the annotation_dir. Default is None.
            annotation_dir (str, optional): The directory containing annotation files
                                            (relative to dataset_dir). Default is 'annotations'.
            image_dir (str, optional): The directory containing image files
                                       (relative to dataset_dir). Default is 'images'.
            min_object (int, optional): The minimum number of pixels required for an object
                                        to be considered valid. Default is 0.
        """
        super().__init__()
        if ids is None:
            ids = [file.replace(".json",'') for file in os.listdir(os.path.join(dataset_dir,annotation_dir))]
            
        self.min_object = min_object
        self.samples = [load_sample(dataset_dir,id,annotation_dir,image_dir) for id in ids]
        self.image_info=[]
        for img,info in self.samples:
            self.image_info.append(json.load(open(info)))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        img = Image.open(self.samples[index][0])
        mask, class_ids = self.load_mask(index)
        return img, mask, class_ids
    
    def load_mask(self, idx):
        """Load instance masks for the given image.
        Different datasets use different ways to store masks. This
        function converts the different mask format to one format
        in the form of a bitmap [height, width, instances].
        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a COCO image, delegate to parent class.
        image_info = self.image_info[idx]
        
        instance_masks = []
        class_ids = []
        annotations = image_info["annotations"]
        image_info = image_info['image']
        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.
        for annotation in annotations:
            m = self.annToMask(annotation, image_info["height"],
                                image_info["width"])
            # Some objects are so small that they're less than 1 pixel area
            # and end up rounded out. Skip those objects.
            if m.sum() < self.min_object:
                continue
            class_id = annotation['id']
            instance_masks.append(m)
            class_ids.append(class_id)

        # Pack instance masks into an array
        
        mask = np.stack(instance_masks, axis=2)
        class_ids = np.array(class_ids, dtype=np.int32)
        return mask, class_ids


    # The following two functions are from pycocotools with a few changes.

    def annToRLE(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
        """
        segm = ann['segmentation']
        if isinstance(segm, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(segm, height, width)
            rle = maskUtils.merge(rles)
        elif isinstance(segm['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(segm, height, width)
        else:
            # rle
            rle = ann['segmentation']
        return rle

    def annToMask(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
        :return: binary mask (numpy 2D array)
        """
        rle = self.annToRLE(ann, height, width)
        m = maskUtils.decode(rle)
        return m

    
    