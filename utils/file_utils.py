import os
from fnmatch import fnmatch
import re
  


def get_file_prefixes_from_path(data_path):
    depth_prefixes = {}
    for path, subdirs, files in os.walk(data_path):
        for name in files:
            if fnmatch(name, "NYU*_0000.png"):
                key = name[3:7]
                depth_prefixes[key] = os.path.join(path, name[:-4]) # removes the ".png" from the end of the filename

    return depth_prefixes
#########################

def get_all_preprocessed_prefixes(data_path, criteria="*.npz"):
    prefixes = []
    for path, subdirs, files in os.walk(data_path):
        for name in files:
            if fnmatch(name, criteria):
                prefixes.append(os.path.join(path, name)[:-4])# removes the last 4 characters from the string (which will be ".npz" for a matching file)
                
    return prefixes