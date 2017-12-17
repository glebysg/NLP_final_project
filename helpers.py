import os
import pickle

def save_object(obj, filename):
        with open(filename, 'wb') as output:
                    pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)
def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)

def get_dataset_dict(path, file_reader, dict_name):
    full_dict_name = os.path.join(path,dict_name)
    if os.path.isfile(full_dict_name):
       return load_obj(full_dict_name)
    else:
        dataset_dict = {}
        for feat_out in file_reader:
            key = (int(feat_out)-1)
            if key in dataset_dict:
                dataset_dict[key] += 1
            else:
                dataset_dict[key] = 1
        file_reader.close()
        save_object(dataset_dict,full_dict_name)
        return dataset_dict
