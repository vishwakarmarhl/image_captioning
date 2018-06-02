import shutil
import numpy as np
import json 


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return np.asarray(obj)
        else:
            return super(MyEncoder, self).default(obj)

def train_images_dump():
    lines = []
    with open("train_images.csv") as file:
        for line in file:
            line = line.strip() #or some other preprocessing
            lines.append(line) #storing everything in memory!
            shutil.copy2(line, './out/')


def json_dump_npy():
    results = []
    results.append({'image_id': np.int32(23456), 'caption': 'brown fox jumps'})
    results.append({'image_id': 34456, 'caption': 'brown fox jumps'})
    fp = open('test.json', 'w')
    json.dump(json.dumps(results, cls=MyEncoder), fp)
    fp.close()

    anns    = json.load(open('test.json'))
    assert type(anns) == list, 'results in not an array of objects'

json_dump_npy()
