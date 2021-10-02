from hyperparams import *


def collection_to_list(inits_data, num_inits):
    row = []
    for j in range(MAX_INITS):
        s = []
        v = []
        o = []
        if j < num_inits:
            s, v, o = inits_data.pop()
            s = s.split('¿')
            v = v.split('¿')
            o = o.split('¿')
        for part, max_length in zip([s, v, o], MAX_LENGTH_SVO):
            for k in range(max_length):
                if k < len(part):
                    row.append(part[k])
                else:
                    row.append('')