import json
import numpy as np
import uuid

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, uuid.UUID):
            return str(obj)
        return super(NpEncoder, self).default(obj)