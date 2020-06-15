import json
import pickle


def load_json(path):
    with open(path, "r", encoding="utf8") as f:
        data = json.load(f)
    return data


def dump_json(tgt, path, indent=None):
    with open(path, "w", encoding="utf8") as f:
        json.dump(tgt, f, indent=indent)


def load_pickle(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def dump_pickle(tgt, path):
    with open(path, "wb") as f:
        pickle.dump(tgt, f)