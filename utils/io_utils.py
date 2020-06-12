import json


def load_json(path):
    with open(path, "r", encoding="utf8") as f:
        data = json.load(f)
    return data


def dump_json(tgt, path):
    with open(path, "w", encoding="utf8") as f:
        json.dump(tgt, f)