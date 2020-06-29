import cattr
import json
import os


class Serializable():
    """a base class for serializable objects.
    """

    def save(self, path, make_dir=False):
        if make_dir:
            d = os.path.dirname(path)
            os.makedirs(d, exist_ok=True)

        with open(path, "w") as f:
            json.dump(cattr.unstructure(self), f, indent=4)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cattr.structure(json.load(f), cls)
