class Data:
    __conf = {
        "seeds_path": "data/sudoku_seeds.txt",
        "sudoku_lists_dir": "data/sudoku_lists",
        "train_path": "data/train.pil",
        "val_path": "data/val.pil",
        "test_path": "data/test.pil",
    }
    __setters = ["seeds_path", "sudoku_lists_dir", "train_path",
                 "val_path", "test_path"]

    @staticmethod
    def config(name):
        return Data.__conf[name]

    @staticmethod
    def set(name, value):
        if name in Data.__setters:
            Data.__conf[name] = value
        else:
            raise NameError("Name not accepted in set() method")
