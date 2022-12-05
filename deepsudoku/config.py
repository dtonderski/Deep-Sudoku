class Data:
    __conf = {
        "seeds_path": "data/sudoku_seeds.txt",
        "sudoku_lists_dir": "data/sudoku_lists",
        "train_path": "data/train.pkl",
        "val_path": "data/val.pkl",
        "test_path": "data/test.pkl",
        "difficulty_path": "data/difficulty.pkl"
    }
    __setters = ["seeds_path", "sudoku_lists_dir", "train_path",
                 "val_path", "test_path", "difficulty_path"]

    @staticmethod
    def config(name):
        return Data.__conf[name]

    @staticmethod
    def set(name, value):
        if name in Data.__setters:
            Data.__conf[name] = value
        else:
            raise NameError("Name not accepted in set() method")
