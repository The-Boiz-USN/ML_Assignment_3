import numpy as np
import pickle

"""
This file is contains helper functions to pace the data file and save them as pickled files.
contains a methode to load the pickled object.
the parsing and picketing is just done once, then the pickled files a used to save time.
"""


def get_data(filename: str):
    with open(f"data_parsed/{filename}", "rb") as f:
        ps = pickle.load(f)
    return ps


def parse_file(filename: str) -> np.ndarray:
    with open(filename) as f:
        data = f.read().strip()

    data = data.split("\n")
    tmp = []
    for r in data:
        t = [float(x) for x in r.split("\t")]
        tmp.append(t)

    data = np.array(tmp)
    return data


def main():
    for i in range(1, 7):
        ps = parse_file(f"data/PS{i}.txt")
        with open(f"data_parsed/ps{i}", "wb") as f:
            pickle.dump(ps, f)

    for i in range(1, 5):
        ts = parse_file(f"data/TS{i}.txt")
        with open(f"data_parsed/ts{i}", "wb") as f:
            pickle.dump(ts, f)

    for i in range(1, 3):
        fs = parse_file(f"data/FS{i}.txt")
        with open(f"data_parsed/fs{i}", "wb") as f:
            pickle.dump(fs, f)

    vs = parse_file("data/VS1.txt")
    with open("data_parsed/vs1", "wb") as f:
        pickle.dump(vs, f)

    se = parse_file("data/SE.txt")
    with open("data_parsed/se", "wb") as f:
        pickle.dump(se, f)

    eps1 = parse_file("data/EPS1.txt")
    with open("data_parsed/eps1", "wb") as f:
        pickle.dump(eps1, f)

    cp = parse_file("data/CP.txt")
    with open("data_parsed/cp", "wb") as f:
        pickle.dump(cp, f)

    ce = parse_file("data/CE.txt")
    with open("data_parsed/ce", "wb") as f:
        pickle.dump(ce, f)

    profile = parse_file("data/profile.txt")
    with open("data_parsed/profile", "wb") as f:
        pickle.dump(profile, f)


if __name__ == '__main__':
    main()
