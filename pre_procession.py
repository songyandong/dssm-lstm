import os
import csv

data_dir = './data'


def analyze_dataset():
    idxs = {}
    with open(os.path.join(data_dir, 'all.csv'), 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if int(row[1]) not in idxs.keys():
                idxs[int(row[1])] = [int(row[2])]
            else:
                idxs[int(row[1])].append(int(row[2]))
    with open(os.path.join(data_dir, 'dataset_info.txt'), 'w') as f:
        for idx, sims in idxs:
            f.write('same to' + str(idx) + ': ' + ','.join(sims) + '\n')

if __name__ == '__main__':
    analyze_dataset()