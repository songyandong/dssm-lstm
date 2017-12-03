import os
import csv
import random
import nltk
from nltk import word_tokenize
nltk.download('punkt')

data_dir = './data'


def analyze_dataset():
    idxs = {}
    # debug
    iter = 0
    with open(os.path.join(data_dir, 'all.csv'), 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if int(row[1]) not in idxs.keys():
                idxs[int(row[1])] = [int(row[2])]
            else:
                idxs[int(row[1])].append(int(row[2]))
            # debug
            iter += 1
            print("No.%d" % iter)
    # debug
    print("Finished reading. Wtring...")
    with open(os.path.join(data_dir, 'dataset_info.txt'), 'w') as f:
        for idx, values in idxs.iteritems():
            f.write('similar to ' + str(idx) + ': ' + ','.join(str(v) for v in values) + '\n')


def create_dataset():
    lines = []
    dis_similar = []
    with open(os.path.join(data_dir, 'all.csv'), 'r') as f:
        reader = csv.reader(f)
        for line in reader:
            if int(line[5]) != 1:
                dis_similar.append(line)
            else:
                lines.append(line)

    query_num = 10000
    neg_num = 4
    test_num = 2500
    random.shuffle(lines)
    test_lines = lines[query_num:int(query_num + test_num / 2)]
    lines = lines[:query_num]

    queries = []
    docs = []
    idxs = {}
    for line in lines:
        idx1 = int(line[1])
        idx2 = int(line[2])
        if idx1 not in idxs.keys():
            idxs[idx1] = [idx2]
        else:
            idxs[idx1].append(idx2)
        queries.append(' '.join(word_tokenize(line[3])) + '\n')
        docs.append(' '.join(word_tokenize(line[4])) + '\n')
        negs_count = 0
        while negs_count < neg_num:
            rand_idx = random.randint(0, query_num - 1)
            candidate = lines[rand_idx]
            i1 = int(candidate[1])
            i2 = int(candidate[2])
            if i1 == idx1 or i2 == idx1 or i1 in idxs[idx1] or i2 in idxs[idx1]:
                continue
            docs.append(' '.join(word_tokenize(candidate[3])) + '\n')
            negs_count += 1
    with open(os.path.join(data_dir, 'queries.txt'), 'w') as f:
        f.writelines(queries)
    with open(os.path.join(data_dir, 'docs.txt'), 'w') as f:
        f.writelines(docs)

    random.shuffle(dis_similar)
    test_lines = test_lines + dis_similar[:int(test_num / 2)]

    with open(os.path.join(data_dir, 'test.csv'), 'w') as f:
        writer = csv.writer(f)
        writer.writerows(test_lines)


if __name__ == '__main__':
    create_dataset()