import csv
import random
from collections import defaultdict
from datetime import datetime
import pickle
from nlgeval import NLGEval
import itertools as it
import statistics
import numpy as np

DATASET_FILE_jaccard = './beam_jaccard/result_beamjaccard.csv'
DATASET_FILE_beam = './beam/result_beam.csv'
DATASET_FILE_recurrent = './recurrent/result_recurrent.csv'


def dd():
    return defaultdict(list)


print("====size====")
with open(DATASET_FILE_recurrent, 'r', encoding='utf8') as csvfile:
    data_recurrent = list(csv.reader(csvfile, quotechar='"'))
data_dict_recurrent = defaultdict(dd)
for i in data_recurrent:
    if len(i[4]) > 0:
        # ['id', 'content', 'question', 'answer', 'distractor', 'gold_distractor']
        i[0] = ''.join(e for e in i[0] if e.isalnum())
        data_dict_recurrent[i[0]]['content'] = i[1]
        data_dict_recurrent[i[0]]['race_id'] = str(datetime.now().timestamp())
        data_dict_recurrent[i[0]]["answers"] = i[3]
        data_dict_recurrent[i[0]]["questions"] = i[2]
        data_dict_recurrent[i[0]]["options"].append(i[4])
        data_dict_recurrent[i[0]]["gold_distractor"].append(i[5])
print("data_dict_recurrent", len(data_dict_recurrent))

with open(DATASET_FILE_beam, 'r', encoding='utf8') as csvfile:
    data_beam = list(csv.reader(csvfile, quotechar='"'))
data_dict_beam = defaultdict(dd)
for i in data_beam:
    if len(i[4]) > 0:
        # ['id', 'content', 'question', 'answer', 'distractor', 'gold_distractor']
        i[0] = ''.join(e for e in i[0] if e.isalnum())
        data_dict_beam[i[0]]['content'] = i[1]
        data_dict_beam[i[0]]['race_id'] = str(datetime.now().timestamp())
        data_dict_beam[i[0]]["answers"] = i[3]
        data_dict_beam[i[0]]["questions"] = i[2]
        data_dict_beam[i[0]]["options"].append(i[4])
        data_dict_beam[i[0]]["gold_distractor"].append(i[5])
print("data_dict_beam", len(data_dict_beam))

with open(DATASET_FILE_jaccard, 'r', encoding='utf8') as csvfile:
    data_jaccard = list(csv.reader(csvfile, quotechar='"'))
data_dict_jaccard = defaultdict(dd)
for i in data_jaccard:
    if len(i[4]) > 0 and len(i[5]) > 0:
        # ['id', 'content', 'question', 'answer', 'distractor', 'gold_distractor']
        i[0] = ''.join(e for e in i[0] if e.isalnum())
        data_dict_jaccard[i[0]]['content'] = i[1]
        data_dict_jaccard[i[0]]['race_id'] = str(datetime.now().timestamp())
        data_dict_jaccard[i[0]]["answers"] = i[3]
        data_dict_jaccard[i[0]]["questions"] = i[2]
        data_dict_jaccard[i[0]]["options"].append(i[4])
        data_dict_jaccard[i[0]]["gold_distractor"].append(i[5])
print("data_dict_jaccard", len(data_dict_jaccard))

data_dict_ori = defaultdict(dd)
for i in data_jaccard:
    if len(i[4]) > 0 and len(i[5]) > 0:
        # ['id', 'content', 'question', 'answer', 'distractor', 'gold_distractor']
        i[0] = ''.join(e for e in i[0] if e.isalnum())
        data_dict_ori[i[0]]['content'] = i[1]
        data_dict_ori[i[0]]['race_id'] = str(datetime.now().timestamp())
        data_dict_ori[i[0]]["answers"] = i[3]
        data_dict_ori[i[0]]["questions"] = i[2]
        data_dict_ori[i[0]]["options"].append(i[5])
        data_dict_ori[i[0]]["gold_distractor"].append(i[5])
print("data_dict_ori", len(data_dict_ori))

data_dict_random = defaultdict(dd)
for i in data_jaccard:
    if len(i[4]) > 0 and len(i[5]) > 0:
        # ['id', 'content', 'question', 'answer', 'distractor', 'gold_distractor']
        i[0] = ''.join(e for e in i[0] if e.isalnum())
        data_dict_random[i[0]]['content'] = i[1]
        data_dict_random[i[0]]['race_id'] = str(datetime.now().timestamp())
        data_dict_random[i[0]]["answers"] = i[3]
        data_dict_random[i[0]]["questions"] = i[2]
        data_dict_random[i[0]]["options"].append(random.choice(data_jaccard)[5])
        data_dict_random[i[0]]["gold_distractor"].append(i[5])
print("data_dict_random", len(data_dict_random))

count = 0
for i in list(data_dict_beam):
    if i in data_dict_jaccard and i in data_dict_recurrent and i in data_dict_beam:
        data_dict_jaccard[i]["options"] = data_dict_jaccard[i]["options"][:3]
        data_dict_recurrent[i]["options"] = data_dict_recurrent[i]["options"][:3]
        data_dict_beam[i]["options"] = data_dict_beam[i]["options"][:3]
        data_dict_ori[i]["options"] = data_dict_ori[i]["options"][:3]
        data_dict_random[i]["options"] = data_dict_random[i]["options"][:3]
        count += 1
    else:
        if i in data_dict_jaccard:
            del data_dict_jaccard[i]
        if i in data_dict_recurrent:
            del data_dict_recurrent[i]
        if i in data_dict_beam:
            del data_dict_beam[i]
        if i in data_dict_ori:
            del data_dict_ori[i]
        if i in data_dict_random:
            del data_dict_random[i]
count = 0
for i in list(data_dict_recurrent):
    if i in data_dict_jaccard and i in data_dict_recurrent and i in data_dict_beam:
        data_dict_jaccard[i]["options"] = data_dict_jaccard[i]["options"][:3]
        data_dict_recurrent[i]["options"] = data_dict_recurrent[i]["options"][:3]
        data_dict_beam[i]["options"] = data_dict_beam[i]["options"][:3]
        data_dict_ori[i]["options"] = data_dict_ori[i]["options"][:3]
        data_dict_random[i]["options"] = data_dict_random[i]["options"][:3]
        count += 1
    else:
        if i in data_dict_jaccard:
            del data_dict_jaccard[i]
        if i in data_dict_recurrent:
            del data_dict_recurrent[i]
        if i in data_dict_beam:
            del data_dict_beam[i]
        if i in data_dict_ori:
            del data_dict_ori[i]
        if i in data_dict_random:
            del data_dict_random[i]

print("====example====")
key = list(data_dict_recurrent)[0]
print("ori", data_dict_ori[key]['questions'], data_dict_ori[key]['options'], data_dict_ori[key]['answers'])
print("random", data_dict_random[key]['questions'], data_dict_random[key]['options'], data_dict_random[key]['answers'])
print("beam", data_dict_beam[key]['questions'], data_dict_beam[key]['options'], data_dict_beam[key]['answers'])
print("beam_jaccard", data_dict_jaccard[key]['questions'], data_dict_jaccard[key]['options'],
      data_dict_jaccard[key]['answers'])
print("recurrent", data_dict_recurrent[key]['questions'], data_dict_recurrent[key]['options'],
      data_dict_recurrent[key]['answers'])

import io


def print_to_string(*args, **kwargs):
    output = io.StringIO()
    print(*args, file=output, **kwargs)
    contents = output.getvalue()
    output.close()
    return contents


with open("compare.txt", "w", encoding='utf8') as f:
    for key in list(data_dict_recurrent):
        f.write("===\n")
        f.write(print_to_string("ori", data_dict_ori[key]['questions'], data_dict_ori[key]['options'],
                                data_dict_ori[key]['answers']))
        f.write(print_to_string("random", data_dict_random[key]['questions'], data_dict_random[key]['options'],
                                data_dict_random[key]['answers']))
        f.write(print_to_string("beam", data_dict_beam[key]['questions'], data_dict_beam[key]['options'],
                                data_dict_beam[key]['answers']))
        f.write(print_to_string("beam_jaccard", data_dict_jaccard[key]['questions'], data_dict_jaccard[key]['options'],
                                data_dict_jaccard[key]['answers']))
        f.write(print_to_string("recurrent", data_dict_recurrent[key]['questions'], data_dict_recurrent[key]['options'],
                                data_dict_recurrent[key]['answers']))
        f.write("+++\n")

print("====filtered====")
print(len(data_dict_jaccard))
print(len(data_dict_recurrent))
print(len(data_dict_beam))
print(len(data_dict_ori))
print(len(data_dict_random))

n = NLGEval(
    metrics_to_omit=['METEOR', 'EmbeddingAverageCosineSimilairty', 'SkipThoughtCS', 'VectorExtremaCosineSimilarity',
                     'GreedyMatchingScore'])
# n = NLGEval()

print("====similarity between gold and generated distractor====")
overall_dict_jaccard = defaultdict(list)
overall_result_jaccard = dict()
for k, v in data_dict_jaccard.items():
    example_dict = defaultdict(list)
    for i in v['options']:
        metrics_dict = n.compute_individual_metrics(v["gold_distractor"], i)
        for mk, mv in metrics_dict.items():
            example_dict[mk].append(mv)
    for mk, mv in example_dict.items():
        # if not 0 in mv and not any(n < 0 for n in mv):
        #     overall_dict_jaccard[mk].append(statistics.harmonic_mean(mv))
        # else:
        #     overall_dict_jaccard[mk].append(np.mean(mv))
        overall_dict_jaccard[mk].append(np.mean(mv))
for mk, mv in overall_dict_jaccard.items():
    overall_result_jaccard[mk] = np.mean(mv)
print("jaccard", overall_result_jaccard)

overall_dict_ori = defaultdict(list)
overall_result_ori = dict()
for k, v in data_dict_ori.items():
    example_dict = defaultdict(list)
    for i in v['options']:
        metrics_dict = n.compute_individual_metrics(v["gold_distractor"], i)
        for mk, mv in metrics_dict.items():
            example_dict[mk].append(mv)
    for mk, mv in example_dict.items():
        # if not 0 in mv and not any(n < 0 for n in mv):
        #     overall_dict_ori[mk].append(statistics.harmonic_mean(mv))
        # else:
        #     overall_dict_ori[mk].append(np.mean(mv))
        overall_dict_ori[mk].append(np.mean(mv))
for mk, mv in overall_dict_ori.items():
    overall_result_ori[mk] = np.mean(mv)
print("ori", overall_result_ori)

overall_dict_beam = defaultdict(list)
overall_result_beam = dict()
for k, v in data_dict_beam.items():
    example_dict = defaultdict(list)
    for i in v['options']:
        metrics_dict = n.compute_individual_metrics(v["gold_distractor"], i)
        for mk, mv in metrics_dict.items():
            example_dict[mk].append(mv)
    for mk, mv in example_dict.items():
        # if not 0 in mv and not any(n < 0 for n in mv):
        #     overall_dict_beam[mk].append(statistics.harmonic_mean(mv))
        # else:
        #     overall_dict_beam[mk].append(np.mean(mv))
        overall_dict_beam[mk].append(np.mean(mv))
for mk, mv in overall_dict_beam.items():
    overall_result_beam[mk] = np.mean(mv)
print("beam", overall_result_beam)

overall_dict_recurrent = defaultdict(list)
overall_result_recurrent = dict()
for k, v in data_dict_recurrent.items():
    example_dict = defaultdict(list)
    for i in v['options']:
        metrics_dict = n.compute_individual_metrics(v["gold_distractor"], i)
        for mk, mv in metrics_dict.items():
            example_dict[mk].append(mv)
    for mk, mv in example_dict.items():
        # if not 0 in mv and not any(n < 0 for n in mv):
        #     overall_dict_recurrent[mk].append(statistics.harmonic_mean(mv))
        # else:
        #     overall_dict_recurrent[mk].append(np.mean(mv))
        overall_dict_recurrent[mk].append(np.mean(mv))
for mk, mv in overall_dict_recurrent.items():
    overall_result_recurrent[mk] = np.mean(mv)
print("recurrent", overall_result_recurrent)

overall_dict_random = defaultdict(list)
overall_result_random = dict()
for k, v in data_dict_random.items():
    example_dict = defaultdict(list)
    for i in v['options']:
        metrics_dict = n.compute_individual_metrics(v["gold_distractor"], i)
        for mk, mv in metrics_dict.items():
            example_dict[mk].append(mv)
    for mk, mv in example_dict.items():
        # if not 0 in mv and not any(n < 0 for n in mv):
        #     overall_dict_random[mk].append(statistics.harmonic_mean(mv))
        # else:
        #     overall_dict_random[mk].append(np.mean(mv))
        overall_dict_random[mk].append(np.mean(mv))
for mk, mv in overall_dict_random.items():
    overall_result_random[mk] = np.mean(mv)
print("random", overall_result_random)

print("====similarity between distractor====")
overall_dict_jaccard = defaultdict(list)
overall_result_jaccard = dict()
for k, v in data_dict_jaccard.items():
    if len(v['options']) < 4:
        # {'Bleu_1': 0.19999999996000023, 'Bleu_2': 7.071067810274489e-09, 'Bleu_3': 2.5543647739782087e-11, 'Bleu_4': 1.699044244302013e-12, 'METEOR': 0.0547945205479452, 'ROUGE_L': 0.26180257510729615, 'CIDEr': 0.0, 'SkipThoughtCS': 0.41264296, 'EmbeddingAverageCosineSimilairty': 0.804388, 'VectorExtremaCosineSimilarity': 0.650115, 'GreedyMatchingScore': 0.655746}
        example_dict = defaultdict(list)
        for i in set(it.combinations(v['options'], 2)):
            assert len(i[0]) != 0, i
            assert len(i[1]) != 0, i
            metrics_dict = n.compute_individual_metrics([i[0]], i[1])
            for mk, mv in metrics_dict.items():
                example_dict[mk].append(mv)
        for mk, mv in example_dict.items():
            # if not 0 in mv and not any(n < 0 for n in mv):
            #     overall_dict_jaccard[mk].append(statistics.harmonic_mean(mv))
            # else:
            #     overall_dict_jaccard[mk].append(np.mean(mv))
            overall_dict_jaccard[mk].append(np.mean(mv))
for mk, mv in overall_dict_jaccard.items():
    overall_result_jaccard[mk] = np.mean(mv)
print("jaccard", overall_result_jaccard)

overall_dict_ori = defaultdict(list)
overall_result_ori = dict()
for k, v in data_dict_ori.items():
    if len(v['options']) < 4:
        # {'Bleu_1': 0.19999999996000023, 'Bleu_2': 7.071067810274489e-09, 'Bleu_3': 2.5543647739782087e-11, 'Bleu_4': 1.699044244302013e-12, 'METEOR': 0.0547945205479452, 'ROUGE_L': 0.26180257510729615, 'CIDEr': 0.0, 'SkipThoughtCS': 0.41264296, 'EmbeddingAverageCosineSimilairty': 0.804388, 'VectorExtremaCosineSimilarity': 0.650115, 'GreedyMatchingScore': 0.655746}
        example_dict = defaultdict(list)
        for i in set(it.combinations(v['options'], 2)):
            assert len(i[0]) != 0, i
            assert len(i[1]) != 0, i
            metrics_dict = n.compute_individual_metrics([i[0]], i[1])
            for mk, mv in metrics_dict.items():
                example_dict[mk].append(mv)
        for mk, mv in example_dict.items():
            # if not 0 in mv and not any(n < 0 for n in mv):
            #     overall_dict_ori[mk].append(statistics.harmonic_mean(mv))
            # else:
            #     overall_dict_ori[mk].append(np.mean(mv))
            overall_dict_ori[mk].append(np.mean(mv))
for mk, mv in overall_dict_ori.items():
    overall_result_ori[mk] = np.mean(mv)
print("ori", overall_result_ori)

overall_dict_beam = defaultdict(list)
overall_result_beam = dict()
for k, v in data_dict_beam.items():
    if len(v['options']) < 4:
        # {'Bleu_1': 0.19999999996000023, 'Bleu_2': 7.071067810274489e-09, 'Bleu_3': 2.5543647739782087e-11, 'Bleu_4': 1.699044244302013e-12, 'METEOR': 0.0547945205479452, 'ROUGE_L': 0.26180257510729615, 'CIDEr': 0.0, 'SkipThoughtCS': 0.41264296, 'EmbeddingAverageCosineSimilairty': 0.804388, 'VectorExtremaCosineSimilarity': 0.650115, 'GreedyMatchingScore': 0.655746}
        example_dict = defaultdict(list)
        for i in set(it.combinations(v['options'], 2)):
            assert len(i[0]) != 0, i
            assert len(i[1]) != 0, i
            metrics_dict = n.compute_individual_metrics([i[0]], i[1])
            for mk, mv in metrics_dict.items():
                example_dict[mk].append(mv)
        for mk, mv in example_dict.items():
            # if not 0 in mv and not any(n < 0 for n in mv):
            #     overall_dict_beam[mk].append(statistics.harmonic_mean(mv))
            # else:
            #     overall_dict_beam[mk].append(np.mean(mv))
            overall_dict_beam[mk].append(np.mean(mv))
for mk, mv in overall_dict_beam.items():
    overall_result_beam[mk] = np.mean(mv)
print("beam", overall_result_beam)

overall_dict_recurrent = defaultdict(list)
overall_result_recurrent = dict()
for k, v in data_dict_recurrent.items():
    if len(v['options']) < 4:
        # {'Bleu_1': 0.19999999996000023, 'Bleu_2': 7.071067810274489e-09, 'Bleu_3': 2.5543647739782087e-11, 'Bleu_4': 1.699044244302013e-12, 'METEOR': 0.0547945205479452, 'ROUGE_L': 0.26180257510729615, 'CIDEr': 0.0, 'SkipThoughtCS': 0.41264296, 'EmbeddingAverageCosineSimilairty': 0.804388, 'VectorExtremaCosineSimilarity': 0.650115, 'GreedyMatchingScore': 0.655746}
        example_dict = defaultdict(list)
        for i in set(it.combinations(v['options'], 2)):
            assert len(i[0]) != 0, i
            assert len(i[1]) != 0, i
            metrics_dict = n.compute_individual_metrics([i[0]], i[1])
            for mk, mv in metrics_dict.items():
                example_dict[mk].append(mv)
        for mk, mv in example_dict.items():
            # if not 0 in mv and not any(n < 0 for n in mv):
            #     overall_dict_recurrent[mk].append(statistics.harmonic_mean(mv))
            # else:
            #     overall_dict_recurrent[mk].append(np.mean(mv))
            overall_dict_recurrent[mk].append(np.mean(mv))
for mk, mv in overall_dict_recurrent.items():
    overall_result_recurrent[mk] = np.mean(mv)
print("recurrent", overall_result_recurrent)

overall_dict_random = defaultdict(list)
overall_result_random = dict()
for k, v in data_dict_random.items():
    if len(v['options']) < 4:
        # {'Bleu_1': 0.19999999996000023, 'Bleu_2': 7.071067810274489e-09, 'Bleu_3': 2.5543647739782087e-11, 'Bleu_4': 1.699044244302013e-12, 'METEOR': 0.0547945205479452, 'ROUGE_L': 0.26180257510729615, 'CIDEr': 0.0, 'SkipThoughtCS': 0.41264296, 'EmbeddingAverageCosineSimilairty': 0.804388, 'VectorExtremaCosineSimilarity': 0.650115, 'GreedyMatchingScore': 0.655746}
        example_dict = defaultdict(list)
        for i in set(it.combinations(v['options'], 2)):
            assert len(i[0]) != 0, i
            assert len(i[1]) != 0, i
            metrics_dict = n.compute_individual_metrics([i[0]], i[1])
            for mk, mv in metrics_dict.items():
                example_dict[mk].append(mv)
        for mk, mv in example_dict.items():
            # if not 0 in mv and not any(n < 0 for n in mv):
            #     overall_dict_random[mk].append(statistics.harmonic_mean(mv))
            # else:
            #     overall_dict_random[mk].append(np.mean(mv))
            overall_dict_random[mk].append(np.mean(mv))
for mk, mv in overall_dict_random.items():
    overall_result_random[mk] = np.mean(mv)
print("random", overall_result_random)

# with open('data_dict_jaccard.pickle', 'wb') as file:
#     pickle.dump(data_dict_jaccard, file)
# with open('data_dict_recurrent.pickle', 'wb') as file:
#     pickle.dump(data_dict_recurrent, file)
# with open('data_dict_beam.pickle', 'wb')as file:
#     pickle.dump(data_dict_beam, file)
# with open('data_dict_ori.pickle', 'wb')as file:
#     pickle.dump(data_dict_ori, file)
# with open('data_dict_random.pickle', 'wb')as file:
#     pickle.dump(data_dict_random, file)
