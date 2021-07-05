import pickle
import random
import json

MIN_TOKENS = 20
N_EXAMPLES = 30

with open("chintang_vocab.p", "rb") as f:
    mapWordIndo = pickle.load(f)

lstWords = []
for strWord, mapInfo in mapWordIndo.items():
    mapInfo["word"] = strWord
    lstWords.append(mapInfo)

lstWords.sort(key=lambda mapInfo: -mapInfo["n"])

lstFrequentEnough = [ mapInfo for mapInfo in lstWords if mapInfo["n"] >= MIN_TOKENS]

random.shuffle(lstFrequentEnough)

lstExamples = [ ]

for i, mapInfo in enumerate(lstFrequentEnough):
    if i >= MIN_TOKENS:
        break

    lstExamples.append(mapInfo["word"])

print(json.dumps(lstExamples, ensure_ascii=False).encode('utf-8').decode())
