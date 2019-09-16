import faiss
import sys
import os
import numpy as np
import random
import logging

ids = []
vectors = []
with open('./video_embedding_test.csv','r') as f:
    for eachline in f:
        line = eachline.replace('\n','').split(':')
        ids.append(int(line[0]))
        vectors.append([float(x) for x in line[1].split(',')])

vectors = np.array(vectors).reshape((len(vectors),128)).astype('float32')
ids = np.array(ids).reshape((len(ids),)).astype('int64')
    
index = faiss.IndexFlatIP(128)
index_with_id = faiss.IndexIDMap(index)
faiss.normalize_L2(vectors)
index_with_id.add_with_ids(vectors,ids)

search_vectors = vectors[:100]
faiss.normalize_L2(search_vectors)

D, I = index_with_id.search(search_vectors,10)
result = zip(I,D)
f = open('./sim_result.csv','w')
for ids, sims in result:
    f.write("{}\n".format(','.join(["{}:{}".format(v,sim) for (v,sim) in zip(ids,sims)])))
    
