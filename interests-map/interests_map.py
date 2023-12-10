import csv
import seaborn as sns
from scipy import spatial
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer
import umap.umap_ as umap
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np

def load_data():
  person_to_interests_map = {}

  with open('data/ppl.csv', newline='') as csvfile:
      file = csv.reader(csvfile, delimiter=',', quotechar='"')
      next(file)  # Skip the header row
      
      for row in file:
        name, personality, sp_id = row
        person_to_interests_map[sp_id] = (name, personality)
      
  return person_to_interests_map

def load_embeddings():
  embedding_map = {}

  with open('data/embeddings.csv', newline='') as csvfile:
      file = csv.reader(csvfile, delimiter=',', quotechar='"')
      next(file)  # Skip the header row
      
      for row in file:
        sp_id, embedding = row
        embedding_map[sp_id] = np.fromstring(embedding, sep=',')
      
  return embedding_map

def generate_interest_embedding(interest_list):
  model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
  embeddings = model.encode(interest_list)
      
  return embeddings


def create_visualization(person_interest_map):
  embeddings = np.array(list(load_embeddings().values()))

  reducer = umap.UMAP(n_neighbors=4)
  scaler = StandardScaler()
  scaled_data = scaler.fit_transform(embeddings)

  reduced_data = reducer.fit_transform(scaled_data)
  
  xs = [row[0] for row in reduced_data]
  ys = [row[1] for row in reduced_data]
  
  plt.scatter(xs, ys)
  for i, (name, _) in enumerate(list(person_interest_map.values())):
    plt.annotate(name, (xs[i], ys[i]))
  
  plt.axis('off')
  plt.savefig('data/visualization.png')
  plt.clf()
  
  
def provide_top_matches(person_interest_map, n_mactches=5):
  # TODO: create persistent embeddings store
  embeddings = np.array(list(load_embeddings().values()))

  embeddings_map = {spid: (name, embedding) for spid, (name, _), embedding in zip(person_interest_map.items(), embeddings)}

  top_matches = {}
  all_personal_pairs = defaultdict(list)
  
  ids = list(person_interest_map.keys())
  for id1 in ids:
    for id2 in ids:
        all_personal_pairs[id1].append([spatial.distance.cosine(embeddings_map[id1], embeddings_map[id2]), person_interest_map[id1][0]])

  for sp_id in ids:
    top_matches[sp_id] = sorted(all_personal_pairs[sp_id], key=lambda x: x[0])
    
  return {spid: matches[:n_mactches] for sp_id, matches in top_matches.items()}

def add_interests(interests_array):
  for name, interest, sp_id in interests_array:
    with open('data/ppl.csv', 'a', newline='') as csvfile:
      writer = csv.writer(csvfile, delimiter=',', quotechar='"')
      writer.writerow([name, interest, sp_id])
      
    with open('data/embeddings.csv', 'a', newline='') as f:
        writer = csv.writer(f, delimiter=',', quotechar='"')
        emb = generate_interest_embedding([interest])[0]
        writer.writerow([sp_id, np.array2string(emb, separator=',').replace('\n', '').strip('[]')])