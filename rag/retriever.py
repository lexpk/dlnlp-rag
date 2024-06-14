import chromadb
from chromadb.utils import embedding_functions
from datasets import load_dataset
from torch import cuda
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm


class Retriever:
    def __call__(self, query, k=10):
        return self.db.query(    
            query_texts=[query],
            n_results=k,
        )['documents'][0]


class Wiki10k(Retriever):
    def __init__(self, sentence_transformer="all-MiniLM-L6-v2"):
        self.client = chromadb.PersistentClient(path='./chroma')
        sentence_transformer = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=sentence_transformer,
            device='cuda' if cuda.is_available() else 'cpu'
        )
        try:
            self.db = self.client.get_collection('Wiki10k')
        except:
            self.db = self.client.create_collection('Wiki10k', embedding_function=sentence_transformer)
            wikitext = load_dataset('sentence-transformers/wikipedia-en-sentences')
            loader = DataLoader(wikitext['train']['sentence'][:10000], batch_size=100)
            for i, batch in enumerate(tqdm(loader, desc='Embedding Wiki10k')):
                self.db.add(ids=[f"{i*100 + j}" for j in range(100)], documents=batch)
            


class Wikipedia(Retriever):
    def __init__(self, sentence_transformer="all-MiniLM-L6-v2"):
        self.client = chromadb.PersistentClient(path='./chroma')
        sentence_transformer = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=sentence_transformer,
            device='cuda' if cuda.is_available() else 'cpu'
        )
        try:
            self.db = self.client.get_collection('Wikipedia')
        except:
            self.db = self.client.create_collection('Wikipedia', embedding_function=sentence_transformer)
            wikitext = load_dataset('sentence-transformers/wikipedia-en-sentences')
            loader = DataLoader(wikitext['train']['sentence'], batch_size=100)
            for i, batch in enumerate(tqdm(loader, desc='Embedding Wikipedia')):
                self.db.add(ids=[f"{i*100 + j}" for j in range(100)], documents=batch)


class Medical(Retriever):
    def __init__(self, sentence_transformer="all-MiniLM-L6-v2"):
        self.client = chromadb.PersistentClient(path='./chroma')
        sentence_transformer = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=sentence_transformer,
            device='cuda' if cuda.is_available() else 'cpu'
        )
        try:
            self.db = self.client.get_collection('Medical')
        except:
            self.db = self.client.create_collection('Medical', embedding_function=sentence_transformer)
            dataset = load_dataset("MedRAG/textbooks")         
            loader = DataLoader(dataset['train']['contents'], batch_size=100)
            for i, batch in enumerate(tqdm(loader, desc='Embedding Medical documents')):
                batch_size = len(batch)
                self.db.add(ids=[f"{i*100 + j}" for j in range(batch_size)], documents=batch)


# Retriever but instead of a predefined db we use a WebSearch
# Save your APi key as an environment variable: WEB_SEARCH_TOKEN=<token>

import requests
import json

class WebSearch():
    def __init__(self):
      pass

    def __call__(self, query, k=10):
      url = 'https://api.tavily.com/search'
      parameters = {
        "api_key": os.environ.get("WEB_SEARCH_TOKEN"),
        "query": query,
        "search_depth": "basic",
        "include_answer": False,
        "include_images": False,
        "include_raw_content": False,
        "max_results": k,
        "include_domains": [],
        "exclude_domains": []
      }

      response = requests.post(url, json = parameters)
      result = response.json()

      contextEntries = []

      for entry in result["results"]:
        content = entry["content"]
        if not content or content=="[Removed]":
          continue

        # optionally trim content
        # content = content[:50]

        contextEntries.append(content)
        if len(contextEntries) >= 10:
          break

      return "\n".join(contextEntries)

