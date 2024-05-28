import chromadb
from chromadb.utils import embedding_functions
from datasets import load_dataset
from torch import cuda
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm


device = 'cuda' if cuda.is_available() else 'cpu'
sentence_transformer = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2", device='cuda')


class Database:
    def query(self, query, k=10):
        return self.db.query(    
            query_texts=[query],
            n_results=k,
        )['documents'][0]


class Wiki10k(Database):
    def __init__(self):
        self.client = chromadb.PersistentClient(path='./chroma')
        try:
            self.db = self.client.get_collection('wiki10k')
        except:
            self.db = self.client.create_collection('wiki10k', embedding_function=sentence_transformer)
            wikitext = load_dataset('sentence-transformers/wikipedia-en-sentences')
            loader = DataLoader(wikitext['train']['sentence'][:10000], batch_size=100)
            for i, batch in enumerate(tqdm(loader, desc='Embedding Wiki10k')):
                self.db.add(ids=[f"{i*100 + j}" for j in range(100)], documents=batch)
            


class Wikipedia(Database):
    def __init__(self):
        self.client = chromadb.PersistentClient(path='./chroma')
        try:
            self.db = self.client.get_collection('wikipedia')
        except:
            self.db = self.client.create_collection('wikipedia', embedding_function=sentence_transformer)
            wikitext = load_dataset('sentence-transformers/wikipedia-en-sentences')
            loader = DataLoader(wikitext['train']['sentence'], batch_size=100)
            for i, batch in enumerate(tqdm(loader, desc='Embedding Wikipedia')):
                self.db.add(ids=[f"{i*100 + j}" for j in range(100)], documents=batch)
