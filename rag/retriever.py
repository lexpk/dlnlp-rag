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
            loader = DataLoader(wikitext['train']['contents'], batch_size=100)
            for i, batch in enumerate(tqdm(loader, desc='Embedding Medical documents')):
                self.db.add(ids=[f"{i*100 + j}" for j in range(100)], documents=batch)

