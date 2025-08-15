import "dotenv/config";
import { OllamaEmbeddings } from "@langchain/ollama";
import { QdrantVectorStore } from "@langchain/qdrant";

export const embeddings = new OllamaEmbeddings({
  model: process.env.EMBEDDINGS_MODEL,
});

export const getVectorStore = (collectionName: string) => {
    return new QdrantVectorStore(embeddings, {
      url: process.env.VECTOR_STORE_URL,
      collectionName,
    });
  };
