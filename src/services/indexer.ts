import type { QdrantVectorStore } from "@langchain/qdrant";
import { DirectoryLoader } from "langchain/document_loaders/fs/directory";
import { TextLoader } from "langchain/document_loaders/fs/text";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { CSVLoader } from "@langchain/community/document_loaders/fs/csv";
import { DocxLoader } from "@langchain/community/document_loaders/fs/docx";
import { JSONLoader } from "langchain/document_loaders/fs/json";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import type { Document } from "@langchain/core/documents";
import { logger, type Logger } from "../utils/logger";
import type { IndexerConfig } from "../types";
import { vectorStore } from "../lib/llm";

export class Indexer {
  private vectorStore?: QdrantVectorStore;
  private config: Required<IndexerConfig>;
  private logger: Logger;

  constructor(vectorStore?: QdrantVectorStore, config?: IndexerConfig) {
    this.vectorStore = vectorStore;
    this.config = {
      chunkSize: config?.chunkSize || 500,
      chunkOverlap: config?.chunkOverlap || 100,
      enableLogging: config?.enableLogging || true,
      batchSize: config?.batchSize || 10,
    };
    this.logger = logger;
  }

  private log(message: string) {
    if (this.config.enableLogging) {
      this.logger.info(`[Indexer] ${message}`);
    }
  }

  private logError(message: string) {
    if (this.config.enableLogging) {
      this.logger.error(`[Indexer] ${message}`);
    }
  }

  private logDebug(message: string, data?: any) {
    if (this.config.enableLogging) {
      this.logger.debug(`[Indexer] ${message}`, data);
    }
  }

  async loadDocuments(folderPath: string) {
    try {
      const loader = new DirectoryLoader(
        folderPath,
        {
          ".txt": (path) => new TextLoader(path),
          ".md": (path) => new TextLoader(path),
          ".pdf": (path) => new PDFLoader(path),
          ".csv": (path) => new CSVLoader(path),
          ".docx": (path) => new DocxLoader(path),
          ".json": (path) => new JSONLoader(path),
        },
        true // recursive
      );

      const documents = await loader.load();

      // Add account metadata to each document
      documents.forEach((doc) => {
        doc.metadata = {
          ...doc.metadata,
          loadedFrom: "local_folder",
          folderPath,
        };
      });

      this.log(
        `Loaded ${documents.length} documents from folder: ${folderPath}`
      );
      return documents;
    } catch (error) {
      this.logError(`Error loading documents from ${folderPath}: ${error}`);
      throw new Error(
        `Failed to load documents for folder ${folderPath}: ${error}`
      );
    }
  }

  /**
   * Split documents into chunks
   */
  async splitDocuments(documents: Document[]): Promise<Document[]> {
    if (documents.length === 0) {
      this.log("No documents to split");
      return [];
    }

    const textSplitter = new RecursiveCharacterTextSplitter({
      chunkSize: this.config.chunkSize,
      chunkOverlap: this.config.chunkOverlap,
    });

    this.log(
      `Splitting ${documents.length} documents (chunk size: ${this.config.chunkSize}, overlap: ${this.config.chunkOverlap})...`
    );

    const chunks = await textSplitter.splitDocuments(documents);

    this.log(`Split into ${chunks.length} chunks`);

    // Log sample chunk for debugging
    if (this.config.enableLogging && chunks.length > 10) {
      const sampleChunk = chunks[10];
      this.logDebug(
        "Sample chunk:",
        sampleChunk?.pageContent.substring(0, 200) + "..."
      );
      this.logDebug("Sample metadata:", sampleChunk?.metadata);
    }

    return chunks;
  }

  async indexDocuments(chunks: Document[]): Promise<void> {
    if (chunks.length === 0) {
      this.log("No chunks to save");
      return;
    }

    this.log(`Saving ${chunks.length} chunks to vector store...`);

    // Process in batches to avoid overwhelming the vector store
    const batchSize = this.config.batchSize;
    for (let i = 0; i < chunks.length; i += batchSize) {
      const batch = chunks.slice(i, i + batchSize);
      this.log(
        `Processing batch ${Math.floor(i / batchSize) + 1}/${Math.ceil(
          chunks.length / batchSize
        )} (${batch.length} chunks)`
      );

      try {
        await this.vectorStore?.addDocuments(batch);
      } catch (error) {
        this.log(`Error processing batch: ${error}`);
        throw error;
      }
    }

    this.log("Successfully saved all chunks to vector store");
  }

  async indexFolder(folderPath: string) {
    const documents = await this.loadDocuments(folderPath);
    const chunks = await this.splitDocuments(documents);
    await this.indexDocuments(chunks);
  }
}

if (import.meta.main) {
  const indexer = new Indexer(vectorStore);
  await indexer.indexFolder("./data");
}