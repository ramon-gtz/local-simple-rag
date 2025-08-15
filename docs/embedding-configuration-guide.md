# Optimal Embedding Configuration Guide for RAG Systems

## Overview
This guide provides best practices for configuring text embeddings in RAG (Retrieval-Augmented Generation) systems, specifically optimized for Ollama embeddings with Qdrant vector store.

## 1. Embedding Model Selection

### Recommended Models for Ollama
```typescript
// High-quality general-purpose models
const recommendedModels = {
  // Best overall performance
  "nomic-embed-text": "v1.5", // 768 dimensions, excellent quality
  "all-MiniLM-L6-v2": "latest", // 384 dimensions, fast and accurate
  "all-mpnet-base-v2": "latest", // 768 dimensions, high quality
  
  // Multilingual support
  "paraphrase-multilingual-MiniLM-L12-v2": "latest",
  
  // Domain-specific (if available)
  "sentence-transformers": "latest" // Generic sentence transformer
};
```

### Model Configuration
```typescript
export const embeddings = new OllamaEmbeddings({
  model: process.env.EMBEDDINGS_MODEL || "nomic-embed-text",
  // Additional parameters for better performance
  maxConcurrency: 5, // Parallel processing
  batchSize: 32, // Optimal batch size for most models
  timeout: 30000, // 30 second timeout
});
```

## 2. Text Chunking Strategy

### Optimal Chunking Parameters
```typescript
interface OptimalChunkingConfig {
  // For general documents
  general: {
    chunkSize: 512, // ~512 tokens (roughly 400-600 characters)
    chunkOverlap: 128, // 25% overlap for context preservation
  },
  
  // For technical/scientific documents
  technical: {
    chunkSize: 768, // Larger chunks for complex content
    chunkOverlap: 192, // 25% overlap
  },
  
  // For conversational/chat data
  conversational: {
    chunkSize: 256, // Smaller chunks for dialogue
    chunkOverlap: 64, // 25% overlap
  },
  
  // For code/documentation
  code: {
    chunkSize: 1024, // Larger chunks to preserve code structure
    chunkOverlap: 256, // 25% overlap
  }
}
```

### Advanced Chunking Strategy
```typescript
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";

const createOptimalSplitter = (contentType: 'general' | 'technical' | 'conversational' | 'code') => {
  const configs = {
    general: { chunkSize: 512, chunkOverlap: 128 },
    technical: { chunkSize: 768, chunkOverlap: 192 },
    conversational: { chunkSize: 256, chunkOverlap: 64 },
    code: { chunkSize: 1024, chunkOverlap: 256 }
  };
  
  const config = configs[contentType];
  
  return new RecursiveCharacterTextSplitter({
    chunkSize: config.chunkSize,
    chunkOverlap: config.chunkOverlap,
    separators: [
      "\n\n", // Paragraphs
      "\n",   // Lines
      ". ",   // Sentences
      "! ",   // Exclamations
      "? ",   // Questions
      ", ",   // Clauses
      " ",    // Words
      ""      // Characters
    ],
    lengthFunction: (text) => text.length, // Character-based for consistency
  });
};
```

## 3. Vector Store Configuration

### Qdrant Optimization
```typescript
import { QdrantVectorStore } from "@langchain/qdrant";

export const vectorStore = new QdrantVectorStore(embeddings, {
  url: process.env.VECTOR_STORE_URL,
  collectionName: process.env.VECTOR_STORE_COLLECTION,
  // Performance optimizations
  collectionConfig: {
    vectors: {
      size: 768, // Match your embedding model dimension
      distance: "Cosine", // Best for semantic similarity
      // Alternative: "Dot" for normalized embeddings
    },
    optimizers_config: {
      memmap_threshold: 20000, // Use memory mapping for large collections
    },
    quantization_config: {
      scalar: {
        type: "int8", // Reduce memory usage
        quantile: 0.99,
      },
    },
  },
});
```

## 4. Retrieval Parameters

### Optimal Retrieval Configuration
```typescript
interface RetrievalConfig {
  // Number of documents to retrieve
  topK: 5, // Start with 5, adjust based on context window
  
  // Similarity threshold
  similarityThreshold: 0.7, // Minimum similarity score
  
  // Reranking (if available)
  enableReranking: true,
  rerankTopK: 20, // Retrieve more, then rerank
  
  // Hybrid search
  enableHybridSearch: true,
  alpha: 0.5, // Balance between dense and sparse retrieval
}

// Enhanced retrieval function
async function enhancedRetrieval(
  query: string, 
  config: RetrievalConfig = { topK: 5, similarityThreshold: 0.7 }
) {
  // Retrieve more candidates initially
  const candidates = await vectorStore.similaritySearchWithScore(
    query, 
    config.topK * 2
  );
  
  // Filter by similarity threshold
  const filtered = candidates.filter(([_, score]) => score >= config.similarityThreshold);
  
  // Return top K results
  return filtered.slice(0, config.topK);
}
```

## 5. Performance Optimization

### Batch Processing
```typescript
interface BatchConfig {
  // For indexing
  indexingBatchSize: 50, // Process 50 documents at a time
  
  // For embedding generation
  embeddingBatchSize: 32, // Optimal for most models
  
  // For retrieval
  retrievalBatchSize: 10, // Process queries in batches
}

// Optimized indexing
async function optimizedIndexing(documents: Document[], config: BatchConfig) {
  const batches = [];
  for (let i = 0; i < documents.length; i += config.indexingBatchSize) {
    batches.push(documents.slice(i, i + config.indexingBatchSize));
  }
  
  for (const batch of batches) {
    await vectorStore.addDocuments(batch);
    // Add delay to prevent overwhelming the system
    await new Promise(resolve => setTimeout(resolve, 100));
  }
}
```

## 6. Quality Metrics & Monitoring

### Embedding Quality Assessment
```typescript
interface QualityMetrics {
  // Semantic similarity tests
  semanticTests: {
    query: string;
    expectedResults: string[];
    minSimilarity: number;
  }[];
  
  // Diversity metrics
  diversityThreshold: 0.3, // Minimum diversity between results
  
  // Coverage metrics
  coverageThreshold: 0.8, // Minimum coverage of relevant content
}

// Quality assessment function
async function assessEmbeddingQuality(
  testQueries: string[], 
  expectedResults: string[][],
  config: QualityMetrics
) {
  const results = [];
  
  for (let i = 0; i < testQueries.length; i++) {
    const query = testQueries[i];
    const expected = expectedResults[i];
    
    const retrieved = await vectorStore.similaritySearch(query, 5);
    const retrievedTexts = retrieved.map(doc => doc.pageContent);
    
    // Calculate overlap with expected results
    const overlap = calculateOverlap(retrievedTexts, expected);
    results.push({ query, overlap, retrieved: retrievedTexts });
  }
  
  return results;
}
```

## 7. Environment-Specific Configurations

### Development Environment
```typescript
const devConfig = {
  chunkSize: 256, // Smaller chunks for faster processing
  chunkOverlap: 64,
  batchSize: 10,
  topK: 3,
  similarityThreshold: 0.6, // Lower threshold for more results
};
```

### Production Environment
```typescript
const prodConfig = {
  chunkSize: 512, // Optimal chunk size
  chunkOverlap: 128,
  batchSize: 50,
  topK: 5,
  similarityThreshold: 0.7, // Higher threshold for quality
  enableCaching: true,
  cacheTTL: 3600, // 1 hour cache
};
```

### High-Performance Environment
```typescript
const highPerfConfig = {
  chunkSize: 768, // Larger chunks for better context
  chunkOverlap: 192,
  batchSize: 100,
  topK: 10,
  similarityThreshold: 0.8, // Very high threshold
  enableParallelProcessing: true,
  maxConcurrency: 10,
};
```

## 8. Recommended Configuration for Your Setup

Based on your current Ollama + Qdrant setup:

```typescript
// Recommended configuration
export const recommendedConfig = {
  // Embedding model
  embeddingModel: "nomic-embed-text",
  
  // Chunking
  chunkSize: 512,
  chunkOverlap: 128,
  
  // Processing
  batchSize: 32,
  maxConcurrency: 5,
  
  // Retrieval
  topK: 5,
  similarityThreshold: 0.7,
  
  // Vector store
  vectorSize: 768,
  distanceMetric: "Cosine",
  
  // Performance
  enableQuantization: true,
  memmapThreshold: 20000,
};
```

## 9. Troubleshooting Common Issues

### Low Retrieval Quality
- Increase `chunkOverlap` to preserve context
- Decrease `similarityThreshold` for more results
- Try different embedding models
- Adjust chunk size based on content type

### Slow Performance
- Enable quantization in Qdrant
- Reduce batch size
- Use memory mapping for large collections
- Implement caching for frequent queries

### Memory Issues
- Reduce batch size
- Enable quantization
- Use streaming for large datasets
- Implement pagination for retrieval

## 10. Monitoring & Maintenance

### Key Metrics to Track
- Embedding generation time
- Retrieval latency
- Similarity score distribution
- Chunk overlap effectiveness
- Memory usage
- Query success rate

### Regular Maintenance
- Monitor embedding model updates
- Retrain/reindex periodically
- Clean up low-quality embeddings
- Optimize vector store configuration
- Update chunking strategy based on content changes
