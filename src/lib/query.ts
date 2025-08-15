import "dotenv/config";
import { ChatOllama } from "@langchain/ollama";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { vectorStore } from "./llm";
import { logger } from "../utils/logger";

const PROMPT_TEMPLATE = `
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
`;

interface QueryResult {
  response: string;
  sources: (string | null)[];
}

export async function queryDatabase(queryText: string): Promise<QueryResult | null> {
  // Search the DB.
  const results = await vectorStore.similaritySearchWithScore(queryText, 5);
  logger.debug(results);

  if (results.length === 0 || (results[0] && results[0][1] < 0.6)) {
    logger.error("Unable to find matching results.");
    return null;
  }

  const contextText = results
    .map(([doc, _score]) => doc.pageContent)
    .join("\n\n---\n\n");

  const promptTemplate = ChatPromptTemplate.fromTemplate(PROMPT_TEMPLATE);
  const prompt = await promptTemplate.format({
    context: contextText,
    question: queryText,
  });

  const model = new ChatOllama({
    model: process.env.MODEL,
    baseUrl: process.env.MODEL_BASE_URL,
  });

  const response = await model.invoke(prompt);
  const responseText = response.content as string;

  const sources = results.map(([doc, _score]) => doc.metadata.source || null);
  const formattedResponse = `Response: ${responseText}\nSources: ${JSON.stringify(
    sources
  )}`;

  logger.info(formattedResponse);

  return {
    response: responseText,
    sources: sources,
  };
}

export async function main(): Promise<void> {
  // Get query text from command line arguments
  const queryText = process.argv[2];

  if (!queryText) {
    logger.error("Usage: bun run src/lib/query.ts <query_text>");
    process.exit(1);
  }

  await queryDatabase(queryText);
}

// Run main if this file is executed directly
if (import.meta.main) {
  await main();
}
