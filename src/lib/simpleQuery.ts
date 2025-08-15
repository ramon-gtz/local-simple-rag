import "dotenv/config";
import { ChatOllama } from "@langchain/ollama";
import { logger } from "../utils/logger";

export async function main(): Promise<void> {
  // Get query text from command line arguments
  const queryText = process.argv[2];

  if (!queryText) {
    logger.error("Usage: bun run src/lib/simpleQuery.ts <query_text>");
    process.exit(1);
  }

  const model = new ChatOllama({
    model: process.env.MODEL,
    baseUrl: process.env.MODEL_BASE_URL,
  });

  const response = await model.invoke(queryText);
  logger.info(response.content);
}

// Run main if this file is executed directly
if (import.meta.main) {
  await main();
}
