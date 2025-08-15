import { Logger as TsLogger } from "tslog";

export const logger = new TsLogger({
  prettyLogTemplate: "{{yyyy}}-{{mm}}-{{dd}} {{hh}}:{{MM}}:{{ss}} [{{name}}] {{logLevelName}} ",
  type: "pretty",
  name: "SimpleRAG"
});

export type Logger = typeof logger;