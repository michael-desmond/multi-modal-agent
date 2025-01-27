import "dotenv/config";
import { z } from "zod";
import { Workflow } from "bee-agent-framework/experimental/workflows/workflow";
import { UnconstrainedMemory } from "bee-agent-framework/memory/unconstrainedMemory";
import { createConsoleReader } from "./helpers/reader.js";
import { BaseMessage, Role } from "bee-agent-framework/llms/primitives/message";
import { JsonDriver } from "bee-agent-framework/llms/drivers/json";
import { getChatLLM } from "src/helpers/llm.js";
import { ReadOnlyMemory } from "bee-agent-framework/memory/base";
import { PromptTemplate } from "bee-agent-framework";

interface LLMRoute {
  name: string;
  description: string;
}

// Available routes w/ description
const routes: LLMRoute[] = [
  {
    name: "language",
    description:
      "Handles general language queries. Best at language-based tasks like text generation or comprehension.",
  },
  {
    name: "vision",
    description:
      "Handles queries about the content of an image. Capable of understaning objects and scenes within an image.",
  },
  {
    name: "timeseries",
    description:
      "Handles queries that requires analyzing trends, patterns, and behaviors over time.",
  },
];

// Routing prompt, looks at the user message and decides where to route
// Includes message history for better contextual reasoning
const ROUTE_MESSAGE_PROMPT = new PromptTemplate({
  schema: z.object({
    message: z.string(),
    memory: z.array(z.instanceof(BaseMessage)),
    routes: z.array(
      z.object({
        name: z.string().min(1),
        description: z.string().min(1),
      }),
    ),
  }),
  template: `You are an assistant tasked with routing the users query based on their most recent message.

The following routes are available:"
{{#routes}}
Name: {{name}}
Description: {{description}}
{{/routes}}

Message History:
{{#memory}}
{{role}}: {{text}}
{{/memory}}

User Message: 
{{message}}

Choose the most appropriate route for the user message.
`,
});

// Agent schema
// Could include image data in here
const workflowSchema = z.object({
  response: z.string().optional(),
  memory: z.instanceof(ReadOnlyMemory),
});

const routeSchema = z.union([z.literal("language"), z.literal("vision"), z.literal("timeseries")]);

// Agent workflow
const workflow = new Workflow({
  schema: workflowSchema,
  outputSchema: workflowSchema.required({ response: true }),
})
  .addStep("routeUserMessage", async (state) => {
    const driver = new JsonDriver(getChatLLM());
    const { parsed } = await driver.generate(
      z.object({
        route: routeSchema,
      }),
      [
        BaseMessage.of({
          role: Role.USER,
          text: ROUTE_MESSAGE_PROMPT.render({
            message: state.memory.messages[state.memory.messages.length - 1].text,
            memory: state.memory.messages.slice(0, -1),
            routes: routes,
          }),
        }),
      ],
    );
    return { next: parsed.route };
  })
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  .addStep("language", async (state) => {
    console.log("Routed to language node.");
    return { next: "generateResponse" };
  })
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  .addStep("vision", async (state) => {
    console.log("Routed to vision node.");
    return { next: "generateResponse" };
  })
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  .addStep("timeseries", async (state) => {
    console.log("Routed to timeseries node.");
    return { next: "generateResponse" };
  })
  .addStep("generateResponse", async (state) => {
    const driver = new JsonDriver(getChatLLM());
    const { parsed } = await driver.generate(
      z.object({
        response: z.string(),
      }),
      [
        BaseMessage.of({
          role: Role.SYSTEM,
          text: `You are a helpful AI assistant.`,
        }),
        ...state.memory,
      ],
    );
    return { update: { response: parsed.response }, next: Workflow.END };
  });

const memory = new UnconstrainedMemory();
const reader = createConsoleReader();

for await (const { prompt } of reader) {
  const userMessage = BaseMessage.of({
    role: Role.ASSISTANT,
    text: prompt,
  });

  await memory.add(userMessage);

  const response = await workflow.run({
    memory: memory.asReadOnly(),
  });
  // .observe((emitter) => {
  //   emitter.on("start", ({ step, run }) => {
  //     reader.write(`-> ▶️ ${step}`, JSON.stringify(run.state));
  //   });
  // });

  await memory.add(
    BaseMessage.of({
      role: Role.ASSISTANT,
      text: response.state.response!,
    }),
  );

  reader.write("🤖 Answer", response.state.response!);
}
