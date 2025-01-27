import "dotenv/config";
import { z } from "zod";
import { Workflow } from "bee-agent-framework/experimental/workflows/workflow";
import { BeeAgent } from "bee-agent-framework/agents/bee/agent";
import { UnconstrainedMemory } from "bee-agent-framework/memory/unconstrainedMemory";
import { createConsoleReader } from "./helpers/reader.js";
import { BaseMessage } from "bee-agent-framework/llms/primitives/message";
import { JsonDriver } from "bee-agent-framework/llms/drivers/json";
import { isEmpty, pick } from "remeda";
import { LLMTool } from "bee-agent-framework/tools/llm";
import { DuckDuckGoSearchTool } from "bee-agent-framework/tools/search/duckDuckGoSearch";
import { getChatLLM } from "src/helpers/llm.js";

const schema = z.object({
  input: z.string(),
  output: z.string().optional(),

  topic: z.string().optional(),
  notes: z.array(z.string()).default([]),
  plan: z.string().optional(),
  draft: z.string().optional(),
});

const workflow = new Workflow({
  schema,
  outputSchema: schema.required({ output: true }),
})
  .addStep("preprocess", async (state) => {
    const llm = getChatLLM();
    const driver = new JsonDriver(llm);

    const { parsed } = await driver.generate(
      z
        .object({
          error: z
            .string()
            .describe(
              "Use this field only if the user message is not a valid topic and is not a note to an existing blog post.",
            ),
        })
        .or(schema.pick({ topic: true, notes: true })),
      [
        BaseMessage.of({
          role: `user`,
          text: [
            "Your task is to rewrite the user query so that it guides the content planner and editor to craft a blog post that perfectly aligns with the user's needs. Notes should be used only if the user complains about something.",
            "If the user query does ",
            "",
            ...[state.topic && ["# Previous Topic", state.topic, ""]],
            ...[!isEmpty(state.notes) && ["# Previous Notes", ...state.notes, ""]],
            "# User Query",
            state.input || "empty",
          ]
            .filter(Boolean)
            .join("\n"),
        }),
      ],
    );

    return "error" in parsed
      ? { update: { output: parsed.error }, next: Workflow.END }
      : { update: pick(parsed, ["notes", "topic"]) };
  })
  .addStrictStep("planner", schema.required({ topic: true }), async (state) => {
    const llm = getChatLLM();
    const agent = new BeeAgent({
      llm,
      memory: new UnconstrainedMemory(),
      tools: [new DuckDuckGoSearchTool(), new LLMTool({ llm })],
    });

    const { result } = await agent.run({
      prompt: [
        `You are a Content Planner. Your task is to write a content plan for "${state.topic}" topic in Markdown format.`,
        ``,
        `# Objectives`,
        `1. Prioritize the latest trends, key players, and noteworthy news.`,
        `2. Identify the target audience, considering their interests and pain points.`,
        `3. Develop a detailed content outline including introduction, key points, and a call to action.`,
        `4. Include SEO keywords and relevant sources.`,
        ``,
        ...[!isEmpty(state.notes) && ["# Notes", ...state.notes, ""]],
        `Provide a structured output that covers the mentioned sections.`,
      ].join("\n"),
    });

    return {
      update: {
        plan: result.text,
      },
    };
  })
  .addStrictStep("writer", schema.required({ plan: true }), async (state) => {
    const llm = getChatLLM();
    const output = await llm.generate([
      BaseMessage.of({
        role: `system`,
        text: [
          `You are a Content Writer. Your task is to write a compelling blog post based on the provided context.`,
          ``,
          `# Context`,
          `${state.plan}`,
          ``,
          `# Objectives`,
          `- An engaging introduction`,
          `- Insightful body paragraphs (2-3 per section)`,
          `- Properly named sections/subtitles`,
          `- A summarizing conclusion`,
          `- Format: Markdown`,
          ``,
          ...[!isEmpty(state.notes) && ["# Notes", ...state.notes, ""]],
          `Ensure the content flows naturally, incorporates SEO keywords, and is well-structured.`,
        ].join("\n"),
      }),
    ]);

    return {
      update: { draft: output.getTextContent() },
    };
  })
  .addStrictStep("editor", schema.required({ draft: true }), async (state) => {
    const llm = getChatLLM();
    const output = await llm.generate([
      BaseMessage.of({
        role: `system`,
        text: [
          `You are an Editor. Your task is to transform the following draft blog post to a final version.`,
          ``,
          `# Draft`,
          `${state.draft}`,
          ``,
          `# Objectives`,
          `- Fix Grammatical errors`,
          `- Journalistic best practices`,
          ``,
          ...[!isEmpty(state.notes) && ["# Notes", ...state.notes, ""]],
          ``,
          `IMPORTANT: The final version must not contain any editor's comments.`,
        ].join("\n"),
      }),
    ]);

    return {
      update: { output: output.getTextContent() },
    };
  });

let lastResult = {} as Workflow.output<typeof workflow>;
const reader = createConsoleReader();
reader.write(
  "ℹ️ ",
  "I am a content creator agent. Please give me a topic for which I will write a blog post..",
);

for await (const { prompt } of reader) {
  const { result } = await workflow
    .run({
      input: prompt,
      notes: lastResult?.notes,
      topic: lastResult?.topic,
    })
    .observe((emitter) => {
      emitter.on("start", ({ step, run }) => {
        reader.write(`-> ▶️ ${step}`, JSON.stringify(run.state));
      });
    });

  lastResult = result;
  reader.write("🤖 Answer", lastResult.output);
}