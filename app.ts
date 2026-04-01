import fs, { stat } from "fs";
import path from "path";
import dotenv from "dotenv";
import express, { Request, Response } from "express";
import multer from "multer";

import { ChatGoogle } from "@langchain/google";
import { HumanMessage, AIMessage, SystemMessage } from "@langchain/core/messages";
import { tool } from "@langchain/core/tools";
import { createAgent } from "langchain";
import { ToolNode } from "@langchain/langgraph/prebuilt";
import { StateGraph, START, END,StateSchema } from "@langchain/langgraph";
import { AzureOpenAI as LangChainAzureOpenAI } from "@langchain/openai";
import { AzureOpenAI } from "openai";
import { createClient } from "redis";
import { z } from "zod";

// language-mandated adaptation: dotenv for environment variables
const envCandidates = [
  path.resolve(process.cwd(), ".env"),
  path.resolve(__dirname, ".env"),
  path.resolve(__dirname, "../.env"),
];
const envPath = envCandidates.find((p) => fs.existsSync(p));
dotenv.config({ path: envPath, override: true });

// ============ Zod Schemas (synced from Python data classes) ============

// LongTermMemory Schema
const LongTermMemory = z.object({
  user_name: z.string().optional().nullable().describe("The name of the user"),
  tone: z.string().default("neutral").describe("The preferred tone of communication (casual, formal, playful, etc.)"),
  preferences: z.array(z.string()).default([]).describe("List of user preferences including learning interests, favorite subjects, favorite things"),
  favorite_topics: z.array(z.string()).optional().nullable().describe("Topics the student loves learning about (science, history, math, art, etc.)"),
  learning_style: z.string().optional().nullable().describe("How the student prefers to learn (visual examples, step-by-step, stories, hands-on, etc.)"),
  goals: z.array(z.string()).optional().nullable().describe("Learning goals or areas they want to improve"),
});

// Quiz Generation Schema
const quiz_generation = z.object({
  question: z.string().describe("The quiz question"),
  options: z.array(z.string()).describe("List of answer options"),
});

const quizstructure = z.object({
  quiz: z.array(quiz_generation),
});

// Evaluation Response Schema
const evaluation_response = z.object({
  wrong: z.string().describe("The wrong answer given by the student"),
  corrections: z.array(z.string()).describe("explanations of the wrong answer"),
  explaination: z.string().describe("you are explaining it to a kid so do it in a fun way like a game"),
  advice: z.string().describe("Advice for improvement"),
});

// Sub Module Schema
const sub_module = z.object({
  sub_module_title: z.string().describe("The title of the sub-module"),
  content: z.string().describe("The content of the sub-module"),
  conclusion: z.string().describe("The conclusion of the sub-module"),
});

// Modules Schema
const modules = z.object({
  module_name: z.string().describe("The name of the module"),
  introduction: z.string().describe("write  a simple introduction to the module to engage the student"),
  sub_modules: z.array(sub_module).optional().nullable().describe("Sub-modules if the topic is complex enough to warrant them. Leave empty/None for simple modules that don't need subdivisions."),
  content: z.string().describe("write  a parragraph about the module content"),
  conclusion: z.string().describe("summerizing what you wrote in the module content through a fun way to make the student engaged and wanting to learn more about the topic,"),
});

// Documentation Schema
const documentation = z.object({
  title: z.string().describe("The title of the documentation"),
  topic: z.string().describe("The topic of the documentation"),
  introduction: z.string().describe("A brief introduction to the documentation topic"),
  modules: z.array(modules).describe("List of modules"),
  conclusion: z.string().describe("The conclusion of the whole document"),
  motivation: z.string().describe("A motivational note to encourage further learning on the topic"),
});

// Evaluation Response Structure Schema
const evaluation_response_structure = z.object({
  correct_answers: z.number().describe("Number of correct answers"),
  wrong_answers: z.number().describe("Number of wrong answers"),
  eval: z.array(evaluation_response),
  what_should_i_do_to_improve: z.string().describe("General advice for improvement"),
  level: z.string().describe("The level of the student based on his answers"),
  diffuculty_recommendation: z.string().describe("Recommended difficulty level for the next quiz"),
  document: z.array(documentation),
});

// Document Schema
const document = z.object({
  document: z.array(documentation),
});

// Sub Module Avatar Schema
const sub_module_avatar = z.object({
  sub_module_title: z.string().describe("The title of the sub-module"),
  content: z.string().describe("The content of the sub-module"),
  conclusion: z.string().describe("The conclusion of the sub-module"),
});

// Modules Avatar Schema
const modules_avatar = z.object({
  module_name: z.string().describe("The name of the module"),
  module_goal: z.string().describe("what the student should understand from this section"),
  introduction: z.string().describe("write  a simple introduction to the module to engage the student"),
  sub_modules: z.array(sub_module_avatar).optional().nullable().describe("Sub-modules if the topic is complex enough to warrant them. Leave empty/None for simple modules that don't need subdivisions."),
  content: z.string().describe("write the module content"),
  conclusion: z.string().describe("summerizing what you wrote in the module content through a fun way to make the student engaged and wanting to learn more about the topic,"),
});

// Documentation Avatar Schema
const documentation_avatar = z.object({
  title: z.string().describe("The title of the documentation"),
  topic: z.string().describe("The topic of the documentation"),
  document_goal: z.string().describe("what the student should understand or achieve after this document"),
  introduction: z.string().describe("A brief introduction to the documentation topic"),
  modules: z.array(modules_avatar).describe("List of modules"),
  conclusion: z.string().describe("The conclusion of the whole document"),
  motivation: z.string().describe("A motivational note to encourage further learning on the topic"),
  suggested_reading: z.array(z.string()).optional().nullable().describe("A list of suggested reading materials or resources for further exploration of the topic."),
  suggested: z.string().describe("If you want to add any new field suggest it here"),
});

const state_assistant = z.object({
  client_id: z.string().optional().nullable(),
  tools: z.string().optional().nullable(),
  long_term_memory: LongTermMemory.optional().nullable(),
  system_prompt: z.string().optional().nullable(),
  input: z.string(),
});

const state_python = z.object({
  page: z.string(),
  pdf_content: z.record(z.string(), z.any()),
  script_avatar: z.string(),
  avatar_output: z.string(),
  target_age: z.string(),
  id: z.string(),
  main_language: z.string(),
  second_language: z.string(),
  main_topic: z.string(),
  error: z.string(),
});

// Document Avatar Schema
const document_avatar = z.object({
  document: z.array(documentation_avatar),
});

// Content Safety Response Schema
const content_safety_response = z.object({
  safe: z.boolean().describe("Whether the content is safe for children under 18"),
  violations: z.array(z.string()).describe("List of triggered content violation categories"),
  severity: z.string().describe("Severity level of the violations (low, medium, high, critical)"),
  reason: z.string().describe("Brief explanation of why the content was classified as safe or unsafe"),
});

// Mental Health Risk Response Schema
const mental_health_risk_response = z.object({
  risk_detected: z.boolean().describe("Whether distress/crisis risk signals are detected"),
  risk_types: z.array(z.string()).describe("Detected risk categories such as hyper_stress, severe_anxiety, hopelessness, self_harm, suicidal_ideation"),
  severity: z.string().describe("Risk severity level (low, medium, high, critical)"),
  requires_immediate_escalation: z.boolean().describe("Whether immediate escalation to a human safety response is needed"),
  reason: z.string().describe("Brief explanation for the risk classification"),
});

// ============ End of Zod Schemas ============

// FastAPI app -> Express app (language/framework adaptation)
const app = express();
app.use(express.json({ limit: "20mb" }));

const upload = multer({ dest: "uploads/" });

// Upload folder configuration
const UPLOAD_FOLDER = "uploads";
if (!fs.existsSync(UPLOAD_FOLDER)) {
  fs.mkdirSync(UPLOAD_FOLDER, { recursive: true });
}

// azure api configuration
const key = process.env.AZURE_AI_INFERENCE_CREDENTIAL?.trim();
const api = process.env.AZURE_AI_INFERENCE_ENDPOINT?.trim();

// python: AzureAIOpenAIApiChatModel (from langchain_azure_ai.chat_models)
// TS adaptation: keep same variable names and invocation flow with Azure-compatible chat model object.
const llm: any = new LangChainAzureOpenAI({
  azureOpenAIApiKey: key,
  openAIApiVersion: "2025-01-01-preview",
  azureOpenAIApiInstanceName: "issam-mmottwx5-eastus2",
  azureOpenAIApiDeploymentName: "Mistral-Large-3",
  temperature: 0.7,
});

const hack_api = process.env.hack_api?.trim();
const hack_key = process.env.hack_key?.trim();

const scriptllm: any = new LangChainAzureOpenAI({
  azureOpenAIApiKey: hack_key,
  openAIApiVersion: "2025-01-01-preview",
  azureOpenAIApiInstanceName: "issam-mmottwx5-eastus2",
  azureOpenAIApiDeploymentName: "Llama-4-Maverick-17B-128E-Instruct-FP8",
  temperature: 0.7,
});

const emotionsllm: any = new LangChainAzureOpenAI({
  azureOpenAIApiKey: key,
  openAIApiVersion: "2025-01-01-preview",
  azureOpenAIApiInstanceName: "issam-mmottwx5-eastus2",
  azureOpenAIApiDeploymentName: "Llama-4-Maverick-17B-128E-Instruct-FP8",
  temperature: 0.7,
});

const tts_client = new AzureOpenAI({
  apiKey: key,
  apiVersion: "2025-01-01-preview",
  endpoint: "https://issam-mmottwx5-eastus2.cognitiveservices.azure.com",
});

const embedding: any = new LangChainAzureOpenAI({
  azureOpenAIApiKey: hack_key,
  openAIApiVersion: "2025-01-01-preview",
  azureOpenAIApiInstanceName: "issam-mmottwx5-eastus2",
  azureOpenAIApiDeploymentName: "text-embedding-3-large",
  temperature: 0,
});

// keep same startup TTS side-effect as python
(async () => {
  try {
    const tts: any = await (tts_client as any).audio.speech.create({
      model: "gpt-4o-mini-tts",
      voice: "alloy",
      input: "good moning hope u doing well",
      instructions: "Speak in a friendly, calm tone.",
      response_format: "mp3",
    });

    if (!fs.existsSync("./recording")) {
      fs.mkdirSync("./recording", { recursive: true });
    }

    const out = "./recording/avatar_response.mp3";
    if (tts?.stream_to_file) {
      await tts.stream_to_file(out);
    } else if (tts?.arrayBuffer) {
      const buf = Buffer.from(await tts.arrayBuffer());
      fs.writeFileSync(out, buf);
    }
  } catch {
    // keep runtime tolerant as python-style script boot side-effect
  }
})();

// Google API configuration
const google_api_key = process.env.GOOGLE_API_KEY;
if (!google_api_key) {
  throw new Error("GOOGLE_API_KEY not found in environment variables");
}

// Initialize Google LLM (replacing Azure OpenAI)
const gllm: any = new ChatGoogle({
  model: "gemini-2.5-flash",
  temperature: 0.7,
});

// Structured LLM for more complex outputs
const struct_llm: any = new ChatGoogle({
  model: "gemini-2.5-flash",
  temperature: 0.7,
});
/*
const doc_embeddings: any = new ChatGoogle({
  model: "gemini-embedding-001",
  taskType: "RETRIEVAL_DOCUMENT",
});

const query_embeddings: any = new ChatGoogle({
  model: "gemini-embedding-001",
  taskType: "RETRIEVAL_QUERY",
});


*/
// system prompts
const sys_prompt_avatar = `Transform the following educational content into a comprehensive, well-structured learning document.

Structure Requirements:
- Organize content into logical modules with clear hierarchies
- Create sub-modules for complex topics to ensure digestibility
- Each module must include:
  * Clear learning objective (what the student will master)
  * Engaging introduction that sparks curiosity
  * Detailed, educational content with concrete examples
  * Summary conclusion reinforcing key concepts

Tone & Style:
- Use accessible, age-appropriate language
- Include relatable analogies and real-world examples
- Maintain an encouraging, supportive tone throughout
- End with an inspiring motivational note that celebrates learning and encourages exploration

Goal: Create documentation that transforms complex topics into exciting learning adventures.
`;

const sys_prompt = `
You are an expert AI Learning Companion designed to make education engaging, effective, and personalized.

Core Responsibilities:
1. Provide clear, student-appropriate explanations tailored to the learner's level
2. Generate educational quizzes using the generate_quiz tool when requested
3. Evaluate student understanding with constructive, encouraging feedback
4. Adapt communication style based on user preferences and learning pace
5. Use concrete examples, analogies, and visual descriptions to clarify abstract concepts

MEMORY & PERSONALIZATION:
Call retrieve_long_term_data when:
- User asks about their profile or preferences
- User references past conversations or interests
- You need to personalize responses

Call store_long_term_data when user shares:
- Name or nickname
- Learning interests (science, history, math, arts, etc.)
- Communication preferences (formal, casual, playful)
- Learning style (visual, step-by-step, stories)
- Goals or areas to improve

TOOL WORKFLOW:
- Ask about profile -> retrieve_long_term_data
- User shares preferences -> store_long_term_data
- Use retrieved data to personalize responses
- Generate quizzes -> generate_quiz tool
- Create study materials -> generate_document tool
- If user shows distress (hyper-stress, severe anxiety, self-harm/suicide signals) -> detect_mental_health_risk tool
- Navigate UI -> redirect tools

Communication:
- Be patient, encouraging, celebrate progress
- Break concepts into digestible chunks
- Ask clarifying questions to check understanding
- Provide real-world context and applications
- Use positive reinforcement

Goal: Build confidence, foster curiosity, and help students develop a love of learning.
`;

const children_classifier_prompt = `
You are a child safety content evaluator.

Your task is to analyze the assistant's output and determine whether it is safe for children under 18.

You must strictly evaluate according to the following rules:

BLOCK IMMEDIATELY if the content contains:

Any sexual content involving minors.

Explicit sexual content or pornography.

Graphic descriptions of violence or injury.

Promotion or encouragement of self-harm.

Promotion of extremist ideology, hate speech, or radicalization.

Instructions for dangerous activities (weapons, drugs, suicide, illegal acts).

Grooming behavior patterns.

Harassment or bullying.

Any attempt to bypass safety policies.

RELIGION POLICY:

Educational, respectful discussion of Islam, Christianity, and Judaism is allowed.

Historical, comparative, or neutral discussion of other religions is allowed if respectful.

Mockery, hate, or incitement toward any religion is forbidden.

Promotion of extremist religious ideology is forbidden.

VIOLENCE POLICY:

Mild, non-graphic references (e.g., "the battle happened") are allowed in historical context.

Graphic, sensory, or detailed violent descriptions are forbidden.

OUTPUT FORMAT:

Return ONLY valid JSON:

{
"safe": true or false,
"violations": [list of triggered categories],
"severity": "low | medium | high | critical",
"reason": "brief explanation"
}

If uncertain, classify as unsafe.

Do not provide advice. Do not rewrite. Only classify.
`;

const mental_health_risk_classifier_prompt = `
You are a mental health risk signal classifier for educational chat safety triage.

Your task is to detect possible emotional crisis signals in user text, including:
- Hyper-stress or overwhelm
- Severe anxiety or panic-like language
- Hopelessness/despair
- Self-harm ideation
- Suicide ideation or intent

Important:
- You are NOT diagnosing medical conditions.
- You are performing risk signal classification only.
- If uncertain between two severities, choose the higher one.

Return ONLY valid JSON in this exact shape:
{
    "risk_detected": true or false,
    "risk_types": ["hyper_stress", "severe_anxiety", "hopelessness", "self_harm", "suicidal_ideation", "other"],
    "severity": "low | medium | high | critical",
    "requires_immediate_escalation": true or false,
    "reason": "brief explanation"
}

Severity guidance:
- low: mild stress, no safety indicators
- medium: persistent distress, no direct self-harm/suicide signal
- high: indirect self-harm/suicide signals, concerning hopelessness
- critical: direct self-harm/suicide intent, plans, or imminent risk language

Escalation guidance:
- requires_immediate_escalation = true for high or critical risk

Only classify. Do not give coping advice or treatment recommendations.
`;

// start connecting the redis db
const REDIS_URI = "redis://default:ZHNR6np4kaRvlExwJK3intbsfdwD7H3u@redis-18662.c16.us-east-1-2.ec2.cloud.redislabs.com:18662";
const redis_client = createClient({ url: REDIS_URI });
redis_client.connect().catch(() => undefined);

// schemas/types (python BaseModel -> zod schema + inferred TS type)
const ContactInfo = z.object({
  name: z.string().describe("The name of the person"),
  email: z.string().describe("The email address of the person"),
  phone: z.string().describe("The phone number of the person"),
});

const LongTermMemorySchema = z.object({
  user_name: z.string().optional(),
  tone: z.string().optional(),
  preferences: z.array(z.string()),
  favorite_topics: z.array(z.string()).optional(),
  learning_style: z.string().optional(),
  goals: z.array(z.string()).optional(),
});
type LongTermMemory = z.infer<typeof LongTermMemorySchema>;

const QuizGenerationSchema = z.object({
  question: z.string(),
  options: z.array(z.string()),
});
type quiz_generation = z.infer<typeof QuizGenerationSchema>;

const QuizStructureSchema = z.object({
  quiz: z.array(QuizGenerationSchema),
});
type quizstructure = z.infer<typeof QuizStructureSchema>;

const EvaluationResponseSchema = z.object({
  wrong: z.string(),
  corrections: z.array(z.string()),
  explaination: z.string(),
  advice: z.string(),
});
type evaluation_response = z.infer<typeof EvaluationResponseSchema>;

const SubModuleSchema = z.object({
  sub_module_title: z.string(),
  content: z.string(),
  conclusion: z.string(),
});
type sub_module = z.infer<typeof SubModuleSchema>;

const ModulesSchema = z.object({
  module_name: z.string(),
  introduction: z.string(),
  sub_modules: z.array(SubModuleSchema).optional(),
  content: z.string(),
  conclusion: z.string(),
});
type modules = z.infer<typeof ModulesSchema>;

const DocumentationSchema = z.object({
  title: z.string(),
  topic: z.string(),
  introduction: z.string(),
  modules: z.array(ModulesSchema),
  conclusion: z.string(),
  motivation: z.string(),
});
type documentation = z.infer<typeof DocumentationSchema>;

const EvaluationResponseStructureSchema = z.object({
  correct_answers: z.number(),
  wrong_answers: z.number(),
  eval: z.array(EvaluationResponseSchema),
  what_should_i_do_to_improve: z.string(),
  level: z.string(),
  diffuculty_recommendation: z.string(),
  document: z.array(DocumentationSchema),
});
type evaluation_response_structure = z.infer<typeof EvaluationResponseStructureSchema>;

const DocumentSchema = z.object({
  document: z.array(DocumentationSchema),
});
type document = z.infer<typeof DocumentSchema>;

const SubModuleAvatarSchema = z.object({
  sub_module_title: z.string(),
  content: z.string(),
  conclusion: z.string(),
});
type sub_module_avatar = z.infer<typeof SubModuleAvatarSchema>;

const ModulesAvatarSchema = z.object({
  module_name: z.string(),
  module_goal: z.string(),
  introduction: z.string(),
  sub_modules: z.array(SubModuleAvatarSchema).optional(),
  content: z.string(),
  conclusion: z.string(),
});
type modules_avatar = z.infer<typeof ModulesAvatarSchema>;

const DocumentationAvatarSchema = z.object({
  title: z.string(),
  topic: z.string(),
  document_goal: z.string(),
  introduction: z.string(),
  modules: z.array(ModulesAvatarSchema),
  conclusion: z.string(),
  motivation: z.string(),
  suggested_reading: z.array(z.string()).optional(),
  suggested: z.string(),
});
type documentation_avatar = z.infer<typeof DocumentationAvatarSchema>;

const DocumentAvatarSchema = z.object({
  document: z.array(DocumentationAvatarSchema),
});
type document_avatar = z.infer<typeof DocumentAvatarSchema>;

const ContentSafetyResponseSchema = z.object({
  safe: z.boolean(),
  violations: z.array(z.string()),
  severity: z.string(),
  reason: z.string(),
});
type content_safety_response = z.infer<typeof ContentSafetyResponseSchema>;

const MentalHealthRiskResponseSchema = z.object({
  risk_detected: z.boolean(),
  risk_types: z.array(z.string()),
  severity: z.string(),
  requires_immediate_escalation: z.boolean(),
  reason: z.string(),
});
type mental_health_risk_response = z.infer<typeof MentalHealthRiskResponseSchema>;

type state = z.infer<typeof state_python>;
type state_document = {
  pdf_path: string;
  main_language: string;
  second_language: string;
  pdf_content: string[];
  pages: string[];
  target_age: string;
  document_output: string;
};
type state_RAG = {
  pdf_path: string;
  pages: string[];
  embeddings: number[][];
  final_output: any[];
  error?: string;
  pls_work: any;
};



const state= new StateSchema({ 
  page: z.string(),
  pdf_content: z.any(),
  script_avatar: z.string(),
  avatar_output: z.string(),
  target_age: z.string(),
  id: z.string(),
  main_language: z.string(),
  second_language: z.string(),
  main_topic: z.string(),
  error: z.string()
})

const state_document= new StateSchema({
  pdf_path: z.string(),
  main_language: z.string(),
  second_language: z.string(),
  pdf_content: z.array(z.string()),
  pages: z.array(z.string()),
  target_age: z.string(),
  document_output: z.string(),// we will change it later
})

const state_RAG = new StateSchema({
  pdf_path: z.string(),
  pages: z.array(z.string()),
  embeddings: z.array(z.array(z.number())),
  final_output: z.array(z.any()),
  error: z.string().optional(),
  pls_work: z.any(),
});
const context = z.object({
  user_name: z.string(),
});
// start of the llm with structured output (minimum adaptation with zod)
const long_term_memory_structured_llm: any = llm.withStructuredOutput?.(
  z.object({
    user_name: z.string().optional(),
    tone: z.string().optional(),
    preferences: z.array(z.string()).default([]),
    favorite_topics: z.array(z.string()).optional(),
    learning_style: z.string().optional(),
    goals: z.array(z.string()).optional(),
  })
) ?? llm;

const avatar_summirization_structured_llm: any = llm;
const document_structured_llm: any = llm;
const quiz_structured_llm: any = llm;
const evaluation_structured_llm: any = llm;
const children_safety_structured_llm: any = llm;
const mental_health_risk_structured_llm: any = llm;

// vector db placeholders with same variable names (language adaptation only)
const schema = {
  index: {
    name: "documents_index",
    prefix: "docs_6",
  },
  fields: [
    { name: "id", type: "tag" },
    { name: "paragraph", type: "text" },
    {
      name: "para_embedding",
      type: "vector",
      attrs: {
        dims: 3072,
        distance_metric: "COSINE",
        algorithm: "FLAT",
        datatype: "FLOAT32",
      },
    },
  ],
};

const index: any = {
  create: async () => undefined,
  load: async (rows: any[]) => rows.map((_: any, i: number) => `key-${i}`),
  query: async () => [],
  schema: { fields: schema.fields },
};

// start tool calling
const store_long_term_data = tool(
  async (input: { messages: any[]; user_id: string }): Promise<string> => {
    const messages = input.messages ?? [];
    const user_id = input.user_id;
    const namespace = `${user_id}:memories`;

    const latest = messages.length >= 2 ? messages[messages.length - 2]?.content : "";
    const clean_messages = [
      new SystemMessage("Extract user preferences from this message."),
      new HumanMessage(String(latest ?? "")),
    ];

    const response = await long_term_memory_structured_llm.invoke(clean_messages);
    await redis_client.hSet(namespace, "a-memory", JSON.stringify(response));
    return "data stored inside the db";
  },
  {
    name: "store_long_term_data",
    description: "SAVE USER PROFILE AND LEARNING PREFERENCES TO MEMORY.",
    schema: z.object({
      messages: z.array(z.any()),
      user_id: z.string(),
    }),
  }
);

const retrieve_long_term_data = tool(
  async (input: { user_id: string }): Promise<string> => {
    const namespace = `${input.user_id}:memories`;
    const personal_data = await redis_client.hGet(namespace, "a-memory");
    if (!personal_data) return "No profile data found.";
    return personal_data;
  },
  {
    name: "retrieve_long_term_data",
    description: "Retrieve user profile and learning preferences from memory.",
    schema: z.object({ user_id: z.string() }),
  }
);

function trim_messages(messages: any[]): any[] {
  return messages.slice(-20);
}

function count_tokens_approximately(_: any): number {
  return 0;
}

async function chat_assistant(state_assistant_in: { messages: any[] }): Promise<{ messages: any[] }> {
  const messages = trim_messages(state_assistant_in.messages);
  const response = await run_agent.invoke({ messages });
  const ai_message = response.messages[response.messages.length - 1];
  return { messages: [new AIMessage(String(ai_message?.content ?? ""))] };
}

async function load_pdf(state_in: state_document): Promise<state_document> {
  // minimum adaptation: local file read fallback for TS without PyPDFLoader
  const pages: string[] = [];
  try {
    const raw = fs.readFileSync(state_in.pdf_path, "utf8");
    pages.push(raw);
  } catch {
    pages.push("");
  }
  state_in.pages = pages;
  return state_in;
}

async function sumurize_avatar(state_in: state): Promise<state> {
  const data = [
    new SystemMessage(sys_prompt_avatar),
    new HumanMessage(state_in.page),
  ];
  const avatar = await avatar_summirization_structured_llm.invoke(data);
  state_in.pdf_content = avatar;
  return state_in;
}

async function script(state_in: state): Promise<state> {
  const main_language = state_in.main_language;
  const second_language = state_in.second_language;

  let multilingual_instructions = "";
  if (main_language !== "English" || second_language !== "English") {
    multilingual_instructions = `

MULTILINGUAL HYBRID LANGUAGE INSTRUCTIONS:
============================================
Main Language (Student's Primary Language): ${main_language}
Secondary Language (Documentation/Scientific Language): ${second_language}

LANGUAGE MIXING STRATEGY:
1. Use ${main_language} for ALL explanations, examples, step-by-step instructions, and illustrations
2. Use ${second_language} ONLY for:
   - Scientific/technical terminology that has no direct ${main_language} equivalent
   - Mathematical formulas and symbols
   - Chemical names and formulas
   - Standard technical definitions
   - Proper nouns and scientific names

3. When introducing scientific terms:
   - First explain the concept fully in ${main_language}
   - Provide concrete examples in ${main_language}
   - Then give the ${second_language} scientific term in parentheses
   - Example: "This process (called [scientific_term]) works by..."

4. HYBRID OUTPUT PATTERN:
   - Narrative & Examples: 100% ${main_language}
   - Complex Explanations: 60% ${main_language} + 40% ${second_language} terminology
   - Definitions: State in ${main_language}, reference ${second_language} term
   - Analogies: Always in ${main_language}, use familiar student experiences
   - Step-by-step methods: Completely in ${main_language}

5. PRIORITY:
   - Student comprehension in ${main_language} is paramount
   - Use ${main_language} to explain complicated methods and examples
   - Use ${second_language} only for scientific accuracy and technical precision
   - Maintain flow and readability in ${main_language}`;
  }

  const system_prompt_local = `Transform this educational page into a clear, engaging learning script.

STRICT BOUNDARY (VERY IMPORTANT):
- Explain ONLY the page content provided.
- Do NOT add any opening greeting, intro line, title card, welcome text, or scene-setting.
- Do NOT add any ending, conclusion, wrap-up, farewell, thank-you line, or motivational closing.
- Do NOT prepend or append anything outside the page explanation itself.
- Start directly from explaining the first idea in the page, and stop when the page explanation is complete.

REQUIREMENTS:

1. CLARITY:
   - Use simple, accessible language for students
   - Break complex ideas into understandable chunks
   - Define technical terms as you use them
   - Use concrete examples to illustrate
   - Show connections between concepts

2. ENGAGEMENT:
   - Make content relatable with real-world applications
   - Use analogies to familiar concepts
   - Include interesting details and facts
   - Maintain encouraging, supportive tone
   - Ask thought-provoking questions where appropriate

3. FLOW:
   - Create logical progression through the content
   - Use smooth transitions between ideas
   - Highlight key concepts and important points
   - Build understanding step-by-step

4. TONE:
   - Professional yet friendly and approachable
   - Enthusiastic and engaging
   - Patient and supportive
   - Foster genuine curiosity about the topic

Transform this page into an engaging educational explanation by enhancing and clarifying only what is already in the page content.${multilingual_instructions}`;

  const user_message = `PAGE CONTENT:
${state_in.page}

TASK:
Create a smooth, flowing educational explanation of this page's content.`;

  const messages = [
    new SystemMessage(system_prompt_local),
    new HumanMessage(user_message),
  ];

  const script_response = await scriptllm.invoke(messages);
  state_in.script_avatar = String(script_response?.content ?? "");
  return state_in;
}

async function avatar(state_in: state): Promise<state> {
  const system_prompt_local = `You are a voice director for Gemini TTS.
Your input will be a fully written script that must be performed with lifelike delivery.
Target audience: ${state_in.target_age}

Your job:
- Add performance markup to the script to control delivery: tone, pace, volume, whisper, and precise pauses.
- Keep the original wording unless a tiny tweak is needed for speakability (e.g., splitting long sentences).
- Never add new ideas or facts. Do not summarize or remove content; only direct the performance.

Markup syntax to use (plain text, no XML):
- [tone: ...]      -> overall emotion/attitude (e.g., warm, excited, serious, reassuring, curious).
- [pace: ...]      -> speaking speed for the next phrase or until changed (slow | medium | fast).
- [whisper]      -> whisper the next phrase; cancel with [voice: normal].
- [volume: ...]    -> soft | normal | loud (use sparingly for emphasis).
- [pause: Ns]    -> silent pause of N seconds (e.g., [pause: 2.0s]).
- [emphasize]...[/emphasize] -> stress key words.
- [voice: name]  -> optional, if using multiple speakers/voices; cancel with [voice: default].
- Section cues: ### Heading -> keep headings if present, but add appropriate tone/pace.

Rules:
1) Insert concise, targeted directions before the lines they affect.
2) Vary tone and pace to maintain attention; slow down for definitions, formulas, or key steps; speed up lightly on summaries.
3) Add [pause: 0.6-1.0s] between logical steps; use longer pauses (1.5-3.0s) before/after crucial takeaways.
4) Use [whisper] only for short aside moments or suspense; immediately return to [voice: normal].
5) Keep directions readable and minimal.
6) Output ONLY the marked performance script. No explanations or extra text.`;

  const user_message = `Now, take the following script and produce the marked performance version, ready for Gemini TTS:
---SCRIPT START---
${state_in.script_avatar}
---SCRIPT END---`;

  const messages = [
    new SystemMessage(system_prompt_local),
    new HumanMessage(user_message),
  ];

  const avatar_response = await emotionsllm.invoke(messages);
  state_in.avatar_output = String(avatar_response?.content ?? "");
  return state_in;
}

async function tts_avatar(state_in: state): Promise<state> {
  try {
    const tts_response: any = await (tts_client as any).audio.speech.create({
      model: "gpt-4o-mini-tts",
      voice: "alloy",
      input: state_in.avatar_output,
      instructions: "Speak in a friendly, calm tone.",
      response_format: "mp3",
    });

    if (!fs.existsSync("./recording")) {
      fs.mkdirSync("./recording", { recursive: true });
    }

    const file_name = `./recording/c${state_in.id}.wav`;
    if (tts_response?.stream_to_file) {
      await tts_response.stream_to_file(file_name);
    } else if (tts_response?.arrayBuffer) {
      const buf = Buffer.from(await tts_response.arrayBuffer());
      fs.writeFileSync(file_name, buf);
    }
  } catch {
    // keep tolerant, same flow continuation
  }

  return state_in;
}

async function second_page_summary(state_in: state_document): Promise<Partial<state_document>> {
  const page = state_in.pages[1] ?? "";
  const response = await one_page_chain.invoke({ page, target_age: state_in.target_age, main_language: state_in.main_language, second_language: state_in.second_language, id: "2" });
  return { document_output: response.avatar_output };
}

async function third_page_summary(state_in: state_document): Promise<Partial<state_document>> {
  const page = state_in.pages[2] ?? "";
  const response = await one_page_chain.invoke({ page, target_age: state_in.target_age, main_language: state_in.main_language, second_language: state_in.second_language, id: "3" });
  return { document_output: response.avatar_output };
}

async function fourth_page_summary(state_in: state_document): Promise<Partial<state_document>> {
  const page = state_in.pages[3] ?? "";
  const response = await one_page_chain.invoke({ page, target_age: state_in.target_age, main_language: state_in.main_language, second_language: state_in.second_language, id: "4" });
  return { document_output: response.avatar_output };
}

async function fifth_page_summary(state_in: state_document): Promise<Partial<state_document>> {
  const page = state_in.pages[4] ?? "";
  const response = await one_page_chain.invoke({ page, target_age: state_in.target_age, main_language: state_in.main_language, second_language: state_in.second_language, id: "5" });
  return { document_output: response.avatar_output };
}

async function sixth_page_summary(state_in: state_document): Promise<Partial<state_document>> {
  const page = state_in.pages[5] ?? "";
  const response = await one_page_chain.invoke({ page, target_age: state_in.target_age, main_language: state_in.main_language, second_language: state_in.second_language, id: "6" });
  return { document_output: response.avatar_output };
}

async function seventh_page_summary(state_in: state_document): Promise<Partial<state_document>> {
  const page = state_in.pages[6] ?? "";
  const response = await one_page_chain.invoke({ page, target_age: state_in.target_age, main_language: state_in.main_language, second_language: state_in.second_language, id: "7" });
  return { document_output: response.avatar_output };
}

async function eighth_page_summary(state_in: state_document): Promise<Partial<state_document>> {
  const page = state_in.pages[7] ?? "";
  const response = await one_page_chain.invoke({ page, target_age: state_in.target_age, main_language: state_in.main_language, second_language: state_in.second_language, id: "8" });
  return { document_output: response.avatar_output };
}

async function ninth_page_summary(state_in: state_document): Promise<Partial<state_document>> {
  const page = state_in.pages[8] ?? "";
  const response = await one_page_chain.invoke({ page, target_age: state_in.target_age, main_language: state_in.main_language, second_language: state_in.second_language, id: "9" });
  return { document_output: response.avatar_output };
}

async function tenth_page_summary(state_in: state_document): Promise<Partial<state_document>> {
  const page = state_in.pages[9] ?? "";
  const response = await one_page_chain.invoke({ page, target_age: state_in.target_age, main_language: state_in.main_language, second_language: state_in.second_language, id: "10" });
  return { document_output: response.avatar_output };
}

async function eleventh_page_summary(state_in: state_document): Promise<Partial<state_document>> {
  const page = state_in.pages[10] ?? "";
  const response = await one_page_chain.invoke({ page, target_age: state_in.target_age, main_language: state_in.main_language, second_language: state_in.second_language, id: "11" });
  return { document_output: response.avatar_output };
}

async function twelvth_page_summary(state_in: state_document): Promise<Partial<state_document>> {
  const page = state_in.pages[11] ?? "";
  const response = await one_page_chain.invoke({ page, target_age: state_in.target_age, main_language: state_in.main_language, second_language: state_in.second_language, id: "12" });
  return { document_output: response.avatar_output };
}

async function thirten_page_summary(state_in: state_document): Promise<Partial<state_document>> {
  const page = state_in.pages[12] ?? "";
  const response = await one_page_chain.invoke({ page, target_age: state_in.target_age, main_language: state_in.main_language, second_language: state_in.second_language, id: "13" });
  return { document_output: response.avatar_output };
}

async function fourteen_page_summary(state_in: state_document): Promise<Partial<state_document>> {
  const page = state_in.pages[13] ?? "";
  const response = await one_page_chain.invoke({ page, target_age: state_in.target_age, main_language: state_in.main_language, second_language: state_in.second_language, id: "14" });
  return { document_output: response.avatar_output };
}

async function fifteen_page_summary(state_in: state_document): Promise<Partial<state_document>> {
  const page = state_in.pages[14] ?? "";
  const response = await one_page_chain.invoke({ page, target_age: state_in.target_age, main_language: state_in.main_language, second_language: state_in.second_language, id: "15" });
  return { document_output: response.avatar_output };
}

async function sixteenth_page_summary(state_in: state_document): Promise<Partial<state_document>> {
  const page = state_in.pages[15] ?? "";
  const response = await one_page_chain.invoke({ page, target_age: state_in.target_age, main_language: state_in.main_language, second_language: state_in.second_language, id: "16" });
  return { document_output: response.avatar_output };
}

async function seventeenth_page_summary(state_in: state_document): Promise<Partial<state_document>> {
  const page = state_in.pages[16] ?? "";
  const response = await one_page_chain.invoke({ page, target_age: state_in.target_age, main_language: state_in.main_language, second_language: state_in.second_language, id: "17" });
  return { document_output: response.avatar_output };
}

async function start_conversation(state_in: state_document): Promise<Partial<state_document>> {
  const first_page = state_in.pages[0] ?? "";
  const main_language = state_in.main_language || "English";
  const second_language = state_in.second_language || "English";

  let multilingual_note = "";
  if (main_language !== "English" || second_language !== "English") {
    multilingual_note = `\nUse ${main_language} for all welcoming statements and examples. Use ${second_language} only for any technical terms if needed.`;
  }

  const system_prompt_local = `You are an enthusiastic educational host introducing a fascinating learning journey!

Your job is to create an ENGAGING, EXCITING introduction that:
1. Greets the student warmly and excitingly (e.g., "Hello! Today we're going to explore the marvelous universe of programming!")
2. Mentions this is the beginning of an exciting learning adventure
3. Provides a brief, attractive hook that makes the topic sound amazing
4. Explains what document/file we're about to explore
5. Builds excitement and curiosity for what's to come

TONE: Enthusiastic, friendly, inspiring, like a passionate teacher welcoming students to class
${multilingual_note}

Generate the opening introduction speech (2-3 sentences max, keep it punchy and exciting!):`;

  const user_message = `This is the FIRST PAGE of the content:\n${first_page}`;
  const messages = [new SystemMessage(system_prompt_local), new HumanMessage(user_message)];
  const response = await llm.invoke(messages);
  const intro_text = String(response?.content ?? "");

  try {
    const tts_response: any = await (tts_client as any).audio.speech.create({
      model: "gpt-4o-mini-tts",
      voice: "alloy",
      input: intro_text,
      instructions: "Speak in a friendly, calm tone.",
      response_format: "mp3",
    });

    if (!fs.existsSync("./recording")) fs.mkdirSync("./recording", { recursive: true });
    const file_name = "./recording/c1.wav";
    if (tts_response?.stream_to_file) {
      await tts_response.stream_to_file(file_name);
    } else if (tts_response?.arrayBuffer) {
      const buf = Buffer.from(await tts_response.arrayBuffer());
      fs.writeFileSync(file_name, buf);
    }
  } catch {
    // keep flow
  }

  return { document_output: intro_text };
}

async function end_conversation(state_in: state_document): Promise<Partial<state_document>> {
  const last_page = state_in.pages[state_in.pages.length - 1] ?? "";
  const main_language = state_in.main_language;
  const second_language = state_in.second_language;

  let multilingual_note = "";
  if (main_language !== "English" || second_language !== "English") {
    multilingual_note = `\nUse ${main_language} for all closing statements and gratitude. Use ${second_language} only for any technical terms if needed.`;
  }

  const prompt = `You are an enthusiastic educational host concluding an amazing learning journey!

 YOUR TASK:
You are ending an educational session. This is the LAST PAGE of the content.
Your job is to create a MEANINGFUL, INSPIRING conclusion that:

1. Summarizes what we learned from the final content
2. Explains the last page/section we explored
3. Celebrates the learning journey and accomplishments
4. Ends with an inspiring thank you message (e.g., "Thanks for exploring this amazing journey with us!" or "Thanks for hearing us, keep learning!")
5. Encourages continued exploration and curiosity

 TONE: Warm, celebratory, grateful, like ending a great class with motivation
 ${multilingual_note}

 LAST PAGE CONTENT:
${last_page}

Generate the closing speech (2-3 sentences max, memorable and motivational):
`;

  const response = await llm.invoke(prompt);
  const end_text = String(response?.content ?? response ?? "");

  try {
    const tts_response: any = await (tts_client as any).audio.speech.create({
      model: "gpt-4o-mini-tts",
      voice: "alloy",
      input: end_text,
      instructions: "Speak in a friendly, calm tone.",
      response_format: "mp3",
    });
    if (!fs.existsSync("./recording")) fs.mkdirSync("./recording", { recursive: true });
    const file_name = "./recording/cend.wav";
    if (tts_response?.stream_to_file) {
      await tts_response.stream_to_file(file_name);
    } else if (tts_response?.arrayBuffer) {
      const buf = Buffer.from(await tts_response.arrayBuffer());
      fs.writeFileSync(file_name, buf);
    }
  } catch {
    // keep flow
  }

  return { document_output: end_text };
}

function conditional_edge(state_in: state_document): string[] {
  const page = state_in.pages.length;
  const pages: string[] = [];
  for (let i = 0; i < page - 1; i += 1) {
    pages.push(String(i + 2));
  }
  pages.push("end_conversation");
  return pages;
}

function get_embedding(state_in: state_RAG): Partial<state_RAG> {
/*
    try {
    const data = doc_embeddings.embedDocuments(state_in.pages);
    return { embeddings: data as any };
  } catch (e: any) {
    return { error: String(e) };
  }
    */
   return { embeddings: state_in.pages.map(() => Array(3072).fill(0)) };
}

function prepare_for_vector_store(state_in: state_RAG): Partial<state_RAG> {
  const vector_store_data: any[] = [];
  for (let i = 0; i < state_in.pages.length; i += 1) {
    const text = state_in.pages[i];
    const emb = state_in.embeddings[i];
    try {
      vector_store_data.push({
        id: `kid_${i}`,
        paragraph: text,
        para_embedding: emb,
      });
    } catch (e: any) {
      return { error: `Error processing embedding for paragraph ${i}: ${String(e)}` };
    }
  }
  return { final_output: vector_store_data };
}

async function store_redis_db(state_in: state_RAG): Promise<Partial<state_RAG>> {
  try {
    const keys = await index.load(state_in.final_output);
    return { pls_work: keys };
  } catch (e: any) {
    return { error: String(e) };
  }
}

async function children_safety_check(input: string): Promise<content_safety_response> {
  const data = [new SystemMessage(children_classifier_prompt), new HumanMessage(input)];
  const response = await children_safety_structured_llm.invoke(data);
  return response as content_safety_response;
}

const detect_mental_health_risk = tool(
  async (input: { user_text: string }): Promise<any> => {
    const data = [
      new SystemMessage(mental_health_risk_classifier_prompt),
      new HumanMessage(input.user_text),
    ];
    const response = await mental_health_risk_structured_llm.invoke(data);
    return response;
  },
  {
    name: "detect_mental_health_risk",
    description: "Detect possible distress crisis signals.",
    schema: z.object({ user_text: z.string() }),
  }
);

const generate_quiz = tool(
  async (input: { quiz_topic: string; quiz_subject: string; num_questions: number }): Promise<any> => {
    const prompt = `Generate 30 well-crafted multiple-choice quiz questions.

Subject: ${input.quiz_subject}
Topic: ${input.quiz_topic}

REQUIREMENTS:
- Create questions that test understanding of ${input.quiz_topic} within ${input.quiz_subject}
- 4 answer options (A, B, C, D) per question
- One clearly correct answer, plausible but distinct distractors
- Clear, age-appropriate language
- Cover diverse aspects of the topic
- Encourage learning, not just memorization
- Mix difficulty levels: basic recall, application, critical thinking

Create engaging, fair questions that accurately assess knowledge.
`;
    const response = await quiz_structured_llm.invoke(prompt);
    return response;
  },
  {
    name: "generate_quiz",
    description: "Generate an educational quiz on a specific topic and subject.",
    schema: z.object({
      quiz_topic: z.string(),
      quiz_subject: z.string(),
      num_questions: z.number(),
    }),
  }
);

const generate_document = tool(
  async (input: { topic: string }): Promise<any> => {
    const prompt = `Create a comprehensive educational document on '${input.topic}'.

 DOCUMENT REQUIREMENTS:

1. STRUCTURE:
   - Clear, descriptive title
   - Engaging introduction explaining relevance
   - Organized modules with clear progression
   - Sub-modules for complex topics
   - Comprehensive conclusion synthesizing learnings
   - Motivational note encouraging further exploration

2. CONTENT QUALITY:
   - Clear, age-appropriate language
   - Concrete examples and relatable analogies
   - Explain why and how, not just what
   - Storytelling to make concepts memorable
   - Real-world applications and context
   - Break complex ideas into digestible chunks

3. ENGAGEMENT:
   - Maintain enthusiastic, encouraging tone
   - Use vivid descriptions
   - Pose thought-provoking questions
   - Make abstract concepts concrete
   - Celebrate learning moments

4. EACH MODULE:
   - Clear name and preview of concepts
   - Detailed content (3-5+ paragraphs)
   - Sub-modules for complex topics
   - Summary conclusion

5. LEARNING PROGRESSION:
   - Build logically from basics to advanced
   - Emphasize understanding over memorization
   - Create aha moments
   - Foster critical thinking

Create an exciting, comprehensive learning adventure.
`;
    const response = await document_structured_llm.invoke(prompt);
    return response;
  },
  {
    name: "generate_document",
    description: "Generate comprehensive educational documentation on a specific topic.",
    schema: z.object({ topic: z.string() }),
  }
);

const evaluate_student_and_generate_documentation = tool(
  async (input: { quiz_answers: Record<string, any> }): Promise<any> => {
    const answers_json = JSON.stringify(input.quiz_answers);
    const prompt = `Evaluate this student quiz performance and provide comprehensive feedback.

QUIZ ANSWERS:
${answers_json}

EVALUATION TASKS:

1. ANALYZE PERFORMANCE:
   - Count correct and incorrect answers
   - Identify error patterns (conceptual gaps, careless mistakes, misunderstandings)
   - Determine proficiency level (beginner, intermediate, advanced, expert)
   - Recommend difficulty for next quiz (easier, same, harder)

2. PROVIDE FEEDBACK (For Each Wrong Answer):
   - Identify the incorrect answer
   - Explain why it's wrong in clear, simple terms
   - Provide correct answer with thorough explanation
   - Use analogies, examples, or stories
   - Frame mistakes as learning opportunities, not failures
   - Maintain encouraging, supportive tone

3. GENERATE COMPREHENSIVE DOCUMENTATION:
   - Focus on topics where student struggled
   - Create detailed, substantial modules (not summaries)

4. PROVIDE ACTIONABLE ADVICE:
   - Specific study recommendations based on performance
   - Practice strategies for weak areas

5. TONE & STYLE:
   - Encouraging and supportive throughout

Transform quiz mistakes into comprehensive learning opportunities that build mastery and confidence.
`;
    const response = await evaluation_structured_llm.invoke(prompt);
    return response;
  },
  {
    name: "evaluate_student_and_generate_documentation",
    description: "Evaluate quiz performance and generate targeted learning materials.",
    schema: z.object({ quiz_answers: z.record(z.string(), z.any()) }),
  }
);

async function avatar_script_generator(content: string): Promise<any> {
  const prompt = `Create an engaging, natural-sounding avatar script for this learning content.

CONTENT TO SCRIPT:
${content}

SCRIPT REQUIREMENTS:

1. CONVERSATIONAL TONE:
   - Write as natural speech, not formal text
   - Use contractions and casual phrasing ("Let's explore...", "You know what's cool?")
   - Include rhetorical questions to engage viewer
   - Add natural pauses with ellipses (...) for pacing
   - Vary sentence length for dynamic delivery

2. ENGAGEMENT TECHNIQUES:
   - Grab attention with a hook
   - Use "you" and "we" to create connection
   - Pose thought-provoking questions
   - Show enthusiasm and excitement
   - Build curiosity before revealing concepts
   - Use storytelling and real-world examples

3. PACING & STRUCTURE:
   - Break complex explanations into digestible chunks
   - Signal transitions: "Now, let's move on to...", "But here's where it gets interesting..."
   - Add emphasis with natural speech patterns: "And this is REALLY important..."
   - Include brief summaries: "So remember...", "To sum up..."
   - Create natural pauses for reflection

4. EDUCATIONAL VALUE:
   - Maintain accuracy while simplifying
   - Use analogies for abstract ideas
   - Explain why things matter
   - Connect to viewer's life and experiences
   - Encourage critical thinking with open questions

5. PERSONALITY & EMOTION:
   - Express enthusiasm and curiosity
   - Show wonder at interesting facts: "Isn't that incredible?"
   - Be warm and supportive: "Don't worry if this seems tricky at first..."
   - Celebrate learning moments: "You're getting it!"
   - Maintain energy without being overwhelming

6. TECHNICAL CONSIDERATIONS:
   - Use pronunciation-friendly words
   - Spell out acronyms phonetically if needed
   - Break up long numbers: "one hundred fifty" not "150"
   - Indicate emphasis where important
   - Keep sentences speakable (not too long or complex)

7. STRUCTURE:
   - Opening hook (grab attention)
   - Clear introduction with relevance
   - Main content in clear segments
   - Regular engagement touchpoints
   - Strong conclusion
   - Motivational closing

Create a script that sounds like an enthusiastic friend explaining fascinating concepts-engaging, clear, and memorable.
`;
  const response = await llm.invoke(prompt);
  return response;
}

const redirect_to_quiz = tool(
  async (input: { subject: string; topic: string }): Promise<any> => {
    const value = { action: "redirect", target: `/quiz?subject=${input.subject}&topic=${input.topic}` };
    return value;
  },
  {
    name: "redirect_to_quiz",
    description: "Redirect user to quiz generation interface.",
    schema: z.object({ subject: z.string(), topic: z.string() }),
  }
);

const redirect_to_document = tool(
  async (input: { topic: string }): Promise<any> => {
    const value = { action: "redirect", target: `/document?topic=${input.topic}` };
    return value;
  },
  {
    name: "redirect_to_document",
    description: "Redirect user to document generation interface.",
    schema: z.object({ topic: z.string() }),
  }
);

const redirect_to_tabs = tool(
  async (input: { tab_name: string }): Promise<any> => {
    const normalized_tab = input.tab_name.toLowerCase().replace(/ /g, "-");
    const value = { action: "redirect", target: `/tabs?name=${normalized_tab}` };
    return value;
  },
  {
    name: "redirect_to_tabs",
    description: "Redirects the user to a specific tab, page, or course within the application interface.",
    schema: z.object({ tab_name: z.string() }),
  }
);

const tool_long_term = [
  store_long_term_data,
  retrieve_long_term_data,
  detect_mental_health_risk,
  redirect_to_quiz,
  redirect_to_document,
  redirect_to_tabs,
];

const tools = [redirect_to_quiz];

const run_agent: any = createAgent({
  model: llm,
  tools: tool_long_term,
  systemPrompt: sys_prompt,
});

const Tollnode = new ToolNode(tool_long_term as any);

const one_page_chain: any = new StateGraph<any>(state as any)
  .addNode("emotionless script", script)
  .addNode("avatar script", avatar)
  .addNode("tts avatar", tts_avatar)
  .addEdge(START, "emotionless script")
  .addEdge("emotionless script", "avatar script")
  .addEdge("avatar script", "tts avatar")
  .addEdge("tts avatar", END)
  .compile();

const avatar_chain: any = new StateGraph<any>(state_document as any)
  .addNode("load_pdf", load_pdf)
  .addNode("start_conversation", start_conversation)
  .addNode("2", second_page_summary)
  .addNode("3", third_page_summary)
  .addNode("4", fourth_page_summary)
  .addNode("5", fifth_page_summary)
  .addNode("6", sixth_page_summary)
  .addNode("7", seventh_page_summary)
  .addNode("8", eighth_page_summary)
  .addNode("9", ninth_page_summary)
  .addNode("10", tenth_page_summary)
  .addNode("11", eleventh_page_summary)
  .addNode("12", twelvth_page_summary)
  .addNode("13", thirten_page_summary)
  .addNode("14", fourteen_page_summary)
  .addNode("15", fifteen_page_summary)
  .addNode("16", sixteenth_page_summary)
  .addNode("17", seventeenth_page_summary)
  .addNode("end_conversation", end_conversation)
  .addNode("avatar script", avatar)
  .addNode("tts avatar", tts_avatar)
  .addEdge(START, "load_pdf")
  .addEdge("load_pdf", "start_conversation")
  .addConditionalEdges("start_conversation", conditional_edge as any)
  .addEdge("end_conversation", END)
  .compile();

const RAG_chain: any = new StateGraph<any>(state_RAG as any)
  .addNode("load_pdf", load_pdf)
  .addNode("get_embedding", get_embedding)
  .addNode("prepare_for_vector_store", prepare_for_vector_store)
  .addNode("store_redis_db", store_redis_db)
  .addEdge(START, "load_pdf")
  .addEdge("load_pdf", "get_embedding")
  .addEdge("get_embedding", "prepare_for_vector_store")
  .addEdge("prepare_for_vector_store", "store_redis_db")
  .addEdge("store_redis_db", END)
  .compile();

app.get("/", async (req: Request, res: Response) => {
  const user_id = String(req.query.user_id ?? "");
  const namespace = `${user_id}:memories`;
  const personal_data_raw = await redis_client.hGet(namespace, "a-memory");
  const personal_data = personal_data_raw ? JSON.parse(personal_data_raw) : null;
  res.json(personal_data);
});

app.post("/quiz/", async (req: Request, res: Response) => {
  const subject = String(req.body.subject ?? "");
  const topic = String(req.body.topic ?? "");
  const result = await generate_quiz.invoke({ quiz_topic: topic, quiz_subject: subject, num_questions: 30 });
  res.json(result);
});

app.post("/document/", async (req: Request, res: Response) => {
  const topic = String(req.body.topic ?? "");
  const document_out = await generate_document.invoke({ topic });
  res.json(document_out);
});

app.post("/chatbot/", upload.single("file"), async (req: Request, res: Response) => {
  const user_input = String(req.body.user_input ?? "");
  const thr_id = String(req.body.thr_id ?? "");
  const usr_id = String(req.body.usr_id ?? "");

  if (req.file?.filename) {
    const file_path = path.join(UPLOAD_FOLDER, req.file.filename);
    const test_input = {
      pdf_path: file_path,
      target_age: "high school students",
      main_language: "English",
      second_language: "English",
    };
    const avatar_result = await avatar_chain.invoke(test_input);
    res.json(avatar_result.document_output);
    return;
  }

  const test = await run_agent.invoke({ messages: [new HumanMessage(user_input)] });
  const messages = test.messages ?? [];
  const tool_message = messages.length >= 2 ? messages[messages.length - 2] : null;
  const tool_content = tool_message?.content;

  if (tool_content) {
    try {
      const action = typeof tool_content === "string" ? JSON.parse(tool_content) : tool_content;
      if (action?.target) {
        res.redirect(action.target);
        return;
      }
    } catch {
      // tool content not JSON
    }
  }

  res.json({ aimessage: test });
});

app.post("/avatar/", upload.single("file"), async (req: Request, res: Response) => {
  const target_age = String(req.body.target_age ?? "");
  const main_language = String(req.body.main_language ?? "English");
  const second_language = String(req.body.second_language ?? "English");

  if (req.file?.filename) {
    const file_path = path.join(UPLOAD_FOLDER, req.file.filename);
    const test_input = { pdf_path: file_path, target_age, main_language, second_language };
    const avatar_result = await avatar_chain.invoke(test_input);
    res.json(avatar_result.document_output);
    return;
  }

  res.status(400).json({ error: "file is required" });
});

app.post("/evaluate/", async (req: Request, res: Response) => {
  const data = req.body.data ?? req.body;
  const feedback = await evaluate_student_and_generate_documentation.invoke({ quiz_answers: data });
  const script_out = await avatar_script_generator(String((feedback as any)?.document ?? ""));
  res.json({ feedback, script: script_out });
});

app.post("/code/correct/", async (req: Request, res: Response) => {
  const code = String(req.body.code ?? "");
  const role = String(req.body.role ?? "");
  const prompt = `
    You are an expert on ${role}.
    Your task is to review the following code snippet, identify any syntax errors, logical mistakes, or inefficiencies, and provide a corrected version of the code along with explanations for the changes made.
    Here is the code to review: ${code}

    `;
  const correction = await llm.invoke(prompt);
  res.json({ correction });
});

app.post("/code/review/", async (req: Request, res: Response) => {
  const code = String(req.body.code ?? "");
  const role = String(req.body.role ?? "");
  const prompt = `
    You are an expert on ${role}.
    Your task is to review the following code snippet for best practices, readability, and maintainability .
    Provide feedback on how to improve the code structure, naming conventions, and overall design, along with a revised version of the code that incorporates these improvements.
    Here is the code to review: ${code}

    `;
  const review = await llm.invoke(prompt);
  res.json({ review });
});

app.post("/process_pdf/", upload.single("pdf"), async (req: Request, res: Response) => {
  if (!req.file?.filename) {
    res.status(400).json({ error: "pdf file is required" });
    return;
  }
  const pdf_path = path.join(UPLOAD_FOLDER, req.file.filename);
  await RAG_chain.invoke({ pdf_path, pages: [], embeddings: [], final_output: [], pls_work: {} });
  res.json({ status: "File processed successfully and indexed in vector database" });
});

app.post("/retrieve/", async (req: Request, res: Response) => {
  const query = String(req.body.query ?? "");
  const input = String(req.body.input ?? "");

  let results: any[] = [];
  let paragraphs: string[] = [];

  try {
    //const query_embedding = await query_embeddings.embedQuery(query);
    //results = await index.query({ vector: query_embedding, num_results: 3 });
    paragraphs = results.map((item: any) => String(item.paragraph ?? ""));

    const retrieved_context = paragraphs.join("\n");
    const system_prompt_local = `You are an expert assistant that explains complex information in a clear and concise manner.
            Your role is to analyze user queries and explain them based on retrieved relevant data.
            Provide accurate, helpful explanations that connect the user's input with the retrieved information.`;

    const user_prompt = `Based on the following retrieved data, please explain the user's query in detail.

            User Query: ${input}

            Retrieved Data:
            ${retrieved_context}

            Please provide a comprehensive explanation that addresses the user's query using the retrieved information.`;

    const messages = [new SystemMessage(system_prompt_local), new HumanMessage(user_prompt)];
    const explanation = await llm.invoke(messages);

    res.json({ results, paragraphs, explanation });
  } catch (e: any) {
    res.status(500).json({ error: `Query error: ${String(e)}` });
  }
});

const PORT = Number(process.env.PORT ?? 3000);
app.listen(PORT, () => {
  // keep startup behavior
});
