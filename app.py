import json
import os
from dotenv import load_dotenv
from shutil import copyfileobj
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain.tools import tool, ToolRuntime  
from fastapi.encoders import jsonable_encoder
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field
from typing import Optional, TypedDict, Annotated, Union, Literal
import operator
from langchain.agents import create_agent
from langgraph.checkpoint.redis import RedisSaver
from langgraph.store.redis import RedisStore
from fastapi import FastAPI, UploadFile
from langgraph.prebuilt import ToolNode
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.messages.utils import trim_messages, count_tokens_approximately
from langgraph.graph import MessagesState
from dataclasses import dataclass
from langgraph.types import Command
from redisvl.index import SearchIndex
from redis import Redis
from redisvl.query import VectorQuery


# Load environment variables
load_dotenv()

# FastAPI app
app = FastAPI()

# Upload folder configuration
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Google API configuration
google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    raise ValueError("GOOGLE_API_KEY not found in environment variables")

# Initialize Google LLM (replacing Azure OpenAI)
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7,
)

# Structured LLM for more complex outputs
struct_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7,
)
doc_embeddings= GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    task_type="RETRIEVAL_DOCUMENT"
)
query_embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    task_type="RETRIEVAL_QUERY"
)
tts= ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-preview-tts",
	response_modalities=["AUDIO"],
    model_kwargs={
        "speech_config": {
            "voice_config": {"prebuilt_voice_config": {"voice_name": "Kore"}}
        },
    },
)


#system prompts
sys_prompt_avatar="""Transform the following educational content into a comprehensive, well-structured learning document.

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
"""  
sys_prompt = """
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
- Ask about profile → retrieve_long_term_data
- User shares preferences → store_long_term_data
- Use retrieved data to personalize responses
- Generate quizzes → generate_quiz tool
- Create study materials → generate_document tool
- Navigate UI → redirect tools

Communication:
- Be patient, encouraging, celebrate progress
- Break concepts into digestible chunks
- Ask clarifying questions to check understanding
- Provide real-world context and applications
- Use positive reinforcement

Goal: Build confidence, foster curiosity, and help students develop a love of learning.
"""
children_classifier_prompt = """
You are a child safety content evaluator.

Your task is to analyze the assistant’s output and determine whether it is safe for children under 18.

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

Mild, non-graphic references (e.g., “the battle happened”) are allowed in historical context.

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
"""

#end system prompts
# start connecting the redis db 
REDIS_URI = "redis://default:ZHNR6np4kaRvlExwJK3intbsfdwD7H3u@redis-18662.c16.us-east-1-2.ec2.cloud.redislabs.com:18662"
with RedisSaver.from_conn_string(REDIS_URI) as checkpointer:
    checkpointer.setup()
with RedisStore.from_conn_string(REDIS_URI) as store:
    store.setup()
class LongTermMemory(BaseModel):
    user_name: Optional[str] = Field(default=None, description="The name of the user")
    tone: Optional[str] = Field(default="neutral", description="The preferred tone of communication (casual, formal, playful, etc.)")
    preferences: list[str] = Field(default_factory=list, description="List of user preferences including learning interests, favorite subjects, favorite things")
    favorite_topics: Optional[list[str]] = Field(default=None, description="Topics the student loves learning about (science, history, math, art, etc.)")
    learning_style: Optional[str] = Field(default=None, description="How the student prefers to learn (visual examples, step-by-step, stories, hands-on, etc.)")
    goals: Optional[list[str]] = Field(default=None, description="Learning goals or areas they want to improve")

# end connecting the redis db
#start writing the data structure
#start writing teh schema for both the Quiz structure
class quiz_generation(BaseModel):
    question: str = Field(..., description="The quiz question")
    options: list[str] = Field(..., description="List of answer options")
    #correct_answer: str = Field(..., description="The correct answer")
class quizstructure(BaseModel):
    quiz: list[quiz_generation]
#end writing teh schema for both the Quiz structure
#start writing the schema for both the documentation and the evaluation response structure
class evaluation_response(BaseModel):
    wrong: str = Field(..., description="The wrong answer given by the student")
    corrections: list[str] = Field(..., description="explanations of the wrong answer")
    explaination: str = Field(..., description="you are explaining it to a kid so do it in a fun way like a game")
    advice: str = Field(..., description="Advice for improvement")
class sub_module(BaseModel):
    sub_module_title: str = Field(..., description="The title of the sub-module")
    content: str = Field(..., description="The content of the sub-module")
    conclusion: str = Field(..., description="The conclusion of the sub-module")

class modules(BaseModel):
    module_name: str = Field(..., description="The name of the module")
    introduction: str = Field(..., description="write  a simple introduction to the module to engage the student")
    sub_modules: Optional[list[sub_module]] = Field(default=None, description="Sub-modules if the topic is complex enough to warrant them. Leave empty/None for simple modules that don't need subdivisions.")
    content: str = Field(..., description="write  a parragraph about the module content")
    conclusion: str = Field(..., description="summerizing what you wrote in the module content through a fun way to make the student engaged and wanting to learn more about the topic,")

class documentation(BaseModel):#probably gonna improve it later
    title: str = Field(..., description="The title of the documentation")
    topic: str = Field(..., description="The topic of the documentation")
    introduction: str = Field(..., description="A brief introduction to the documentation topic")
    modules: list[modules] 
    conclusion: str = Field(..., description="The conclusion of the whole document")
    motivation: str = Field(..., description="A motivational note to encourage further learning on the topic")

class evaluation_response_structure(BaseModel):
    correct_answers: int = Field(..., description="Number of correct answers")
    wrong_answers: int = Field(..., description="Number of wrong answers")
    eval: list[evaluation_response]
    what_should_i_do_to_improve: str = Field(..., description="General advice for improvement")
    level: str = Field(..., description="The level of the student based on his answers")
    diffuculty_recommendation: str = Field(..., description="Recommended difficulty level for the next quiz")
    document: list[documentation]
class document(BaseModel):
    document: list[documentation]
# maybe we will improve the schemma later for the documentation and the eval  to  be under diiferent structure
#end writing the schema for both the documentation and the evaluation response structure
#start documentation avatar structure
class sub_module_avatar(BaseModel):
    sub_module_title: str = Field(..., description="The title of the sub-module")
    content: str = Field(..., description="The content of the sub-module")
    conclusion: str = Field(..., description="The conclusion of the sub-module")

class modules_avatar(BaseModel):
    module_name: str = Field(..., description="The name of the module")
    module_goal: str = Field(..., description="what the student should understand from this section")
    introduction: str = Field(..., description="write  a simple introduction to the module to engage the student")
    sub_modules: Optional[list[sub_module_avatar]] = Field(default=None, description="Sub-modules if the topic is complex enough to warrant them. Leave empty/None for simple modules that don't need subdivisions.")
    content: str = Field(..., description="write the module content")
    conclusion: str = Field(..., description="summerizing what you wrote in the module content through a fun way to make the student engaged and wanting to learn more about the topic,")

class documentation_avatar(BaseModel):#probably gonna improve it later
    title: str = Field(..., description="The title of the documentation")
    topic: str = Field(..., description="The topic of the documentation")
    document_goal: str = Field(...,description="what the student should understand or achieve after this document")
    introduction: str = Field(..., description="A brief introduction to the documentation topic")
    modules: list[modules_avatar] 
    conclusion: str = Field(..., description="The conclusion of the whole document")
    motivation: str = Field(..., description="A motivational note to encourage further learning on the topic")
    suggested_reading: Optional[list[str]] = Field(default=None, description="A list of suggested reading materials or resources for further exploration of the topic.")
    suggested: str = Field(..., description="If you want to add any new field suggest it here")

class document_avatar(BaseModel):
    document: list[ documentation_avatar]
# end documentation avatar structure
#data structure to protect the kids
class content_safety_response(BaseModel):
    safe: bool = Field(..., description="Whether the content is safe for children under 18")
    violations: list[str] = Field(..., description="List of triggered content violation categories")
    severity: str = Field(..., description="Severity level of the violations (low, medium, high, critical)")
    reason: str = Field(..., description="Brief explanation of why the content was classified as safe or unsafe")
#end of the data structure to protect the kids
#end writing the data structure
#states
class state_assistant(MessagesState):
    client_id: Optional[str] = None
    tools: Optional[str] = None
    long_term_memory: Optional[LongTermMemory] = None
    system_prompt: Optional[str] = None
    input: str
class state(TypedDict):
    page: str
    pdf_content: dict#maybe we will change it to list later
    script_avatar: str
    avatar_output: str
    target_age: str
    id: str

class state_document(MessagesState):
    pdf_path: str
    pdf_content: list[str]
    pages: dict
    target_age: str
    document_output: Annotated[str, operator.add]
class state_RAG(TypedDict):
    pdf_path: str
    paragraphs: list
    embeddings: list
    final_output: list[dict]
    error: Optional[Annotated[str, operator.add]]  
    pls_work: dict 

@dataclass
class Context:
    user_id: str 
#end of states  
#start of the llm with structured output
long_term_memory_structured_llm = llm.with_structured_output(LongTermMemory)# we need to either improve it later or do a lot of work on it 
avatar_summirization_structured_llm= llm.with_structured_output(document_avatar)
document_structured_llm = llm.with_structured_output(document)
quiz_structured_llm = llm.with_structured_output(quizstructure)
evaluation_structured_llm = llm.with_structured_output(evaluation_response_structure)
children_safety_structured_llm = llm.with_structured_output(content_safety_response)
#end of the llm with structured output
#creation a vector database with Redis
schema = {
    "index": {
        "name": "documents_index",
        "prefix": "docs_6",
    },
    "fields": [
        {"name": "id", "type": "tag"},
        {"name": "paragraph", "type": "text"},
        {
            "name": "para_embedding",
            "type": "vector",
            "attrs": {
                "dims": 3072,
                "distance_metric": "COSINE",
                "algorithm": "FLAT",
                "datatype": "FLOAT32",
            }
        }
    ]
}
client = Redis.from_url(REDIS_URI)
index = SearchIndex.from_dict(schema, redis_client=client, validate_on_load=True)
index.create(overwrite=True)
#end creating a vector database with Redis

#start tool calling
#tools for storing long and short term memory  for teh bot
@tool
def store_long_term_data(runtime: ToolRuntime[Context]) -> Union[dict, state_assistant]:
    """SAVE USER PROFILE AND LEARNING PREFERENCES TO MEMORY.
    
     CALL THIS TOOL IMMEDIATELY WHEN the user shares:
    - Name, username, or preferred nickname → Extract this for user_name
    - Learning interests and favorite subjects (science, history, math, arts, etc.) → Add to preferences
    - Communication preferences (formal, casual, playful tone) → Set as tone
    - Learning style preferences (visual examples, step-by-step, stories) → Add to learning_style
    - Goals or areas they want to improve → Add to goals
    - Any persistent profile information for personalization
    
    Examples of when to call:
    "My name is Alex" --> store user_name="Alex"
    "I love science and history" --> store preferences=["science", "history"]
    "I prefer a casual tone" --> store tone="casual"
    "I learn best with examples" --> store learning_style="visual examples"
    
    PURPOSE: Enable personalized, contextual learning experiences across sessions.
    This is CRITICAL for the learning companion experience!
    """
    messages = runtime.state.get("messages", [])
    print("messages in the store tool:", messages)
    user_id=runtime.context.user_id
    namespace=(user_id,"memories")
    #user_message=runtime.state_assistant["messages"]
    #response = long_term_memory_structured_llm.invoke(messages)
    #we will defintly change this code later
    messages_for_memory = []
    #print("ai messages:", messages[-2].content)
    # print("content", messages.content)
    # for message in messages:
    #    if isinstance(message, HumanMessage):
    #        messages_for_memory.append(message)
    #    elif isinstance(message, AIMessage) and not getattr(message, "tool_calls", None):
    #        messages_for_memory.append(message)
    print("messages_for_memory:", messages[-2].content)
    # Create a clean message history without tool_calls
    clean_messages = [
        SystemMessage(content="Extract user preferences from this message."),
        HumanMessage(content=messages[-2].content)
    ]
    response = long_term_memory_structured_llm.invoke(clean_messages)#rember to  check the db now if anythign is missing 
    #don t forget to change it 
    runtime.store.put( 
        namespace,
        "a-memory",
        jsonable_encoder(response),
    )
    #return ToolMessage(
    #    content="data stored inside the db",
     #   tool_call_id=runtime.tool_call_id  # Critical: pairs response with the call
    #)
    return ToolMessage(
        content="data stored inside the db",
        tool_call_id=runtime.tool_call_id  # Critical: pairs response with the call
    )
@tool
def retrieve_long_term_data(runtime: ToolRuntime[Context]) -> Union[dict, state_assistant]:
    """Retrieve user profile and learning preferences from memory.
    
    Call when user asks about their profile, preferences, or references past information.
    Returns stored user data including name, interests, tone preference, learning style, and goals.
    """
    
    user_id=runtime.context.user_id
    namespace=(user_id,"memories")
    personal_data=runtime.store.get(namespace,"a-memory")
    # Convert dict to JSON string to ensure ToolMessage content is string type
    
    return ToolMessage(
        content=personal_data if personal_data else "No profile data found.",
        tool_call_id=runtime.tool_call_id  # Critical: pairs response with the call
    )
#@tool    
from langchain.agents.middleware import before_model, ModelRequest
#@before_model
#def trim(request: ModelRequest):#we will see later if we need to  change the params
def chat_assistant(state_assistant: MessagesState) -> Union[dict, state_assistant]:
    messages = trim_messages(  
        state_assistant["messages"],
        strategy="last",
        token_counter=count_tokens_approximately,
        max_tokens=2000,
        start_on="human",
        end_on=("human", "tool"),
    )
    #messages= state_assistant["messages"]
    response = run_agent.invoke({"messages": messages})
    ai_message = response["messages"][-1]
    tool_message = response["messages"][-2]
    return {
        "messages": [AIMessage(content=str(ai_message.content))],
    }
#end of tools for short term memory for the bot 
#not techniclly tools but we are gonna use them for the documentation
async def load_pdf(state: state_document) -> state_document:
    loader = PyPDFLoader(state["pdf_path"])
    pagey = loader.load_and_split()
    pages = []
    for i, page in enumerate(pagey):
        print(f"--- Page {i + 1} ---")
        pages.append(str(page.page_content))
        print("end of page\n")
    state["pages"] = pages
    return state

def sumurize_avatar(state: state) -> state:
    #avatar = run_agent.invoke({"messages": [HumanMessage(content=state["pages"])]})
    data=[]
    sys=SystemMessage(content=sys_prompt_avatar)
    human=HumanMessage(content=state["pages"])
    data.append(sys)
    data.append(human)
    avatar = avatar_summirization_structured_llm.invoke(data)
    #ai_message = avatar["messages"][-1]
    state["pdf_content"] = avatar
    return state

async def script(state: state) -> state:
    system_prompt = """Transform this educational page into a clear, engaging learning script.

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

Transform this page into an engaging educational explanation without artificially adding introduction/conclusion—just enhance and clarify what's already here."""
	
    user_message = f"""PAGE CONTENT:
{state["page"]}

TASK:
Create a smooth, flowing educational explanation of this page's content."""
	
    messages = [
		SystemMessage(content=system_prompt),
		HumanMessage(content=user_message)
	]
	
    script_response = await llm.ainvoke(messages)
    state["script_avatar"] = str(script_response.content)
    return state
async def avatar(state: state) -> state:
    system_prompt = f"""You are a voice director for Gemini TTS.
Your input will be a fully written script that must be performed with lifelike delivery.
Target audience: {state['target_age']}

Your job:
- Add performance markup to the script to control delivery: tone, pace, volume, whisper, and precise pauses.
- Keep the original wording unless a tiny tweak is needed for speakability (e.g., splitting long sentences).
- Never add new ideas or facts. Do not summarize or remove content; only direct the performance.

Markup syntax to use (plain text, no XML):
- [tone: …]      → overall emotion/attitude (e.g., warm, excited, serious, reassuring, curious).
- [pace: …]      → speaking speed for the next phrase or until changed (slow | medium | fast).
- [whisper]      → whisper the next phrase; cancel with [voice: normal].
- [volume: …]    → soft | normal | loud (use sparingly for emphasis).
- [pause: Ns]    → silent pause of N seconds (e.g., [pause: 2.0s]).
- [emphasize]…[/emphasize] → stress key words.
- [voice: name]  → optional, if using multiple speakers/voices; cancel with [voice: default].
- Section cues: ### Heading — keep headings if present, but add appropriate tone/pace.

Rules:
1) Insert concise, targeted directions before the lines they affect. 
2) Vary tone and pace to maintain attention; slow down for definitions, formulas, or key steps; speed up lightly on summaries.
3) Add [pause: 0.6–1.0s] between logical steps; use longer pauses (1.5–3.0s) before/after crucial takeaways.
4) Use [whisper] only for short aside moments or suspense; immediately return to [voice: normal].
5) Keep directions readable and minimal—no nesting beyond what’s defined above.
6) Output ONLY the marked performance script. No explanations or extra text."""
	
    user_message = f"""Now, take the following script and produce the marked performance version, ready for Gemini TTS:
---SCRIPT START---
{state['script_avatar']}
---SCRIPT END---"""
	
    messages = [
		SystemMessage(content=system_prompt),
		HumanMessage(content=user_message)
	]
	
    avatar_response = await llm.ainvoke(messages)
    state["avatar_output"] = str(avatar_response.content)
    return state
import asyncio
from scipy.io import wavfile 
import numpy as np
import wave
def wave_file(filename, pcm, channels=1, rate=24000, sample_width=2):
   with wave.open(filename, "wb") as wf:
      wf.setnchannels(channels)
      wf.setsampwidth(sample_width)
      wf.setframerate(rate)
      wf.writeframes(pcm)
async def tts_avatar(state: state) -> state:
	"""Convert marked performance script to audio using Gemini TTS."""
	result = await tts.ainvoke(state["avatar_output"])
	
	# Save audio to WAV file
	file_name = f'./recording/{state["id"]}.wav'
	wave_file(file_name, result.additional_kwargs["audio"])
	return state

async def second_page_summary(state: state_document) -> state_document:
	page = state["pages"][1]
	response = await one_page_chain.ainvoke({"page": page, "target_age": state["target_age"], "id": "2"})
	return {"document_output": response["avatar_output"]}


async def third_page_summary(state: state_document) -> state_document:
	page = state["pages"][2]
	response = await one_page_chain.ainvoke({"page": page, "target_age": state["target_age"], "id": "3"})
	return {"document_output": response["avatar_output"]}


async def fourth_page_summary(state: state_document) -> state_document:
	page = state["pages"][3]
	response = await one_page_chain.ainvoke({"page": page, "target_age": state["target_age"], "id": "4"})
	return {"document_output": response["avatar_output"]}
async def fifth_page_summary(state: state_document) -> state_document:
	page = state["pages"][4]
	response = await one_page_chain.ainvoke({"page": page, "target_age": state["target_age"], "id": "5"})
	return {"document_output": response["avatar_output"]}

async def sixth_page_summary(state: state_document) -> state_document:
	page = state["pages"][5]
	response = await one_page_chain.ainvoke({"page": page, "target_age": state["target_age"], "id": "6"})
	return {"document_output": response["avatar_output"]}
async def seventh_page_summary(state: state_document) -> state_document:
	page = state["pages"][6]
	response = await one_page_chain.ainvoke({"page": page, "target_age": state["target_age"], "id": "7"})
	return {"document_output": response["avatar_output"]}
async def eighth_page_summary(state: state_document) -> state_document:
	page = state["pages"][7]
	response = await one_page_chain.ainvoke({"page": page, "target_age": state["target_age"], "id": "8"})
	return {"document_output": response["avatar_output"]}
async def ninth_page_summary(state: state_document) -> state_document:
	page = state["pages"][8]
	response = await one_page_chain.ainvoke({"page": page, "target_age": state["target_age"], "id": "9"})
	return {"document_output": response["avatar_output"]}
async def tenth_page_summary(state: state_document) -> state_document:
	page = state["pages"][9]
	response = await one_page_chain.ainvoke({"page": page, "target_age": state["target_age"], "id": "10"})
	return {"document_output": response["avatar_output"]}
async def eleventh_page_summary(state: state_document) -> state_document:
	page = state["pages"][10]
	response = await one_page_chain.ainvoke({"page": page, "target_age": state["target_age"], "id": "11"})
	return {"document_output": response["avatar_output"]}
async def twelvth_page_summary(state: state_document) -> state_document:
	page = state["pages"][11]
	response = await one_page_chain.ainvoke({"page": page, "target_age": state["target_age"], "id": "12"})
	return {"document_output": response["avatar_output"]}
async def thirten_page_summary(state: state_document) -> state_document:
	page = state["pages"][12]
	response = await one_page_chain.ainvoke({"page": page, "target_age": state["target_age"], "id": "13"})
	return {"document_output": response["avatar_output"]}
async def fourteen_page_summary(state: state_document) -> state_document:
	page = state["pages"][13]
	response = await one_page_chain.ainvoke({"page": page, "target_age": state["target_age"], "id": "14"})
	return {"document_output": response["avatar_output"]}
async def fifteen_page_summary(state: state_document) -> state_document:
    page = state["pages"][14]
    response = await one_page_chain.ainvoke({"page": page, "target_age": state["target_age"], "id": "15"})
    return {"document_output": response["avatar_output"]}
async def sixteenth_page_summary(state: state_document) -> state_document:
    page = state["pages"][15]
    response = await one_page_chain.ainvoke({"page": page, "target_age": state["target_age"], "id": "16"})
    return {"document_output": response["avatar_output"]}
async def seventeenth_page_summary(state: state_document) -> state_document:
    page = state["pages"][16]
    response = await one_page_chain.ainvoke({"page": page, "target_age": state["target_age"], "id": "17"})
    return {"document_output": response["avatar_output"]}
async def start_conversation(state: state_document) -> state_document:
	"""Generate an engaging introduction for the first page of the document."""
	first_page = state["pages"][0]

	system_prompt = """You are an enthusiastic educational host introducing a fascinating learning journey!

Your job is to create an ENGAGING, EXCITING introduction that:
1. Greets the student warmly and excitingly (e.g., "Hello! Today we're going to explore the marvelous universe of programming!")
2. Mentions this is the beginning of an exciting learning adventure
3. Provides a brief, attractive hook that makes the topic sound amazing
4. Explains what document/file we're about to explore
5. Builds excitement and curiosity for what's to come

TONE: Enthusiastic, friendly, inspiring, like a passionate teacher welcoming students to class

Generate the opening introduction speech (2-3 sentences max, keep it punchy and exciting!):"""
	
	user_message = f"""This is the FIRST PAGE of the content:
{first_page}"""
	
	messages = [
		SystemMessage(content=system_prompt),
		HumanMessage(content=user_message)
	]
	
	response = llm.invoke(messages)
	intro_text = response.content if hasattr(response, "content") else str(response)
	return {"avatar_output": intro_text}

async def end_conversation(state: state_document) -> state_document:
    """Generate a meaningful closing for the last page of the document."""
    last_page = state["pages"][-1]
    prompt = f"""You are an enthusiastic educational host concluding an amazing learning journey!

 YOUR TASK:
You are ending an educational session. This is the LAST PAGE of the content.
Your job is to create a MEANINGFUL, INSPIRING conclusion that:

1. Summarizes what we learned from the final content
2. Explains the last page/section we explored
3. Celebrates the learning journey and accomplishments
4. Ends with an inspiring thank you message (e.g., "Thanks for exploring this amazing journey with us!" or "Thanks for hearing us, keep learning!")
5. Encourages continued exploration and curiosity

 TONE: Warm, celebratory, grateful, like ending a great class with motivation

 LAST PAGE CONTENT:
{last_page}

Generate the closing speech (2-3 sentences max, memorable and motivational):
"""
    
    response = llm.invoke(prompt)
    #state["avatar_output"] = response.content
    return {"avatar_output": response.content}

def conditional_edge(state: state_document) -> Literal["2", "3", "4", "5", "6", "7", "8", "9", "10", "end_conversation"]:
    """Determine if we should ship to user based on overall score."""
    page = len(state["pages"])
    pages = []
    for i in range(page-1):# check it later if it s correct or not like this 
        pages.append(str(i+2))
    pages.append("end_conversation")
    return pages 
# end of the script avatar workglow
# start of the RAG functions
def get_embedding(state: state_RAG) -> state_RAG:
    try:
        data=doc_embeddings.embed_documents( state["paragraphs"])
    except Exception as e:
        return {"error": str(e)}
    return {"embeddings": data}
import numpy as np
def prepare_for_vector_store(state: state_RAG) -> state_RAG:
    vector_store_data=[]
    i=-1
    for text in state["paragraphs"]:
        i+=1
        embedding = state["embeddings"][i]
        try:
            vector_store_data.append({
                "id": f"kid_{i}",
                "paragraph": text,
                "para_embedding": np.array(embedding, dtype=np.float32).tobytes(),
            })
        except Exception as e:
            b=f"Error processing embedding for paragraph {i}: {e}"
            return {"error": str(b)}
    return {"final_output": vector_store_data}
def store_redis_db(state: state_RAG) -> state_RAG:
    i=-1
    try:
        keys = index.load(state["final_output"])
    except Exception as e:
        return {"error": str(e)}
    yo=index.schema.fields
    return {"pls_work": keys}
#end of the RAG functions 

#end of the not techniclly tools 
#start the defr to  protect the children from any harmful content
def children_safety_check(input: str) -> content_safety_response:
    data=[]
    sys=SystemMessage(content=children_classifier_prompt)
    human=HumanMessage(content=input)
    data.append(sys)
    data.append(human)
    response = children_safety_structured_llm.invoke(data)
    return response
#end of the defr to  protect the children from any harmful content
#start  for teh tools for teh quiz and the document generation
@tool
def generate_quiz(quiz_topic: str, quiz_subject: str, num_questions: int) -> quizstructure:
    """Generate an educational quiz on a specific topic and subject.
    
    args:
    - quiz_topic: Specific topic (e.g., "photosynthesis", "fractions")
    - quiz_subject: Subject area (e.g., "biology", "mathematics")
    - num_questions: Number of questions to generate
    
    Use when user requests a quiz or wants to assess their knowledge.
    """
    prompt = f"""Generate 30 well-crafted multiple-choice quiz questions.

Subject: {quiz_subject}
Topic: {quiz_topic}

REQUIREMENTS:
- Create questions that test understanding of {quiz_topic} within {quiz_subject}
- 4 answer options (A, B, C, D) per question
- One clearly correct answer, plausible but distinct distractors
- Clear, age-appropriate language
- Cover diverse aspects of the topic
- Encourage learning, not just memorization
- Mix difficulty levels: basic recall, application, critical thinking

Create engaging, fair questions that accurately assess knowledge.
"""
    print("prompt:", prompt)
    response = jsonable_encoder(quiz_structured_llm.invoke(prompt))
    
    return response
@tool
def generate_document(topic: str) -> document:
    """Generate comprehensive educational documentation on a specific topic.
    
    Parameter:
    - topic: Subject to document (e.g., "solar system", "water cycle")
    
    Generates structured learning document with modules, sub-modules, examples, and motivational content.
    Use when user requests study materials, notes, or learning guides.
    """
    prompt = f"""Create a comprehensive educational document on '{topic}'.

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
"""
    response = jsonable_encoder(document_structured_llm.invoke(prompt))
    return response

#end of the tools for the quiz and the document generation
#tool for evaluating the quiz result probably we will need to cut it to three parts
@tool
def evaluate_student_and_generate_documentation(quiz_answers: dict) -> evaluation_response_structure: # maybe I will make it a function later not a tool still thinking or maybe make 3 or 4 tool out of it 
    """Evaluate quiz performance and generate targeted learning materials.
    
    Use when:
    - Student completes a quiz
    - Assessment of understanding is needed
    
    Evaluates:
    1. Correctness of quiz answers
    2. Knowledge gaps and misconceptions
    3. Provides specific feedback for each incorrect answer
    4. Generates comprehensive documentation on weak areas
    5. Recommends difficulty level for next quiz
    6. Assigns proficiency level
    
    Input:
    - quiz_answers: Dictionary with responses {"question_id": "student_answer", ...}
    
    Output:
    - Correct/incorrect counts
    - Detailed explanations for wrong answers
    - Actionable improvement advice
    - Proficiency level assessment
    - Difficulty recommendation
    - Comprehensive study documentation
    
    Feedback approach:
    - Constructive, never discouraging
    - Specific corrections with clear explanations
    - Celebrate correct answers
    - Frame mistakes as learning opportunities
    - Provide path forward for improvement
    
    CRITICAL: Generated documentation is comprehensive with multiple detailed paragraphs per module, thorough examples, complete sub-modules. Substantive educational content, not summaries.
    """
    answers_json = json.dumps(quiz_answers, ensure_ascii=False)
    prompt = f"""Evaluate this student quiz performance and provide comprehensive feedback.

QUIZ ANSWERS:
{answers_json}

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
   - Create detailed, substantial modules (not summaries):
     * 3-5+ full paragraphs per module
     * Rich examples, analogies, and real-world applications
     * Sub-modules for complex concepts
     * Each sub-module: intro, detailed content, conclusion
   
   - Each module has clear learning objectives
   - Engaging introductions establishing relevance
   - Comprehensive explanations from basics to advanced
   - Concrete examples
   - Practice scenarios or thought experiments
   - Summary conclusions
   
   - Documentation should be textbook-quality:
     * Educational depth for mastery
     * Deep dives, not surface-level summaries
     * 10-15 minutes to read and absorb per module
     * Storytelling and engagement throughout

4. PROVIDE ACTIONABLE ADVICE:
   - Specific study recommendations based on performance
   - Practice strategies for weak areas
   - Encourage strengths while addressing gaps
   - Recommend additional topics for study
   - Set achievable improvement goals

5. TONE & STYLE:
   - Encouraging and supportive throughout
   - Celebrate correct answers and effort
   - Frame errors as valuable learning steps
   - Build confidence with honest assessment
   - Age-appropriate, accessible language
   - Maintain enthusiasm for learning

REQUIREMENTS:
- Documentation is study material for mastery—comprehensive, detailed
- Each module: 4-6+ rich paragraphs minimum
- Sub-modules are complete educational units
- Explanations are thorough enough for mastery
- Quality over brevity

Transform quiz mistakes into comprehensive learning opportunities that build mastery and confidence.
"""
    response = evaluation_structured_llm.invoke(prompt)
    return response

def avatar_script_generator(content: str) -> dict:    
    
    prompt = f"""Create an engaging, natural-sounding avatar script for this learning content.

CONTENT TO SCRIPT:
{content}

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

Create a script that sounds like an enthusiastic friend explaining fascinating concepts—engaging, clear, and memorable.
"""
    response = llm.invoke(prompt)
    return response
#end tool evaluation
#redirection tools 

from langgraph.types import Command
@tool
def redirect_to_quiz(subject: str, topic: str, runtime: ToolRuntime[Context]) -> Command:
    """Redirect user to quiz generation interface.
    
    IF the user asked you of a quiz u must use this tool and no other tool except it.
    
     IMPORTANT: Before calling this tool, if the user mentions "my favorite subject" or personal preferences,
    you MUST call retrieve_long_term_data FIRST to get their profile information.
    Then use that information to determine the subject/topic.
    
    args:
    subject: the subject of the quiz
    topic: the topic of the quiz. if this is missing choose a random topic in the subject.

    """   
    value= {"action": "redirect", "target": f"/quiz?subject={subject}&topic={topic}"}
    #return {"messages": [ToolMessage(content=json.dumps(value), tool_call_id=runtime.tool_call_id)]}
    return  Command(
        update={
            "messages": [ToolMessage(content=json.dumps(value), tool_call_id=runtime.tool_call_id)],
            "redirect_target": value["target"]  # State for graph logic
        }
    )
@tool
def redirect_to_document(topic: str, runtime: ToolRuntime[Context]) -> Command:
    """
    IF the user asked you of a document u must use this tool.
    
     WHAT THIS DOES:
    - Navigates user to document generation interface
    - Triggers creation of structured educational content
    - Presents learning material in readable, organized format
    - Ensures proper UI/UX for document consumption
    
     CRITICAL: Use this tool EXCLUSIVELY for document requests—handles proper routing and interface loading. Do not generate long-form content directly in chat.
    
     EXAMPLE USAGE:
    User: "Can you create a study guide on the water cycle?"
    → redirect_to_document(topic="water cycle")
    
    User: "I need notes about ancient Egypt"
    → redirect_to_document(topic="ancient Egypt")
     IMPORTANT: Before calling this tool, if the user mentions "my favorite topic" or personal preferences,
    you MUST call retrieve_long_term_data FIRST to get their profile information.
    Then use that information to determine the topic.
    args:
    topic: the topic of the document.
    """     
    value= {"action": "redirect", "target": f"/document?topic={topic}"}
   # return {"messages": [ToolMessage(content=json.dumps(value), tool_call_id=runtime.tool_call_id)]}
    return  Command(
        update={
            "messages": [ToolMessage(content=json.dumps(value), tool_call_id=runtime.tool_call_id)],
            "redirect_target": value["target"]  # State for graph logic
        }
    )
#end redirection tools
@tool 
def redirect_to_tabs(tab_name: str, runtime: ToolRuntime[Context]) :
    """
    Redirects the user to a specific tab, page, or course within the application interface.
    
    USER REQUESTS HANDLED:
    - Navigation to main sections: "home page", "dashboard", "profile", "settings"
    - Navigation to courses: "python essential", "web development", "data science" etc.
    - Navigation to features: "my quizzes", "my documents", "learning progress"
    
    EXAMPLES:
    User says: "Take me to home page" → redirect_to_tabs(tab_name="home")
    User says: "I want to go to Python Essential course" → redirect_to_tabs(tab_name="python-essential")
    User says: "Show me my profile" → redirect_to_tabs(tab_name="profile")
    User says: "Let's try the web development course" → redirect_to_tabs(tab_name="web-development")
    
    args:
    tab_name: the name of the tab to redirect to (e.g., "profile", "settings", "dashboard")
    
    This tool is used for navigating users to different sections of the app, courses, or pages based on their requests.
    It handles both built-in app navigation and dynamic course routing.
    """
    # Normalize tab_name: convert to lowercase and replace spaces with hyphens
    normalized_tab = tab_name.lower().replace(" ", "-")
    
    value= {"action": "redirect", "target": f"/tabs?name={normalized_tab}"}
    #return  Command(
    #    update={
    #        "messages": [ToolMessage(content=json.dumps(value), tool_call_id=runtime.tool_call_id)],
    #        "redirect_target": value["target"]  # State for graph logic
    #    }
    #)
    return {"messages": [AIMessage(content=json.dumps(value), tool_call_id=runtime.tool_call_id)]}
# end tools 

tool_long_term=[
    store_long_term_data,
    retrieve_long_term_data, 
    redirect_to_quiz,  
    redirect_to_document, 
    redirect_to_tabs,
]
tools=[
    redirect_to_quiz,
]
run_agent = create_agent(llm,system_prompt=SystemMessage(content=sys_prompt), context_schema= Context , tools=tool_long_term, checkpointer=checkpointer, store=store)

#agent = create_agent(llm, tools=tools, system_prompt= SYSTEM_PROMPT)
Tollnode=ToolNode(tool_long_term)
from langgraph.graph import StateGraph, MessagesState, START, END
one_page_workflow=StateGraph(state)
#one_page_workflow.add_node("summrize document",sumurize_avatar)
one_page_workflow.add_node("emotionless script",script)
one_page_workflow.add_node("avatar script",avatar)
one_page_workflow.add_node("tts avatar", tts_avatar)
one_page_workflow.add_edge(START,  "emotionless script")
#one_page_workflow.add_edge("load_pdf", "emotionless script")
#one_page_workflow.add_edge("summrize document", "emotionless script")
one_page_workflow.add_edge("emotionless script","avatar script")
one_page_workflow.add_edge("avatar script", "tts avatar")
one_page_workflow.add_edge("tts avatar", END)
one_page_chain = one_page_workflow.compile()

agent_avatar = StateGraph(state_document)
agent_avatar.add_node("load_pdf",load_pdf)
agent_avatar.add_node("start_conversation",start_conversation)
agent_avatar.add_node("2",second_page_summary)
agent_avatar.add_node("3",third_page_summary)
agent_avatar.add_node("4",fourth_page_summary)
agent_avatar.add_node("5",fifth_page_summary)
agent_avatar.add_node("6",sixth_page_summary)
agent_avatar.add_node("7",seventh_page_summary)
agent_avatar.add_node("8",eighth_page_summary)
agent_avatar.add_node("9",ninth_page_summary)
agent_avatar.add_node("10",tenth_page_summary)
agent_avatar.add_node("11",eleventh_page_summary)
agent_avatar.add_node("12",twelvth_page_summary)
agent_avatar.add_node("13",thirten_page_summary)
agent_avatar.add_node("14",fourteen_page_summary)
agent_avatar.add_node("15",fifteen_page_summary)
agent_avatar.add_node("16",sixteenth_page_summary)
agent_avatar.add_node("17",seventeenth_page_summary)
agent_avatar.add_node("end_conversation",end_conversation)
agent_avatar.add_edge(START, "load_pdf")
agent_avatar.add_edge("load_pdf", "start_conversation")
agent_avatar.add_conditional_edges("start_conversation", conditional_edge)
agent_avatar.add_edge("end_conversation", END)
avatar_chain = agent_avatar.compile()
#ent sub Graph

#RAG workflow 
RAG = StateGraph(state_RAG)
RAG.add_node("load_pdf", load_pdf)
RAG.add_node("get_embedding", get_embedding)
RAG.add_node("prepare_for_vector_store", prepare_for_vector_store)
RAG.add_node("store_redis_db", store_redis_db)

# Add edges
RAG.add_edge(START, "load_pdf")
RAG.add_edge("load_pdf", "get_embedding")
RAG.add_edge("get_embedding", "prepare_for_vector_store")
RAG.add_edge("prepare_for_vector_store", "store_redis_db")
RAG.add_edge("store_redis_db", END)
#RAG.add_edge("get_embedding", END)

RAG_chain = RAG.compile()
#ent RAG workflow 
@app.get("/")
async def index(user_id: str):
    namespace=(user_id,"memories")
    personal_data=store.get(namespace,"a-memory")
    prompt = f"""Welcome back, {personal_data}!

Great to see you again! I'm ready to continue your learning journey.

I remember you're interested in: {', '.join(personal_data['preferences'])}
You prefer a {personal_data['tone']} tone in conversations.

What would you like to explore today? I can:
- Generate study materials and documentation
- Create quizzes to test your knowledge
- Explain concepts and answer questions
- Help with difficult topics
- Provide personalized learning recommendations

What's on your mind today?
"""
    return personal_data   
@app.post('/quiz/')# theres still work on it so to make it cheapear
async def quizroute(subject: str,topic: str):
    prompt = f"""Generate an educational quiz with these specifications:
    
    Subject: {subject}
    Topic: {topic}
    Number of Questions: 30
    
    Create engaging, well-crafted multiple-choice questions that accurately assess understanding.
    """
    print("prompt:", prompt)
    result = generate_quiz.invoke({"quiz_topic": topic, "quiz_subject":subject ,"num_questions": 30 })
    print("result:", result)
    return result
@app.post('/document/')# I rembered need to make this with inside the chatbto not in it s own
async def doucument_route(topic:str):
    document= generate_document.invoke({"topic": topic})
    return document
@app.post('/chatbot/')#we are gonna ue langraph later here
async def chatbot(user_input: str, thr_id:str, usr_id: str, file: UploadFile | None = None):
    if file and file.filename:
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        with open(file_path, "wb") as buffer:
            copyfileobj(file.file, buffer)
        test_input = {"pdf_path": file_path, "target_age": "high school students"}#exchange it later to be compatible with  state or else the error will persist
        avatar_result  = await avatar_chain.ainvoke(test_input)
        return avatar_result["document_output"]
    test = run_agent.invoke(
        {"messages": [HumanMessage(content=user_input)]},
        config={"configurable": {"thread_id": thr_id}},
        context=Context(user_id=usr_id),
    )
    tool_message = test["messages"][-2]
    tool = tool_message.content
    if tool:
        print("tool is not None")
        try:
            action = json.loads(tool) 
            print("still working")  
            if "target" in action:
                print("here s the error ")
                print("action:", action)
                print("action target:", action["target"])
                return RedirectResponse(url=action["target"])
        except (json.JSONDecodeError, TypeError):
            # Tool content is not JSON (e.g., memory tools return plain text)
            print("Tool content is not JSON, skipping redirect check")
            pass
    return {"aimessage": test }#, "content_safety_response": child_protector}
@app.post('/avatar/')
async def avatar_script(target_age: str, file: UploadFile | None = None):
   if file and file.filename:
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        with open(file_path, "wb") as buffer:
            copyfileobj(file.file, buffer)
        test_input = {"pdf_path": file_path, "target_age": target_age }#exchange it later to be compatible with  state or else the error will persist
        avatar_result  = await avatar_chain.ainvoke(test_input)
        return avatar_result["document_output"]
@app.post('/evaluate/')
async def evaluate_route(data: dict):
    feedback = evaluate_student_and_generate_documentation.invoke({"quiz_answers": data})
    script= avatar_script_generator.invoke({"content": str(feedback.document)})
    print("script:", script)
    return {"feedback": feedback, "script": script}
@app.post('/code/correct/')
async def code_correction_route(code: str, role: str):
    prompt = f"""
    You are an expert on {role}. 
    Your task is to review the following code snippet, identify any syntax errors, logical mistakes, or inefficiencies, and provide a corrected version of the code along with explanations for the changes made.
    Here is the code to review: {code}

    """# maybe we will review the role part later to make it more specific
    correction = llm.invoke(prompt)
    return {"correction": correction}
@app.post('/code/review/')
async def code_review_route(code: str, role: str):
    prompt = f"""
    You are an expert on {role}. 
    Your task is to review the following code snippet for best practices, readability, and maintainability . 
    Provide feedback on how to improve the code structure, naming conventions, and overall design, along with a revised version of the code that incorporates these improvements.
    Here is the code to review: {code}

    """# maybe we will review the role part later to make it more specific
    review = llm.invoke(prompt)
    return {"review": review}
@app.post("/process_pdf/")
async def process_pdf(pdf: UploadFile):
    # Save the uploaded PDF file
    pdf_path = os.path.join(UPLOAD_FOLDER, pdf.filename)
    with open(pdf_path, "wb") as f:
        copyfileobj(pdf.file, f)
    final_state = RAG_chain.invoke({"pdf_path": pdf_path})
    #Return the final output as JSON response
    return {"status": "File processed successfully and indexed in vector database"}
@app.post("/retrieve/")
async def retrieve(query: str, input: str):
    query_embedding = query_embeddings.embed_query(query)
    query = VectorQuery(
    vector=query_embedding,
    vector_field_name="para_embedding",
    return_fields=["paragraph", "id", "vector_distance"],
    num_results=3
    )
    try:
        results = index.query(query)
    except Exception as e:
        return {"error": f"Query error: {e}"}
    return {"results": results}


