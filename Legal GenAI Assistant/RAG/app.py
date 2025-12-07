# Public IP for Texas EC2 Instance: 18.188.120.165

import streamlit as st
import os
import torch
from haystack import Pipeline
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.builders import ChatPromptBuilder
from haystack.dataclasses import ChatMessage
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.converters import PyPDFToDocument
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from haystack import component
from typing import List, Dict, Any
import re
import unicodedata

# Page Config
st.set_page_config(page_title="Texas Family Law Bot", layout="wide")
st.title("⚖️ Texas Family Law RAG Chatbot")

# --- Configuration ---
PDF_DIR = "./texas-family-code" # Ensure this folder exists on the server
HF_TOKEN = os.environ.get("HF_TOKEN") # Set via env variable

# --- Helper Functions (Your Cleaners) ---
def normalize(s:str)->str:
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\u00A0", " ").replace("\u200b", "")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\r\n|\r", "\n", s)
    return s

def fix_texas_A_artifacts(s:str)->str:

    s = re.sub(r'\bSec\.\s*A(\d+)\.\s*(\d{2})', r'Sec. \1.\2', s)
    s = re.sub(r'(?<=\.)A(?=\d)', ' ', s)
    s = re.sub(r'(?<=\.)A{1,3}(?=[A-Z])', ' ', s)
    s = re.sub(r'(\([a-zA-Z0-9\-]+\))\s*A{1,3}\s+(?=[A-Z])', r'\1 ', s)
    s = re.sub(r'(?<=\s)A{2,}(?=[A-Z])', ' ', s)
    s = re.sub(r'([()\[\].,:;-])\s*A\s+(?=[A-Z0-9])', r'\1 ', s)
    return s

def drop_legislative_history_lines(s:str)->str:
    out=[]
    for ln in s.splitlines():
        t = ln.strip()
        if not t: continue
        if re.match(r"^(Added|Amended|Acts)\b", t): continue
        if re.match(r"^(Source:|Notes?:)\b", t): continue
        if re.match(r"^https?://", t): continue
        if re.match(r"^Page\s+\d+\s+of\s+\d+$", t, flags=re.I): continue
        if t.isupper() and len(t)<=40 and re.search(r"(CODE|CHAPTER|TITLE)", t): continue
        out.append(ln)
    return "\n".join(out)

def fix_hyphenation(s:str)->str:
    s = re.sub(r'(\w+)-\n(\w+)', r'\1\2', s)
    s = re.sub(r'(?<=\w)- (?=\w)', '-', s)
    return s

def tidy_punctuation(s:str)->str:
    s = re.sub(r'\.(?=[A-Za-z0-9])', '. ', s)
    s = s.replace("“","\"").replace("”","\"").replace("’","'").replace("–","-").replace("—","-")
    s = re.sub(r"[ \t]{2,}", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def clean_text(raw:str)->str:
    s = normalize(raw)
    s = fix_texas_A_artifacts(s)
    s = drop_legislative_history_lines(s)
    s = fix_hyphenation(s)
    s = tidy_punctuation(s)
    return s

# --- Custom Component: Local LLM ---
@component
class LocalHFLLM:
    outgoing_edges = 1

    def __init__(self, pipe):
        self.pipe = pipe

    @component.output_types(replies=List[str])
    def run(self, messages: List[str], **kwargs) -> Dict[str, Any]:
        prompt = "\n".join(messages) if isinstance(messages, list) else str(messages)
        output = self.pipe(prompt)[0]["generated_text"]
        # Cleanup output logic
        if "---END---" in output:
            output = output.split("---END---", 1)[0].strip()
        # Remove the prompt from the output (common in Llama pipelines)
        if output.startswith(prompt):
            output = output[len(prompt):].strip()
        return {"replies": [output]}

@component
class ChatMessageToStr:
    outgoing_edges = 1
    @component.output_types(messages=List[str])
    def run(self, prompt: List[ChatMessage], **kwargs) -> Dict[str, Any]:
        return {"messages": [msg.text for msg in prompt]}

# --- CACHED RESOURCE LOADING ---
@st.cache_resource(show_spinner="Loading Model & Indexing Documents...")
def load_rag_pipeline():
    # 1. Load PDFs
    pdf_files = [os.path.join(PDF_DIR, f) for f in os.listdir(PDF_DIR) if f.endswith(".pdf")]
    converter = PyPDFToDocument()
    docs = converter.run(sources=pdf_files)["documents"]
    
    # 2. Clean & Split
    for d in docs:
        if d.content:
            d.content = clean_text(d.content)
            
    splitter = DocumentSplitter(split_by="word", split_length=250, split_overlap=50)
    docs_chunks = splitter.run(documents=docs)["documents"]
    
    # 3. Embed & Store
    document_store = InMemoryDocumentStore()
    doc_embedder = SentenceTransformersDocumentEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
    doc_embedder.warm_up()
    docs_with_embeddings = doc_embedder.run(docs_chunks)
    document_store.write_documents(docs_with_embeddings["documents"])
    
    # 4. Load LLM (Quantized for GPU Efficiency)
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )
    
    model_id = "meta-llama/Llama-3.1-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=HF_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quant_config,
        device_map="auto",
        token=HF_TOKEN
    )
    
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        temperature=0.1,
        return_full_text=False
    )
    
    llm = LocalHFLLM(pipe=pipe)
    
    # 5. Build Pipeline
    rag_pipeline = Pipeline()
    rag_pipeline.add_component("text_embedder", SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2"))
    rag_pipeline.add_component("retriever", InMemoryEmbeddingRetriever(document_store=document_store))
    
    template = [
        ChatMessage.from_user(
            """
You are a highly experienced and cautious **Texas Family Law Expert**.
Your primary goal is to provide a brief but legally rigorous, and factually accurate responses to the legal questions. 
The answer should be complete and cater to the main question asked but should not reference any section codes or Chapters.
Keep the language user friendly that can be understood by a non-legal reader.
Mention the Section number of the law at the end as a note.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know — don't try to make one up.
STOP ANSWERING ONCE THE ANSWER IS COMPLETE. Output the exact sequence ---END--- immediately after the Note.
Context:
{% for document in documents %}
    {{ document.content }}
{% endfor %}
Question: {{question}}
Answer:
            """
        )
    ]
    rag_pipeline.add_component("prompt_builder", ChatPromptBuilder(template=template))
    rag_pipeline.add_component("converter", ChatMessageToStr())
    rag_pipeline.add_component("llm", llm)
    
    rag_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
    rag_pipeline.connect("retriever", "prompt_builder.documents")
    rag_pipeline.connect("prompt_builder.prompt", "converter.prompt")
    rag_pipeline.connect("converter.messages", "llm.messages")
    
    return rag_pipeline

# --- Main App Logic ---
try:
    pipeline_obj = load_rag_pipeline()
    st.success("System Ready!")
except Exception as e:
    st.error(f"Error loading system: {e}")

# Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about Texas Family Law..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing legal code..."):
            response = pipeline_obj.run({
                "text_embedder": {"text": prompt},
                "prompt_builder": {"question": prompt}
            })
            answer = response["llm"]["replies"][0]
            st.markdown(answer)
            
    st.session_state.messages.append({"role": "assistant", "content": answer})