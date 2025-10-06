import os
import tempfile
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import CharacterTextSplitter 
from langchain_community.embeddings import HuggingFaceEmbeddings 
from langchain_community.vectorstores import FAISS
from langchain.schema.document import Document

import streamlit as st
from gtts import gTTS 
from io import BytesIO
import base64

# --- Configuration ---
load_dotenv()
api_key = os.getenv('GEMINI_API_KEY')

if api_key:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.5-flash")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
else:
    model = None
    embeddings = None

# --- Status Functions ---
def get_model_status():
    if api_key and embeddings:
        return "ready"
    elif not api_key:
        return "no_key"
    else:
        return "embed_fail"

def get_embed_model_name():
    return embeddings.model_name if embeddings else "N/A"

# --- Core Logic ---

def process_documents(document_list, source_name="Document"):
    """Splits documents and creates a FAISS vector store."""
    if not document_list or not embeddings:
        return None, "Embedding model not ready or document list empty."

    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000, 
        chunk_overlap=200, 
        length_function=len
    )
    
    texts = text_splitter.split_documents(document_list)
    vectorstore = FAISS.from_documents(texts, embeddings)
    
    return vectorstore, f"{source_name} processed successfully!"

def generate_first_question(vectorstore):
    """Generates the opening question based on document content."""
    if not model:
        return "Could you please introduce yourself and tell me about your background?"

    sample_docs = vectorstore.similarity_search("experience skills education", k=3)
    
    if not sample_docs:
        return "The document was processed, but I couldn't find enough content to ask a specific first question. Could you start by summarizing your professional experience?"
        
    context = "\n\n".join([doc.page_content for doc in sample_docs])
    
    prompt = f"""You are a professional interviewer. Based on the following document content, ask ONE specific, relevant opening question to start the interview. 
    The question should be directly related to the information in the document.
    
    DOCUMENT CONTEXT:
    {context}
    
    Ask only ONE question. Be professional and direct. Do not include any preamble or explanation, just the question."""
    
    response = model.generate_content(prompt)
    return response.text.strip()

def load_and_process_file(uploaded_file):
    """Loads, processes, and chunks an uploaded file (PDF, DOCX, DOC, TXT)."""
    if not uploaded_file:
        return None, "No file uploaded."

    file_extension = os.path.splitext(uploaded_file.name)[1].lower()

    # Save to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    try:
        loaders = {
            ".pdf": PyPDFLoader,
            ".txt": lambda path: TextLoader(path, encoding="utf-8"),
            ".docx": Docx2txtLoader,
            ".doc": Docx2txtLoader,
        }
        
        if file_extension not in loaders:
            return None, f"Unsupported file type: {file_extension}. Please use PDF, DOCX, DOC, or TXT."

        loader = loaders[file_extension](tmp_file_path)
        document = loader.load()
        
        if not document or not any(doc.page_content.strip() for doc in document):
            return None, "Could not extract any text from the document."
            
        return process_documents(document, uploaded_file.name)
    
    except Exception as e:
        return None, f"File processing error for {file_extension}: {str(e)}"
    
    finally:
        # Cleanup the temporary file
        if os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)

def process_raw_text(raw_text):
    """Processes user-pasted text into a document for vector storage."""
    if not raw_text.strip():
        return None, "Please enter some text before submitting."
    
    doc_obj = [Document(page_content=raw_text, metadata={"source": "user_text_input"})]
    return process_documents(doc_obj, "Pasted Text")

def generate_response(query, vectorstore, history, max_turns=12):
    """Generates the AI's question or final evaluation."""
    if not model:
        return "The AI model is not configured. Please check your GEMINI_API_KEY setting."

    if not vectorstore:
        return "Please upload a document first to start the interview."
    
    # Retrieve relevant document context
    relevant_docs = vectorstore.similarity_search(query, k=3)
    context = "\n\n".join([doc.page_content for doc in relevant_docs])

    turn_number = len([msg for msg in history if msg[0] == "You"])
    
    if turn_number >= max_turns:
        # Final Evaluation Prompt
        prompt = f"""**SYSTEM INSTRUCTION: AI INTERVIEWER - FINAL EVALUATION**

The interview has reached its conclusion after {max_turns} turns.
Based on the full conversation history provided below and the document context, you must now provide a final evaluation.

## DOCUMENT CONTEXT (Used for knowledge check):
{context}

## FULL INTERVIEW HISTORY:
{history}

## USER'S FINAL RESPONSE:
{query}

Use the following FINAL EVALUATION FORMAT exactly:
--- INTERVIEW CONCLUDED ---

Analysis:
[2-3 sentence summary of the candidate's performance, including strengths and weaknesses based on their answers and knowledge reflected in the document.]

Overall Rating: [Score]/10"""
    else:
        # Standard Interview Question Prompt
        prompt = f"""**SYSTEM INSTRUCTION: AI INTERVIEWER AND SUBJECT MATTER EXPERT**

You are a professional interviewer conducting a structured interview based on the document content provided.

## DOCUMENT CONTEXT:
{context}

## INTERVIEW RULES:
1. Ask ONE question at a time based on the document content
2. Analyze the user's response critically
3. Ask follow-up questions if the answer is vague, incomplete, or needs clarification
4. Current turn: {turn_number}/{max_turns}

## USER'S RESPONSE: 
{query}

Respond with either:
- A follow-up question if more clarification is needed
- The next interview question based on the document"""
    
    response = model.generate_content(prompt)
    return response.text.strip()


# =========================================================================
# --- Streamlit UI and TTS Logic ---
# =========================================================================

# --- TTS Function ---
def text_to_audio_base64(text):
    """Convert text to audio and return base64 for embedding in HTML."""
    try:
        tts = gTTS(text=text, lang='en', slow=False)
        fp = BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        return base64.b64encode(fp.read()).decode()
    except Exception:
        return None

# --- Callback Functions for Reruns (to fix red warning) ---

def start_interview_from_file(uploaded_file):
    """Processes uploaded file and updates session state."""
    with st.spinner("Processing..."):
        vectorstore, msg = load_and_process_file(uploaded_file)
        if vectorstore:
            st.session_state.vectorstore = vectorstore
            st.session_state.doc_uploaded = True
            st.session_state.history = []
            st.session_state.interview_started = False
            st.success(msg)
        else:
            st.error(msg or "Failed to process")

def start_interview_from_text(raw_text):
    """Processes pasted text and updates session state."""
    with st.spinner("Processing..."):
        vectorstore, msg = process_raw_text(raw_text)
        if vectorstore:
            st.session_state.vectorstore = vectorstore
            st.session_state.doc_uploaded = True
            st.session_state.history = []
            st.session_state.interview_started = False
            st.success(msg)
        else:
            st.warning(msg or "Enter valid text")

def handle_chat_submit():
    """Handles user text submission during the interview."""
    user_input = st.session_state.text_input.strip()
    if user_input:
        # 1. Add user message to history
        st.session_state.history.append(("You", user_input))
        
        # 2. Generate AI response
        try:
            answer = generate_response(user_input, st.session_state.vectorstore, st.session_state.history)
            # 3. Add bot's response/question to history
            st.session_state.history.append(("Bot", answer))
        except Exception as e:
            # Handle API/Model error
            st.session_state.history.append(("Bot", f"Error generating response: {str(e)}"))
        
        # 4. Clear the input box (Important: must be after accessing the value)
        st.session_state.text_input = ""


# --- Page Config ---
st.set_page_config(page_title="AI Interview Bot", page_icon="🤖", layout="wide")

# --- Session State ---
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "history" not in st.session_state:
    st.session_state.history = []
if "doc_uploaded" not in st.session_state:
    st.session_state.doc_uploaded = False
if "interview_started" not in st.session_state:
    st.session_state.interview_started = False
if "audio_cache" not in st.session_state:
    st.session_state.audio_cache = {}
if "audio_played" not in st.session_state:
    st.session_state.audio_played = set()

# --- Check API ---
if get_model_status() == "no_key":
    st.error("🚨 GEMINI_API_KEY not found. Please create a .env file and add your key.")
    st.stop()

# --- CSS ---
st.markdown("""
    <style>
    .stApp { background: #ffffff; font-family: 'Segoe UI', sans-serif; }
    [data-testid="stSidebar"] { background: #f8f9fa; border-right: 1px solid #e2e8f0; }
    .sidebar-title { font-size: 24px; font-weight: bold; color: #2563eb; }
    .main-content { padding-bottom: 150px; max-width: 1200px; margin: 0 auto; }
    .user-msg {
        background: #f0f7ff;
        padding: 15px 20px;
        border-radius: 20px 20px 5px 20px;
        margin: 10px 0 10px auto;
        max-width: 70%;
        border: 1px solid #dbeafe;
        width: fit-content;
        margin-left: auto;
    }
    .bot-audio {
        background: #f8fafc;
        padding: 15px 20px;
        border-radius: 20px 20px 20px 5px;
        margin: 10px auto 10px 0;
        max-width: 70%;
        border: 1px solid #e2e8f0;
        width: fit-content;
    }
    .audio-label { color: #2563eb; font-weight: bold; margin-bottom: 8px; font-size: 14px; }
    audio { width: 100%; outline: none; }
    .fixed-bottom {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        padding: 15px 20px 15px calc(250px + 20px);
        background: #ffffff;
        border-top: 2px solid #2563eb;
        box-shadow: 0 -4px 20px rgba(0,0,0,0.08);
        z-index: 1000;
    }
    .upload-container {
        background: #ffffff;
        padding: 40px;
        border-radius: 16px;
        max-width: 800px;
        margin: 20px auto;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        border: 2px solid #e2e8f0;
    }
    .stButton > button {
        background: #2563eb !important;
        color: white !important;
        font-weight: bold;
        border-radius: 12px;
        padding: 12px 24px;
        border: none;
    }
    .stButton > button:hover {
        background: #1d4ed8 !important;
        transform: scale(1.02);
    }
    h1, h3 { color: #1e293b !important; }
    </style>
""", unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.markdown("<div class='sidebar-title'>🤖 AI Interview Bot</div>", unsafe_allow_html=True)
    st.markdown("**Voice Questions • Text Answers**")
    st.markdown("<hr>", unsafe_allow_html=True)
    
    if st.session_state.doc_uploaded:
        st.success("✅ Interview active!")
    else:
        st.info("📄 Upload a document or paste text to begin")
    
    st.success(f"Embed Model: **{get_embed_model_name()}**")
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("👩‍💻 **Ruhani Gera**")
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # Corrected f-string issue: removed the unnecessary 'f' prefix
    st.markdown("<a href='https://www.linkedin.com/in/ruhani-gera-851454300/' target='_blank'><button style='background:#2563eb;color:white;padding:10px;border:none;border-radius:8px;width:100%;'>📩 Contact</button></a>", unsafe_allow_html=True)
    
    st.markdown("<hr>", unsafe_allow_html=True)
    
    if st.button("🗑️ Clear & Reset", use_container_width=True):
        # Clears all session state variables and reruns
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# --- Main ---
st.markdown("<h1 style='text-align:center;'>💬 AI Interview Simulation</h1>", unsafe_allow_html=True)

# --- Upload Screen (First Page) ---
if not st.session_state.doc_uploaded:
    st.markdown("<div class='upload-container'>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align:center;'>📄 Upload Your Document</h3>", unsafe_allow_html=True)
    
    # File Uploader
    uploaded_file = st.file_uploader("Choose file (PDF, DOCX, TXT, DOC)", type=["pdf", "txt", "docx", "doc"])
    
    # Button for File Upload (Centered)
    col_file_pre, col_file_btn, col_file_post = st.columns([1, 1, 1])
    with col_file_btn:
        if uploaded_file:
            # Call the dedicated callback function with arguments
            st.button("🚀 Start Interview from File", use_container_width=True, on_click=start_interview_from_file, args=(uploaded_file,))
        else:
            # Display a disabled button or spacer if no file is uploaded
            st.button("🚀 Start Interview from File", use_container_width=True, disabled=True)
    
    st.markdown("<h5 style='text-align:center; margin:20px 0;'>— OR —</h5>", unsafe_allow_html=True)
    
    # Text Area
    raw_text = st.text_area("📝 Paste Resume/Job Description", height=200, placeholder="Paste text here...", key="raw_text_input")
    
    # Button for Text Input (Centered)
    col_text_pre, col_text_btn, col_text_post = st.columns([1, 1, 1])
    with col_text_btn:
        if raw_text.strip():
            # Call the dedicated callback function with arguments
            st.button("✨ Process Text Input", use_container_width=True, on_click=start_interview_from_text, args=(st.session_state.raw_text_input,))
        else:
            # Display a disabled button or spacer if no text is entered
            st.button("✨ Process Text Input", use_container_width=True, disabled=True)

    st.markdown("</div>", unsafe_allow_html=True)
    
# --- Chat Interface (Second Page) ---
else:
    # 1. Start interview (initial question)
    if not st.session_state.interview_started:
        with st.spinner("Preparing opening question..."):
            first_q = generate_first_question(st.session_state.vectorstore)
            st.session_state.history.append(("Bot", first_q))
            st.session_state.interview_started = True
            # No rerun needed here, the state change causes a natural rerun
    
    # 2. Display history
    st.markdown("<div class='main-content'>", unsafe_allow_html=True)
    
    for i, (speaker, text) in enumerate(st.session_state.history):
        if speaker == "You":
            st.markdown(f"<div class='user-msg'>{text}</div>", unsafe_allow_html=True)
        else:
            # Bot message (Voice Question)
            msg_id = f"msg_{i}"
            
            # Cache the audio to avoid re-generating on every rerun
            if msg_id not in st.session_state.audio_cache:
                audio = text_to_audio_base64(text)
                if audio:
                    st.session_state.audio_cache[msg_id] = audio
            else:
                audio = st.session_state.audio_cache[msg_id]
            
            if audio:
                # Autoplay the latest bot message once
                is_latest_message = i == len(st.session_state.history) - 1
                has_not_played = msg_id not in st.session_state.audio_played
                
                autoplay = "autoplay" if is_latest_message and has_not_played else ""
                
                if autoplay:
                    # Mark as played so it doesn't try to autoplay again on future reruns
                    st.session_state.audio_played.add(msg_id)
                
                # HTML with base64 embedded audio source
                st.markdown(f"""
                    <div class='bot-audio'>
                        <div class='audio-label'>🤖 Interviewer:</div>
                        <audio controls {autoplay}>
                            <source src="data:audio/mp3;base64,{audio}" type="audio/mp3">
                            Your browser does not support the audio element.
                        </audio>
                    </div>
                """, unsafe_allow_html=True)
            else:
                 # Fallback for text if TTS fails
                st.markdown(f"<div class='bot-audio'>🤖 Interviewer (Text Fallback):<br>{text}</div>", unsafe_allow_html=True)

    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # 3. Text input at the bottom (User Answer)
    st.markdown("<div class='fixed-bottom'>", unsafe_allow_html=True)
    col1, col2 = st.columns([0.85, 0.15])
    
    with col1:
        # Key 'text_input' links this text_area to st.session_state for the callback
        st.text_area("User Answer", placeholder="Type your answer...", height=50, label_visibility="collapsed", key="text_input")
    
    with col2:
        # Use on_click to handle the submission and clearing without explicit st.rerun() in the main flow
        # This prevents the red warning message.
        st.button("Send", use_container_width=True, type="primary", on_click=handle_chat_submit)
    
    st.markdown("</div>", unsafe_allow_html=True)