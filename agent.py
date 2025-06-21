import streamlit as st
from groq import Groq
import os
import PyPDF2 # Import PyPDF2 for PDF text extraction
import pandas as pd # Import pandas for CSV processing
import base64 # For encoding images to base64
import requests # For making HTTP requests to Gemini API
import io # For handling image bytes

# --- API Key Configuration ---
# IMPORTANT: Replace "YOUR_GROQ_API_KEY" with your actual Groq API Key.
# It's recommended to set this as an environment variable (e.g., GROQ_API_KEY)
groq_api_key = "YOURGROKIAPIKEY" # <<<--- REPLACE THIS WITH YOUR ACTUAL GROQ API KEY ---<<<

# IMPORTANT: Replace "YOUR_GEMINI_API_KEY" with your actual Gemini API Key.
# This is required for image understanding features.
gemini_api_key = "YOURGEMINIAPIKEY" # <<<--- REPLACE THIS WITH YOUR ACTUAL GEMINI API KEY ---<<<


# --- Initialize Groq Client ---
# The check for placeholder API keys has been removed as per your request.
try:
    groq_client = Groq(api_key=groq_api_key)
except Exception as e:
    st.error(f"Failed to initialize Groq client. Please check your Groq API key and internet connection: {e}")
    st.stop()

# --- Utility Functions for Chat Actions ---
def clear_chat_history():
    """Clears the chat history and all loaded document/data/image info in session state."""
    st.session_state.messages = []
    if 'show_copy_area' in st.session_state:
        del st.session_state['show_copy_area']
    
    # Clear PDF-related session state
    if 'pdf_text' in st.session_state:
        del st.session_state['pdf_text']
    if 'pdf_name' in st.session_state:
        del st.session_state['pdf_name']
    
    # Clear CSV-related session state
    if 'csv_data' in st.session_state:
        del st.session_state['csv_data']
    if 'csv_name' in st.session_state:
        del st.session_state['csv_name']
    if 'csv_overview_text' in st.session_state: # New: Clear CSV overview text
        del st.session_state['csv_overview_text']
    
    # Clear Image-related session state
    if 'image_data_base64' in st.session_state:
        del st.session_state['image_data_base64']
    if 'image_name' in st.session_state:
        del st.session_state['image_name']
    if 'image_mime_type' in st.session_state:
        del st.session_state['image_mime_type']
    if 'image_analysis_result' in st.session_state: # New: Clear image analysis result
        del st.session_state['image_analysis_result']
    
    st.rerun() # Rerun to reflect the cleared state immediately

def get_chat_history_as_text():
    """Converts the chat history into a plain text string."""
    history_text = ""
    # Include PDF info if available
    if 'pdf_name' in st.session_state and st.session_state.pdf_name:
        history_text += f"--- Document: {st.session_state.pdf_name} ---\n\n"
    # Include CSV info if available
    if 'csv_name' in st.session_state and st.session_state.csv_name:
        history_text += f"--- Data File: {st.session_state.csv_name} ---\n\n"
    # Include Image info if available
    if 'image_name' in st.session_state and st.session_state.image_name:
        history_text += f"--- Image: {st.session_state.image_name} (Analyzed by AI) ---\n\n"
    
    for message in st.session_state.messages:
        role = "You" if message["role"] == "user" else "Bot"
        history_text += f"{role}: {message['content']}\n\n"
    return history_text.strip()

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    """Extracts text from a PDF file."""
    text = ""
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text += page.extract_text() or "" # Handle potential None if page has no text
    return text

# Functions for CSV handling
def prepare_csv_for_llm(df):
    """Prepares a digestible string representation of CSV data for the LLM."""
    summary_parts = []
    summary_parts.append("--- CSV Data Overview ---")
    summary_parts.append(f"Number of rows: {len(df)}")
    summary_parts.append(f"Number of columns: {len(df.columns)}")
    summary_parts.append("\nColumn Names and Data Types:")
    for col, dtype in df.dtypes.items():
        summary_parts.append(f"- '{col}': {dtype}")
    
    summary_parts.append("\nFirst 5 rows (head of the data):")
    summary_parts.append(df.head().to_markdown(index=False))
    
    numerical_cols = df.select_dtypes(include=['number']).columns
    if not numerical_cols.empty:
        summary_parts.append("\nDescriptive Statistics for Numerical Columns:")
        summary_parts.append(df[numerical_cols].describe().transpose().to_markdown())
        
    summary_parts.append("\n--- End CSV Data Overview ---")
    
    return "\n".join(summary_parts)

def summarize_csv_with_groq(df_content, csv_name):
    """Uses Groq API to summarize the prepared CSV content."""
    prepared_csv_text = prepare_csv_for_llm(df_content)
    
    chat_messages_for_summary = [
        {"role": "system", "content": "You are an AI assistant specialized in summarizing tabular data. Analyze the provided CSV data description and give a concise, insightful summary highlighting key insights, trends, or important aspects of the data. Focus on what the data represents and its potential implications."},
        {"role": "user", "content": f"Please provide a summary of the data from the file '{csv_name}' based on the following overview:\n\n{prepared_csv_text}"}
    ]
    
    with st.spinner(f"Generating summary for {csv_name}..."):
        try:
            chat_completion = groq_client.chat.completions.create(
                messages=chat_messages_for_summary,
                model="llama3-8b-8192",
                temperature=0.5,
                max_tokens=1500,
                top_p=1,
                stream=False,
                stop=None,
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            st.error(f"Error summarizing CSV: {e}")
            return f"Sorry, I could not generate a summary for the CSV due to an error: {e}"

# New function for image handling
def image_to_base64(image_file):
    """Converts an uploaded Streamlit image file to a base64 string."""
    bytes_data = image_file.getvalue()
    base64_encoded_data = base64.b64encode(bytes_data).decode('utf-8')
    return base64_encoded_data

def analyze_image_with_gemini(base64_image_data, mime_type, prompt_text="Describe this image in detail and identify any key objects or scenes."):
    """Sends an image to the Gemini API for analysis."""
    if gemini_api_key == "YOUR_GEMINI_API_KEY" or not gemini_api_key:
        st.error("Gemini API Key is not configured. Please add your Gemini API Key to enable image analysis.")
        return "I cannot analyze the image. Gemini API Key is missing."

    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={gemini_api_key}"
    
    # Construct the payload as required by Gemini API for image understanding
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": prompt_text},
                    {
                        "inlineData": {
                            "mimeType": mime_type,
                            "data": base64_image_data
                        }
                    }
                ]
            }
        ]
    }

    with st.spinner("Analyzing image with Gemini..."):
        try:
            response = requests.post(api_url, headers={'Content-Type': 'application/json'}, json=payload)
            response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)
            result = response.json()

            if result.get("candidates") and result["candidates"][0].get("content") and result["candidates"][0]["content"].get("parts"):
                return result["candidates"][0]["content"]["parts"][0]["text"]
            else:
                st.error(f"Unexpected response from Gemini API: {result}")
                return "Sorry, I received an unexpected response when analyzing the image."
        except requests.exceptions.RequestException as req_e:
            st.error(f"Network or API request error: {req_e}")
            return f"Sorry, I could not analyze the image due to a network or API error: {req_e}"
        except Exception as e:
            st.error(f"An error occurred during image analysis: {e}")
            return f"Sorry, I could not analyze the image due to an unexpected error: {e}"


# --- Streamlit UI Setup ---
st.set_page_config(page_title="Dynamic Groq Chatbot", layout="centered")

# --- Custom CSS for enhanced UI ---
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Inter', sans-serif;
        /* Vibrant multi-color gradient background */
        background: linear-gradient(135deg, #FF6B6B, #4ECDC4, #45B7D1, #FFA07A, #98D8AA);
        background-size: 400% 400%; /* Make gradient larger for animation */
        animation: gradientAnimation 15s ease infinite; /* Animation for a dynamic look */
        background-attachment: fixed;
    }

    @keyframes gradientAnimation {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* Main container styling - Chatbot's overall surrounding */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 1.5rem;
        padding-right: 1.5rem;
        max-width: 750px; /* Slightly wider for better content display */
        margin: auto;
        background-color: rgba(255, 255, 255, 0.2); /* Subtle white overlay to complement gradient */
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3); /* More prominent shadow */
        color: #FFFFFF; /* Ensures general text inside this container (like spinner) is white */
        backdrop-filter: blur(5px); /* Add a subtle blur effect to the background */
    }

    /* Chat message styling - Base for all messages */
    [data-testid="stChatMessage"] {
        background-color: #FFFFFF; /* White background for all messages */
        border-radius: 18px;
        padding: 12px 18px;
        margin-bottom: 12px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.15); /* Improved shadow for messages */
        transition: transform 0.2s ease-in-out;
    }

    /* Apply black color to the text content within chat messages */
    [data-testid="stChatMessage"] p {
        color: #000000 !important; /* Black text color for paragraph elements, using !important for strong specificity */
    }

    [data-testid="stChatMessage"]:hover {
        transform: translateY(-2px); /* Slight lift on hover */
    }

    /* User messages - Specific alignment */
    [data-testid="stChatMessage"].st-chat-message-user {
        border-bottom-right-radius: 8px;
        text-align: right;
        margin-left: 20%; /* Keep user messages to the right */
        border: 1px solid #CCCCCC; /* Light gray border */
    }

    /* Assistant messages - Specific alignment */
    [data-testid="stChatMessage"].st-chat-message-assistant {
        border-bottom-left-radius: 8px;
        text-align: left;
        margin-right: 20%; /* Keep assistant messages to the left */
        border: 1px solid #CCCCCC; /* Light gray border */
    }

    /* Input text area styling */
    [data-testid="stTextInput"] div div input {
        border-radius: 25px;
        padding: 12px 20px;
        border: 2px solid #6A5ACD; /* Thicker, colored border */
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        font-size: 1rem;
        color: #333;
        background-color: #ffffff;
        transition: border-color 0.3s ease, box-shadow 0.3s ease;
    }

    [data-testid="stTextInput"] div div input:focus {
        border-color: #4A90E2; /* Blue on focus */
        box-shadow: 0 0 0 3px rgba(74, 144, 226, 0.3); /* Glow effect on focus */
        outline: none;
    }

    /* Streamlit header alignment and styling */
    h1 {
        text-align: center;
        color: #FFFFFF; /* White for title */
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 5px rgba(0,0,0,0.3);
    }

    /* Markdown text below title */
    .stMarkdown {
        text-align: center;
        color: #E0E0E0; /* Light gray for subtitle */
        margin-bottom: 2rem;
        font-size: 1.1rem;
    }

    /* Spinner style */
    .stSpinner {
        text-align: center;
        color: #ADD8E6; /* Light blue for spinner */
        font-size: 1.1rem;
    }

    /* Clear chat button styling */
    .stButton > button {
        background-color: #4A90E2; /* Brighter blue */
        color: white;
        border-radius: 12px;
        padding: 10px 20px;
        border: none;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        cursor: pointer;
        transition: background-color 0.3s ease, transform 0.2s ease, box-shadow 0.3s ease;
        font-weight: 600;
        letter-spacing: 0.5px;
        width: 100%; /* Make button fill its column */
    }

    .stButton > button:hover {
        background-color: #357ABD; /* Darker blue on hover */
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.3);
    }

    .stButton > button:active {
        transform: translateY(0);
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }

    /* Responsive adjustments for smaller screens */
    @media (max-width: 600px) {
        .main .block-container {
            padding-left: 0.5rem;
            padding-right: 0.5rem;
            border-radius: 0;
            box-shadow: none;
        }
        [data-testid="stChatMessage"].st-chat-message-user,
        [data-testid="stChatMessage"].st-chat-message-assistant {
            margin-left: 0;
            margin-right: 0;
            border-radius: 10px;
        }
        .stButton > button {
            padding: 8px 15px;
            font-size: 0.9rem;
        }
    }

    </style>
    """,
    unsafe_allow_html=True
)

st.title("🤝 AI chatbot")
st.markdown("Feel free to chat with me! I'm powered by Groq's fast LLMs.")

# --- Document/Media Upload Section ---
st.sidebar.header("Upload Document/Media")
uploaded_pdf_file = st.sidebar.file_uploader("Upload a PDF document", type=["pdf"], key="pdf_uploader")
uploaded_csv_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"], key="csv_uploader")
uploaded_image_file = st.sidebar.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"], key="image_uploader")


# --- Handle Uploads (Prioritize latest upload and clear others) ---

# Handle Image Upload
if uploaded_image_file is not None:
    if 'image_name' not in st.session_state or st.session_state.image_name != uploaded_image_file.name:
        # Clear other loaded docs/data
        if 'pdf_text' in st.session_state: del st.session_state['pdf_text']
        if 'pdf_name' in st.session_state: del st.session_state['pdf_name']
        if 'csv_data' in st.session_state: del st.session_state['csv_data']
        if 'csv_name' in st.session_state: del st.session_state['csv_name']
        if 'csv_overview_text' in st.session_state: del st.session_state['csv_overview_text'] # Clear CSV overview

        with st.spinner(f"Processing {uploaded_image_file.name}..."):
            try:
                base64_data = image_to_base64(uploaded_image_file)
                st.session_state.image_data_base64 = base64_data
                st.session_state.image_name = uploaded_image_file.name
                st.session_state.image_mime_type = uploaded_image_file.type # Store MIME type for Gemini
                st.session_state.messages.append({"role": "assistant", "content": f"Successfully loaded image **{uploaded_image_file.name}**. You can now click 'Analyze Image' to get details."})
                st.sidebar.success(f"Loaded: {uploaded_image_file.name}")
                st.rerun() 
            except Exception as e:
                st.sidebar.error(f"Error processing image: {e}")
                # Clear potentially corrupted state
                if 'image_data_base64' in st.session_state: del st.session_state['image_data_base64']
                if 'image_name' in st.session_state: del st.session_state['image_name']
                if 'image_mime_type' in st.session_state: del st.session_state['image_mime_type']

# Handle PDF Upload
elif uploaded_pdf_file is not None:
    if 'pdf_name' not in st.session_state or st.session_state.pdf_name != uploaded_pdf_file.name:
        # Clear other loaded docs/data
        if 'csv_data' in st.session_state: del st.session_state['csv_data']
        if 'csv_name' in st.session_state: del st.session_state['csv_name']
        if 'csv_overview_text' in st.session_state: del st.session_state['csv_overview_text'] # Clear CSV overview
        if 'image_data_base64' in st.session_state: del st.session_state['image_data_base64']
        if 'image_name' in st.session_state: del st.session_state['image_name']
        if 'image_mime_type' in st.session_state: del st.session_state['image_mime_type']
        if 'image_analysis_result' in st.session_state: del st.session_state['image_analysis_result'] # Clear image analysis

        with st.spinner(f"Processing {uploaded_pdf_file.name}..."):
            try:
                pdf_text = extract_text_from_pdf(uploaded_pdf_file)
                st.session_state.pdf_text = pdf_text
                st.session_state.pdf_name = uploaded_pdf_file.name
                st.session_state.messages.append({"role": "assistant", "content": f"Successfully loaded **{uploaded_pdf_file.name}**. You can now ask questions about its content."})
                st.sidebar.success(f"Loaded: {uploaded_pdf_file.name}")
                st.rerun() 
            except Exception as e:
                st.sidebar.error(f"Error processing PDF: {e}")
                if 'pdf_text' in st.session_state: del st.session_state['pdf_text']
                if 'pdf_name' in st.session_state: del st.session_state['pdf_name']

# Handle CSV Upload
elif uploaded_csv_file is not None:
    if 'csv_name' not in st.session_state or st.session_state.csv_name != uploaded_csv_file.name:
        # Clear other loaded docs/data
        if 'pdf_text' in st.session_state: del st.session_state['pdf_text']
        if 'pdf_name' in st.session_state: del st.session_state['pdf_name']
        if 'image_data_base64' in st.session_state: del st.session_state['image_data_base64']
        if 'image_name' in st.session_state: del st.session_state['image_name']
        if 'image_mime_type' in st.session_state: del st.session_state['image_mime_type']
        if 'image_analysis_result' in st.session_state: del st.session_state['image_analysis_result'] # Clear image analysis

        with st.spinner(f"Processing {uploaded_csv_file.name}..."):
            try:
                df = pd.read_csv(uploaded_csv_file)
                st.session_state.csv_data = df
                st.session_state.csv_name = uploaded_csv_file.name
                st.session_state.csv_overview_text = prepare_csv_for_llm(df) # Store overview directly
                st.session_state.messages.append({"role": "assistant", "content": f"Successfully loaded **{uploaded_csv_file.name}**. You can now ask me to summarize it or ask questions about its structure."})
                st.sidebar.success(f"Loaded: {uploaded_csv_file.name}")
                st.rerun()
            except Exception as e:
                st.sidebar.error(f"Error processing CSV: {e}")
                if 'csv_data' in st.session_state: del st.session_state['csv_data']
                if 'csv_name' in st.session_state: del st.session_state['csv_name']
                if 'csv_overview_text' in st.session_state: del st.session_state['csv_overview_text']


# --- Display current loaded Document/Data/Image info and action buttons ---
if 'pdf_name' in st.session_state and st.session_state.pdf_name:
    st.sidebar.write(f"**Document Loaded:** {st.session_state.pdf_name}")
    if st.sidebar.button("Clear Loaded Document"):
        clear_chat_history() # Use the consolidated clear function
        st.session_state.messages.append({"role": "assistant", "content": "Document has been cleared. I will now answer general questions."})
        st.rerun()

elif 'csv_name' in st.session_state and st.session_state.csv_name:
    st.sidebar.write(f"**Data Loaded:** {st.session_state.csv_name}")
    
    # Button to generate CSV summary
    if st.sidebar.button("Generate CSV Summary"):
        # Add a placeholder message for the summary generation
        st.session_state.messages.append({"role": "assistant", "content": f"Generating a summary for '{st.session_state.csv_name}'... This might take a moment."})
        # Directly display the thinking message in chat without rerun to avoid flashing
        st.chat_message("assistant", avatar="🤖").markdown(f"Generating a summary for '{st.session_state.csv_name}'... This might take a moment.")
        
        summary = summarize_csv_with_groq(st.session_state.csv_data, st.session_state.csv_name)
        st.session_state.messages.append({"role": "assistant", "content": summary})
        # No need for a separate 'csv_summary_generated' flag, the summary is now part of chat history
        st.rerun() # Rerun to display the summary in chat

    if st.sidebar.button("Clear Loaded Data"):
        clear_chat_history() # Use the consolidated clear function
        st.session_state.messages.append({"role": "assistant", "content": "Data has been cleared. I will now answer general questions."})
        st.rerun()

elif 'image_name' in st.session_state and st.session_state.image_name:
    st.sidebar.write(f"**Image Loaded:** {st.session_state.image_name}")
    # Corrected: Replaced use_column_width with use_container_width
    st.sidebar.image(base64.b64decode(st.session_state.image_data_base64), caption="Uploaded Image", use_container_width=True)
    
    image_analysis_prompt = st.sidebar.text_input(
        "Enter prompt for image analysis (optional):",
        value="Describe this image in detail and identify any key objects or scenes.",
        key="image_analysis_prompt_input"
    )

    if st.sidebar.button("Analyze Image"):
        if not (gemini_api_key and gemini_api_key != "YOUR_GEMINI_API_KEY"):
            st.error("Gemini API Key is not configured. Please add your Gemini API Key to enable image analysis.")
            st.session_state.messages.append({"role": "assistant", "content": "I cannot analyze the image. Gemini API Key is missing."})
        else:
            # Add a placeholder message for the analysis generation
            st.session_state.messages.append({"role": "assistant", "content": f"Analyzing image '{st.session_state.image_name}' with Gemini... This might take a moment."})
            st.chat_message("assistant", avatar="🤖").markdown(f"Analyzing image '{st.session_state.image_name}' with Gemini... This might take a moment.")

            analysis_result = analyze_image_with_gemini(
                st.session_state.image_data_base64,
                st.session_state.image_mime_type,
                prompt_text=image_analysis_prompt
            )
            st.session_state.image_analysis_result = analysis_result # Store analysis result for context
            st.session_state.messages.append({"role": "assistant", "content": analysis_result})
            st.rerun() # Rerun to display the analysis in chat

    if st.sidebar.button("Clear Loaded Image"):
        clear_chat_history() # Use the consolidated clear function
        st.session_state.messages.append({"role": "assistant", "content": "Image has been cleared. I will now answer general questions."})
        st.rerun()


# --- Chat History Management (Remains the same) ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    if message["role"] == "user":
        with st.chat_message("user", avatar="😀"):
            st.markdown(message["content"])
    else:
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(message["content"])

# --- User Input and Response Generation ---
if prompt := st.chat_input("Type your message here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="😀"):
        st.markdown(prompt)

    with st.spinner("Thinking..."):
        try:
            chat_messages = []
            
            # Add context based on what's currently loaded
            if 'pdf_text' in st.session_state and st.session_state.pdf_text:
                max_pdf_chars = 30000 
                context_pdf_text = st.session_state.pdf_text[:max_pdf_chars]
                if len(st.session_state.pdf_text) > max_pdf_chars:
                    context_pdf_text += "\n\n... (document truncated for context. For very large documents, consider a RAG solution for better performance and accuracy.)"

                chat_messages.append({"role": "system", "content": f"The user has provided the following document content: '{context_pdf_text}'."})
                chat_messages.append({"role": "system", "content": "Answer questions based on the provided document content first. If the question is not related to the document, answer generally."})
            
            elif 'csv_data' in st.session_state and st.session_state.csv_overview_text: # Use csv_overview_text directly
                chat_messages.append({"role": "system", "content": f"The user has loaded a CSV file named '{st.session_state.csv_name}'. Here's an overview of the data: {st.session_state.csv_overview_text}"})
                chat_messages.append({"role": "system", "content": "Answer questions based on this CSV data overview first. If the question is not related to the CSV, answer generally."})

            elif 'image_name' in st.session_state and st.session_state.image_analysis_result: # Use image_analysis_result directly
                image_analysis = st.session_state.image_analysis_result
                chat_messages.append({"role": "system", "content": f"The user has loaded and analyzed an image named '{st.session_state.image_name}'. The analysis is: '{image_analysis}'."})
                chat_messages.append({"role": "system", "content": "Answer questions based on this image analysis first. If the question is not related to the image, answer generally."})


            # Append existing chat history (user's previous questions and bot's previous answers/summaries/analyses)
            for msg in st.session_state.messages:
                if msg["role"] in ["user", "assistant"]:
                    chat_messages.append({"role": msg["role"], "content": msg["content"]})

            chat_completion = groq_client.chat.completions.create(
                messages=chat_messages,
                model="llama3-8b-8192",
                temperature=0.7,
                max_tokens=1024,
                top_p=1,
                stream=False,
                stop=None,
            )

            response_content = chat_completion.choices[0].message.content

            st.session_state.messages.append({"role": "assistant", "content": response_content})
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(response_content)

        except Exception as e:
            st.error(f"An error occurred: {e}. Please check your Groq API key, network connection, or if the content is too large for the model's context window.")
            st.session_state.messages.append({"role": "assistant", "content": f"Error: {e}"})

# --- Chat Action Buttons (Remains the same) ---
st.markdown("---")
col1, col2, col3 = st.columns([1,1,1])

with col1:
    st.button("Clear Chat", on_click=clear_chat_history, use_container_width=True)

with col2:
    if st.button("Copy Chat", use_container_width=True):
        st.session_state['show_copy_area'] = True
        st.session_state['copy_text'] = get_chat_history_as_text()

with col3:
    chat_text_to_save = get_chat_history_as_text()
    st.download_button(
        label="Save Chat",
        data=chat_text_to_save,
        file_name="groq_chatbot_chat_history.txt",
        mime="text/plain",
        use_container_width=True
    )

if st.session_state.get('show_copy_area'):
    st.text_area(
        "Copy Chat History (Ctrl+C or Cmd+C to copy)",
        value=st.session_state.get('copy_text', ''),
        height=200,
        key="chat_copy_area"
    )
    if st.button("Hide Copy Area", use_container_width=True):
        st.session_state['show_copy_area'] = False
        st.rerun()
