import torch
import gradio as gr
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# ✅ Check and use GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ✅ Load model and tokenizer with GPU optimization
MODEL_NAME = "google/flan-t5-large"  # Large model for balance of speed & accuracy
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME, 
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32  # Use float16 for GPU speedup
).to(device)  # Move model to GPU/CPU

# ✅ Optimized function for fast responses
def chatbot_response(message, history):
    with torch.no_grad():  # Disable gradients for speed
        inputs = tokenizer(message, return_tensors="pt").to(device)  # Move input to GPU
        reply_ids = model.generate(
            **inputs,
            max_length=200,  # Shorter responses for speed
            temperature=0.7,  # Balanced creativity
            top_p=0.8,  # Nucleus sampling for faster output
            top_k=30,  # Consider fewer words for faster processing
            num_beams=2,  # Speeds up response generation
            do_sample=True  # Ensures diverse responses
        )
        response = tokenizer.decode(reply_ids[0], skip_special_tokens=True)
    return response  # Return response only

# ✅ Enhanced Gradio Chat Interface with Error Fix
interface = gr.ChatInterface(
    chatbot_response,
    title="🩺 Health Chatbot",
    description="💬 **Ask me health-related questions!**\n\n⚠️ *Disclaimer: This chatbot provides general information and is NOT a substitute for professional medical advice.*",
    examples=[
        "What are the symptoms of flu?",
        "How can I boost my immune system?",
        "Is it safe to take paracetamol for a headache?",
        "What are some home remedies for sore throat?"
    ],
)

# ✅ Launch the Gradio Interface with Debug Mode for Errors
interface.launch(debug=True, share=True)

