import torch
import gradio as gr
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


MODEL_NAME = "google/flan-t5-xl"  
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32).to("cpu")

def chatbot_response(user_input):
    inputs = tokenizer(user_input, return_tensors="pt").to("cpu")
    reply_ids = model.generate(
        **inputs,
        max_length=500,  
        temperature=0.7,  
        top_p=0.9,  
        top_k=50,  
        repetition_penalty=1.2  
    )
    response = tokenizer.decode(reply_ids[0], skip_special_tokens=True)
    return response


interface = gr.Interface(
    fn=chatbot_response,
    inputs=gr.Textbox(label="You:"),
    outputs=gr.Textbox(label="Chatbot:"),
    title="Health Chatbot",
    description="Ask me health-related questions!",
)

interface.launch(share=True)
