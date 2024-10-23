from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import torch
import json
from flask import Flask, request, jsonify
import gradio as gr
import os

# Load model XGLM
local_model_dir = "./local_model_bloomz"
if not os.path.exists(local_model_dir):
    model_id = "bigscience/bloomz-560m"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    tokenizer.save_pretrained(local_model_dir)
    model.save_pretrained(local_model_dir)
else:
    tokenizer = AutoTokenizer.from_pretrained(local_model_dir, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(local_model_dir, local_files_only=True)

# Load dataset function
def load_dataset():
    with open('bps_dataset.json', 'r') as file:
        data = json.load(file)
    return data

dataset = load_dataset()
questions = dataset['questions']
answers = dataset['answers']

# Retrieval Function
def retrieve_answer(user_question):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(questions + [user_question])
    cosine_similarities = cosine_similarity(vectors[-1], vectors[:-1]).flatten()
    best_match_idx = cosine_similarities.argmax()
    return answers[best_match_idx]

# Generate response using XGLM
def generate_response(user_input, retrieved_answer):
    input_text = f"{retrieved_answer}"
    inputs = tokenizer(input_text, return_tensors="pt")
    outputs = model.generate(
        inputs['input_ids'], 
        max_length=300,
        num_return_sequences=1,
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        do_sample=True  
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Flask API
app = Flask(__name__)
@app.route('/chatbot', methods=['POST'])
def chatbot():
    data = request.json
    user_input = data.get("message", "")
    if user_input:
        retrieved_answer = retrieve_answer(user_input)
        response = generate_response(user_input, retrieved_answer)
        return jsonify({"response": response, "retrieved_answer": retrieved_answer})
    return jsonify({"response": "Mohon ajukan pertanyaan!"})

# Gradio Interface
def chatbot_response(user_input):
    if user_input:
        retrieved_answer = retrieve_answer(user_input)
        response = generate_response(user_input, retrieved_answer)
        return f"Chatbot: {response}\n\nRetrieved Answer: {retrieved_answer}"

def chat_interface(user_input, history=[]):
    history.append(("Anda: " + user_input, "Chatbot: " + chatbot_response(user_input)))
    return history, history

with gr.Blocks() as demo:
    gr.Markdown("# Chatbot Sederhana dengan Gradio")
    chatbot = gr.Chatbot(label="Chatbot")
    with gr.Row():
        txt_input = gr.Textbox(show_label=False, placeholder="Tulis pesan Anda di sini...")
        submit_btn = gr.Button("Kirim")
    submit_btn.click(chat_interface, inputs=[txt_input, chatbot], outputs=[chatbot, chatbot])
    txt_input.submit(chat_interface, inputs=[txt_input, chatbot], outputs=[chatbot, chatbot])

# Jalankan Flask dan Gradio
if __name__ == '__main__':
    demo.launch(server_name="0.0.0.0", server_port=7860)
    app.run(port=5000)
