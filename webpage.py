import gradio as gr
from transformers import AutoModel, AutoTokenizer

model_id = "THUDM/chatglm-6b-int4"



tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModel.from_pretrained(model_id, trust_remote_code=True).half().cuda()

print("Model have been loaded!")

first_start = True
history = []

def chat(text):
    response, his = model.chat(tokenizer, text, history=history)
    history.append(his)
    return response


demo = gr.Interface(fn=chat, 
                    inputs=gr.Textbox(lines=2, placeholder="chat"),
                    outputs='text')

demo.launch()