import os
from queue import Queue
from threading import Thread
import textwrap

import gradio as gr
from transformers import LlamaForCausalLM, LlamaTokenizer


SYSTEM_PROMPT = """\
    You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

    If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""
SYSTEM_PROMPT = textwrap.dedent(SYSTEM_PROMPT).strip()


def format_prompt(history, message, system_prompt):
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

    prompt = f"{B_INST} {B_SYS}{system_prompt}{E_SYS} "
    for user_msg, asst_msg in history:
        user_msg = str(user_msg).strip()
        asst_msg = str(asst_msg).strip()
        prompt += f"{user_msg} {E_INST} {asst_msg} </s><s> {B_INST} "

    message = str(message).strip()
    prompt += f"{message} {E_INST} "
    return prompt


class LLama2:
    def __init__(self, model_name, auth_token, temperature=0.6, top_p=0.9, max_gen_len=4096) -> None:
        self.temperature = temperature
        self.top_p = top_p
        self.max_gen_len = max_gen_len
        self.model = LlamaForCausalLM.from_pretrained(
            model_name, use_auth_token=auth_token, load_in_4bit=True, device_map="auto"
        ).eval()
        self.tokenizer = LlamaTokenizer.from_pretrained(model_name, use_auth_token=auth_token)
     
    def generate(self, prompt, history):
        prompt = format_prompt(history, prompt, SYSTEM_PROMPT)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        model_resp = self.model.generate(
            **inputs,
            max_new_tokens=self.max_gen_len,
            temperature=self.temperature,
            top_p=self.top_p,
        )
        # get text from tokenizer
        model_resp = self.tokenizer.decode(model_resp[0], batched=True)
        
        # try to remove the prompt from the output
        model_resp = model_resp.replace(prompt, '')
        
        return model_resp


auth_token = "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"
model_name = "meta-llama/Llama-2-7b-chat-hf"

model = LLama2(model_name=model_name, auth_token=auth_token)

