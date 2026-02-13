from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    set_seed,
)
import gradio as gr
import mdtex2html
import torch
import os
import sys
from ptuning.arguments import ModelArguments, DataTrainingArguments
import time
'''
parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
    # If we pass only one argument to the script and it's the path to a json file,
    # let's parse it to get our arguments.
    model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
else:
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    #output\checkpoint-1000
'''
tokenizer = AutoTokenizer.from_pretrained("output\checkpoint-100", trust_remote_code=True)
config = AutoConfig.from_pretrained("output\checkpoint-100", trust_remote_code=True)
config.pre_seq_len = 128
# config.prefix_projection = model_args.prefix_projection

model = AutoModel.from_pretrained("model", config=config, trust_remote_code=True).half().cuda()
prefix_state_dict = torch.load(os.path.join("output\checkpoint-100", "pytorch_model.bin"))
new_prefix_state_dict = {}
for k, v in prefix_state_dict.items():
    new_prefix_state_dict[k[len("transformer.prefix_encoder."):]] = v
model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)

model = model.eval()

"""Override Chatbot.postprocess"""


def postprocess(self, y):
    if y is None:
        return []
    for i, (message, response) in enumerate(y):
        y[i] = (
            None if message is None else mdtex2html.convert((message)),
            None if response is None else mdtex2html.convert(response),
        )
    return y


gr.Chatbot.postprocess = postprocess


def parse_text(text):
    """copy from https://github.com/GaiZhenbiao/ChuanhuChatGPT/"""
    lines = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0
    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            items = line.split('`')
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = f'<br></code></pre>'
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace("`", "\`")
                    line = line.replace("<", "&lt;")
                    line = line.replace(">", "&gt;")
                    line = line.replace(" ", "&nbsp;")
                    line = line.replace("*", "&ast;")
                    line = line.replace("_", "&lowbar;")
                    line = line.replace("-", "&#45;")
                    line = line.replace(".", "&#46;")
                    line = line.replace("!", "&#33;")
                    line = line.replace("(", "&#40;")
                    line = line.replace(")", "&#41;")
                    line = line.replace("$", "&#36;")
                lines[i] = "<br>"+line
    text = "".join(lines)
    return text


def predict(input, chatbot, max_length, top_p, temperature, history):
    time_now=str(int(time.time()))
    print(time_now +"input: "+input)
    chatbot.append((parse_text(input), ""))
    response00=""
    for response, history in model.stream_chat(tokenizer, input, history, max_length=max_length, top_p=top_p,
                                               temperature=temperature):
        response00=response
        chatbot[-1] = (parse_text(input), parse_text(response))       

        yield chatbot, history
    print(time_now +"response: "+response00)


def reset_user_input():
    return gr.update(value='')


def reset_state():
    return [], []

def handle_input(input_text):
    # 输入的文本会被传递到这个函数中
    # 这个函数返回输入文本的长度
    print("反馈："+str(input_text))
def reset_handle_input():
    return gr.update(value='')

with gr.Blocks() as demo:
    gr.HTML("""<h1 align="center">尹一达数字分身测试 V1.0</h1>""")
    gr.HTML("""<h2 align="center">TEL：13614964477  EMAIL：yidalinyu@gmail.com  抖音：yida_linyu</h2>""")
    text_input = gr.Textbox(lines=2, placeholder="提供反馈与建议...")
    submit_button = gr.Button("提交", variant="primary")
    submit_button.click(handle_input,[text_input],[])
    submit_button.click(reset_handle_input,[],[text_input])


    chatbot = gr.Chatbot()
    with gr.Row():
        with gr.Column(scale=4):
            with gr.Column(scale=12):
                user_input = gr.Textbox(show_label=False, placeholder="Input...", lines=10).style(
                    container=False)
            with gr.Column(min_width=32, scale=1):
                submitBtn = gr.Button("Submit", variant="primary")
        with gr.Column(scale=1):
            emptyBtn = gr.Button("Clear History")
            max_length = gr.Slider(0, 4096, value=2048, step=1.0, label="Maximum length", interactive=False)
            top_p = gr.Slider(0, 1, value=0.5, step=0.01, label="Top P", interactive=False)
            temperature = gr.Slider(0, 1, value=0.5, step=0.01, label="Temperature", interactive=False)

    history = gr.State([])

    submitBtn.click(predict, [user_input, chatbot, max_length, top_p, temperature, history], [chatbot, history],
                    show_progress=True)
    submitBtn.click(reset_user_input, [], [user_input])

    emptyBtn.click(reset_state, outputs=[chatbot, history], show_progress=True)

demo.queue().launch(share=True, inbrowser=True)
