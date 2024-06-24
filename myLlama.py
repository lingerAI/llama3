import torch
from pathlib import Path
import os
import time
import json

from llama.model import ModelArgs, Transformer
from llama.tokenizer import ChatFormat, Tokenizer

MODEL_DIR = "D:\AI\meta-llama\Meta-Llama-3-8B-Instruct\original"
MAX_SEQ_LEN = 512
MAX_BATCH_SIZE = 1
SEED = 42
TEMPERATURE = 0.6
TOP_P = 0.9
DEVICE = "cuda" if torch.cuda.is_available else "cpu"

def loadModel():
    model_path = os.path.join(MODEL_DIR, "consolidated.00.pth")
    params_path = os.path.join(MODEL_DIR, "params.json")
    tokenizer_path = os.path.join(MODEL_DIR, "tokenizer.model")
    torch.manual_seed(SEED)
    start_time = time.time()
    tokenizer = Tokenizer(model_path=tokenizer_path)
    with open(params_path, "r") as f:
        params = json.loads(f.read())
    model_args: ModelArgs = ModelArgs(
        max_seq_len=MAX_SEQ_LEN,
        max_batch_size=MAX_BATCH_SIZE,
        **params,
    )
    if torch.cuda.is_bf16_supported():
        torch.set_default_tensor_type(torch.cuda.BFloat16Tensor)
    else:
        torch.set_default_tensor_type(torch.cuda.HalfTensor)
    checkpoint = torch.load(model_path, map_location="cpu")
    model = Transformer(model_args)
    model.load_state_dict(checkpoint, strict=False)
    #model = model.to(DEVICE)
    # tokenizer = tokenizer.to(DEVICE)
    print(f"Loaded in {time.time() - start_time:.2f} seconds")
    return model, tokenizer

def sample_top_p(probs, p):
    """
    Perform top-p (nucleus) sampling on a probability distribution.

    Args:
        probs (torch.Tensor): Probability distribution tensor.
        p (float): Probability threshold for top-p sampling.

    Returns:
        torch.Tensor: Sampled token indices.

    Note:
        Top-p sampling selects the smallest set of tokens whose cumulative probability mass
        exceeds the threshold p. The distribution is renormalized based on the selected tokens.
    """
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token


def select_next_token(logits):
    if TEMPERATURE > 0:
        probs = torch.softmax(logits[:, -1] / TEMPERATURE, dim=-1)
        next_token = sample_top_p(probs, TOP_P)
    else:
        next_token = torch.argmax(logits[:, -1], dim=-1)
    return next_token

def decodeUtf8(bstr):
    try:
        return bstr.decode("utf-8")
    except UnicodeDecodeError:
        return None


def chat(model, tokenizer, prompt):
    formatter = ChatFormat(tokenizer)
    dialog = [{"role": "user", "content": prompt}]
    prompt_token = formatter.encode_dialog_prompt(dialog)
    prompt_len = len(prompt_token)
    if prompt_len >= MAX_SEQ_LEN:
        print(f"\33[31m你丫问题太长了， 我很难办！\033[0m")
        return
    total_len = MAX_SEQ_LEN
    pad_id = tokenizer.pad_id
    size = (1, total_len) # batch_size = 1
    token_list = torch.full(size, pad_id, dtype=torch.long, device=DEVICE)
    token_list[0, :prompt_len] = torch.tensor(prompt_token, dtype=torch.long, device=DEVICE)
    prev_pos = 0
    stop_tokens = list(tokenizer.stop_tokens)
    bstr = b""
    print("\033[34mLlama3: ", end='', flush=True)
    for cur_pos in range(prompt_len, total_len):
        logits = model.forward(token_list[:, prev_pos:cur_pos], prev_pos)
        next_token = select_next_token(logits)
        token_list[:, cur_pos] = next_token
        
        if next_token[0] in stop_tokens:
            break
        
        ansbytes = tokenizer.decode_single_token_bytes(next_token)
        bstr += ansbytes
        s = decodeUtf8(bstr)
        if s:
            print(s, end='', flush=True)

            bstr = b""
        prev_pos = cur_pos
    print("\033[0m")

def chatLoop(model, tokenizer):
    while True:
        prompt = input("\33[32mprompt: ") ##要想生活过的去， prompt必须来点绿
        if prompt == "exit":
            break
        chat(model, tokenizer, prompt)

def test_tokenizer():
    tokenizer_path = os.path.join(MODEL_DIR, "tokenizer.model")
    tokenizer = Tokenizer(model_path=tokenizer_path)
    #t = tokenizer.encode("hello", bos=False, eos=False)
    t = tokenizer.encode("饕餮", bos=False, eos=False)
    print(t)
    #s = tokenizer.decode_single_token_bytes(104145)
    s = tokenizer.decode_single_token_bytes(104145)
    print(s)
    s = tokenizer.decode_single_token_bytes(243)
    print(s)
    bstr = b'\xe9\xa5'
    bstr += b'\x95'
    s = decodeUtf8(bstr)
    if s:
        print(s)
    else:
        print("decode error")

if __name__ == "__main__":
    model, tokenizer = loadModel()
    chatLoop(model, tokenizer)

    #test_tokenizer()
