import os

import matplotlib.pyplot as plt
import requests
import torch
from hyperopt import hp
from torch.optim.adamw import AdamW

from models import GPT
from tokenizers import Tokenizer
from utils import BatchLoader, architecture_search, get_model_with_importances, save, train_loop

# hyperparameters
num_trials = 500
batch_size = 16  # number of independent sequences that'll be processed in parallel
block_size = 128 # maximum context length for the preds
max_iters = 1000
eval_interval = 200
learning_rate = 3e-4
device = (
    "mps"
    if torch.backends.mps.is_available()
    else ("cuda:0" if torch.cuda.is_available() else "cpu")
)
eval_iters = 200
n_embd = 384
n_head = 4
n_blocks = 6
dropout = 0.2
# --------------

torch.manual_seed(1337)

if not os.path.exists("tinyshakespeare.txt"):
    import requests

    dataset_url = "https://gist.github.com/alperiox/1b85fb55ac6d39e513b8de5617ce1898/raw/546439f414a887d31d6034e45c35ab57e724d540/tinyshakespeare.txt"

    r = requests.get(dataset_url)
    with open("tinyshakespeare.txt", "wb") as f:
        f.write(r.content)

# data preparation
text = open("tinyshakespeare.txt", "r").read()
# set up the vocabulary
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
tokenizer = Tokenizer(chars)

data = torch.tensor(tokenizer.encode(text), dtype=torch.long)

n = int(0.9 * len(data))  # first 90% will be the training set
n1 = int(
    0.95 * len(data)
)  # 90-95% will be the validation set and the last 5% will be the calibration set for the paper

train_data = data[:n]
val_data = data[n:n1]
calibrate_data = data[n1:]

train_loader = BatchLoader(train_data, block_size, batch_size, device, name="train")
val_loader = BatchLoader(val_data, block_size, batch_size, device, name="val")
calibration_loader = BatchLoader(
    calibrate_data, block_size, batch_size, device, name="calibrate"
)
model = GPT(vocab_size, block_size, n_embd, n_head, n_blocks, device, dropout)
model.to(device)

optimizer = AdamW(model.parameters(), lr=learning_rate)

training_losses = train_loop(
    model,
    optimizer,
    vocab_size,
    train_loader,
    [train_loader, val_loader],
    max_iters,
    eval_interval,
    eval_iters,
)

print("training is done!")

plt.title("training losses")
plt.plot(training_losses)
plt.savefig("training_losses.png")

idx = torch.zeros((1, 1), dtype=torch.long, device=device)
print(tokenizer.decode(model.generate(idx, max_new_tokens=500)[0].tolist()))

model_params = {
    "params": {
    "vocab_size": vocab_size,
    "block_size": block_size,
    "n_embd": n_embd,
    "n_head": n_head,
    "n_blocks": n_blocks,
    "dropout": dropout,
    "device": device,
    }
}

save(model, tokenizer, model_params, "model")

print("-" * 50)
print("The experiment is going to start soon!")

model, num_params = get_model_with_importances(device, "model", calibration_loader, batch_size, block_size)

training_arguments = {
    "optimizer": optimizer,
    "vocab_size": vocab_size,
    "train_loader": calibration_loader,
    "batch_loaders": [calibration_loader, val_loader],
    "max_iters": 200,  # 64 batchsize x 200 max iters = 12800 token for the retraining
    "teacher_model": model,
    "eval_interval": 50,
    "eval_iters": 50,
}

upper_bound = int(num_params*.55)
lower_bound = int(num_params*.45)

ratios = [k*.1 for k in range(10)]

search_space = hp.choice(
    "parameters",
    [
        (
            upper_bound,
            lower_bound,
            [
                ("width_head", hp.choice("head_ratio", ratios)),
                ("width_neuron", hp.choice("neuron_ratio", ratios)), 
                ("width_embedding", hp.choice("embedding_ratio", ratios)),
                ],
            training_arguments,
        )
    ],
)


results_df, best_strategy = architecture_search(search_space, num_evals=num_trials)

del best_strategy["parameters"]

model_params["optimal_pruning_strategy"] = best_strategy

print(results_df.head())

idx = torch.zeros((1, 1), dtype=torch.long, device=device)
print(tokenizer.decode(model.generate(idx, max_new_tokens=500)[0].tolist()))






