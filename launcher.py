import datetime
import os.path

import torch
from datasets import load_dataset
from torch.utils.tensorboard import SummaryWriter

from tokenizer import Tokenizer
from trainer import StandardTrainer, pad_to_size
from transformer import StdDETransformer

ds = load_dataset('json', data_files={
    "train": "tokens.json"
})

board_writer = SummaryWriter(f'runs/de_tf/{datetime.datetime.now().isoformat().replace(":", "-")}')

vocab_size = 100
context_window = 64

tokenizer = Tokenizer(vocab_size)

torch.autograd.set_detect_anomaly(True)

device = "cpu"
if torch.cuda.is_available():
    print("CUDA device selected")
    device = torch.cuda.current_device()

debug = False

network = StdDETransformer(
    vocab_size=vocab_size,
    d_embedding=64,
    max_context_window=context_window,
    transformer_depth=1,
    dropout=0.1
)

saved_models_folder = "./saved_models"

if not os.path.isfile("./saved_models/encoder_decoder_std.pt"):
    training_loss_history, training_val_history = StandardTrainer(
        network,
        device,
        tokenizer,
        context_window=context_window,
        epochs=30,
        batch_size=64,
        training_size=10,
        writer=board_writer
    ).train_ds(ds=ds)

network.load_state_dict(torch.load(f"{saved_models_folder}/encoder_decoder_std.pt", weights_only=True))
network.to(device)
network.eval()


def generate_padding_mask(sequence, pad_token=0):
    # Mask out padding tokens (assumes pad_token is 0)
    return (sequence != pad_token).unsqueeze(1).unsqueeze(2)


def get_padded_tensor(list_to_pad: [int]):
    return torch.tensor(pad_to_size(list_to_pad, context_window), dtype=torch.long)


def infer_text(input_text):
    with torch.no_grad():
        input_tokens = tokenizer.tokenize(input_text, with_sod=False) + Tokenizer.END

        encoder_input_batch = get_padded_tensor(input_tokens).view(1, -1).to(device)

        collected_tokens: [int] = [Tokenizer.START_IDX]
        idx = 0
        while Tokenizer.END_IDX not in collected_tokens and not idx >= 10:
            print(
                f"Iteration {idx}, Decoder gets:{tokenizer.textual(collected_tokens)}, Encoder get: {tokenizer.textual(encoder_input_batch.flatten().cpu().numpy())}")
            decoder_input_batch = torch.tensor([collected_tokens], dtype=torch.long, device=device)
            decoder_output = network(encoder_input_batch, decoder_input_batch)

            board_writer.add_graph(network, [encoder_input_batch, decoder_input_batch])
            board_writer.flush()

            all_tokens = decoder_output.argmax(dim=-1).flatten().cpu().numpy()
            new_token = all_tokens[-1]
            collected_tokens.append(new_token)
            print(
                f"Iteration {idx} new token: {tokenizer.textual([new_token])}, full output: {tokenizer.textual(all_tokens)}")

            idx += 1

        return tokenizer.decode(collected_tokens)


while True:
    query = input("Enter a text or write 'exit' to stop session: ")
    if query == "exit":
        break
    else:
        print(f"Network wrote: {infer_text(query)}")
