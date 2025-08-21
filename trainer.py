import math
import os

import datasets
import numpy as np
import torch
from torch import nn, optim
from torch.distributed import init_process_group
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from preprocess_data import ARTICLE_TOKENS, TITLE_TOKENS
from tokenizer import Tokenizer
from transformer import StdDETransformer


def pad_to_size(list_to_pad: [int], target_size: int):
    current_size = len(list_to_pad)
    return np.pad(list_to_pad, (0, target_size - current_size), constant_values=Tokenizer.PAD_IDX)


def ddp_setup(rank: int, world_size: int):
    """
    Args:
        rank: Unique identifier of each process
       world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    torch.cuda.set_device(rank)
    init_process_group(backend="nccl", rank=rank, world_size=world_size)


class StandardTrainer:
    def __init__(self, net: StdDETransformer, device, tokenizer: Tokenizer, context_window: int = 1024,
                 batch_size: int = 64, training_size: int = 50, epochs: int = 50, writer: SummaryWriter = None):
        super().__init__()
        self.net = net
        self.optimizer = optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-5)
        # self.optimizer = optim.SGD(net.parameters(), lr=1e-3, momentum=0.1)

        self.loss_algorithm = nn.CrossEntropyLoss(ignore_index=Tokenizer.PAD_IDX)
        self.device = device
        self.tokenizer = tokenizer
        self.context_window = context_window
        self.max_tokens = self.context_window - 4
        self.batch_size = batch_size
        self.training_size = training_size
        self.epochs = epochs
        self.writer: SummaryWriter = writer

        # self.scheduler = transformers.get_linear_schedule_with_warmup(
        #     self.optimizer,
        #     self.training_size,
        #     self.epochs * self.training_size
        # )

    def train_start(self):
        self.net.to(self.device)
        self.net.train()

    def train_step(self, encoder_input: torch.Tensor, decoder_input: torch.Tensor, decoder_output: torch.Tensor):
        # Pass to GPU if available.
        encoder_input, decoder_input, decoder_output = encoder_input.to(self.device), decoder_input.to(
            self.device), decoder_output.to(self.device)

        # Zero out the gradients of the optimizer
        self.optimizer.zero_grad()

        outputs = self.net(encoder_input, decoder_input)
        loss = self.loss_algorithm(
            outputs.view(-1, outputs.size(-1)),
            decoder_output.view(-1)
        )

        # print(f"[AFTER] Loss: {loss.data.item()}")

        # Compute the loss gradient using the backward method and have the optimizer take a step
        loss.backward()
        self.optimizer.step()
        # self.scheduler.step()

        out_classes = outputs.argmax(-1)
        expected_classes = decoder_output

        diff_matrix = (expected_classes - out_classes)

        print(diff_matrix)

        return loss.item()

    def eval_step(self, encoder_input: torch.Tensor, decoder_input: torch.Tensor, decoder_output: torch.Tensor):
        # Pass to GPU if available.
        encoder_input, decoder_input, decoder_output = encoder_input.to(self.device), decoder_input.to(
            self.device), decoder_output.to(self.device)

        # Get the outputs of your model and compute your loss
        outputs = self.net(encoder_input, decoder_input)

        out_classes = outputs.argmax(-1)
        expected_classes = decoder_output

        diff_matrix = (expected_classes - out_classes)

        print(diff_matrix)
        return self.loss_algorithm(
            outputs.view(-1, self.net.vocab_size),
            decoder_output.view(-1)
        )

    def save_model(self):
        saved_models_folder = "./saved_models"
        if not os.path.exists(saved_models_folder):
            os.makedirs(saved_models_folder)

        torch.save(self.net.state_dict(), f"{saved_models_folder}/encoder_decoder_std.pt")

    def eval_ds(self, input_tokens: [int], output_tokens: [int]):
        eval_loss = 0
        with torch.no_grad():
            decoder_input_batch, decoder_output_batch, encoder_input_batch = self.prepare_batches(input_tokens, output_tokens)

            if len(encoder_input_batch) == 0:
                return 0

            for batchIdx, encoder_input in enumerate(encoder_input_batch):
                eval_loss += self.eval_step(
                    encoder_input,
                    decoder_input_batch[batchIdx],
                    decoder_output_batch[batchIdx]
                )
        return eval_loss

    def train_end(self):
        self.net.eval()
        print("Finished training")

    def get_padded_tensor(self, list_to_pad: [int]):
        return torch.tensor(pad_to_size(list_to_pad, self.context_window), dtype=torch.long)

    def train_ds(self, ds: datasets.Dataset):
        self.train_start()
        train_loss_history = list()
        train_val_history = list()
        lowest_epoch_loss = 10000
        lowest_val_loss = 10000

        for epoch in tqdm(range(self.epochs), desc="Epoch", position=0):

            ds = ds.shuffle()
            training_ds = ds["train"]
            cutoff_idx = math.floor(len(training_ds) * 0.8)
            train_ds = training_ds[:cutoff_idx]
            eval_ds = training_ds[cutoff_idx:]

            self.net.train()
            train_loss = 0.0
            train_length = len(train_ds[TITLE_TOKENS])
            for idx in tqdm(range(train_length), desc="Training", position=0):
                output_tokens = train_ds[ARTICLE_TOKENS][idx]
                input_tokens = train_ds[TITLE_TOKENS][idx]
                decoder_input_batch, decoder_output_batch, encoder_input_batch = self.prepare_batches(input_tokens, output_tokens)

                if len(encoder_input_batch) == 0:
                    continue

                for batchIdx, encoder_input in enumerate(encoder_input_batch):
                    train_loss += self.train_step(
                        encoder_input,
                        decoder_input_batch[batchIdx],
                        decoder_output_batch[batchIdx]
                    )

            self.net.eval()
            val_loss = 0
            validation_length = len(eval_ds[TITLE_TOKENS])
            for idx in tqdm(range(validation_length), desc="Validation", position=0):
                output_tokens = eval_ds[ARTICLE_TOKENS][idx]
                input_tokens = eval_ds[TITLE_TOKENS][idx]
                val_loss += self.eval_ds(input_tokens, output_tokens)

            epoch_loss = train_loss / train_length
            if self.writer is not None:
                self.writer.add_scalar("Training Loss", epoch_loss, epoch)

            if epoch_loss < lowest_epoch_loss:
                lowest_epoch_loss = epoch_loss
                print(f"\nLowest Epoch {epoch} loss is: {lowest_epoch_loss}.")

            self.log_parameters("network", self.net, epoch)

            val_loss /= validation_length
            if self.writer is not None:
                self.writer.add_scalar("Validation Loss", val_loss, epoch)
            if val_loss < lowest_val_loss:
                lowest_val_loss = val_loss
                print(f"\nLowest Validation {epoch} loss is: {lowest_val_loss}. Saving model...")
                self.save_model()

            train_loss_history.append(epoch_loss)
            train_val_history.append(val_loss)

        self.train_end()

        return train_loss_history, train_val_history

    def prepare_batches(self, input_tokens, output_tokens):
        encoder_input_batch = []
        decoder_input_batch = []
        decoder_output_batch = []
        for batchIdx in range(0, math.ceil(len(output_tokens) / self.max_tokens)):
            train_tokens = output_tokens[
                           batchIdx * self.max_tokens:min((batchIdx + 1) * self.max_tokens, len(output_tokens))
                           ]

            encoder_input_batch.append(self.get_padded_tensor(
                input_tokens + Tokenizer.END
            ))

            decoder_input_batch.append(self.get_padded_tensor(
                Tokenizer.START + train_tokens + Tokenizer.END
            ))

            decoder_output_batch.append(self.get_padded_tensor(
                train_tokens + Tokenizer.END
            ))
        encoder_input_batch = torch.stack(encoder_input_batch).split(self.batch_size)
        decoder_input_batch = torch.stack(decoder_input_batch).split(self.batch_size)
        decoder_output_batch = torch.stack(decoder_output_batch).split(self.batch_size)
        return decoder_input_batch, decoder_output_batch, encoder_input_batch

    def log_parameters(self, parent_name: str, current_module: nn.Module, epoch: int):
        for param_name, param in current_module.named_parameters():
            self.writer.add_histogram(parent_name + "/" + param_name, param, epoch)
            self.writer.add_histogram(parent_name + "/" + param_name + "/grad", param.grad, epoch)
