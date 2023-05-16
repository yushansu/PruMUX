from dis import dis
from multiprocessing import reduction
import numpy as np
import torch
from torch import nn
from torch import import_ir_module
import torch.nn.functional as F
from transformers import (
    DataCollatorForLanguageModeling,
)
from dataclasses import dataclass
import tracemalloc
from transformers.utils import logging
from transformers.modeling_outputs import ModelOutput
from typing import Optional, Tuple
from transformers import ElectraModel, ElectraPreTrainedModel
from transformers.models.electra.modeling_electra import (
    ElectraDiscriminatorPredictions,
    ElectraForPreTrainingOutput,
    ElectraClassificationHead,
    ElectraGeneratorPredictions,
    ElectraLayer,
)
import math
from transformers.activations import gelu
import time
from utils.utils_pretraining import (
    random_encoding,
    binary_encoding,
)
from models.multiplexing import SequenceClassifierOutputMuxed
from scipy.stats import ortho_group, special_ortho_group
from utils.utils_pretraining import gen_attn_mask

logger = logging.get_logger(__name__)

class DataCollatorForLanguageModelingMuxed(DataCollatorForLanguageModeling):
    def __init__(self, tokenizer, mlm_probability=0.15, max_pad_tokens=0):
        super().__init__(tokenizer, mlm_probability=mlm_probability)
        self.max_pad_tokens = max_pad_tokens
    
    def mask_tokens(
        self, inputs: torch.Tensor, special_tokens_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()
        # potentially add pad tokens at the end of the sequence
        batch, seq_len = labels.shape
        pad_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)
        pad_lens = torch.randint(0, self.max_pad_tokens + 1, (batch,))
        non_pad_lens = seq_len - pad_lens
        non_pad_attn_mask = gen_attn_mask(non_pad_lens, seq_len)
        pad_attn_mask = ~non_pad_attn_mask
        
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        masked_indices = masked_indices & non_pad_attn_mask
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]
        
        inputs[pad_attn_mask] = pad_token_id
        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels    

class DataCollatorElectra(DataCollatorForLanguageModeling):
    def __init__(self, tokenizer, mlm_probability=0.15, max_pad_tokens=0):
        super().__init__(tokenizer, mlm_probability=mlm_probability)
        self.max_pad_tokens = max_pad_tokens

    def mask_tokens(
        self, inputs: torch.Tensor, special_tokens_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(
                    val, already_has_special_tokens=True
                )
                for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        # potentially add pad tokens at the end of the sequence
        batch, seq_len = labels.shape
        pad_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)
        pad_lens = torch.randint(0, self.max_pad_tokens + 1, (batch,))
        non_pad_lens = seq_len - pad_lens
        non_pad_attn_mask = gen_attn_mask(non_pad_lens, seq_len)
        pad_attn_mask = ~non_pad_attn_mask

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        masked_indices = masked_indices & non_pad_attn_mask
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 85% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = (
            torch.bernoulli(torch.full(labels.shape, 0.85)).bool() & masked_indices
        )
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.mask_token
        )

        inputs[pad_attn_mask] = pad_token_id
        # 10% of the time, we replace masked input tokens with random word
        # indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        # random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        # inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels


class DataCollatorForLanguageModelingMultiSentenceSingleMask(
    DataCollatorForLanguageModeling
):
    def __init__(self, tokenizer, mlm_probability=0.15, num_instances=5):
        super().__init__(tokenizer, mlm_probability=mlm_probability)
        self.num_instances = num_instances

    def mask_tokens(self, inputs, special_tokens_mask=None):
        """
        Override default behavior for multisentence, mask num_instances * mlm_probability number of tokens for batches of N sentences
        Steps:
        1) Divide the batch into chunks of N sentences
        2) For each chunk, ensure we are only masking only a single word in all the sentences ie. 1 single token in a column
        """
        batch, L = inputs.shape
        assert self.mlm_probability * self.num_instances <= 1
        if batch % self.num_instances != 0:
            logger.warning(
                "Can't create equal sized chunks, might be the last batch in training/evaluation"
            )
        # calculate the number of chunks
        num_chunks = math.ceil(batch / self.num_instances)
        # calculate masked positions for each of the chunks
        probability_matrix = torch.full(
            (num_chunks, L), self.mlm_probability * self.num_instances
        )
        masked_pos = torch.bernoulli(probability_matrix).bool()
        # now within each chunk, we need to assign each masked position to a sentence in the chunk
        # sample integers from 0 to N -1 and add a mulitple of N
        batch_indices = torch.randint(self.num_instances, probability_matrix.shape)
        # the last chunk can have less N sentences, clamp the value to less than the batch size
        batch_indices = batch_indices + (
            torch.arange(num_chunks) * self.num_instances
        ).unsqueeze(1)
        batch_indices = torch.clamp(batch_indices, max=batch - 1)
        all_masked_indices = torch.full((batch, L), False)

        all_masked_indices[
            batch_indices[masked_pos], masked_pos.nonzero(as_tuple=False)[:, -1]
        ] = True

        # 90% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        labels = torch.full((batch, L), -100)
        labels = inputs.clone()
        labels[~all_masked_indices] = -100
        indices_replaced = (
            torch.bernoulli(torch.full((batch, L), 0.8)).bool() & all_masked_indices
        )
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.mask_token
        )

        # 10% of the time, we replace masked input tokens with random word
        indices_random = (
            torch.bernoulli(torch.full((batch, L), 0.5)).bool()
            & all_masked_indices
            & ~indices_replaced
        )
        random_words = torch.randint(
            len(self.tokenizer), labels.shape, dtype=torch.long
        )
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels


class DataCollatorForLanguageModelingMultiSentenceSingleMaskEval(
    DataCollatorForLanguageModeling
):
    def __init__(self, tokenizer, mlm_probability=0.15, num_instances=5):
        super().__init__(tokenizer, mlm_probability=mlm_probability)
        self.num_instances = num_instances

    def mask_tokens(self, inputs, special_tokens_mask=None):
        """
        Override default behavior for multisentence, mask num_instances * mlm_probability number of tokens for batches of N sentences
        Steps:
        1) Divide the batch into chunks of N sentences
        2) For each chunk, ensure we are only masking only a single word in all the sentences ie. 1 single token in a column
        """
        batch, L = inputs.shape
        assert self.mlm_probability * self.num_instances <= 1
        if batch % self.num_instances != 0:
            logger.warning(
                "Can't create equal sized chunks, might be the last batch in training/evaluation"
            )
        # calculate the number of chunks
        num_chunks = math.ceil(batch / self.num_instances)
        # calculate masked positions for each of the chunks
        probability_matrix = torch.full(
            (num_chunks, L), self.mlm_probability * self.num_instances
        )
        masked_pos = torch.bernoulli(probability_matrix).bool()
        # now within each chunk, we need to assign each masked position to a sentence in the chunk
        # sample integers from 0 to N -1 and add a mulitple of N
        batch_indices = torch.randint(self.num_instances, probability_matrix.shape)
        # the last chunk can have less N sentences, clamp the value to less than the batch size
        batch_indices = batch_indices + (
            torch.arange(num_chunks) * self.num_instances
        ).unsqueeze(1)
        batch_indices = torch.clamp(batch_indices, max=batch - 1)
        all_masked_indices = torch.full((batch, L), False)

        all_masked_indices[
            batch_indices[masked_pos], masked_pos.nonzero(as_tuple=False)[:, -1]
        ] = True

        labels = torch.full((batch, L), -100)
        labels = inputs.clone()
        labels[~all_masked_indices] = -100
        indices_replaced = (
            torch.bernoulli(torch.full((batch, L), 1.0)).bool() & all_masked_indices
        )
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.mask_token
        )

        return inputs, labels


class DistillationModel(nn.Module):
    def __init__(
        self,
        student,
        student_config,
        tokenizer,
        teacher=None,
        teacher_config=None,
        alpha_cos=0.0,
        alpha_mse=0.0,
        alpha_ce=0.0,
        temperature=1.0,
    ):
        super().__init__()
        self.student, self.teacher = student, teacher
        self.tokenizer = tokenizer
        self.student_config = student_config
        self.generator_config = teacher_config
        self.cos_loss_fc = nn.CosineEmbeddingLoss(reduction="mean")
        self.mse_loss_fc = nn.MSELoss(reduction="sum")
        self.ce_loss_fc = nn.KLDivLoss(reduction="batchmean")
        self.alpha_cos = alpha_cos
        self.alpha_mse = alpha_mse
        self.alpha_ce = alpha_ce
        self.temperature = temperature

    def forward(
        self, input_ids=None, attention_mask=None, labels=None, return_dict=None
    ):
        student_outputs = self.student(
            input_ids, attention_mask=attention_mask, labels=labels, return_dict=True
        )
        s_logits, s_hidden_states = (
            student_outputs["logits"],
            student_outputs["hidden_states"],
        )
        with torch.no_grad():
            teacher_outputs = self.teacher(
                input_ids,
                attention_mask=attention_mask,
                labels=labels,
                return_dict=True,
                output_hidden_states=True,
            )
            t_logits, t_hidden_states = (
                teacher_outputs["logits"],
                teacher_outputs["hidden_states"],
            )
            t_hidden_states = t_hidden_states[-1]
        # exxtract the cls hidden state from the student and teacher
        s_hidden_states = s_hidden_states[:, 0, :]
        t_hidden_states = t_hidden_states[:, 0, :]
        assert s_logits.shape == t_logits.shape
        assert len(s_logits.shape) == 2
        loss_ce, loss_mse, loss_cos = None, None, None
        loss_ce = self.alpha_ce * self.ce_loss_fc(
            nn.functional.log_softmax(s_logits / self.temperature, dim=-1),
            nn.functional.softmax(t_logits / self.temperature, dim=-1),
        )
        loss_ce = loss_ce * self.temperature * self.temperature
        loss = self.alpha_ce * loss_ce
        if self.alpha_mse > 0.0:
            loss_mse = self.mse_loss_fc(s_logits, t_logits) / s_logits.size(
                0
            )  # Reproducing batchmean reduction
            loss = loss + self.alpha_mse * loss_mse
        if self.alpha_cos > 0.0:
            # TODO break point here, check dimensions
            assert s_hidden_states.size() == t_hidden_states.size()
            bs, _ = s_hidden_states.shape
            target = torch.ones(bs, device=input_ids.device).long()  # (bs,)
            loss_cos = self.cos_loss_fc(s_hidden_states, t_hidden_states, target)
            loss = loss + self.alpha_cos * loss_cos

        # add retrieval loss is model is muxed
        if "retrieval_loss" in student_outputs:
            loss += 0.2 * student_outputs["retrieval_loss"]
        # visualize retrieval loss
        return DistillationOutput(
            loss=loss,
            logits=s_logits,
            loss_ce=loss_ce,
            loss_mse=loss_mse,
            loss_cos=loss_cos,
            retrieval_loss=student_outputs["retrieval_loss"]
            if "retrieval_loss" in student_outputs
            else None,
        )


class ELECTRAModel(nn.Module):
    def __init__(
        self,
        discriminator,
        discriminator_config,
        tokenizer,
        generator=None,
        generator_config=None,
        loss_weights=(1.0, 50.0),
    ):
        super().__init__()
        self.generator, self.discriminator = generator, discriminator
        # self.gumbel_dist = torch.distributions.gumbel.Gumbel(0.0, 1.0)
        self.tokenizer = tokenizer
        self.discriminator_config = discriminator_config
        self.generator_config = generator_config
        self.gen_loss_fc = nn.CrossEntropyLoss()
        self.disc_loss_fc = nn.BCEWithLogitsLoss()
        self.loss_weights = loss_weights
        self.snapshots = []
        # tracemalloc.start()

    def collect_stats(self):
        self.snapshots.append(tracemalloc.take_snapshot())
        if len(self.snapshots) > 1:
            stats = self.snapshots[-1].compare_to(self.snapshots[-2], "filename")
            for stat in stats[:10]:
                print(
                    "{} new KiB {} total KiB {} new {} total memory blocks: ".format(
                        stat.size_diff / 1024,
                        stat.size / 1024,
                        stat.count_diff,
                        stat.count,
                    )
                )
            for line in stat.traceback.format():
                print(line)

    def forward(
        self, input_ids=None, attention_mask=None, labels=None, return_dict=None
    ):
        mask_token = 103
        is_mlm_applied = input_ids == mask_token
        mlm_gen_logits = None
        gen_loss = None
        gen_retrieval_loss = None
        gen_mlm_loss = None
        if self.generator is not None:
            # s_t = time.time()
            # self.collect_stats()
            gen_outputs = self.generator(
                input_ids,
                attention_mask=attention_mask,
                labels=labels,
                return_dict=True,
            )
            # (B, L, vocab size)
            # reduce size to save space and speed
            gen_logits = gen_outputs["logits"]
            mlm_gen_logits = gen_logits[
                is_mlm_applied, :
            ]  # ( #mlm_positions, vocab_size)
            # loss calculation
            gen_loss = gen_outputs["loss"]
            gen_retrieval_loss = None
            gen_mlm_loss = None
            if "retrieval_loss" in gen_outputs:
                gen_retrieval_loss = gen_outputs["retrieval_loss"]
            if "task_loss" in gen_outputs:
                gen_mlm_loss = gen_outputs["task_loss"]
            with torch.no_grad():
                # sampling
                # sample_start = time.time()
                pred_toks = self.sample(mlm_gen_logits)  # ( #mlm_positions, )
                # produce inputs for discriminator
                # logger.warn("Sampling took %.2f seconds" % (time.time() - sample_start))
                generated = input_ids.clone()  # (B,L)
                generated[is_mlm_applied] = pred_toks  # (B,L)
                # produce labels for discriminator
                is_replaced = is_mlm_applied.clone()  # (B,L)
                is_replaced[is_mlm_applied] = (
                    pred_toks != labels[is_mlm_applied]
                )  # (B,L)
            # logger.warn(f"Generator time: {time.time() - s_t}")
        else:
            # replace the masked tokens with other random tokens from the vocab
            generated = input_ids.clone()  # (B,L)
            rand_toks = torch.randint_like(
                generated, self.discriminator_config.vocab_size
            )
            generated[is_mlm_applied] = rand_toks[is_mlm_applied]  # (B,L)
            is_replaced = is_mlm_applied.clone()  # (B,L)
        # pass the generated tokens to the discriminator only if loss weight is non-zero
        discriminator_hidden_states = None
        if self.loss_weights[1] > 0:
            # s_t = time.time()
            outputs = self.discriminator(generated, attention_mask, return_dict=True)
            disc_logits = outputs["logits"]
            discriminator_hidden_states = outputs["hidden_states"] if "hidden_states" in outputs else None
            retrieval_loss = (
                None if "retrieval_loss" not in outputs else outputs["retrieval_loss"]
            )
            if self.discriminator_config.demuxing_variant == "index":
                attention_mask = attention_mask[:, : -(self.discriminator_config.num_instances + 1)]
                is_replaced = is_replaced[:, : -(self.discriminator_config.num_instances + 1)]
            disc_logits_flat = disc_logits.masked_select(
                attention_mask.bool()
            )  # -> 1d tensor
            is_replaced_flat = is_replaced.masked_select(attention_mask.bool())  # -> 1d
            disc_loss = self.disc_loss_fc(
                disc_logits_flat.float(), is_replaced_flat.float()
            )
            # add retrieval loss here
            disc_retrieval_loss = (
                self.discriminator_config.retrieval_loss_coeff * retrieval_loss
                + self.discriminator_config.task_loss_coeff * disc_loss
                if retrieval_loss is not None
                else disc_loss
            )
            # logger.warn(f"Discriminator time: {time.time() - s_t}")
        else:
            disc_logits = None
            retrieval_loss = None
            disc_logits = None
            disc_loss = None
            disc_retrieval_loss = 0

        loss = (
            gen_loss * self.loss_weights[0] + disc_retrieval_loss * self.loss_weights[1]
            if gen_loss is not None
            else disc_retrieval_loss * self.loss_weights[1]
        )

        if not return_dict:
            return (
                loss,
                disc_loss,
                gen_loss,
                retrieval_loss,
                gen_retrieval_loss,
                gen_mlm_loss,
                disc_logits,
                None,  # mlm_gen_logits,
                generated,
                is_replaced,
            )
        return ElectraOutput(
            loss=loss,
            disc_loss=disc_loss,
            gen_loss=gen_loss,
            retrieval_loss=retrieval_loss,
            gen_retrieval_loss=gen_retrieval_loss,
            gen_mlm_loss=gen_mlm_loss,
            disc_logits=disc_logits,
            # mlm_gen_logits=mlm_gen_logits,
            sampled_input_ids=generated,
            corruption_applied=is_replaced,
            hidden_states=discriminator_hidden_states,
        )

    def sample(self, logits):
        # return torch.multinomial(F.softmax(logits,1), 1).squeeze(1)
        return F.gumbel_softmax(logits).argmax(dim=-1)
        # gumbel = self.gumbel_dist.sample(logits.shape).to(logits.device)
        # return (logits.float() + gumbel).argmax(dim=-1)


@dataclass
class ElectraOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    disc_loss: Optional[torch.FloatTensor] = None
    gen_loss: Optional[torch.FloatTensor] = None
    retrieval_loss: Optional[torch.FloatTensor] = None
    gen_retrieval_loss: Optional[torch.FloatTensor] = None
    gen_mlm_loss: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    disc_logits: torch.FloatTensor = None
    mlm_gen_logits: torch.FloatTensor = None
    sampled_input_ids: torch.LongTensor = None
    corruption_applied: torch.BoolTensor = None


@dataclass
class DistillationOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    loss_ce: Optional[torch.FloatTensor] = None
    loss_mse: Optional[torch.FloatTensor] = None
    loss_cos: Optional[torch.FloatTensor] = None
    retrieval_loss: Optional[torch.FloatTensor] = None


class MuxedElectraForPreTraining(ElectraPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.config = config

        self.num_instances = config.num_instances
        self.muxing_variant = config.muxing_variant
        self.demuxing_variant = config.demuxing_variant
        self.retrieval_loss_coeff = config.retrieval_loss_coeff
        self.task_loss_coeff = config.task_loss_coeff

        self.electra = ElectraModel(config)

        if config.demuxing_variant == "index":
            self.demultiplexer = IndexDemultiplexerTokenLevel(config)
            self.retrieval_head = RetrievalHeadIndexDemultiplexing(
                config
            )
        elif config.demuxing_variant == "mlp":
            self.demux_module = MLPDemuxModule(config)
            self.demultiplexer = MLPDemultiplexerTokenLevel(config, self.demux_module)
            self.retrieval_head = RetrievalHeadMLPDemultiplexing(
                config, self.demux_module
            )
        elif config.demuxing_variant == "index_pos":
            self.demux_module = IndexPosDemuxModule(config)
            self.demultiplexer = IndexPosDemultiplexerTokenLevel(
                config, self.demux_module
            )
            self.retrieval_head = RetrievalHeadIndexPosDemultiplexing(
                config, self.demux_module
            )
        else:
            raise NotImplementedError()

        self.init_weights()

        self.discriminator_predictions = ElectraDiscriminatorPredictions(config)

        # multiplexing initialization

        d_model = config.embedding_size
        instance_embedding = None

        if self.muxing_variant == "gaussian_hadamard":
            instance_embedding = random_encoding(
                self.num_instances, d_model, norm=config.gaussian_hadamard_norm
            )
        elif self.muxing_variant == "random_ortho":
            instance_embedding = [
                torch.from_numpy(ortho_group.rvs(config.hidden_size)).float()
                for _ in range(self.num_instances)
            ]
            instance_embedding = torch.stack(instance_embedding, dim=0)

        elif self.muxing_variant == "random_ortho_low_rank":
            # create low rank matrix
            instance_embedding = []
            H_ = special_ortho_group.rvs(dim=config.hidden_size)
            U_ = special_ortho_group.rvs(dim=config.hidden_size)
            for i in range(self.num_instances):
                G_ = np.zeros((config.hidden_size, config.hidden_size))
                l = i * (config.hidden_size // self.num_instances)
                r = l + (config.hidden_size // self.num_instances)
                H_copy = H_.copy()
                G_[:, l:r] = H_copy[:, l:r]
                G_ = U_ @ G_.T
                instance_embedding.append(torch.from_numpy(G_).float())
            instance_embedding = torch.stack(instance_embedding, dim=0)

        elif self.muxing_variant == "binary_hadamard":
            instance_embedding = binary_encoding(
                self.num_instances, d_model, epsilon=config.binary_hadamard_epsilon
            )
        elif self.muxing_variant == "gaussian_attention":
            instance_embedding = random_encoding(
                self.num_instances, d_model, norm=config.gaussian_hadamard_norm
            )
            self.muxing_attention = ElectraLayer(config)
            self.cross_instances_linear = nn.Linear(config.embedding_size, d_model)
            self.cross_instances_layernorm = nn.LayerNorm(d_model)

        else:
            raise NotImplementedError()

        if instance_embedding is not None:
            self.instance_embedding = torch.nn.Parameter(instance_embedding)
        else:
            instance_embedding = random_encoding(
                self.num_instances, d_model, norm=self.gaussian_hadamard_norm
            )

        if not config.learn_muxing:
            self.instance_embedding.requires_grad = False
        else:
            self.instance_embedding.requires_grad = True

        # Initialize weights and apply final processing
        # self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        labels=None,
        return_dict=None,
    ):

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # get input embeddings and average over N instances
        input_shape = input_ids.size()

        batch_size, seq_length = input_shape
        num_instances = self.num_instances
        past_key_values_length = 0

        modified_batch_size = batch_size // num_instances
        modified_seq_length = None
        special_tokens_end_position = None
        if self.demuxing_variant == "index":

            # add the prefix
            # [CLS1, <s>, <s>, <s>, <s>]
            # [<s>, CLS2, <s>, <s>, <s>]
            # [<s>, <s>, CLS3, <s>, <s>]
            # [<s>, <s>, <s>, CLS4, <s>]
            # [<s>, <s>, <s>, <s>, CLS5]
            # let us just assume the last 5 tokens barring the masked token
            # are the cls tokens (easiest way to make use of existing vocab)

            # prefix 5 x 5

            prefix = torch.full(
                (num_instances, num_instances), 680, device=input_ids.device
            )
            prefix[
                torch.arange(num_instances, device=input_ids.device),
                torch.arange(num_instances, device=input_ids.device),
            ] = (
                -(torch.arange(num_instances, device=input_ids.device) + 2)
                + 680
            )

            # [-2   <s>, <s>, <s>, <s>]
            # [<s>, -3, <s>, <s>, <s>]
            # [<s>, <s>, -4, <s>, <s>]
            # [<s>, <s>, <s>, -5, <s>]
            # [<s>, <s>, <s>, <s>, -6]
            # +  size of vocab
            cls_tokens = torch.full((num_instances, 1), 101, device=input_ids.device)
            prefix = torch.cat([prefix, cls_tokens], dim=1)

            prefix = prefix.repeat(modified_batch_size, 1)
            input_ids = input_ids[: (modified_batch_size * num_instances)]
            input_ids = torch.cat([prefix, input_ids[:, :-(num_instances + 1)]], dim=1)
            modified_seq_length = seq_length
            special_tokens_end_position = num_instances + 1

        elif self.demuxing_variant == "mlp" or self.demuxing_variant == "index_pos":
            cls_tokens = torch.full((num_instances, 1), 101, device=input_ids.device)
            cls_tokens = cls_tokens.repeat(modified_batch_size, 1)
            input_ids = input_ids[: (modified_batch_size * num_instances)]
            input_ids[:, 0:1] = cls_tokens
            modified_seq_length = seq_length
            special_tokens_end_position = 0

        else:
            raise NotImplementedError()
        # concatenate
        embedding_output = self.electra.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )

        _, _, embedding_dim = embedding_output.shape
        if (
            self.muxing_variant == "random_ortho"
            or self.muxing_variant == "random_ortho_low_rank"
        ):
            embedding_output = embedding_output.view(
                modified_batch_size,
                num_instances,
                modified_seq_length,
                embedding_dim,
            )
            embedding_output = torch.matmul(
                self.instance_embedding, embedding_output.permute(0, 1, 3, 2)
            )
            # swap the last 2 dimensions again
            embedding_output = embedding_output.permute(0, 1, 3, 2)
            # average across the instances
            embedding_output = torch.sum(embedding_output, dim=1) / math.sqrt(
                self.num_instances
            )
        elif self.muxing_variant == "gaussian_attention":
            embedding_output_intermediate = embedding_output.view(
                modified_batch_size,
                num_instances,
                modified_seq_length,
                embedding_dim,
            )
            # extract relevant instance embeddings
            instance_embed = self.instance_embedding[:num_instances, :]
            instance_embed = instance_embed.unsqueeze(1).expand(
                num_instances, modified_seq_length, embedding_dim
            )
            embedding_output_intermediate = (
                embedding_output_intermediate * instance_embed.unsqueeze(0)
            )

            embedding_output_cross_instance = torch.mean(
                embedding_output_intermediate, dim=1
            )
            embedding_output_cross_instance = self.cross_instances_linear(
                embedding_output_cross_instance
            )
            embedding_output_cross_instance = gelu(embedding_output_cross_instance)
            embedding_output_cross_instance = self.cross_instances_layernorm(
                embedding_output_cross_instance
            )

            embedding_output_intermediate = embedding_output_intermediate.view(
                modified_batch_size * num_instances, modified_seq_length, embedding_dim
            )
            # pass throughh attention layer
            embedding_output_attention = self.muxing_attention(
                embedding_output_intermediate
            )
            embedding_output_attention = embedding_output_attention[0]
            embedding_output_cross_instance = embedding_output_cross_instance.unsqueeze(
                1
            ).expand(
                modified_batch_size, num_instances, modified_seq_length, embedding_dim
            )
            # average across the instances, and add the cross instance attention
            embedding_output_attention = embedding_output_attention.view(
                modified_batch_size, num_instances, modified_seq_length, embedding_dim
            )
            embedding_output_attention = (
                embedding_output_attention + embedding_output_cross_instance
            )
            embedding_output = torch.mean(embedding_output_attention, dim=1)
        else:
            embedding_output = embedding_output.view(
                modified_batch_size,
                num_instances,
                modified_seq_length,
                embedding_dim,
            )
            # extract relevant instance embeddings
            instance_embed = self.instance_embedding[:num_instances, :]
            instance_embed = instance_embed.unsqueeze(1).expand(
                num_instances, modified_seq_length, embedding_dim
            )
            embedding_output = embedding_output * instance_embed.unsqueeze(0)

            embedding_output = torch.mean(embedding_output, dim=1)

        outputs = self.electra(
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=position_ids,
            inputs_embeds=embedding_output,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        # fancy indexing to get the instance position embedding

        demuxed_sequence_output = self.demultiplexer(sequence_output)
        logits = self.discriminator_predictions(demuxed_sequence_output)
        # retrieval loss calculation
        instance_labels = torch.full(
            (modified_batch_size, modified_seq_length),
            0,
            device=input_ids.device,
        ).long()
        # skip the cls and prefix tokens
        instance_labels[:, special_tokens_end_position:] = torch.randint(
            num_instances,
            (modified_batch_size, modified_seq_length - special_tokens_end_position),
            device=input_ids.device,
        )

        # index into input ids to get the corresponding labels
        input_ids = input_ids.view(modified_batch_size, num_instances, -1)
        input_ids = input_ids.permute(0, 2, 1)

        retrieval_labels = input_ids[
            torch.arange(modified_batch_size, device=input_ids.device)
            .unsqueeze(1)
            .expand(modified_batch_size, modified_seq_length),
            torch.arange(modified_seq_length, device=input_ids.device)
            .unsqueeze(0)
            .expand(modified_batch_size, modified_seq_length),
            instance_labels,
        ]
        retrieval_labels[:, :special_tokens_end_position] = -100

        # pad_mask = retrieval_labels == 1
        # # wipe of 1 - (0.1  *  retrieval percentage) of pad tokens
        # pad_mask_wipe = pad_mask
        # non_pad_mask_wipe = (
        #     ~pad_mask
        #     & torch.bernoulli(
        #         torch.full(
        #             retrieval_labels.shape,
        #             1 - self.config.retrieval_percentage,
        #             device=input_ids.device,
        #         )
        #     ).bool()
        # )
        # retrieval_labels[non_pad_mask_wipe] = -100

        # retrieval_labels[pad_mask_wipe] = -100

        retrieval_predictions = self.retrieval_head(sequence_output, instance_labels)

        retrieval_loss = None
        task_loss = None
        loss = None

        retrieval_loss_fct = nn.CrossEntropyLoss()

        retrieval_loss = retrieval_loss_fct(
            retrieval_predictions.view(-1, self.config.vocab_size),
            retrieval_labels.view(-1),
        )

        if labels is not None:
            labels = labels[: (modified_batch_size * num_instances)]

            loss_fct = nn.BCEWithLogitsLoss()

            if attention_mask is not None:
                loss_fct = nn.BCEWithLogitsLoss()

                active_loss = (
                    attention_mask.view(-1, demuxed_sequence_output.shape[1]) == 1
                )
                active_logits = logits.view(-1, demuxed_sequence_output.shape[1])[
                    active_loss
                ]
                active_labels = labels[active_loss]

                task_loss = loss_fct(active_logits, active_labels.float())
                loss = (self.task_loss_coeff * task_loss) + (
                    self.retrieval_loss_coeff * retrieval_loss
                )

        # make logits align with the length of the input sequence
        logits = logits[:, special_tokens_end_position:]
        if not return_dict:
            output = (logits,)
            return ((loss,) + output) if loss is not None else output

        return MuxedElectraForPreTrainingOutput(
            loss=loss, logits=logits, task_loss=task_loss, retrieval_loss=retrieval_loss, hidden_states=demuxed_sequence_output
        )


class MuxedElectraForSequenceClassification(ElectraPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.config = config

        self.num_instances = config.num_instances
        self.muxing_variant = config.muxing_variant
        self.demuxing_variant = config.demuxing_variant
        self.retrieval_loss_coeff = config.retrieval_loss_coeff
        self.task_loss_coeff = config.task_loss_coeff

        self.head_temperature = config.head_temperature
        self.num_labels = config.num_labels
        self.electra = ElectraModel(config)

        if config.demuxing_variant == "index":
            self.demultiplexer = IndexDemultiplexerTokenLevel(config)
            self.retrieval_head = RetrievalHeadIndexDemultiplexing(
                config
            )
        elif config.demuxing_variant == "mlp":
            self.demux_module = MLPDemuxModule(config)
            self.demultiplexer = MLPDemultiplexerTokenLevel(config, self.demux_module)
            self.retrieval_head = RetrievalHeadMLPDemultiplexing(
                config, self.demux_module
            )
        elif config.demuxing_variant == "index_pos":
            self.demux_module = IndexPosDemuxModule(config)
            self.demultiplexer = IndexPosDemultiplexerTokenLevel(
                config, self.demux_module
            )
            self.retrieval_head = RetrievalHeadIndexPosDemultiplexing(
                config, self.demux_module
            )
        else:
            raise NotImplementedError()

        self.init_weights()

        self.classifier = ElectraClassificationHead(config)

        # multiplexing initialization

        d_model = config.embedding_size
        instance_embedding = None

        if self.muxing_variant == "gaussian_hadamard":
            instance_embedding = random_encoding(
                self.num_instances, d_model, norm=config.gaussian_hadamard_norm
            )
        elif self.muxing_variant == "random_ortho":
            instance_embedding = [
                torch.from_numpy(ortho_group.rvs(config.hidden_size)).float()
                for _ in range(self.num_instances)
            ]
            instance_embedding = torch.stack(instance_embedding, dim=0)
        elif self.muxing_variant == "random_ortho_low_rank":
            # create low rank matrix
            instance_embedding = []
            H_ = special_ortho_group.rvs(dim=config.hidden_size)
            U_ = special_ortho_group.rvs(dim=config.hidden_size)
            for i in range(self.num_instances):
                G_ = np.zeros((config.hidden_size, config.hidden_size))
                l = i * (config.hidden_size // self.num_instances)
                r = l + (config.hidden_size // self.num_instances)
                H_copy = H_.copy()
                G_[:, l:r] = H_copy[:, l:r]
                G_ = U_ @ G_.T
                instance_embedding.append(torch.from_numpy(G_).float())
            instance_embedding = torch.stack(instance_embedding, dim=0)
        elif self.muxing_variant == "binary_hadamard":
            instance_embedding = binary_encoding(
                self.num_instances, d_model, epsilon=config.binary_hadamard_epsilon
            )
        elif self.muxing_variant == "gaussian_attention":
            instance_embedding = random_encoding(
                self.num_instances, d_model, norm=config.gaussian_hadamard_norm
            )
            self.muxing_attention = ElectraLayer(config)
            self.cross_instances_linear = nn.Linear(config.embedding_size, d_model)
            self.cross_instances_layernorm = nn.LayerNorm(d_model)
        else:
            raise NotImplementedError()

        if instance_embedding is not None:
            self.instance_embedding = torch.nn.Parameter(instance_embedding)
        else:
            instance_embedding = random_encoding(
                self.num_instances, d_model, norm=self.gaussian_hadamard_norm
            )

        if not config.learn_muxing:
            self.instance_embedding.requires_grad = False
        else:
            self.instance_embedding.requires_grad = True

        self.head_accuracies = None
        self.head_acc_smooth_coeff = 0.8
        self.head_acc_gamma = 10
        # Initialize weights and apply final processing
        # self.post_init()
        # self.anchor = None

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        labels=None,
        return_dict=None,
    ):

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        dummy_samples_added = 0    
        if input_ids.shape[0] % self.num_instances != 0:
            dummy_samples_added = self.num_instances - (input_ids.shape[0] % self.num_instances)
            pad_input_ids = input_ids[
                torch.randint(
                    0, input_ids.shape[0], (dummy_samples_added,)
                )
            ]
            input_ids = torch.cat([input_ids, pad_input_ids], dim=0)
            pad_attention_mask = attention_mask[
                torch.randint(
                    0, attention_mask.shape[0], (dummy_samples_added,)
                )
            ]
            attention_mask = torch.cat([attention_mask, pad_attention_mask], dim=0)
        # get input embeddings and average over N instances
        input_shape = input_ids.size()

        batch_size, seq_length = input_shape
        num_instances = self.num_instances
        past_key_values_length = 0

        modified_batch_size = batch_size // num_instances
        modified_seq_length = None
        special_tokens_end_position = None
        if self.demuxing_variant == "index":

            # add the prefix
            # [CLS1, <s>, <s>, <s>, <s>]
            # [<s>, CLS2, <s>, <s>, <s>]
            # [<s>, <s>, CLS3, <s>, <s>]
            # [<s>, <s>, <s>, CLS4, <s>]
            # [<s>, <s>, <s>, <s>, CLS5]
            # let us just assume the last 5 tokens barring the masked token
            # are the cls tokens (easiest way to make use of existing vocab)

            # prefix 5 x 5

            prefix = torch.full(
                (num_instances, num_instances), 680, device=input_ids.device
            )
            prefix[
                torch.arange(num_instances, device=input_ids.device),
                torch.arange(num_instances, device=input_ids.device),
            ] = (
                -(torch.arange(num_instances, device=input_ids.device) + 2)
                + 680 
            )

            # [-2   <s>, <s>, <s>, <s>]
            # [<s>, -3, <s>, <s>, <s>]
            # [<s>, <s>, -4, <s>, <s>]
            # [<s>, <s>, <s>, -5, <s>]
            # [<s>, <s>, <s>, <s>, -6]
            # +  size of vocab
            cls_tokens = torch.full((num_instances, 1), 101, device=input_ids.device)
            prefix = torch.cat([prefix, cls_tokens], dim=1)

            prefix = prefix.repeat(modified_batch_size, 1)
            input_ids = input_ids[: (modified_batch_size * num_instances)]
            input_ids = torch.cat([prefix, input_ids], dim=1)
            modified_seq_length = seq_length + num_instances + 1
            special_tokens_end_position = num_instances + 1

        elif self.demuxing_variant == "mlp" or self.demuxing_variant == "index_pos":
            cls_tokens = torch.full((num_instances, 1), 101, device=input_ids.device)
            cls_tokens = cls_tokens.repeat(modified_batch_size, 1)
            input_ids = input_ids[: (modified_batch_size * num_instances)]
            # input_ids = torch.cat([cls_tokens, input_ids], dim=1)
            # modified_seq_length = seq_length + 1
            modified_seq_length = seq_length
            special_tokens_end_position = 1

        else:
            raise NotImplementedError()

        # if self.anchor is None:
        #     self.anchor = input_ids[0:1]

        # anchor_mask = (input_ids == self.anchor).all(dim=1)
        # if anchor_mask.sum() > 0:
        #     logger.warning("Anchor token found in input_ids")
        #     logger.warning(f"Anchor id: {torch.where(anchor_mask)[0]}")
        # concatenate
        embedding_output = self.electra.embeddings(
            input_ids=input_ids,
            position_ids=None,
            token_type_ids=None,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )

        _, _, embedding_dim = embedding_output.shape
        if (
            self.muxing_variant == "random_ortho"
            or self.muxing_variant == "random_ortho_low_rank"
        ):
            embedding_output = embedding_output.view(
                modified_batch_size,
                num_instances,
                modified_seq_length,
                embedding_dim,
            )
            embedding_output = torch.matmul(
                self.instance_embedding, embedding_output.permute(0, 1, 3, 2)
            )
            # swap the last 2 dimensions again
            embedding_output = embedding_output.permute(0, 1, 3, 2)
            # average across the instances
            embedding_output = torch.sum(embedding_output, dim=1) / math.sqrt(
                self.num_instances
            )
        elif self.muxing_variant == "gaussian_attention":
            embedding_output_intermediate = embedding_output.view(
                modified_batch_size,
                num_instances,
                modified_seq_length,
                embedding_dim,
            )
            # extract relevant instance embeddings
            instance_embed = self.instance_embedding[:num_instances, :]
            instance_embed = instance_embed.unsqueeze(1).expand(
                num_instances, modified_seq_length, embedding_dim
            )
            embedding_output_intermediate = (
                embedding_output_intermediate * instance_embed.unsqueeze(0)
            )

            embedding_output_cross_instance = torch.mean(
                embedding_output_intermediate, dim=1
            )
            embedding_output_cross_instance = self.cross_instances_linear(
                embedding_output_cross_instance
            )
            embedding_output_cross_instance = gelu(embedding_output_cross_instance)
            embedding_output_cross_instance = self.cross_instances_layernorm(
                embedding_output_cross_instance
            )

            embedding_output_intermediate = embedding_output_intermediate.view(
                modified_batch_size * num_instances, modified_seq_length, embedding_dim
            )
            # pass throughh attention layer
            embedding_output_attention = self.muxing_attention(
                embedding_output_intermediate, attention_mask=self.get_extended_attention_mask(attention_mask, input_shape, device=input_ids.device))
            embedding_output_attention = embedding_output_attention[0]
            embedding_output_cross_instance = embedding_output_cross_instance.unsqueeze(
                1
            ).expand(
                modified_batch_size, num_instances, modified_seq_length, embedding_dim
            )
            # average across the instances, and add the cross instance attention
            embedding_output_attention = embedding_output_attention.view(
                modified_batch_size, num_instances, modified_seq_length, embedding_dim
            )
            embedding_output_attention = (
                embedding_output_attention + embedding_output_cross_instance
            )
            embedding_output = torch.mean(embedding_output_attention, dim=1)
        else:
            embedding_output = embedding_output.view(
                modified_batch_size,
                num_instances,
                modified_seq_length,
                embedding_dim,
            )
            # extract relevant instance embeddings
            instance_embed = self.instance_embedding[:num_instances, :]
            instance_embed = instance_embed.unsqueeze(1).expand(
                num_instances, modified_seq_length, embedding_dim
            )
            embedding_output = embedding_output * instance_embed.unsqueeze(0)

            embedding_output = torch.mean(embedding_output, dim=1)

        outputs = self.electra(
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=position_ids,
            inputs_embeds=embedding_output,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        # fancy indexing to get the instance position embedding
        if self.demuxing_variant != "index":
            demuxed_sequence_output = self.demultiplexer(sequence_output[:, 0:1, :])
        else:
            demuxed_sequence_output = self.demultiplexer(sequence_output)
            demuxed_sequence_output = demuxed_sequence_output[:, num_instances:num_instances+1, :]
        logits = self.classifier(demuxed_sequence_output)

        if labels is not None:
            # retrieval loss calculation
            instance_labels = torch.full(
                (modified_batch_size, modified_seq_length),
                0,
                device=input_ids.device,
            ).long()
            # skip the cls and prefix tokens
            instance_labels[:, special_tokens_end_position:] = torch.randint(
                num_instances,
                (
                    modified_batch_size,
                    modified_seq_length - special_tokens_end_position,
                ),
                device=input_ids.device,
            )

            # index into input ids to get the corresponding labels
            input_ids = input_ids.view(modified_batch_size, num_instances, -1)
            input_ids = input_ids.permute(0, 2, 1)

            retrieval_labels = input_ids[
                torch.arange(modified_batch_size, device=input_ids.device)
                .unsqueeze(1)
                .expand(modified_batch_size, modified_seq_length),
                torch.arange(modified_seq_length, device=input_ids.device)
                .unsqueeze(0)
                .expand(modified_batch_size, modified_seq_length),
                instance_labels,
            ]
            retrieval_labels[:, :special_tokens_end_position] = -100

            pad_mask = retrieval_labels == 0
            # wipe of 1 - (0.1  *  retrieval percentage) of pad tokens
            pad_mask_wipe = pad_mask
            non_pad_mask_wipe = (
                ~pad_mask
                & torch.bernoulli(
                    torch.full(
                        retrieval_labels.shape,
                        1 - self.config.retrieval_percentage,
                        device=input_ids.device,
                    )
                ).bool()
            )
            retrieval_labels[non_pad_mask_wipe] = -100

            retrieval_labels[pad_mask_wipe] = -100

            retrieval_predictions = self.retrieval_head(
                sequence_output, instance_labels
            )
            # retrieval_loss = None
            # task_loss = None
            # loss = None

            # retrieval_loss_fct = nn.CrossEntropyLoss()

            # retrieval_loss = retrieval_loss_fct(
            #     retrieval_predictions.view(-1, self.config.vocab_size),
            #     retrieval_labels.view(-1),
            # )

        # loss_fct = nn.CrossEntropyLoss()
        # task_loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        # # calculate head accuracies for each instance
        """
        head_accuracies = torch.zeros(num_instances, device=input_ids.device)
        for i in range(num_instances):
            head_accuracies[i] = (
                (
                    torch.argmax(logits[i::num_instances, :], dim=-1)
                    == labels[i::num_instances]
                )
                .float()
                .mean()
            )
        # if self.head_accuracies is None:
        #     self.head_accuracies = head_accuracies
        # else:
        #     # updating EWA
        #     self.head_accuracies = self.head_acc_smooth_coeff * self.head_accuracies + (1 - self.head_acc_smooth_coeff) * head_accuracies
        # # based on head accuracies, calculate the task loss
        # average_head_acc = torch.mean(self.head_accuracies)
        # average_head_mask = self.head_accuracies <=  average_head_acc
        # instance_weights = torch.ones(num_instances).to(input_ids.device)
        # instance_weights[average_head_mask] = 1 + (((average_head_acc - self.head_accuracies[average_head_mask])) * self.head_acc_gamma)

        loss_fct = nn.CrossEntropyLoss(reduction="none")
        task_loss_per_sample = loss_fct(
            logits.view(-1, self.num_labels), labels.view(-1)
        )
        task_loss_per_head = torch.mean(
            task_loss_per_sample.view(modified_batch_size, num_instances), dim=0
        )
        # assert task_loss_per_head.shape[0] == num_instances
        # assert torch.isclose(
        #     torch.mean(task_loss_per_head), torch.mean(task_loss_per_sample)
        # )
        # assert torch.isclose(
        #     task_loss_per_head[0], torch.mean(task_loss_per_sample[0::num_instances])
        # )
        # normalize the loss per head, higher head loss will be weighted more
        instance_weights = F.softmax(
            task_loss_per_head / self.head_temperature
        ).detach()
        task_loss = torch.sum(
            (task_loss_per_sample * instance_weights.repeat(modified_batch_size))
        ) / torch.sum(instance_weights.repeat(modified_batch_size))
        """
        # print("instance weights: ", instance_weights)
        # print("head accuracies: ", self.head_accuracies)
        # print("current head accuracy: ", head_accuracies)
        retrieval_loss = None
        task_loss = None
        loss = None
        if dummy_samples_added != 0:
            logits = logits[:-dummy_samples_added]
        if labels is not None:
            # labels = labels[: (modified_batch_size * num_instances)]
            assert len(labels.shape) == 1  # assert one dimension
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = nn.MSELoss()
                task_loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                task_loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            loss_fct = nn.CrossEntropyLoss()
            retrieval_loss = loss_fct(
                retrieval_predictions.view(-1, self.config.vocab_size),
                retrieval_labels.view(-1),
            )
            loss = (self.task_loss_coeff * task_loss) + (
                self.retrieval_loss_coeff * retrieval_loss
            )
        # make logits align with the length of the input sequence
        # logits = logits[:, special_tokens_end_position:]
        if not return_dict:
            output = (logits,)
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputMuxed(
            loss=loss,
            logits=logits,
            task_loss=task_loss,
            retrieval_loss=retrieval_loss,
            retrieval_predictions=None,
            retrieval_instance_labels=None,
            hidden_states=demuxed_sequence_output,
            head_weights=None,
            train_metrics_per_head=None,
        )


class MuxedElectraForTokenClassification(ElectraPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.config = config

        self.num_instances = config.num_instances
        self.muxing_variant = config.muxing_variant
        self.demuxing_variant = config.demuxing_variant
        self.retrieval_loss_coeff = config.retrieval_loss_coeff
        self.task_loss_coeff = config.task_loss_coeff

        self.head_temperature = config.head_temperature
        self.num_labels = config.num_labels
        self.electra = ElectraModel(config)

        if config.demuxing_variant == "index":
            self.demultiplexer = IndexDemultiplexerTokenLevel(config)
            self.retrieval_head = RetrievalHeadIndexDemultiplexing(
                config
            )
        elif config.demuxing_variant == "mlp":
            self.demux_module = MLPDemuxModule(config)
            self.demultiplexer = MLPDemultiplexerTokenLevel(config, self.demux_module)
            self.retrieval_head = RetrievalHeadMLPDemultiplexing(
                config, self.demux_module
            )
        elif config.demuxing_variant == "index_pos":
            self.demux_module = IndexPosDemuxModule(config)
            self.demultiplexer = IndexPosDemultiplexerTokenLevel(
                config, self.demux_module
            )
            self.retrieval_head = RetrievalHeadIndexPosDemultiplexing(
                config, self.demux_module
            )
        else:
            raise NotImplementedError()

        self.init_weights()
        classifier_dropout = config.hidden_dropout_prob

        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # multiplexing initialization

        d_model = config.embedding_size
        instance_embedding = None

        if self.muxing_variant == "gaussian_hadamard":
            instance_embedding = random_encoding(
                self.num_instances, d_model, norm=config.gaussian_hadamard_norm
            )
        elif self.muxing_variant == "random_ortho":
            instance_embedding = [
                torch.from_numpy(ortho_group.rvs(config.hidden_size)).float()
                for _ in range(self.num_instances)
            ]
            instance_embedding = torch.stack(instance_embedding, dim=0)
        elif self.muxing_variant == "random_ortho_low_rank":
            # create low rank matrix
            instance_embedding = []
            H_ = special_ortho_group.rvs(dim=config.hidden_size)
            U_ = special_ortho_group.rvs(dim=config.hidden_size)
            for i in range(self.num_instances):
                G_ = np.zeros((config.hidden_size, config.hidden_size))
                l = i * (config.hidden_size // self.num_instances)
                r = l + (config.hidden_size // self.num_instances)
                H_copy = H_.copy()
                G_[:, l:r] = H_copy[:, l:r]
                G_ = U_ @ G_.T
                instance_embedding.append(torch.from_numpy(G_).float())
            instance_embedding = torch.stack(instance_embedding, dim=0)
        elif self.muxing_variant == "binary_hadamard":
            instance_embedding = binary_encoding(
                self.num_instances, d_model, epsilon=config.binary_hadamard_epsilon
            )
        elif self.muxing_variant == "gaussian_attention":
            instance_embedding = random_encoding(
                self.num_instances, d_model, norm=config.gaussian_hadamard_norm
            )
            self.muxing_attention = ElectraLayer(config)
            self.cross_instances_linear = nn.Linear(config.embedding_size, d_model)
            self.cross_instances_layernorm = nn.LayerNorm(d_model)
        else:
            raise NotImplementedError()

        if instance_embedding is not None:
            self.instance_embedding = torch.nn.Parameter(instance_embedding)
        else:
            instance_embedding = random_encoding(
                self.num_instances, d_model, norm=self.gaussian_hadamard_norm
            )

        if not config.learn_muxing:
            self.instance_embedding.requires_grad = False
        else:
            self.instance_embedding.requires_grad = True

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        labels=None,
        return_dict=None,
    ):
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        dummy_samples_added = 0    
        if input_ids.shape[0] % self.num_instances != 0:
            dummy_samples_added = self.num_instances - (input_ids.shape[0] % self.num_instances)
            pad_input_ids = input_ids[
                torch.randint(
                    0, input_ids.shape[0], (dummy_samples_added,)
                )
            ]
            input_ids = torch.cat([input_ids, pad_input_ids], dim=0)
            pad_attention_mask = attention_mask[
                torch.randint(
                    0, attention_mask.shape[0], (dummy_samples_added,)
                )
            ]
            attention_mask = torch.cat([attention_mask, pad_attention_mask], dim=0)
        # get input embeddings and average over N instances
        input_shape = input_ids.size()

        batch_size, seq_length = input_shape
        num_instances = self.num_instances
        past_key_values_length = 0

        modified_batch_size = batch_size // num_instances
        modified_seq_length = None
        special_tokens_end_position = None
        if self.demuxing_variant == "index":

            # add the prefix
            # [CLS1, <s>, <s>, <s>, <s>]
            # [<s>, CLS2, <s>, <s>, <s>]
            # [<s>, <s>, CLS3, <s>, <s>]
            # [<s>, <s>, <s>, CLS4, <s>]
            # [<s>, <s>, <s>, <s>, CLS5]
            # let us just assume the last 5 tokens barring the masked token
            # are the cls tokens (easiest way to make use of existing vocab)

            # prefix 5 x 5

            prefix = torch.full(
                (num_instances, num_instances), 680, device=input_ids.device
            )
            prefix[
                torch.arange(num_instances, device=input_ids.device),
                torch.arange(num_instances, device=input_ids.device),
            ] = (
                -(torch.arange(num_instances, device=input_ids.device) + 2)
                + 680 
            )

            # [-2   <s>, <s>, <s>, <s>]
            # [<s>, -3, <s>, <s>, <s>]
            # [<s>, <s>, -4, <s>, <s>]
            # [<s>, <s>, <s>, -5, <s>]
            # [<s>, <s>, <s>, <s>, -6]
            # +  size of vocab
            cls_tokens = torch.full((num_instances, 1), 101, device=input_ids.device)
            prefix = torch.cat([prefix, cls_tokens], dim=1)

            prefix = prefix.repeat(modified_batch_size, 1)
            input_ids = input_ids[: (modified_batch_size * num_instances)]
            input_ids = torch.cat([prefix, input_ids], dim=1)
            modified_seq_length = seq_length + num_instances + 1
            special_tokens_end_position = num_instances + 1

        elif self.demuxing_variant == "mlp" or self.demuxing_variant == "index_pos":
            cls_tokens = torch.full((num_instances, 1), 101, device=input_ids.device)
            cls_tokens = cls_tokens.repeat(modified_batch_size, 1)
            input_ids = input_ids[: (modified_batch_size * num_instances)]
            # input_ids = torch.cat([cls_tokens, input_ids], dim=1)
            # modified_seq_length = seq_length + 1
            modified_seq_length = seq_length
            special_tokens_end_position = 0

        else:
            raise NotImplementedError()

        # concatenate
        embedding_output = self.electra.embeddings(
            input_ids=input_ids,
            position_ids=None,
            token_type_ids=None,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )

        _, _, embedding_dim = embedding_output.shape
        if (
            self.muxing_variant == "random_ortho"
            or self.muxing_variant == "random_ortho_low_rank"
        ):
            embedding_output = embedding_output.view(
                modified_batch_size,
                num_instances,
                modified_seq_length,
                embedding_dim,
            )
            embedding_output = torch.matmul(
                self.instance_embedding, embedding_output.permute(0, 1, 3, 2)
            )
            # swap the last 2 dimensions again
            embedding_output = embedding_output.permute(0, 1, 3, 2)
            # average across the instances
            embedding_output = torch.sum(embedding_output, dim=1) / math.sqrt(
                self.num_instances
            )
        elif self.muxing_variant == "gaussian_attention":
            embedding_output_intermediate = embedding_output.view(
                modified_batch_size,
                num_instances,
                modified_seq_length,
                embedding_dim,
            )
            # extract relevant instance embeddings
            instance_embed = self.instance_embedding[:num_instances, :]
            instance_embed = instance_embed.unsqueeze(1).expand(
                num_instances, modified_seq_length, embedding_dim
            )
            embedding_output_intermediate = (
                embedding_output_intermediate * instance_embed.unsqueeze(0)
            )

            embedding_output_cross_instance = torch.mean(
                embedding_output_intermediate, dim=1
            )
            embedding_output_cross_instance = self.cross_instances_linear(
                embedding_output_cross_instance
            )
            embedding_output_cross_instance = gelu(embedding_output_cross_instance)
            embedding_output_cross_instance = self.cross_instances_layernorm(
                embedding_output_cross_instance
            )

            embedding_output_intermediate = embedding_output_intermediate.view(
                modified_batch_size * num_instances, modified_seq_length, embedding_dim
            )
            # pass throughh attention layer
            embedding_output_attention = self.muxing_attention(
                embedding_output_intermediate, attention_mask=self.get_extended_attention_mask(attention_mask, input_shape, device=input_ids.device))
            embedding_output_attention = embedding_output_attention[0]
            embedding_output_cross_instance = embedding_output_cross_instance.unsqueeze(
                1
            ).expand(
                modified_batch_size, num_instances, modified_seq_length, embedding_dim
            )
            # average across the instances, and add the cross instance attention
            embedding_output_attention = embedding_output_attention.view(
                modified_batch_size, num_instances, modified_seq_length, embedding_dim
            )
            embedding_output_attention = (
                embedding_output_attention + embedding_output_cross_instance
            )
            embedding_output = torch.mean(embedding_output_attention, dim=1)
        else:
            embedding_output = embedding_output.view(
                modified_batch_size,
                num_instances,
                modified_seq_length,
                embedding_dim,
            )
            # extract relevant instance embeddings
            instance_embed = self.instance_embedding[:num_instances, :]
            instance_embed = instance_embed.unsqueeze(1).expand(
                num_instances, modified_seq_length, embedding_dim
            )
            embedding_output = embedding_output * instance_embed.unsqueeze(0)

            embedding_output = torch.mean(embedding_output, dim=1)

        outputs = self.electra(
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=position_ids,
            inputs_embeds=embedding_output,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        # fancy indexing to get the instance position embedding

        demuxed_sequence_output = self.demultiplexer(sequence_output)
        demuxed_sequence_output = self.dropout(demuxed_sequence_output)
        logits = self.classifier(demuxed_sequence_output)


        if labels is not None:
            # retrieval loss calculation
            instance_labels = torch.full(
                (modified_batch_size, modified_seq_length),
                0,
                device=input_ids.device,
            ).long()
            # skip the cls and prefix tokens
            instance_labels[:, special_tokens_end_position:] = torch.randint(
                num_instances,
                (modified_batch_size, modified_seq_length - special_tokens_end_position),
                device=input_ids.device,
            )

            # index into input ids to get the corresponding labels
            input_ids = input_ids.view(modified_batch_size, num_instances, -1)
            input_ids = input_ids.permute(0, 2, 1)

            retrieval_labels = input_ids[
                torch.arange(modified_batch_size, device=input_ids.device)
                .unsqueeze(1)
                .expand(modified_batch_size, modified_seq_length),
                torch.arange(modified_seq_length, device=input_ids.device)
                .unsqueeze(0)
                .expand(modified_batch_size, modified_seq_length),
                instance_labels,
            ]
            retrieval_labels[:, :special_tokens_end_position] = -100

            pad_mask = retrieval_labels == 0
            # wipe of 1 - (0.1  *  retrieval percentage) of pad tokens
            pad_mask_wipe = pad_mask
            non_pad_mask_wipe = (
                ~pad_mask
                & torch.bernoulli(
                    torch.full(
                        retrieval_labels.shape,
                        1 - self.config.retrieval_percentage,
                        device=input_ids.device,
                    )
                ).bool()
            )
            retrieval_labels[non_pad_mask_wipe] = -100

            retrieval_labels[pad_mask_wipe] = -100

            retrieval_predictions = self.retrieval_head(sequence_output, instance_labels)

        retrieval_loss = None
        task_loss = None
        loss = None

        if dummy_samples_added != 0:
            logits = logits[:-dummy_samples_added]
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            logits = logits[:, special_tokens_end_position:, :]
            task_loss = loss_fct(logits.reshape(-1, self.num_labels), labels.view(-1))
            retrieval_loss_fct = nn.CrossEntropyLoss()
            retrieval_loss = retrieval_loss_fct(
                retrieval_predictions.view(-1, self.config.vocab_size),
                retrieval_labels.view(-1),
            )
            loss = (self.task_loss_coeff * task_loss) + (
                self.retrieval_loss_coeff * retrieval_loss
            )

        if not return_dict:
            output = (logits,)
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputMuxed(
            loss=loss,
            logits=logits,
            task_loss=task_loss,
            retrieval_loss=retrieval_loss,
            # retrieval_predictions=torch.argmax(retrieval_predictions, dim=2),
            # retrieval_instance_labels=retrieval_labels,
            head_weights=None,
            train_metrics_per_head=None,
        )


class MuxedElectraForQuestionAnswering(ElectraPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.config = config

        self.num_instances = config.num_instances
        self.muxing_variant = config.muxing_variant
        self.demuxing_variant = config.demuxing_variant
        self.retrieval_loss_coeff = config.retrieval_loss_coeff
        self.task_loss_coeff = config.task_loss_coeff

        self.head_temperature = config.head_temperature
        self.num_labels = config.num_labels
        self.electra = ElectraModel(config)

        if config.demuxing_variant == "index":
            raise NotImplementedError()
        elif config.demuxing_variant == "mlp":
            self.demux_module = MLPDemuxModule(config)
            self.demultiplexer = MLPDemultiplexerTokenLevel(config, self.demux_module)
            self.retrieval_head = RetrievalHeadMLPDemultiplexing(
                config, self.demux_module
            )
        elif config.demuxing_variant == "index_pos":
            self.demux_module = IndexPosDemuxModule(config)
            self.demultiplexer = IndexPosDemultiplexerTokenLevel(
                config, self.demux_module
            )
            self.retrieval_head = RetrievalHeadIndexPosDemultiplexing(
                config, self.demux_module
            )
        else:
            raise NotImplementedError()

        self.init_weights()
        classifier_dropout = config.hidden_dropout_prob

        self.dropout = nn.Dropout(classifier_dropout)
        self.qa_ouptuts = nn.Linear(config.hidden_size, config.num_labels)

        # multiplexing initialization

        d_model = config.embedding_size
        instance_embedding = None

        if self.muxing_variant == "gaussian_hadamard":
            instance_embedding = random_encoding(
                self.num_instances, d_model, norm=config.gaussian_hadamard_norm
            )
        elif self.muxing_variant == "random_ortho":
            instance_embedding = [
                torch.from_numpy(ortho_group.rvs(config.hidden_size)).float()
                for _ in range(self.num_instances)
            ]
            instance_embedding = torch.stack(instance_embedding, dim=0)
        elif self.muxing_variant == "random_ortho_low_rank":
            # create low rank matrix
            instance_embedding = []
            H_ = special_ortho_group.rvs(dim=config.hidden_size)
            U_ = special_ortho_group.rvs(dim=config.hidden_size)
            for i in range(self.num_instances):
                G_ = np.zeros((config.hidden_size, config.hidden_size))
                l = i * (config.hidden_size // self.num_instances)
                r = l + (config.hidden_size // self.num_instances)
                H_copy = H_.copy()
                G_[:, l:r] = H_copy[:, l:r]
                G_ = U_ @ G_.T
                instance_embedding.append(torch.from_numpy(G_).float())
            instance_embedding = torch.stack(instance_embedding, dim=0)
        elif self.muxing_variant == "binary_hadamard":
            instance_embedding = binary_encoding(
                self.num_instances, d_model, epsilon=config.binary_hadamard_epsilon
            )
        elif self.muxing_variant == "gaussian_attention":
            instance_embedding = random_encoding(
                self.num_instances, d_model, norm=config.gaussian_hadamard_norm
            )
            self.muxing_attention = ElectraLayer(config)
            self.cross_instances_linear = nn.Linear(config.embedding_size, d_model)
            self.cross_instances_layernorm = nn.LayerNorm(d_model)
        else:
            raise NotImplementedError()

        if instance_embedding is not None:
            self.instance_embedding = torch.nn.Parameter(instance_embedding)
        else:
            instance_embedding = random_encoding(
                self.num_instances, d_model, norm=self.gaussian_hadamard_norm
            )

        if not config.learn_muxing:
            self.instance_embedding.requires_grad = False
        else:
            self.instance_embedding.requires_grad = True

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        start_positions: Optional[torch.Tensor] = None,
        end_positions: Optional[torch.Tensor] = None,
        return_dict=None,
    ):
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        dummy_samples_added = 0    
        if input_ids.shape[0] % self.num_instances != 0:
            dummy_samples_added = self.num_instances - (input_ids.shape[0] % self.num_instances)
            pad_input_ids = input_ids[
                torch.randint(
                    0, input_ids.shape[0], (dummy_samples_added,)
                )
            ]
            input_ids = torch.cat([input_ids, pad_input_ids], dim=0)
        # get input embeddings and average over N instances
        input_shape = input_ids.size()

        batch_size, seq_length = input_shape
        num_instances = self.num_instances
        past_key_values_length = 0

        modified_batch_size = batch_size // num_instances
        modified_seq_length = None
        special_tokens_end_position = None
        if self.demuxing_variant == "index":

            # add the prefix
            # [CLS1, <s>, <s>, <s>, <s>]
            # [<s>, CLS2, <s>, <s>, <s>]
            # [<s>, <s>, CLS3, <s>, <s>]
            # [<s>, <s>, <s>, CLS4, <s>]
            # [<s>, <s>, <s>, <s>, CLS5]
            # let us just assume the last 5 tokens barring the masked token
            # are the cls tokens (easiest way to make use of existing vocab)

            # prefix 5 x 5

            prefix = torch.full(
                (num_instances, num_instances), 680, device=input_ids.device
            )
            prefix[
                torch.arange(num_instances, device=input_ids.device),
                torch.arange(num_instances, device=input_ids.device),
            ] = (
                -(torch.arange(num_instances, device=input_ids.device) + 2)
                + 680 
            )

            # [-2   <s>, <s>, <s>, <s>]
            # [<s>, -3, <s>, <s>, <s>]
            # [<s>, <s>, -4, <s>, <s>]
            # [<s>, <s>, <s>, -5, <s>]
            # [<s>, <s>, <s>, <s>, -6]
            # +  size of vocab
            cls_tokens = torch.full((num_instances, 1), 101, device=input_ids.device)
            prefix = torch.cat([prefix, cls_tokens], dim=1)

            prefix = prefix.repeat(modified_batch_size, 1)
            input_ids = input_ids[: (modified_batch_size * num_instances)]
            input_ids = torch.cat([prefix, input_ids], dim=1)
            modified_seq_length = seq_length + num_instances + 1
            special_tokens_end_position = num_instances + 1

        elif self.demuxing_variant == "mlp" or self.demuxing_variant == "index_pos":
            cls_tokens = torch.full((num_instances, 1), 101, device=input_ids.device)
            cls_tokens = cls_tokens.repeat(modified_batch_size, 1)
            input_ids = input_ids[: (modified_batch_size * num_instances)]
            # input_ids = torch.cat([cls_tokens, input_ids], dim=1)
            # modified_seq_length = seq_length + 1
            modified_seq_length = seq_length
            special_tokens_end_position = 0

        else:
            raise NotImplementedError()

        # concatenate
        embedding_output = self.electra.embeddings(
            input_ids=input_ids,
            position_ids=None,
            token_type_ids=None,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )

        _, _, embedding_dim = embedding_output.shape
        if (
            self.muxing_variant == "random_ortho"
            or self.muxing_variant == "random_ortho_low_rank"
        ):
            embedding_output = embedding_output.view(
                modified_batch_size,
                num_instances,
                modified_seq_length,
                embedding_dim,
            )
            embedding_output = torch.matmul(
                self.instance_embedding, embedding_output.permute(0, 1, 3, 2)
            )
            # swap the last 2 dimensions again
            embedding_output = embedding_output.permute(0, 1, 3, 2)
            # average across the instances
            embedding_output = torch.sum(embedding_output, dim=1) / math.sqrt(
                self.num_instances
            )
        elif self.muxing_variant == "gaussian_attention":
            embedding_output_intermediate = embedding_output.view(
                modified_batch_size,
                num_instances,
                modified_seq_length,
                embedding_dim,
            )
            # extract relevant instance embeddings
            instance_embed = self.instance_embedding[:num_instances, :]
            instance_embed = instance_embed.unsqueeze(1).expand(
                num_instances, modified_seq_length, embedding_dim
            )
            embedding_output_intermediate = (
                embedding_output_intermediate * instance_embed.unsqueeze(0)
            )

            embedding_output_cross_instance = torch.mean(
                embedding_output_intermediate, dim=1
            )
            embedding_output_cross_instance = self.cross_instances_linear(
                embedding_output_cross_instance
            )
            embedding_output_cross_instance = gelu(embedding_output_cross_instance)
            embedding_output_cross_instance = self.cross_instances_layernorm(
                embedding_output_cross_instance
            )

            embedding_output_intermediate = embedding_output_intermediate.view(
                modified_batch_size * num_instances, modified_seq_length, embedding_dim
            )
            # pass throughh attention layer
            embedding_output_attention = self.muxing_attention(
                embedding_output_intermediate
            )
            embedding_output_attention = embedding_output_attention[0]
            embedding_output_cross_instance = embedding_output_cross_instance.unsqueeze(
                1
            ).expand(
                modified_batch_size, num_instances, modified_seq_length, embedding_dim
            )
            # average across the instances, and add the cross instance attention
            embedding_output_attention = embedding_output_attention.view(
                modified_batch_size, num_instances, modified_seq_length, embedding_dim
            )
            embedding_output_attention = (
                embedding_output_attention + embedding_output_cross_instance
            )
            embedding_output = torch.mean(embedding_output_attention, dim=1)
        else:
            embedding_output = embedding_output.view(
                modified_batch_size,
                num_instances,
                modified_seq_length,
                embedding_dim,
            )
            # extract relevant instance embeddings
            instance_embed = self.instance_embedding[:num_instances, :]
            instance_embed = instance_embed.unsqueeze(1).expand(
                num_instances, modified_seq_length, embedding_dim
            )
            embedding_output = embedding_output * instance_embed.unsqueeze(0)

            embedding_output = torch.mean(embedding_output, dim=1)

        outputs = self.electra(
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=position_ids,
            inputs_embeds=embedding_output,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        # fancy indexing to get the instance position embedding

        demuxed_sequence_output = self.demultiplexer(sequence_output)
        demuxed_sequence_output = self.dropout(demuxed_sequence_output)
        # retrieval loss calculation
        instance_labels = torch.full(
            (modified_batch_size, modified_seq_length),
            0,
            device=input_ids.device,
        ).long()
        # skip the cls and prefix tokens
        instance_labels[:, special_tokens_end_position:] = torch.randint(
            num_instances,
            (modified_batch_size, modified_seq_length - special_tokens_end_position),
            device=input_ids.device,
        )

        # index into input ids to get the corresponding labels
        input_ids = input_ids.view(modified_batch_size, num_instances, -1)
        input_ids = input_ids.permute(0, 2, 1)

        retrieval_labels = input_ids[
            torch.arange(modified_batch_size, device=input_ids.device)
            .unsqueeze(1)
            .expand(modified_batch_size, modified_seq_length),
            torch.arange(modified_seq_length, device=input_ids.device)
            .unsqueeze(0)
            .expand(modified_batch_size, modified_seq_length),
            instance_labels,
        ]
        retrieval_labels[:, :special_tokens_end_position] = -100

        pad_mask = retrieval_labels == 0
        # wipe of 1 - (0.1  *  retrieval percentage) of pad tokens
        pad_mask_wipe = pad_mask
        non_pad_mask_wipe = (
            ~pad_mask
            & torch.bernoulli(
                torch.full(
                    retrieval_labels.shape,
                    1 - self.config.retrieval_percentage,
                    device=input_ids.device,
                )
            ).bool()
        )
        retrieval_labels[non_pad_mask_wipe] = -100

        retrieval_labels[pad_mask_wipe] = -100

        retrieval_predictions = self.retrieval_head(sequence_output, instance_labels)

        retrieval_loss = None
        task_loss = None
        loss = None

        retrieval_loss_fct = nn.CrossEntropyLoss()

        retrieval_loss = retrieval_loss_fct(
            retrieval_predictions.view(-1, self.config.vocab_size),
            retrieval_labels.view(-1),
        )

        logits = self.qa_ouptuts(demuxed_sequence_output)
        logits = logits[:, special_tokens_end_position:, :]
        if dummy_samples_added != 0:
            logits = logits[:-dummy_samples_added]
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()
     
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            task_loss = (start_loss + end_loss) / 2

            retrieval_loss_fct = nn.CrossEntropyLoss()
            retrieval_loss = retrieval_loss_fct(
                retrieval_predictions.view(-1, self.config.vocab_size),
                retrieval_labels.view(-1),
            )
            loss = (self.task_loss_coeff * task_loss) + (
                self.retrieval_loss_coeff * retrieval_loss
            )
        else:
            task_loss = None
            retrieval_loss = None
            loss = None

        if not return_dict:
            output = (
                start_logits,
                end_logits,
            )
            return ((loss,) + output) if loss is not None else output

        return QuestionAnsweringModelOutputMuxed(
            loss=loss,
            start_logits=start_logits,
            end_logits=end_logits,
            task_loss=task_loss,
            retrieval_loss=retrieval_loss,
        )

class MuxedElectraForMaskedLM(ElectraPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids", r"lm_head.decoder.bias"]
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)

        self.config = config

        self.num_instances = config.num_instances
        self.muxing_variant = config.muxing_variant
        self.demuxing_variant = config.demuxing_variant
        self.retrieval_loss_coeff = config.retrieval_loss_coeff
        self.task_loss_coeff = config.task_loss_coeff

        if config.is_decoder:
            logger.warning(
                "If you want to use `RobertaForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )
        if config.demuxing_variant == "index":
            self.demultiplexer = IndexDemultiplexerTokenLevel(config)
            self.retrieval_head = RetrievalHeadIndexDemultiplexing(
                config
            )
        elif config.demuxing_variant == "mlp":
            self.demux_module = MLPDemuxModule(config)
            self.demultiplexer = MLPDemultiplexerTokenLevel(config, self.demux_module)
            self.retrieval_head = RetrievalHeadMLPDemultiplexing(
                config, self.demux_module
            )
        elif config.demuxing_variant == "index_pos":
            self.demux_module = IndexPosDemuxModule(config)
            self.demultiplexer = IndexPosDemultiplexerTokenLevel(
                config, self.demux_module
            )
            # self.retrieval_head = RetrievalHeadIndexPosDemultiplexing(
            #     config, self.demux_module
            # )
            self.retrieval_head = RetrievalHeadIndexPosDemultiplexingHierarchical(
                config, self.demux_module
            )
        else:
            raise NotImplementedError()

        d_model = config.embedding_size
        instance_embedding = None

        if self.muxing_variant == "gaussian_hadamard":
            instance_embedding = random_encoding(
                self.num_instances, d_model, norm=config.gaussian_hadamard_norm
            )
        elif self.muxing_variant == "random_ortho":
            instance_embedding = [
                torch.from_numpy(ortho_group.rvs(config.hidden_size)).float()
                for _ in range(self.num_instances)
            ]
            instance_embedding = torch.stack(instance_embedding, dim=0)

        elif self.muxing_variant == "random_ortho_low_rank":
            # create low rank matrix
            instance_embedding = []
            H_ = special_ortho_group.rvs(dim=d_model)
            U_ = special_ortho_group.rvs(dim=d_model)
            for i in range(self.num_instances):
                G_ = np.zeros((d_model, d_model))
                l = i * (d_model // self.num_instances)
                r = l + (d_model // self.num_instances)
                H_copy = H_.copy()
                G_[:, l:r] = H_copy[:, l:r]
                G_ = U_ @ G_.T
                instance_embedding.append(torch.from_numpy(G_).float())
            instance_embedding = torch.stack(instance_embedding, dim=0)

        elif self.muxing_variant == "binary_hadamard":
            instance_embedding = binary_encoding(
                self.num_instances, d_model, epsilon=config.binary_hadamard_epsilon
            )
        else:
            raise NotImplementedError()

        if instance_embedding is not None:
            self.instance_embedding = torch.nn.Parameter(instance_embedding)
        else:
            instance_embedding = random_encoding(
                self.num_instances, d_model, norm=self.gaussian_hadamard_norm
            )

        if not config.learn_muxing:
            self.instance_embedding.requires_grad = False
        else:
            self.instance_embedding.requires_grad = True

        self.electra = ElectraModel(config)

    # def get_output_embeddings(self):
    # return self.generator_lm_head

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        labels=None,
        return_dict=None,
    ):
        # s_t = time.time()
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # get input embeddings and average over N instances
        input_shape = input_ids.size()

        batch_size, seq_length = input_shape
        num_instances = self.num_instances
        past_key_values_length = 0

        modified_batch_size = batch_size // num_instances
        modified_seq_length = None
        special_tokens_end_position = None
        if self.demuxing_variant == "index":

            # add the prefix
            # [CLS1, <s>, <s>, <s>, <s>]
            # [<s>, CLS2, <s>, <s>, <s>]
            # [<s>, <s>, CLS3, <s>, <s>]
            # [<s>, <s>, <s>, CLS4, <s>]
            # [<s>, <s>, <s>, <s>, CLS5]
            # let us just assume the last 5 tokens barring the masked token
            # are the cls tokens (easiest way to make use of existing vocab)

            # prefix 5 x 5

            prefix = torch.full(
                (num_instances, num_instances), 680, device=input_ids.device
            )
            prefix[
                torch.arange(num_instances, device=input_ids.device),
                torch.arange(num_instances, device=input_ids.device),
            ] = (
                -(torch.arange(num_instances, device=input_ids.device) + 2)
                + 680 
            )

            # [-2   <s>, <s>, <s>, <s>]
            # [<s>, -3, <s>, <s>, <s>]
            # [<s>, <s>, -4, <s>, <s>]
            # [<s>, <s>, <s>, -5, <s>]
            # [<s>, <s>, <s>, <s>, -6]
            # +  size of vocab
            cls_tokens = torch.full((num_instances, 1), 101, device=input_ids.device)
            prefix = torch.cat([prefix, cls_tokens], dim=1)

            prefix = prefix.repeat(modified_batch_size, 1)
            input_ids = input_ids[: (modified_batch_size * num_instances)]
            input_ids = torch.cat([prefix, input_ids], dim=1)
            modified_seq_length = seq_length + num_instances + 1
            special_tokens_end_position = num_instances + 1

        elif self.demuxing_variant == "mlp" or self.demuxing_variant == "index_pos":
            cls_tokens = torch.full((num_instances, 1), 101, device=input_ids.device)
            cls_tokens = cls_tokens.repeat(modified_batch_size, 1)
            input_ids = input_ids[: (modified_batch_size * num_instances)]
            input_ids[:, 0:1] = cls_tokens
            modified_seq_length = seq_length
            special_tokens_end_position = 0

        else:
            raise NotImplementedError()
        # concatenate
        embedding_output = self.electra.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )

        _, _, embedding_dim = embedding_output.shape
        if (
            self.muxing_variant == "random_ortho"
            or self.muxing_variant == "random_ortho_low_rank"
        ):
            embedding_output = embedding_output.view(
                modified_batch_size,
                num_instances,
                modified_seq_length,
                embedding_dim,
            )
            embedding_output = torch.matmul(
                self.instance_embedding, embedding_output.permute(0, 1, 3, 2)
            )
            # swap the last 2 dimensions again
            embedding_output = embedding_output.permute(0, 1, 3, 2)
            # average across the instances
            embedding_output = torch.sum(embedding_output, dim=1) / math.sqrt(
                self.num_instances
            )
        else:
            embedding_output = embedding_output.view(
                modified_batch_size,
                num_instances,
                modified_seq_length,
                embedding_dim,
            )
            # extract relevant instance embeddings
            instance_embed = self.instance_embedding[:num_instances, :]
            instance_embed = instance_embed.unsqueeze(1).expand(
                num_instances, modified_seq_length, embedding_dim
            )
            embedding_output = embedding_output * instance_embed.unsqueeze(0)

            embedding_output = torch.mean(embedding_output, dim=1)
        # logger.warn(f"Time taken to embedding: {time.time() - s_t}")
        # s_t = time.time()
        outputs = self.electra(
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=position_ids,
            inputs_embeds=embedding_output,
            return_dict=return_dict,
        )
        # logger.warn(f"Time taken to forward: {time.time() - s_t}")
        sequence_output = outputs[0]
        # fancy indexing to get the instance position embedding

        # retrieval loss calculation
        instance_labels = torch.full(
            (modified_batch_size, modified_seq_length),
            0,
            device=input_ids.device,
        ).long()
        # skip the cls and prefix tokens
        instance_labels[:, special_tokens_end_position:] = torch.randint(
            num_instances,
            (modified_batch_size, modified_seq_length - special_tokens_end_position),
            device=input_ids.device,
        )

        # index into input ids to get the corresponding labels
        input_ids = input_ids.view(modified_batch_size, num_instances, -1)
        input_ids = input_ids.permute(0, 2, 1)
        # s_t = time.time()
        retrieval_labels = input_ids[
            torch.arange(modified_batch_size, device=input_ids.device)
            .unsqueeze(1)
            .expand(modified_batch_size, modified_seq_length),
            torch.arange(modified_seq_length, device=input_ids.device)
            .unsqueeze(0)
            .expand(modified_batch_size, modified_seq_length),
            instance_labels,
        ]
        retrieval_labels[:, :special_tokens_end_position] = -100

        pad_mask = retrieval_labels == 0
        # wipe of 1 - (0.1  *  retrieval percentage) of pad tokens
        pad_mask_wipe = pad_mask
        non_pad_mask_wipe = (
            ~pad_mask
            & torch.bernoulli(
                torch.full(
                    retrieval_labels.shape,
                    1 - self.config.retrieval_percentage,
                    device=input_ids.device,
                )
            ).bool()
        )
        retrieval_labels[non_pad_mask_wipe] = -100

        retrieval_labels[pad_mask_wipe] = -100

        retrieval_predictions = self.retrieval_head(sequence_output, instance_labels)
        # logger.warn(f"Time taken to retrieval: {time.time() - s_t}")
        # only run the expensive head on masked tokens
        is_mlm_applied = labels != -100
        # s_t = time.time()
        mlm_logits = self.retrieval_head(sequence_output, None, is_mlm_applied, labels)
        logits = torch.zeros(
            batch_size,
            modified_seq_length,
            self.config.vocab_size,
            dtype=mlm_logits.dtype,
            device=mlm_logits.device,
        )
        logits[is_mlm_applied] = mlm_logits
        # logger.warn(f"Time taken to mlm: {time.time() - s_t}")
        retrieval_loss = None
        task_loss = None
        loss = None

        retrieval_loss_fct = nn.CrossEntropyLoss()

        retrieval_loss = retrieval_loss_fct(
            retrieval_predictions.view(-1, self.config.vocab_size),
            retrieval_labels.view(-1),
        )
        if labels is not None:
            labels = labels[: (modified_batch_size * num_instances)]

            if attention_mask is not None:
                loss_fct = nn.CrossEntropyLoss()

                active_loss = attention_mask.view(-1, modified_seq_length) == 1
                active_logits = logits[active_loss]
                active_labels = labels[active_loss]

                task_loss = loss_fct(active_logits, active_labels)
                loss = (self.task_loss_coeff * task_loss) + (
                    self.retrieval_loss_coeff * retrieval_loss
                )

        # make logits align with the length of the input sequence
        logits = logits[:, special_tokens_end_position:]
        if not return_dict:
            output = (logits,)
            return ((loss,) + output) if loss is not None else output

        return MuxedElectraForPreTrainingOutput(
            loss=loss, logits=logits, task_loss=task_loss, retrieval_loss=retrieval_loss
        )

@dataclass
class MuxedElectraForPreTrainingOutput(ElectraForPreTrainingOutput):

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    retrieval_loss: torch.FloatTensor = None
    task_loss: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

@dataclass
class QuestionAnsweringModelOutputMuxed(ModelOutput):

    loss: Optional[torch.FloatTensor] = None
    start_logits: torch.FloatTensor = None
    end_logits: torch.FloatTensor = None
    retrieval_loss: torch.FloatTensor = None
    task_loss: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

# electra muxed pretraining
class MLPDemuxModule(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_instances = config.num_instances
        # initialize different MLPs for different instances
        for sent_id in range(self.num_instances):
            setattr(
                self,
                f"dense_{sent_id}",
                nn.Linear(config.hidden_size, config.hidden_size),
            )

            setattr(
                self,
                f"layer_norm_{sent_id}",
                nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps),
            )

            setattr(self, f"dropout_{sent_id}", nn.Dropout(config.hidden_dropout_prob))


class IndexPosDemuxModule(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_instances = config.num_instances
        # initialize different MLPs for different instances
        self.demux_instance_embedding = torch.nn.Parameter(
            torch.randn(config.num_instances, config.hidden_size)
        )

        num_hidden_demux_layers = (
            config.num_hidden_demux_layers
            if hasattr(config, "num_hidden_demux_layers")
            else 2
        )
        config.num_hidden_demux_layers = num_hidden_demux_layers
        # self.dense_1 = nn.Linear(2 * config.hidden_size, config.hidden_size)
        # self.layer_norm_1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # self.dropout_1 = nn.Dropout(config.hidden_dropout_prob)
        # self.dense_2 = nn.Linear(config.hidden_size, config.hidden_size)
        # self.layer_norm_2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # self.dropout_2 = nn.Dropout(config.hidden_dropout_prob)
        self.dense_1 = nn.Linear(2 * config.hidden_size, config.hidden_size)
        self.layer_norm_1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout_1 = nn.Dropout(config.hidden_dropout_prob)
        for demux_hidden_idx in range(2, config.num_hidden_demux_layers + 1):
            setattr(
                self,
                f"dense_{demux_hidden_idx}",
                nn.Linear(config.hidden_size, config.hidden_size),
            )
            setattr(
                self,
                f"dropout_{demux_hidden_idx}",
                nn.Dropout(config.hidden_dropout_prob),
            )
            setattr(
                self,
                f"layer_norm_{demux_hidden_idx}",
                nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps),
            )

class IndexDemultiplexerTokenLevel(nn.Module):
    """Roberta Head for masked language modeling."""

    def __init__(self, config):
        super().__init__()
        self.num_instances = config.num_instances
        self.dense = nn.Linear(2 * config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # self.decoder = nn.Linear(config.hidden_size, config.num_labels)
        # self.bias = nn.Parameter(torch.zeros(config.num_labels))
        # self.decoder.bias = self.bias

    def forward(self, features, **kwargs):

        # extract the first <num sentence> representations and concatenate with the right word
        batch, seqlength, feature_dim = features.shape
        positional_representations = features[:, : self.num_instances, :]
        # concatenate features with the sentence representations based on sentence_labels
        # don't overwrite sentence labels !!

        # need to expand the batch to the original size, need to make predictions
        # on the original
        positional_representations = positional_representations.unsqueeze(2).expand(
            batch, self.num_instances, seqlength, feature_dim
        )
        features = features.unsqueeze(1).expand(
            batch, self.num_instances, seqlength, feature_dim
        )
        features = torch.cat([positional_representations, features], dim=3)
        # increase the batch size by collapsing the first 2 dimensions
        features = features.view(-1, seqlength, 2 * feature_dim)
        x = self.dense(features)
        x = gelu(x)
        x = self.layer_norm(x)

        # x = self.decoder(x)

        return x

    # def _tie_weights(self):
    #     # To tie those two weights if they get disconnected (on TPU or when the bias is resized)
    #     self.bias = self.decoder.bias

class IndexPosDemultiplexerTokenLevel(nn.Module):
    """Roberta Head for masked language modeling."""

    def __init__(self, config, demux_module):
        super().__init__()
        self.num_instances = config.num_instances
        self.demux_module = demux_module
        self.pre_decoder = nn.Linear(config.hidden_size, config.hidden_size)
        self.pre_decoder_layer_norm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        num_hidden_demux_layers = (
            config.num_hidden_demux_layers
            if hasattr(config, "num_hidden_demux_layers")
            else 2
        )
        self.num_hidden_demux_layers = num_hidden_demux_layers
        self.legacy_demuxing = (
            config.legacy_demuxing if hasattr(config, "legacy_demuxing") else False
        )
        if self.legacy_demuxing:
            logger.warning("Using legacy demuxing!")

    def forward(self, features, **kwargs):

        # extract the first <num sentence> representations and concatenate with the right word
        batch, seqlength, feature_dim = features.shape
        positional_representations = (
            self.demux_module.demux_instance_embedding.unsqueeze(0)
        )
        # concatenate features with the sentence representations based on sentence_labels
        # don't overwrite sentence labels !!

        # need to expand the batch to the original size, need to make predictions
        # on the original
        positional_representations = positional_representations.unsqueeze(2).expand(
            batch, self.num_instances, seqlength, feature_dim
        )
        features = features.unsqueeze(1).expand(
            batch, self.num_instances, seqlength, feature_dim
        )
        features = torch.cat([positional_representations, features], dim=3)
        # increase the batch size by collapsing the first 2 dimensions
        features = features.view(-1, seqlength, 2 * feature_dim)

        if not self.legacy_demuxing:
            x = features
            # run the demux module
            # skip the last layer, one specific layer for the retrieval loss
            for demux_hidden_idx in range(1, self.num_hidden_demux_layers):
                dense_cur = getattr(self.demux_module, f"dense_{demux_hidden_idx}")
                layernorm_cur = getattr(
                    self.demux_module, f"dropout_{demux_hidden_idx}"
                )
                x_in = x
                x = dense_cur(x)
                x = gelu(x)
                x = layernorm_cur(x)
                # add skip connection if not first layer
                if demux_hidden_idx > 1:
                    x = x + x_in

            # project back to the label space
            x_in = x
            x = self.pre_decoder(x)
            x = gelu(x)
            x = self.pre_decoder_layer_norm(x)
            x = x + x_in
            return x
        else:
            x = self.demux_module.dense_1(features)
            x = gelu(x)
            x_demux_1 = self.demux_module.layer_norm_1(x)

            # project back to the label space
            x = self.pre_decoder(x_demux_1)
            x = gelu(x)
            x = self.pre_decoder_layer_norm(x)

            x = x + x_demux_1

            return x

    # def _tie_weights(self):
    #     # To tie those two weights if they get disconnected (on TPU or when the bias is resized)
    #     self.bias = self.decoder.bias


class MLPDemultiplexerTokenLevel(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config, demux_module):
        super().__init__()
        self.num_instances = config.num_instances
        # initialize different MLPs for different sentences
        self.demux_module = demux_module
        # shared vocab layers across different sentences
        self.dense_before_out_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, features):
        # extract the first <num sentence> representations and concatenate with the right word
        _, seq_length, feature_dim = features.shape
        all_feats = []
        for sent_id in range(self.num_instances):
            cur_dense1 = getattr(self.demux_module, f"dense_{sent_id}")
            cur_layer_norm = getattr(self.demux_module, f"layer_norm_{sent_id}")
            dropout = getattr(self.demux_module, f"dropout_{sent_id}")
            inp_feat = features
            x = dropout(inp_feat)
            x = cur_dense1(x)
            x = gelu(x)
            x = cur_layer_norm(x)
            all_feats.append(x.unsqueeze(1))
        #  B x L x dim
        # stack to get B x N X L X dim
        all_feats = torch.cat(all_feats, dim=1)
        # collapse the first 2 dimensions
        all_feats = all_feats.view(-1, seq_length, feature_dim)
        x = self.dense_before_out_proj(all_feats)
        x = gelu(x)
        x = self.layernorm(x)
        return x


class RetrievalHeadMLPDemultiplexing(nn.Module):
    def __init__(self, config, demux_module):
        super().__init__()
        self.num_instances = config.num_instances
        # initialize different MLPs for different instances
        self.demux_module = demux_module

        # shared vocab layers across different instances
        self.dense_pre_vocab = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm_pre_vocab = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.decoder.bias = self.bias

    def forward(self, features, instance_labels, **kwargs):
        # extract the first <num instance> representations and concatenate with the right word
        batch, seqlength, _ = features.shape
        all_feats = torch.zeros_like(features)
        all_feats = all_feats.view(-1, features.shape[-1])

        for sent_id in range(self.num_instances):
            cur_dense1 = getattr(self.demux_module, f"dense_{sent_id}")
            cur_layer_norm = getattr(self.demux_module, f"layer_norm_{sent_id}")
            dropout = getattr(self.demux_module, f"dropout_{sent_id}")

            cur_sent_mask = instance_labels == sent_id
            cur_sent_feats = features[cur_sent_mask]

            x = dropout(cur_sent_feats)
            x = cur_dense1(x)
            x = gelu(x)
            x = cur_layer_norm(x)

            all_feats[cur_sent_mask.view(-1), :] = x

        # reshape into  B x L x V
        all_feats = all_feats.view(batch, seqlength, -1)
        # project back to size of vocabulary with bias
        x = self.dense_pre_vocab(all_feats)
        x = gelu(x)
        x = self.layer_norm_pre_vocab(x)
        x = self.decoder(x)

        return x

    def _tie_weights(self):
        # To tie those two weights if they get disconnected (on TPU or when the bias is resized)
        self.bias = self.decoder.bias


class RetrievalHeadIndexDemultiplexing(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_instances = config.num_instances
        self.dense = nn.Linear(2 * config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.decoder = nn.Linear(config.hidden_size, config.vocab_size)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.decoder.bias = self.bias

    def forward(self, features, instance_labels, mlm_mask=None, labels=None, **kwargs):
        if mlm_mask is None:
            # extract the first <num instance> representations and concatenate with the right word
            batch, seqlength, _ = features.shape
            positional_representations = features[:, : self.num_instances, :]
            # concatenate features with the instance representations based on instance labels
            instance_labels_copy = instance_labels.clone()
            instance_labels_copy[instance_labels == -100] = 0
            positional_embeds = positional_representations[
                torch.arange(batch, device=features.device)
                .unsqueeze(1)
                .repeat(1, seqlength),
                instance_labels_copy,
            ]
            features = torch.cat([positional_embeds, features], dim=2)
            x = self.dense(features)
            x = gelu(x)
            x = self.layer_norm(x)

            # project back to size of vocabulary with bias
            x = self.decoder(x)

            return x
        else:
            # extract the first <num instance> representations and concatenate with the right word
            batch, seqlength, feature_dim = features.shape
            positional_representations = features[:, : self.num_instances, :]
            # concatenate features with the instance representations based on instance labels
            positional_representations = positional_representations.unsqueeze(2).expand(
                batch, self.num_instances, seqlength, feature_dim
            )
            features = features.unsqueeze(1).expand(
                batch, self.num_instances, seqlength, feature_dim
            )
            features = torch.cat([positional_representations, features], dim=3)
            features = features.view(-1, seqlength, 2 * feature_dim)

            # only demux the features that are masked
            features = features[mlm_mask]
            x = self.dense(features)
            x = gelu(x)
            x = self.layer_norm(x)

            # project back to size of vocabulary with bias
            x = self.decoder(x)

            return x

    def _tie_weights(self):
        # To tie those two weights if they get disconnected (on TPU or when the bias is resized)
        self.bias = self.decoder.bias


class RetrievalHeadIndexPosDemultiplexing(nn.Module):
    def __init__(self, config, demux_module):
        super().__init__()
        self.num_instances = config.num_instances
        self.demux_module = demux_module

        self.decoder = nn.Linear(config.hidden_size, config.vocab_size)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.decoder.bias = self.bias
        self.pre_decoder = nn.Linear(config.hidden_size, config.hidden_size)
        self.pre_decoder_layer_norm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        num_hidden_demux_layers = (
            config.num_hidden_demux_layers
            if hasattr(config, "num_hidden_demux_layers")
            else 2
        )
        self.num_hidden_demux_layers = num_hidden_demux_layers
        self.legacy_demuxing = (
            config.legacy_demuxing if hasattr(config, "legacy_demuxing") else False
        )
        if self.legacy_demuxing:
            logger.warning("Using legacy demuxing!")

    def forward(self, features, instance_labels, mlm_mask=None, **kwargs):
        if mlm_mask is None:
            batch, seqlength, _ = features.shape
            positional_representations = (
                self.demux_module.demux_instance_embedding.unsqueeze(0).expand(
                    batch, -1, -1
                )
            )
            # concatenate features with the instance representations based on instance labels
            instance_labels_copy = instance_labels.clone()
            instance_labels_copy[instance_labels == -100] = 0
            positional_embeds = positional_representations[
                torch.arange(batch, device=features.device)
                .unsqueeze(1)
                .repeat(1, seqlength),
                instance_labels_copy,
            ]
            features = torch.cat([positional_embeds, features], dim=2)
            if not self.legacy_demuxing:
                x = features
                for demux_hidden_idx in range(1, self.num_hidden_demux_layers + 1):
                    cur_dense = getattr(self.demux_module, f"dense_{demux_hidden_idx}")
                    cur_layer_norm = getattr(
                        self.demux_module, f"layer_norm_{demux_hidden_idx}"
                    )
                    x_in = x
                    x = cur_dense(x)
                    x = gelu(x)
                    x = cur_layer_norm(x)
                    if demux_hidden_idx > 1:
                        x = x + x_in

                x_in = x
                # project back to the label space
                x = self.pre_decoder(x)
                x = gelu(x)
                x = self.pre_decoder_layer_norm(x)
                x = x + x_in
                x = self.decoder(x)

                return x
            else:
                x = self.demux_module.dense_1(features)
                x = gelu(x)
                x = self.demux_module.layer_norm_1(x)
                x = self.demux_module.dense_2(x)
                x = gelu(x)
                x = self.demux_module.layer_norm_2(x)

                # project back to the label space
                x = self.pre_decoder(x)
                x = gelu(x)
                x = self.pre_decoder_layer_norm(x)
                x = self.decoder(x)

                return x

        else:
            batch, seqlength, feature_dim = features.shape
            positional_representations = (
                self.demux_module.demux_instance_embedding.unsqueeze(0)
            )

            positional_representations = positional_representations.unsqueeze(2).expand(
                batch, self.num_instances, seqlength, feature_dim
            )
            features = features.unsqueeze(1).expand(
                batch, self.num_instances, seqlength, feature_dim
            )
            features = torch.cat([positional_representations, features], dim=3)
            features = features.view(-1, seqlength, 2 * feature_dim)

            # only demux the features that are masked
            features = features[mlm_mask]

            if not self.legacy_demuxing:
                x = features
                for demux_hidden_idx in range(1, self.num_hidden_demux_layers + 1):
                    cur_dense = getattr(self.demux_module, f"dense_{demux_hidden_idx}")
                    cur_layer_norm = getattr(
                        self.demux_module, f"layer_norm_{demux_hidden_idx}"
                    )
                    x_in = x
                    x = cur_dense(x)
                    x = gelu(x)
                    x = cur_layer_norm(x)
                    if demux_hidden_idx > 1:
                        x = x + x_in

                x_in = x
                # project back to the label space
                x = self.pre_decoder(x)
                x = gelu(x)
                x = self.pre_decoder_layer_norm(x)
                x = x + x_in
                x = self.decoder(x)

                return x
            else:
                x = self.demux_module.dense_1(features)
                x = gelu(x)
                x = self.demux_module.layer_norm_1(x)
                x = self.demux_module.dense_2(x)
                x = gelu(x)
                x = self.demux_module.layer_norm_2(x)

                # project back to the label space
                x = self.pre_decoder(x)
                x = gelu(x)
                x = self.pre_decoder_layer_norm(x)
                x = self.decoder(x)

                return x

    def _tie_weights(self):
        # To tie those two weights if they get disconnected (on TPU or when the bias is resized)
        self.bias = self.decoder.bias


class HierachicalSoftmaxLayer(nn.Module):
    def __init__(self, config, num_buckets):
        super().__init__()
        self.num_buckets = num_buckets
        self.ntokens_per_bucket = math.ceil(config.vocab_size / num_buckets)
        # partition the vocab into buckets of size ntokens_per_bucket
        indices = torch.randperm(config.vocab_size)
        self.reverse_indices = torch.zeros(config.vocab_size).long()
        for i in range(config.vocab_size):
            self.reverse_indices[indices[i]] = i
        buckets = torch.zeros(num_buckets, self.ntokens_per_bucket).long()
        for i in range(num_buckets - 1):
            buckets[i, :] = indices[
                i * self.ntokens_per_bucket : (i + 1) * self.ntokens_per_bucket
            ]
        # last bucket is the remaining tokens
        buckets[
            -1, : (config.vocab_size - (num_buckets - 1) * self.ntokens_per_bucket)
        ] = indices[(num_buckets - 1) * self.ntokens_per_bucket :]
        self.buckets = buckets
        self.bucket_layer = nn.Linear(config.hidden_size, num_buckets)
        # second layer of the softmax
        self.output_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        # initalize norm of embeddings to 1
        self.output_embeddings.weight.data = self.output_embeddings.weight.data / (
            torch.norm(self.output_embeddings.weight.data, dim=1).unsqueeze(1) + 1e-6
        )
        self.config = config
        self.indices = indices

    def forward(self, x, labels=None):
        # x : (number of predictions, hidden_size)
        # first layer of the softmax
        x_shape = x.shape
        x = x.view(-1, self.config.hidden_size)
        if labels is not None:
            labels = labels.view(-1)
            # s_t = time.time()
            # map labels to buckets
            mapped_labels = self.reverse_indices[labels]
            gt_buckets = mapped_labels // self.ntokens_per_bucket
            which_bucket_scores = self.bucket_layer(
                x
            )  # (number of predictions, num_buckets)
            # get the bucketed word embeddings
            which_bucket_probs = F.softmax(which_bucket_scores, dim=1)
            which_bucket_probs = torch.clamp(which_bucket_probs, min=1e-6)
            # logger.warn("Time to map labels to buckets: {}".format(time.time() - s_t))
            # s_t = time.time()
            all_bucket_embeddings = self.output_embeddings(self.buckets.to(x.device))
            # (num buckets, num tokens per bucket, hidden size)
            # get embeddings for the gt buckets
            gt_buckets_embeddings = all_bucket_embeddings[gt_buckets]
            # (number of predictions, num tokens per bucket, hidden size)
            # dot product to get scores among tokens in each bucket
            bucket_scores = torch.bmm(
                x.unsqueeze(1),
                gt_buckets_embeddings.permute(0, 2, 1),
            )  # (number of predictions, 1, num_tokens per bucket)
            bucket_scores = bucket_scores.squeeze(1)
            bucket_probs = F.softmax(bucket_scores, dim=1)
            bucket_probs = torch.clamp(bucket_probs, min=1e-6)
            log_bucket_probs = torch.log(bucket_probs)
            # get the bucket for each token
            # generate scores for all tokens by merging the two levels
            # bucket probs:
            # logger.warn("Time to get probs within bucket: {}".format(time.time() - s_t))
            # s_t = time.time()
            probs = which_bucket_probs.repeat_interleave(self.ntokens_per_bucket, dim=1)
            log_probs = torch.log(probs)
            log_probs = log_probs - math.log(self.ntokens_per_bucket)
            # update probs for tokens gt bucket
            gt_buckets_mask = torch.zeros_like(which_bucket_probs)
            gt_buckets_mask[torch.arange(gt_buckets_mask.shape[0]), gt_buckets] = 1

            gt_buckets_token_mask = torch.repeat_interleave(
                gt_buckets_mask, self.ntokens_per_bucket, dim=1
            )
            gt_buckets_token_mask = gt_buckets_token_mask.bool()
            log_probs[gt_buckets_token_mask] = (
                torch.log(
                    which_bucket_probs[torch.arange(x.shape[0]), gt_buckets].unsqueeze(
                        1
                    )
                )
                + log_bucket_probs
            ).view(-1)
            # remap the probabilities to the original vocabulary
            remapped_log_probs = torch.zeros(
                x.shape[0], self.config.vocab_size, device=x.device
            )
            # truncate the dummy tokens in the last bucket
            remapped_log_probs[:, self.indices] = log_probs[:, : self.config.vocab_size]
            # logger.warn("Time to get probs for all tokens: {}".format(time.time() - s_t))
            return remapped_log_probs
        else:
            # hierarchical softmax doesn't save memory during inference
            scores = torch.mm(
                x,
                self.output_embeddings(
                    torch.arange(self.config.vocab_size, device=x.device)
                ).t(),
            )  # (number of predictions, 1, vocab size)
            return scores.view(*x_shape[:-1], self.config.vocab_size)


class RetrievalHeadIndexPosDemultiplexingHierarchical(nn.Module):
    def __init__(self, config, demux_module):
        super().__init__()
        self.num_instances = config.num_instances
        self.demux_module = demux_module
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size)
        self.pre_decoder = nn.Linear(config.hidden_size, config.hidden_size)
        self.pre_decoder_layer_norm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.config = config

    def forward(self, features, instance_labels, mlm_mask=None, labels=None, **kwargs):
        # extract the first <num instance> representations and concatenate with the right word
        if mlm_mask is None:
            batch, seqlength, _ = features.shape
            positional_representations = (
                self.demux_module.demux_instance_embedding.unsqueeze(0).expand(
                    batch, -1, -1
                )
            )
            # concatenate features with the instance representations based on instance labels
            instance_labels_copy = instance_labels.clone()
            instance_labels_copy[instance_labels == -100] = 0
            positional_embeds = positional_representations[
                torch.arange(batch, device=features.device)
                .unsqueeze(1)
                .repeat(1, seqlength),
                instance_labels_copy,
            ]
            features = torch.cat([positional_embeds, features], dim=2)

            x = self.demux_module.dense_1(features)
            x = gelu(x)
            x = self.demux_module.layer_norm_1(x)
            x = self.demux_module.dense_2(x)
            x = gelu(x)
            x = self.demux_module.layer_norm_2(x)

            # project back to the label space
            x = self.pre_decoder(x)
            x = gelu(x)
            x = self.pre_decoder_layer_norm(x)
            x = self.decoder(x)

            return x

        else:
            batch, seqlength, feature_dim = features.shape
            positional_representations = (
                self.demux_module.demux_instance_embedding.unsqueeze(0)
            )

            positional_representations = positional_representations.unsqueeze(2).expand(
                batch, self.num_instances, seqlength, feature_dim
            )
            features = features.unsqueeze(1).expand(
                batch, self.num_instances, seqlength, feature_dim
            )
            features = torch.cat([positional_representations, features], dim=3)
            features = features.view(-1, seqlength, 2 * feature_dim)

            # only demux the features that are masked
            features = features[mlm_mask]
            labels = labels[mlm_mask]
            x = self.demux_module.dense_1(features)
            x = gelu(x)
            x = self.demux_module.layer_norm_1(x)
            x = self.demux_module.dense_2(x)
            x = gelu(x)
            x = self.demux_module.layer_norm_2(x)

            # project back to the label space
            x = self.pre_decoder(x)
            x = gelu(x)
            x = self.pre_decoder_layer_norm(x)
            if self.config.use_hierarchical_softmax:
                x = self.decoder(x, labels)
            else:
                x = self.decoder(x)
            return x

    def _tie_weights(self):
        # To tie those two weights if they get disconnected (on TPU or when the bias is resized)
        self.bias = self.decoder.bias
