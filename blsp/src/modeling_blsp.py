import math
from typing import List, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers import PreTrainedModel, LlamaForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import LlamaConfig, WhisperConfig, ASTConfig

try:
    from .configuration_blsp import BlspConfig
    from .modeling_whisper_encoder import WhisperEncoder
    from .modeling_ast_encoder import ASTModel
except:
    from configuration_blsp import BlspConfig
    from modeling_whisper_encoder import WhisperEncoder
    from modeling_ast_encoder import ASTModel


def lengths_to_padding_mask(lens):
    bsz, max_lens = lens.size(0), torch.max(lens).item()
    mask = torch.arange(max_lens).to(lens.device).view(1, max_lens)
    mask = mask.expand(bsz, -1) >= lens.view(bsz, 1).expand(-1, max_lens)
    return mask

class Conv1dSubsampler(nn.Module):
    """Convolutional subsampler: a stack of 1D convolution (along temporal
    dimension) followed by non-linear activation via gated linear units
    (https://arxiv.org/abs/1911.08460)
    Args:
        in_channels (int): the number of input channels
        mid_channels (int): the number of intermediate channels
        out_channels (int): the number of output channels
        kernel_sizes (List[int]): the kernel size for each convolutional layer
    """

    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        kernel_sizes: List[int] = (3, 3),
    ):
        super(Conv1dSubsampler, self).__init__()
        self.n_layers = len(kernel_sizes)
        self.conv_layers = nn.ModuleList(
            nn.Conv1d(
                in_channels if i == 0 else mid_channels // 2,
                mid_channels if i < self.n_layers - 1 else out_channels * 2,
                k,
                stride=2,
                padding=k // 2,
            )
            for i, k in enumerate(kernel_sizes)
        )

    def get_out_seq_lens_tensor(self, in_seq_lens_tensor):
        out = in_seq_lens_tensor.clone()
        for _ in range(self.n_layers):
            out = ((out.float() - 1) / 2 + 1).floor().long()
        return out

    def forward(self, src_tokens, src_lengths):
        bsz, in_seq_len, _ = src_tokens.size()  # B x T x (C x D)
        x = src_tokens.transpose(1, 2).contiguous()  # -> B x (C x D) x T
        for conv in self.conv_layers:
            x = conv(x)
            x = nn.functional.glu(x, dim=1)
        _, _, out_seq_len = x.size()
        x = x.transpose(1, 2).transpose(0, 1).contiguous()  # -> T x B x (C x D)
        return x, self.get_out_seq_lens_tensor(src_lengths)


class Adapter(nn.Module):
    def __init__(
        self,
        in_dim: int,
        mid_dim: int,
    ):
        super(Adapter, self).__init__()

        self.fc1 = nn.Linear(in_dim, mid_dim, bias=False)
        self.fc2 = nn.Linear(mid_dim, in_dim, bias=False)
        self.activation = nn.GELU()

    def forward(self, x):
        residual = x
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)

        return residual + x


class BlspModel(PreTrainedModel):
    config_class = BlspConfig
    base_model_prefix = "blsp"

    def __init__(self, config: BlspConfig):
        super().__init__(config)
        
        if config.audio_model_type == "whisper":
            AUDIO_MODEL_CONFIG_CLASS = WhisperConfig 
            AUDIO_ENCODER_CLASS = WhisperEncoder
        elif config.audio_model_type == "ast":
            AUDIO_MODEL_CONFIG_CLASS = ASTConfig
            AUDIO_ENCODER_CLASS = ASTModel



        self.audio_config = AUDIO_MODEL_CONFIG_CLASS(**config.audio_config)
        self.llama_config = LlamaConfig(**config.llama_config)

        self.audio_model = AUDIO_ENCODER_CLASS(self.audio_config)
        self.llama_model = LlamaForCausalLM(self.llama_config)

        if config.audio_model_type == "whisper":
            in_d = self.audio_config.d_model
        elif config.audio_model_type == "ast":
            in_d = self.audio_config.hidden_size
            
        out_d = self.llama_config.hidden_size
        mid_d = 256 #2 * in_d # 256
        print(f"in_d: {in_d}, mid_d: {mid_d}, out_d: {out_d}")
        self.subsampler = Conv1dSubsampler(
            in_d,
            mid_d,
            out_d,
            [int(k) for k in config.conv_kernel_sizes.split(",")],
        )
        self.speech_ln = torch.nn.LayerNorm(out_d, 1e-5, True)
        self.adapter = Adapter(out_d, config.adapter_inner_dim)
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        speech_values: Optional[torch.FloatTensor] = None,
        speech_attention_mask: Optional[torch.LongTensor] = None,
        suffix_input_ids: Optional[torch.LongTensor] = None,
        suffix_attention_mask: Optional[torch.LongTensor] = None,
        suffix_labels: Optional[torch.LongTensor] = None,
    ):
        ### 1. forward speech
        speech_embeds, speech_attention_mask = self.get_speech_features(speech_values, speech_attention_mask)
        speech_labels = torch.LongTensor(speech_embeds.size(0), speech_embeds.size(1)).fill_(-100).to(speech_embeds.device)
        
        ### 2. forward llama
        prefix_embeds = self.llama_model.get_input_embeddings()(input_ids)
        suffix_embeds = self.llama_model.get_input_embeddings()(suffix_input_ids)
        
        inputs_embeds = torch.cat([prefix_embeds, speech_embeds, suffix_embeds], dim=1)
        attention_mask = torch.cat([attention_mask, speech_attention_mask, suffix_attention_mask], dim=1)
        labels = torch.cat([labels, speech_labels, suffix_labels], dim=1)
        #print(f"prefix_shape: {prefix_embeds.shape}, speech_shape: {speech_embeds.shape}, suffix_shape: {suffix_embeds.shape}")
        return self.llama_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            labels=labels,
        )


    def get_speech_features(self, speech_values, speech_attention_mask):
        w2v_args = {
            "input_features": speech_values,
            "attention_mask": speech_attention_mask,
        }
        output = self.audio_model(**w2v_args)
        speech_embeds = output.last_hidden_state # B x T x C
        speech_lengths = output.output_lengths

        speech_embeds, speech_lengths = self.subsampler(speech_embeds, speech_lengths)
        speech_embeds = speech_embeds.transpose(0,1) # T x B x C -> B x T x C
        speech_padding_mask = lengths_to_padding_mask(speech_lengths)
        speech_atts = ~speech_padding_mask

        speech_embeds = self.adapter(speech_embeds)
        speech_embeds = self.speech_ln(speech_embeds)

        return speech_embeds, speech_atts

    @torch.no_grad()
    def generate(
        self,
        input_ids,
        suffix_input_ids,
        speech_values=None,
        speech_attention_mask=None,
        generation_config=None
    ):
        inputs_embeds, attention_mask = [], []

        prefix_embeds = self.llama_model.get_input_embeddings()(input_ids)
        prefix_attns = torch.ones(prefix_embeds.size(0), prefix_embeds.size(1), dtype=torch.long).to(prefix_embeds.device)
        inputs_embeds.append(prefix_embeds)
        attention_mask.append(prefix_attns)

        if speech_values is not None:
            speech_embeds, speech_attention_mask = self.get_speech_features(speech_values, speech_attention_mask)
            inputs_embeds.append(speech_embeds)
            attention_mask.append(speech_attention_mask)

        suffix_embeds = self.llama_model.get_input_embeddings()(suffix_input_ids)
        suffix_attns = torch.ones(suffix_embeds.size(0), suffix_embeds.size(1), dtype=torch.long).to(suffix_embeds.device)
        inputs_embeds.append(suffix_embeds)
        attention_mask.append(suffix_attns)

        inputs_embeds = torch.cat(inputs_embeds, dim=1)
        attention_mask = torch.cat(attention_mask, dim=1)

        return self.llama_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            generation_config=generation_config
        )
    
    @torch.no_grad()
    def chat(
        self,
        history,
        generation_config=None
    ):
        inputs_embeds = []

        for h in history:
            if len(h) == 1:
                ### text
                input_ids = h[0]
                embeds = self.llama_model.get_input_embeddings()(input_ids)
                inputs_embeds.append(embeds)
            elif len(h) == 2:
                ### speech
                speech_values, speech_attention_mask = h[0], h[1]
                speech_embeds, _ = self.get_speech_features(speech_values, speech_attention_mask)
                inputs_embeds.append(speech_embeds)
            else:
                raise NotImplementedError
        
        inputs_embeds = torch.cat(inputs_embeds, dim=1)

        return self.llama_model.generate(
            inputs_embeds=inputs_embeds,
            generation_config=generation_config
        )


    
    def initialize_audio_tokenizer(
        self,
        tokenizer,
        mm_use_audio_start_end=True,
    ):
        """Set up the tokenizer to handle the various audio tokens."""
        from .special_tokens import DEFAULT_AUDIO_START_TOKEN, DEFAULT_AUDIO_END_TOKEN

        # tokenizer.add_tokens([DEFAULT_AUDIO_PATCH_TOKEN], special_tokens=True)
        # self.resize_token_embeddings(len(tokenizer))
        # todo: update vocab_size in model.config
        if mm_use_audio_start_end:
            num_new_tokens = tokenizer.add_tokens(
                [DEFAULT_AUDIO_START_TOKEN, DEFAULT_AUDIO_END_TOKEN],
                special_tokens=True,
            )
            self.llama_model.resize_token_embeddings(len(tokenizer))
            (
                self.config.audio_start_token,
                self.config.audio_end_token,
            ) = tokenizer.convert_tokens_to_ids(
                [DEFAULT_AUDIO_START_TOKEN, DEFAULT_AUDIO_END_TOKEN]
            )

            if num_new_tokens > 0:
                input_embeddings = self.llama_model.get_input_embeddings().weight.data
                output_embeddings = self.llama_model.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True
                )

                input_embeddings[-num_new_tokens:] = input_embeddings_avg # use the average of other embeddings to initialize audio start and end token embeddings
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

                # update vocab_size in model.config
                self.config.llama_config["vocab_size"] += num_new_tokens
