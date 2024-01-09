#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for sequence to sequence speech recognition.
"""
# You can also adapt this script on your own sequence to sequence speech
# recognition task. Pointers for this are left as comments.

import logging
import os
import sys
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Union, Tuple
import math
import datasets
import evaluate
import torch
import deepspeed
import transformers
from transformers import (
    HfArgumentParser,
    TrainingArguments,
    Trainer,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers import LlamaTokenizer, LlamaForCausalLM, LlamaConfig, WhisperConfig, WhisperFeatureExtractor, ASTConfig
from transformers.deepspeed import is_deepspeed_zero3_enabled
import torch.distributed as dist
import deepspeed.comm as ds_comm
from src.ast_feature_extractor import ASTFeatureExtractor
from src.speech_text_paired_dataset import load_speech_text_paired_dataset, SpeechTextPairedDataCollator
from src.modeling_blsp import BlspModel
from src.modeling_whisper_encoder import WhisperEncoder
from src.modeling_ast_encoder import ASTModel
from src.configuration_blsp import BlspConfig
from src.lora import convert_linear_layer_to_lora, convert_lora_to_linear_layer, unfuse_lora_weight_from_linear_layer, only_optimize_lora_parameters


logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    blsp_model: str = field(
        default=None, metadata={"help": "the path of blsp model"}
    )
    llama_model: str = field(
        default="decapoda-research/llama-7b-hf", metadata={"help": "the path of base model"}
    )
    audio_model: str = field(
        default="openai/whisper-small", metadata={"help": "the path of audio model"}
    )
    audio_model_type: str = field(
        default="ast", metadata={"help": "the type of audio model"}
    )
    lora_dim: int = field(
        default=0, metadata={"help": "If > 0, use LoRA for efficient training."}
    )
    lora_module_name: str = field(
        default="self_attn", metadata={"help": "part module name to add lora, e.g., self_attn ."}
    )

@dataclass
class ExtraArguments:
    offload: bool = field(
        default=True, metadata={"help": "cpu offload training"}
    )
    zero_stage: int = field(
        default=2, metadata={"help": "zero stage"}
    )

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    data: str = field(
        metadata={
            "help": "the root to load dataset"
        },
    )
    manifest_files: str = field(
        default="",
        metadata={
            "help": "The name of the training unit text paired set split to use."
        },
    )
    instruction: str = field(
        default="",
        metadata={
            "help": "The text prefix instruction before speech input, default None"
        },
    )

def to_device(batch, device):
    output = {}
    for k, v in batch.items():
        try:
            output[k] = v.to(device)
        except:
            output[k] = v
    return output




def print_rank_0(msg='', **kwargs):
    if not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0:
        print(msg, **kwargs)


def all_reduce_mean(tensor):
    with torch.no_grad():
        reduced_tensor = tensor.clone()
        ds_comm.all_reduce(reduced_tensor)
        return reduced_tensor / ds_comm.get_world_size()

def main():

    
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    global_rank = int(os.environ.get("RANK", -1))
    print(f"local_rank: {local_rank}, global_rank: {global_rank}")
    if local_rank == -1:
        device = torch.device("cuda")
    else:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        deepspeed.init_distributed()

    # 1. Parse input arguments
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments, ExtraArguments))
    total_args = parser.parse_args()
    model_args, data_args, training_args, extra_args = parser.parse_args_into_dataclasses()

    # 2. Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Dataset parameters {data_args}")

    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
    logger.info("Training/evaluation parameters %s", training_args)

    # 3. Detecting last checkpoint and eventually continue from last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    
    
    # 4. extractor / audio config / audio encoder
    if model_args.audio_model_type == "whisper":
        EXTRACTOR_CLASS = WhisperFeatureExtractor
        AUDIO_MODEL_CONFIG_CLASS = WhisperConfig 
        AUDIO_ENCODER_CLASS = WhisperEncoder
    elif model_args.audio_model_type == "ast":
        EXTRACTOR_CLASS = ASTFeatureExtractor
        AUDIO_MODEL_CONFIG_CLASS = ASTConfig
        AUDIO_ENCODER_CLASS = ASTModel
    else:
        raise Exception("Unknown audio model type")
    # 5. Load pretrained model
    # Distributed training:
    if model_args.blsp_model is None:
        extractor = EXTRACTOR_CLASS.from_pretrained(model_args.audio_model)
        audio_model_config = AUDIO_MODEL_CONFIG_CLASS.from_pretrained(model_args.audio_model)
        tokenizer = LlamaTokenizer.from_pretrained(model_args.llama_model)
        # The .from_pretrained methods guarantee that only one local process can concurrently
        llama_config = LlamaConfig.from_pretrained(model_args.llama_model)
        blsp_config = BlspConfig(
            audio_model_config.to_dict(),
            llama_config.to_dict(),
            audio_model_type=model_args.audio_model_type # provide audio_model_type to ensure a correct audio_config in checkpoint
        )

        model = BlspModel(blsp_config)
        model.audio_model = None 
        model.llama_model = None # save memory
        model.audio_model = AUDIO_ENCODER_CLASS.from_pretrained(model_args.audio_model)
        model.llama_model = LlamaForCausalLM.from_pretrained(model_args.llama_model, _fast_init=not is_deepspeed_zero3_enabled())
        # add special token for audio and extend embeddings
        model.initialize_audio_tokenizer(tokenizer)
    else:
        extractor = EXTRACTOR_CLASS.from_pretrained(model_args.blsp_model)
        tokenizer = LlamaTokenizer.from_pretrained(model_args.blsp_model)
        model = BlspModel.from_pretrained(model_args.blsp_model)


    # if args.gradient_checkpointing:
    #     if hasattr(model, "enable_input_require_grads"):
    #         model.enable_input_require_grads()
    #     else:
    #         def make_inputs_require_grad(module, input, output):
    #             output.requires_grad_(True)

    #         model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
    
    if model_args.lora_dim > 0:
        model.llama_model = convert_linear_layer_to_lora(model.llama_model, model_args.lora_module_name,
                                             model_args.lora_dim)
    


    # for name, param in model.audio_model.named_parameters():
    #     param.requires_grad = False
    for name, param in model.llama_model.named_parameters():
        if "lora" not in name:
            param.requires_grad = False

    trainable_parameters, trainable_key = 0, []
    for name, param in model.named_parameters():
        if param.requires_grad == True:
            trainable_parameters += param.numel()
            trainable_key.append(name)
    
    print(f"Trainable key: {trainable_key}")
    print(f"Trainable parameters: {trainable_parameters}")
    ### 6. Load dataset
    dataset = load_speech_text_paired_dataset(
        dataroot=data_args.data,
        manifest_files=data_args.manifest_files,
        tokenizer=tokenizer,
        instruction=data_args.instruction
    )

    # Define data collator
    tokenizer.pad_token_id = 0 # llama do not have pad_token_id, need to add manually
    print(f"pad_token_id: {tokenizer.pad_token_id}")
    print(f"bos_token_id: {tokenizer.bos_token_id}")
    print(f"eos_token_id: {tokenizer.eos_token_id}")
    data_collator = SpeechTextPairedDataCollator(
        pad_id=tokenizer.pad_token_id,
        sampling_rate=extractor.sampling_rate,
        extractor=extractor
    )


    from torch.utils.data.distributed import DistributedSampler
    from torch.utils.data import DataLoader
    from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
    from transformers import get_scheduler
    from src.ds_utils import get_train_ds_config
    from deepspeed.utils import log_dist

    if local_rank == -1:
        train_sampler = RandomSampler(dataset)
    else:
        train_sampler = DistributedSampler(dataset, shuffle=False)

    train_dataloader = DataLoader(dataset,
                                  collate_fn=data_collator,
                                  sampler=train_sampler,
                                  batch_size=training_args.per_device_train_batch_size)
    

    if extra_args.offload:
        print("Use DeepSpeedCPUAdam")
        AdamOptimizer = DeepSpeedCPUAdam
    else:
        print("Use FusedAdam")
        AdamOptimizer = FusedAdam

    optimizer = AdamOptimizer(model.parameters(),
                              lr=training_args.learning_rate,
                              betas=(0.9, 0.95))

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / training_args.gradient_accumulation_steps)

    lr_scheduler = get_scheduler(
        name=training_args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=training_args.warmup_steps,
        num_training_steps=num_update_steps_per_epoch * int(training_args.num_train_epochs),
    )

    ds_config = get_train_ds_config(args=training_args,
                                    offload=extra_args.offload,
                                    stage=extra_args.zero_stage,
                                    steps_per_print=training_args.logging_steps,
                                    )

    # optimizer and lr_scheduler are not necessarily specified in the config file.
    model, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        args=total_args,
        config=ds_config,
        lr_scheduler=lr_scheduler,
        dist_init_required=True)

 

    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()


    for epoch in range(int(training_args.num_train_epochs)):

        print_rank_0(
            f"Beginning of Epoch {epoch + 1}/{int(training_args.num_train_epochs)}, Total {num_update_steps_per_epoch} steps per epoch"
        )

        model.train()

        losses = []
        #progress_bar = tqdm(range(num_micro_batches_per_epoch), disable=local_rank != 0)
        global_steps = 0
        for step, batch in enumerate(train_dataloader):
            batch = to_device(batch, training_args.device)
            if training_args.fp16:
                batch["speech_values"] = batch["speech_values"].to(torch.float16)
            elif training_args.bf16:
                batch["speech_values"] = batch["speech_values"].to(torch.bfloat16)
            # print(batch["input_ids"].shape)
            outputs = model(**batch, use_cache=False)
            loss = outputs.loss
            model.backward(loss)
            model.step()
            losses.append(loss)

            if model.is_gradient_accumulation_boundary():
                train_loss_this_step = sum(losses) / len(losses)
                # 1. Logging
                #if model.monitor.enabled:
                #if global_rank == 0 and training_args.local_rank == 0:
                global_steps += 1
                if global_steps % training_args.logging_steps == 0:
                    log_dist(f"Epoch: {epoch} | GlobalSteps: {global_steps} Loss: {train_loss_this_step:.3f}", ranks=[0])
                losses = []

                if global_steps > 0 and global_steps % training_args.save_steps == 0:
                    # 1. reduce mean loss
                    reduced_loss = all_reduce_mean(torch.tensor(train_loss_this_step).to(training_args.device)).item()

                    # 2. saving checkpoint
                    # 3. wait_for_everyone
                    client_sd = dict()

                    # Saving these stats in order to resume training
                    client_sd['global_steps'] = global_steps
                    client_sd['num_epoches'] = epoch

                    ckpt_id = f"epoch{epoch}_iter{global_steps}_loss{reduced_loss:.3f}"


                    checkpoint_dir = f"{training_args.output_dir}/{ckpt_id}"
                    if model_args.lora_dim > 0:
                        lora_fused_model = convert_lora_to_linear_layer(model)
                        #lora_fused_model.save_checkpoint(training_args.output_dir, ckpt_id, client_state=client_sd)

                        state_dict = model.state_dict()
                        for key in list(state_dict.keys()):
                            if "lora" in key:
                                del state_dict[key]
                                continue
                            if key.startswith("module."):
                                new_key = key[7:]
                                state_dict[new_key] = state_dict[key]
                                del state_dict[key]


                        lora_fused_model.save_pretrained(
                                checkpoint_dir, state_dict=state_dict
                        )
                        model = unfuse_lora_weight_from_linear_layer(lora_fused_model)
                    else:
                        #model.save_checkpoint(training_args.output_dir, ckpt_id, client_state=client_sd)
                        state_dict = model.state_dict()
                        for key in list(state_dict.keys()):
                            if key.startswith("module."):
                                new_key = key[7:]
                                state_dict[new_key] = state_dict[key]
                                del state_dict[key]

                        model.save_pretrained(
                                checkpoint_dir, state_dict=state_dict
                        )
                    
                    tokenizer.save_pretrained(checkpoint_dir)
                    extractor.save_pretrained(checkpoint_dir)



    # # 7. Initialize Trainer
    # trainer = Trainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=dataset,
    #     data_collator=data_collator,
    # )

    # # 8. Training
    # if training_args.do_train:
    #     checkpoint = None
    #     if training_args.resume_from_checkpoint is not None:
    #         checkpoint = training_args.resume_from_checkpoint
    #     elif last_checkpoint is not None:
    #         checkpoint = last_checkpoint
    #     train_result = trainer.train(resume_from_checkpoint=checkpoint)
    #     trainer.save_model()
    #     trainer.log_metrics("train", train_result.metrics)
    #     trainer.save_metrics("train", train_result.metrics)
    #     trainer.save_state()

    # results = {}
    # # 9. Save tokenizer for inference load
    # tokenizer.save_pretrained(training_args.output_dir)
    # extractor.save_pretrained(training_args.output_dir)

    # return results


if __name__ == "__main__":
    main()
