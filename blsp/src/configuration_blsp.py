"""BLSP config"""

from transformers import PretrainedConfig, LlamaConfig, WhisperConfig, ASTConfig
from transformers import logging

logger = logging.get_logger(__name__)

class BlspConfig(PretrainedConfig):
    def __init__(
        self, 
        audio_config=None, 
        llama_config=None,
        conv_kernel_sizes="5,5,5,5",
        adapter_inner_dim=512,
        audio_model_type="whisper",
        **kwargs
    ):
        super().__init__(**kwargs)

        if audio_config is None:
            audio_config = {}
            logger.info("whisper config is None. Initializing the WhisperConfig with default values")
        
        if llama_config is None:
            llama_config = {}
            logger.info("llama config is None. Initializing the LlamaConfig with default values")

        if audio_model_type == "whisper":
            AUDIO_MODEL_CONFIG_CLASS = WhisperConfig 
        elif audio_model_type == "ast":
            AUDIO_MODEL_CONFIG_CLASS = ASTConfig
        else:
            raise Exception("Please provide correct audio_model_type")


        self.audio_config = AUDIO_MODEL_CONFIG_CLASS(**audio_config).to_dict()
        self.llama_config = LlamaConfig(**llama_config).to_dict()

        self.conv_kernel_sizes = conv_kernel_sizes
        self.adapter_inner_dim = adapter_inner_dim
        self.audio_model_type = audio_model_type
