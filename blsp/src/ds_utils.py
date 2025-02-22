
GLOBAL_BATCH_SIZE = 32
MICRO_BATCH_SIZE = 4

def get_train_ds_config(args=None,
                        offload=True,
                        stage=2,
                        enable_hybrid_engine=False,
                        inference_tp_size=1,
                        release_inference_cache=False,
                        pin_parameters=True,
                        tp_gather_partition_size=8,
                        max_out_tokens=512, monitor_dir=None, steps_per_print=10,
                        gradient_accumulation_steps=1):

    device = "cpu" if offload else "none"
    zero_opt_dict = {
        "stage": stage,
        "offload_param": {
            "device": device
        },
        "offload_optimizer": {
            "device": device
        },
        "stage3_param_persistence_threshold": 1e4,
        "stage3_max_live_parameters": 3e7,
        "stage3_prefetch_bucket_size": 3e7,
        "memory_efficient_linear": False
    }

    ds_config = {
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": "auto",
        "steps_per_print": steps_per_print,
        "zero_optimization": zero_opt_dict,
        "fp16": {
            "enabled": True,
            "loss_scale_window": 100
        },
        "bf16": {
            "enabled": False
        },
        "gradient_clipping": 1.0,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "prescale_gradients": False,
        "wall_clock_breakdown": False,
        "hybrid_engine": {
            "enabled": enable_hybrid_engine,
            "max_out_tokens": max_out_tokens,
            "inference_tp_size": inference_tp_size,
            "release_inference_cache": release_inference_cache,
            "pin_parameters": pin_parameters,
            "tp_gather_partition_size": tp_gather_partition_size,
        }
    }

    # if args.wandb_enable:
    #     ds_monitoring_config = {
    #         "wandb": {
    #             "enabled": True,
    #             "project": args.wandb_project,
    #             "team": args.wandb_team,
    #             "group": args.wandb_group,
    #         }
    #     }

    #     deep_update(ds_config, ds_monitoring_config)

    return ds_config
