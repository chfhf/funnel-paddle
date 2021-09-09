from paddlenlp.transformers import  FunnelModel,FunnelTokenizerFast

import os
import paddle
import torch
import numpy as np
import time

paddle.set_device("cpu")
model_list=(   "funnel-transformer/small",  # B4-4-4H768
    "funnel-transformer/small-base",  # B4-4-4H768, no decoder
    "funnel-transformer/medium",  # B6-3x2-3x2H768
    "funnel-transformer/medium-base",  # B6-3x2-3x2H768, no decoder
    "funnel-transformer/intermediate",  # B6-6-6H768
    "funnel-transformer/intermediate-base",  # B6-6-6H768, no decoder
    "funnel-transformer/large",  # B8-8-8H1024
    "funnel-transformer/large-base",  # B8-8-8H1024, no decoder
    "funnel-transformer/xlarge-base",  # B10-10-10H1024
    "funnel-transformer/xlarge" )



for model in model_list:


    print("converting ", model)
    # if os.path.isfile(f"data/{model}/tokenizer.json"):
    #     print(model,"is converted, skip")
    #     continue

    tokenizer = FunnelTokenizerFast.from_pretrained(f"{model}")

    PDImplemBertModel = FunnelModel

    pd_model = PDImplemBertModel.from_pretrained(model)
    pd_model.save_pretrained(f"data/{model}/")
    tokenizer.save_pretrained(f"data/{model}/")

