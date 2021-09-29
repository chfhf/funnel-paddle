from paddlenlp.transformers import  FunnelModel,FunnelTokenizerFast
from transformers import FunnelModel as FunnelModel_hg

import os
import paddle
import torch
import numpy as np
import time
from collections import OrderedDict

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



def load_pytorch_pretrain_model_remove_prefix(paddle_model, pytorch_state_dict,pytorch_prefix=""):
    '''
    paddle_model: dygraph layer object
    pytorch_state_dict: pytorch state_dict, assume in CPU device
    '''

    paddle_weight=paddle_model.state_dict()
    print("paddle num_params:",len(paddle_weight))
    print("torch num_params:", len(pytorch_state_dict))
    new_weight_dict=OrderedDict()
    torch_key_list=[]
    for key in pytorch_state_dict.keys():
        if "num_batches_tracked" in key:
            continue
        torch_key_list.append(key.replace(pytorch_prefix,""))
    paddle_key_list = []
    for key in paddle_weight.keys():
        if ".cell" in key:
            continue
        paddle_key_list.append(key.replace(pytorch_prefix,""))
    torch_key_set=set(torch_key_list)
    paddle_key_set=set(paddle_key_list)
    paddle_unique_keys=paddle_key_set-torch_key_set
    print("paddle_unique_keys",paddle_unique_keys)
    missingkeys = torch_key_set - paddle_key_set
    print("torch_unique_keys", missingkeys)
    # _fast_init=True
    # if _fast_init:
    #     # retrieve unintialized modules and initialize
    #     missingkeys=torch_key_set-paddle_key_set
    #     print("torch unique key , checking mis-alignment")
    #     print(missingkeys)
    #     unintialized_modules = paddle_model.retrieve_modules_from_names(
    #         missingkeys, add_prefix="", remove_prefix=""
    #     )
    #     for module in unintialized_modules:
    #         paddle_model._init_weights(module)

    paddle_weight = paddle_model.state_dict()
    for torch_key in torch_key_set:
        # if "linears_prediction.4" not in paddle_key or "weight" not in paddle_key:
        #     continue
        paddle_key=torch_key
        if pytorch_prefix+paddle_key in paddle_weight:
            paddle_key=pytorch_prefix+paddle_key
        if paddle_key not in paddle_weight:
            continue
        if pytorch_prefix+torch_key in pytorch_state_dict:
            torch_key=pytorch_prefix+torch_key

        # print(torch_key, paddle_key, pytorch_state_dict[torch_key].shape,paddle_weight[paddle_key].shape)
        if len(pytorch_state_dict[torch_key].shape)==0:
            continue
        ##handle all FC weight cases
        if (  ("weight" in torch_key and "embed" not in torch_key and "conv" not in torch_key) and (len(pytorch_state_dict[torch_key].shape)==2) and   (pytorch_state_dict[torch_key].shape[0]==pytorch_state_dict[torch_key].shape[1]) ) or (len(pytorch_state_dict[torch_key].shape)==2 and (pytorch_state_dict[torch_key].shape[0]!=pytorch_state_dict[torch_key].shape[0]) and pytorch_state_dict[torch_key].shape[0]==pytorch_state_dict[torch_key].shape[1]):
            new_weight_dict[paddle_key] = pytorch_state_dict[torch_key].cpu().detach().numpy().T.astype("float32")
        elif int(paddle_weight[paddle_key].shape[-1])==int(pytorch_state_dict[torch_key].shape[-1])  :
            new_weight_dict[paddle_key]=pytorch_state_dict[torch_key].cpu().detach().numpy().astype("float32")
        else:
            new_weight_dict[paddle_key] = pytorch_state_dict[torch_key].cpu().detach().numpy().T.astype("float32")
        del pytorch_state_dict[torch_key] ##save memory
    paddle_model.set_dict(new_weight_dict)
    del new_weight_dict
    return paddle_model.state_dict()



for model in model_list:


    print("converting ", model)
    # if os.path.isfile(f"data/{model}/tokenizer.json"):
    #     print(model,"is converted, skip")
    #     continue
    PTImplemBertModel=FunnelModel_hg.from_pretrained(f"{model}")
    tokenizer = FunnelTokenizerFast.from_pretrained(f"{model}")

    PDImplemBertModel = FunnelModel
    pytorch_state_dict=PTImplemBertModel.state_dict()
    pd_model=FunnelModel(**PTImplemBertModel.config.to_dict())
    paddle_state_dict = load_pytorch_pretrain_model_remove_prefix(pd_model, pytorch_state_dict,
                                                                  pytorch_prefix=PTImplemBertModel.base_model_prefix + ".")
    pd_model.set_dict(paddle_state_dict)
    pd_model.save_pretrained(f"data/{model}/")
    tokenizer.save_pretrained(f"data/{model}/")

