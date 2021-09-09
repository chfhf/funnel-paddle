# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
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
import numpy as np
import paddle

from hf_paddle.models.funnel import tokenization_funnel_fast
__all__ = [ 'FunnelTokenizerFast' ]
from collections import Iterable


class FunnelTokenizerFast(tokenization_funnel_fast.FunnelTokenizerFast):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)

    def __call__(self, *args,  **kwargs):
        result = super().__call__(*args,  **kwargs)
        if 'length' in result:
            result['seq_len']=result['length']
            del result['length']
        if 'overflow_to_sample_mapping' in result:
            result['overflow_to_sample']=result['overflow_to_sample_mapping']
            del result['overflow_to_sample_mapping']

        # if "token_type_ids" in result:
        #     if not isinstance(args[0], str):
        #         result['token_type_ids'] = [(np.array(x) > 0).astype("int") for x in result['input_ids']]
        #     else:
        #         try:
        #             result['token_type_ids']= (np.array(result['input_ids'])>0 ).astype("int")
        #         except:
        #             result['token_type_ids'] = [(np.array(x) > 0).astype("int") for x in result['input_ids']]

        if not isinstance(args[0] ,str):
            keylist=list(result.keys())
            dict_list=[]
            for i in range(len(args[0])):
                dict_list.append(dict())
                for k in keylist:
                    dict_list[i][k]=result[k][i]
            result=dict_list
        else:
            keylist = list(result.keys())
            ###remove outter dim
            for k in keylist:
                if isinstance(result[k][0],Iterable):
                    result[k] = result[k][0]
        return result

