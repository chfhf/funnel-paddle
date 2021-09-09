# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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
import paddle
from .. import PretrainedModel, register_base_model

from hf_paddle.models.funnel import modeling_funnel

import requests
from hf_paddle.models.funnel.configuration_funnel import    FUNNEL_PRETRAINED_CONFIG_ARCHIVE_MAP , FunnelConfig
import json
from  collections import defaultdict

__all__ = [
    "FunnelModel",
    "FunnelForSequenceClassification",
    "FunnelForTokenClassification",
    "FunnelForQuestionAnswering"
]
dtype_float = paddle.get_default_dtype()


@register_base_model
class FunnelModel(PretrainedModel):
    def __init__(self, **config):

        super().__init__()
        if len(config)>0: ##
            config_obj = FunnelConfig(**config)
            self.model=modeling_funnel.FunnelModel(config_obj)
            self.init_config=config_obj

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        paddlenlp_model=FunnelModel()
        paddlenlp_model.model=modeling_funnel.FunnelModel.from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        config=dict()
        if pretrained_model_name_or_path in FUNNEL_PRETRAINED_CONFIG_ARCHIVE_MAP:
            config=json.loads(requests.get(FUNNEL_PRETRAINED_CONFIG_ARCHIVE_MAP[pretrained_model_name_or_path]).text)
        config_obj = FunnelConfig(**config)
        paddlenlp_model.init_config=config_obj
        return paddlenlp_model

    def forward(self,*args,
                **kwargs):

        BaseModelOutput=self.model.forward(*args,**kwargs)

        return BaseModelOutput

class FunnelForSequenceClassification(modeling_funnel.FunnelForSequenceClassification):
    def __init__(self, basemodel, num_classes=2 ):
        super().__init__(basemodel.init_config)
        self.funnel=basemodel
        self.num_classes=num_classes

    def forward(self, input_ids, **config):

        outputs = super( ).forward(
            input_ids,
            **config
        )
        return outputs.logits

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args,num_classes=2, **kwargs):
        hf_taskmodel = modeling_funnel.FunnelForSequenceClassification.from_pretrained(pretrained_model_name_or_path,
                                                                                    *args, **kwargs)
        hf_taskmodel.register_forward_post_hook(lambda  self, inputs, outputs : outputs.logits )
        return  hf_taskmodel

class FunnelForTokenClassification(modeling_funnel.FunnelForTokenClassification):
    def __init__(self, basemodel, num_classes=2 ):
        super( ).__init__(basemodel.init_config)
        self.funnel=basemodel
        self.num_classes=num_classes

    def forward( self,input_ids , **config):

        outputs = super( ).forward(
            input_ids,
            **config
        )
        return outputs.logits



class FunnelForQuestionAnswering(modeling_funnel.FunnelForQuestionAnswering):
    def __init__(self, basemodel  ):
        super( ).__init__(basemodel.init_config)
        self.funnel=basemodel


    def forward( self,input_ids , **config):
        outputs = super( ).forward(
            input_ids,
            **config
        )
        return outputs.start_logits,outputs.end_logits



    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        hf_taskmodel = modeling_funnel.FunnelForQuestionAnswering.from_pretrained(pretrained_model_name_or_path,
                                                                                    *args, **kwargs)
        hf_taskmodel.register_forward_post_hook(lambda  self, inputs, outputs :(outputs.start_logits,outputs.end_logits))
        return  hf_taskmodel
