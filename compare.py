import hf_paddle
from paddlenlp.transformers import FunnelTokenizerFast, FunnelModel as PDImplemBertModel
from transformers.models.funnel import FunnelModel as PTImplemBertModel

# from paddlenlp.transformers import FunnelTokenizerFast, FunnelForSequenceClassification as PDImplemBertModel
# from transformers.models.funnel import FunnelForSequenceClassification as PTImplemBertModel


import paddle
import torch
import numpy as np
import time

paddle.set_device("cpu")

model_list = [

    ("funnel-transformer/xlarge" ,),

]

for model in model_list:
    tokenizer = FunnelTokenizerFast.from_pretrained( model[0])

    inputs = tokenizer("it is a nice day today!")["input_ids"]
    pt_inputs = torch.tensor([inputs]*100)
    pd_inputs = paddle.to_tensor([inputs]*100)
    # pt_model = PTImplemBertModel.from_pretrained(f"{model_folder}/pytorch_model.bin",config=f"{model_folder}/torch_config.json" )# model[0])
    pt_model = PTImplemBertModel.from_pretrained(model[0])
    # pt_model.eval()

    start=time.time()
    pt_outputs = pt_model(pt_inputs )[0]
    pt_outputs.sum().backward()
    print("torch time:",time.time()-start)
    try:
        torch_grad = pt_model.funnel.embeddings.word_embeddings.weight.grad.detach().cpu().numpy()
    except:
        torch_grad = pt_model.embeddings.word_embeddings.weight.grad.detach().cpu().numpy()

    # pd_model = PDImplemBertModel.from_pretrained(f"{model_folder}/pytorch_model.bin",config=f"{model_folder}/torch_config.json") # )
    # pd_model = PDImplemBertModel.from_pretrained("data/paddorch_model.pdparams", config="data/config.json")
    # pd_model = PDImplemBertModel.from_pretrained(f"data/{model[0]}" )  # )
    # pd_model = PDImplemBertModel.from_pretrained("glue/qnli/qnli_ft_model_best")  # )qnli_ft_model_best
    pd_model = PDImplemBertModel.from_pretrained(model[0])

    # pd_model= modeling_deberta.DebertaForQuestionAnswering.from_pretrained("data/pytorch_model.bin.pdparams",config="data/config.json")
    pd_model.train()
    for _ in range(5):
        start = time.time()
        # pd_model.eval()
        # pd_outputs =    pd_model(pd_inputs,token_type_ids=pd_inputs_2)[0]
        pd_outputs = pd_model(pd_inputs )[0]
        pd_outputs2 = torch.from_numpy(pd_outputs.numpy())
        print("mean forward psas difference:", (pt_outputs - pd_outputs2).abs().mean())
        print("max forward pass difference:", (pt_outputs - pd_outputs2).abs().max())

        pd_outputs.sum().backward()
        # adam.clear_gradients()
        # pd_model.save_pretrained(f"data/{model}/")
        print("paddle time:", time.time() - start)
        break
    pd_outputs=torch.from_numpy(pd_outputs.numpy())
    try:
        paddle_grad = pd_model.embeddings.word_embeddings.weight.gradient()
    except:
        paddle_grad = pd_model.model.embeddings.word_embeddings.weight.gradient()

    print(f"huggingface {model[0]} vs paddle {model[0]}")
    print("mean forward psas difference:", (pt_outputs - pd_outputs).abs().mean())
    print("max forward pass difference:", (pt_outputs - pd_outputs).abs().max())
    print("mean backward psas difference:", np.abs(torch_grad - paddle_grad).mean())
    print("max backward pass difference:", np.abs(torch_grad - paddle_grad).max())

