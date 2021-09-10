from hf_paddle import DebertaTokenizer, DebertaModel

tokenizer = DebertaTokenizer.from_pretrained("microsoft/deberta-xlarge")
model = DebertaModel.from_pretrained("microsoft/deberta-xlarge")
inputs = tokenizer("Hello world!", return_tensors="pt")
outputs = model(**inputs)