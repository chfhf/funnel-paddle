from transformers import FunnelTokenizerFast as HGCTRLTokenizer
from paddlenlp.transformers.funnel.tokenizer import FunnelTokenizerFast as PDCTRLTokenizer

hg_tokenizer = HGCTRLTokenizer.from_pretrained("funnel-transformer/xlarge")
pd_tokenizer = PDCTRLTokenizer.from_pretrained("funnel-transformer/xlarge")


text = "The BBC News app brings you news from the BBC and our global network of journalists. The app also offers the BBC World Service Radio streamed live, social features and personalisation so you can re-order the news categories to suit your interests."


hg_output = hg_tokenizer(text)

pd_output = pd_tokenizer(text,return_attention_mask=True)

print(hg_output["input_ids"] == pd_output["input_ids"])
print(hg_output["token_type_ids"] == pd_output["token_type_ids"])
print(hg_output["attention_mask"] == pd_output["attention_mask"])

pd_decode = pd_tokenizer.convert_tokens_to_string(pd_tokenizer.convert_ids_to_tokens(pd_output["input_ids"]))
hg_decode = hg_tokenizer.convert_tokens_to_string(hg_tokenizer.convert_ids_to_tokens(hg_output["input_ids"]))

print(pd_decode==hg_decode)