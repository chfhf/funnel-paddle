from transformers import FunnelTokenizerFast as HGCTRLTokenizer
from paddlenlp.transformers.funnel.tokenizer import FunnelTokenizerFast as PDCTRLTokenizer
import itertools
hg_tokenizer = HGCTRLTokenizer.from_pretrained("funnel-transformer/xlarge")
pd_tokenizer = PDCTRLTokenizer.from_pretrained("funnel-transformer/xlarge")

a = "They trained and conducted tests of their spacecraft at North American, and in the altitude chamber at the Kennedy Space Center. How much did Capital Cities Communications purchase ABC and its properties for?"

b = "Due to an FCC ban on same-market ownership of television and radio stations by a single company (although the deal would have otherwise complied with new ownership rules implemented by the FCC in January 1985, that allowed broadcasters to own a maximum of 12 television stations), ABC and Capital Cities respectively decided to sell WXYZ-TV and Tampa independent station WFTS-TV to the E. W. Scripps Company (although Capital Cities/ABC originally intended to seek a cross-ownership waiver to retain WXYZ and Capital Cities-owned radio stations WJR and WHYT)."

c = "The BBC News app brings you news from the BBC and our global network of journalists. The app also offers the BBC World Service Radio streamed live, social features and personalisation so you can re-order the news categories to suit your interests."

test_sample = [a,b,c]

for i in itertools.product(test_sample,test_sample):
    print(i[0])
    print(i[1])
    hg_output = hg_tokenizer(i[0], text_pair=i[1], max_length=64)
    pd_output = pd_tokenizer(i[0], return_attention_mask=True, text_pair=i[1], max_seq_len=64)
    # print(hg_output["input_ids"])
    # print(pd_output["input_ids"])
    print(hg_output["input_ids"] == pd_output["input_ids"])
    print(hg_output["token_type_ids"] == pd_output["token_type_ids"])
    print(hg_output["attention_mask"] == pd_output["attention_mask"])
    pd_decode = pd_tokenizer.convert_tokens_to_string(pd_tokenizer.convert_ids_to_tokens(pd_output["input_ids"]))
    hg_decode = hg_tokenizer.convert_tokens_to_string(hg_tokenizer.convert_ids_to_tokens(hg_output["input_ids"]))
    print(pd_decode == hg_decode)
    # print(hg_output1["input_ids"])
    # print(pd_output1["input_ids"])
    print("==================")

