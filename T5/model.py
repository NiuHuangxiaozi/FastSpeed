
import torch
import torch.nn as nn
from transformers import T5Tokenizer, T5ForConditionalGeneration, AutoConfig







class T5base(nn.Module):
    def __init__(self, config_path):
        super(T5base, self).__init__()
        self.t5= T5ForConditionalGeneration.from_pretrained(config_path)
        self.tokenizer=T5Tokenizer.from_pretrained(config_path, legacy=False)

    def generate(self,input_ids,attention_mask,max_length,num_beams,repetition_penalty,length_penalty,early_stopping):
        generate_code=self.t5.generate(
                input_ids = input_ids,
                attention_mask = attention_mask,
                max_length=max_length,
                num_beams=num_beams,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                early_stopping=early_stopping
                )
        return generate_code
    def forward(self,input):
        y = input['target_ids'].to(dtype=torch.long)
        # print(f"y is {y}")

        y_ids = y[:, :-1].contiguous()
        # print(f"y_ids is {y_ids}")
        lm_labels = y[:, 1:].clone().detach()
        # print(f"lm_labels is {lm_labels}")

        lm_labels[y[:, 1:] == self.tokenizer.pad_token_id] = -100
        # print(f"lm_labels is {lm_labels}")

        ids = input['source_ids'].to(dtype=torch.long)
        # print(f"ids is {ids}")

        mask = input['source_mask'].to(dtype=torch.long)
        # print(f"mask is {mask}")

        outputs = self.t5(input_ids=ids, attention_mask=mask, decoder_input_ids=y_ids, labels=lm_labels)

        return outputs.loss




#下面的代码参考并学习来自李沐的github，网址为：https://github.com/mli/transformers-benchmarks/blob/main/micro_bench.ipynb

#计算一个样例在一个encoder或者decoder上的浮点数计算量
def getT5BlockcalculationCost(hidden_size,seq_lens,batch_sizes,cross_attention):
    h=hidden_size
    s=seq_lens
    b=batch_sizes

    ffn=16 * b * s * h * h

    atten=4 * b * h * s * s  +  8 * b * s * h * h

    oneBlockForward=ffn+(2 if cross_attention else 1)*atten

    return oneBlockForward


#计算一个样例在一个encoder-decoder模型上（例如T5）的浮点数计算量
def getT5calculationCost(configPath,seq_len):
    model_conf = AutoConfig.from_pretrained(configPath)
    get = lambda *keys: max([getattr(model_conf, k) if hasattr(model_conf, k) else 0 for k in keys])
    num_layers = get('num_layers')
    hidden_size = get('hidden_size', 'n_embd', 'd_model')
    vocab_size = get('vocab_size')

    assert(seq_len!=None)

    encoderCost=getT5BlockcalculationCost(hidden_size,seq_len, 1, False)
    decoderCost=getT5BlockcalculationCost(hidden_size,seq_len, 1, True)

    embed=2 * seq_len * hidden_size * vocab_size

    forward = num_layers * (encoderCost + decoderCost) + embed

    # TFLOPs to train one example
    # backward cost is about double forward cost

    flops = 1 * forward+ 2 * forward

    return flops






# def main():
#     # 下载T5模型，存放于某目录下
#     t5_path ="./T5config"
#     tokenizer = T5Tokenizer.from_pretrained(t5_path,legacy=False)
#     model = T5ForConditionalGeneration.from_pretrained(t5_path)
#
#     input_ids = tokenizer("The <extra_id_0> walks in <extra_id_1> park", return_tensors="pt").input_ids
#     labels = tokenizer("<extra_id_0> cute dog <extra_id_1> the <extra_id_2>", return_tensors="pt").input_ids
#     outputs = model(input_ids=input_ids, labels=labels)
#     loss = outputs.loss
#     logits = outputs.logits
#     print("The loss is ",loss)
#     print("The logits is ",logits)
#
#     # 准备输入数据
#     article_text = """
#         Global warming is one of the most serious environmental issues today. As the global average temperature continues to rise, extreme weather events have become more frequent, having a significant impact on human life. Scientists agree that reducing greenhouse gas emissions, especially carbon dioxide emissions, is key to combating global warming. Additionally, protecting forests, conserving water, and reducing plastic use are critical actions that everyone needs to take.
#     """
#     input_ids = tokenizer("summarize: " + article_text, return_tensors="pt").input_ids
#
#     # 生成摘要
#     outputs = model.generate(input_ids, max_length=20, min_length=0, length_penalty=2.0, num_beams=4,
#                              early_stopping=True)
#     summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     print(summary)
#
# if __name__=="__main__":
#     main()
