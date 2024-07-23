from transformers import AutoConfig





#The code is from https://github.com/mli/transformers-benchmarks
def transformerCalculationCost(configpath,seq_len):
        model_conf = AutoConfig.from_pretrained(configpath)
        get = lambda *keys: max([getattr(model_conf, k) if hasattr(model_conf, k) else 0 for k in keys])
        num_layers = get('num_hidden_layers', 'n_layer')
        hidden_size = get('hidden_size', 'n_embd', 'd_model')
        vocab_size = get('vocab_size')
        num_heads = get('num_attention_heads', 'n_head')
        if seq_len is None:
            seq_len = get('max_position_embeddings', 'n_ctx')
        n, h, s, v = num_layers, hidden_size, seq_len, vocab_size
        att, ffn, embed = 4 * h * s ** 2 + 8 * s * h ** 2, 16 * s * h ** 2, 2 * s * h * v
        forward = n * (att + ffn) + embed
        # TFLOPs to train one example
        flops =  3 * forward
        return flops





# def main():
#         configpath="./Bert_base_uncasted/config.json"
#         seq_len=512
#         tflops=transformerCalculationCost(configpath, seq_len)
#         print("The tflops of bert_base_uncasted is",tflops)
#
#
#
# if __name__=="__main__":
#     main()