# measure_gpu.py
import torch, time, os, gc
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer, DistilBertConfig

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device:', device)
if device.type!='cuda':
    print('CUDA not available; aborting.')
    raise SystemExit

def count_params(m): return sum(p.numel() for p in m.parameters())

def bench_model_forward(model, inputs, runs=10):
    model.eval()
    with torch.no_grad():
        # warmup
        for _ in range(2):
            _ = model(**inputs) if isinstance(inputs, dict) else model(inputs)
        torch.cuda.reset_peak_memory_stats()
        times = []
        for _ in range(runs):
            t0 = time.time()
            _ = model(**inputs) if isinstance(inputs, dict) else model(inputs)
            torch.cuda.synchronize()
            times.append(time.time()-t0)
        peak = torch.cuda.max_memory_allocated()
    return sum(times)/len(times), times, peak

results = {}

# WAV2VEC2
wav_dir = 'best_wav2vec2_model_7p5h'
if os.path.isdir(wav_dir):
    try:
        m = AutoModel.from_pretrained(wav_dir, local_files_only=True).to(device)
        params = count_params(m)
        # dummy waveform (1s=16000) adjust if your model expects different
        inp = torch.randn(1, 16000).to(device)
        # wrap to call signature used: input_values
        class Wrapper(torch.nn.Module):
            def __init__(self, model): super().__init__(); self.m=model
            def forward(self, input_values): return self.m(input_values=input_values)
        w = Wrapper(m).to(device)
        tmean, times, peak = bench_model_forward(w, inp)
        results['wav2vec'] = {'params':params, 'time_mean_s':tmean, 'times_s':times, 'peak_gpu_bytes':peak}
        del w, m, inp; gc.collect(); torch.cuda.empty_cache()
    except Exception as e:
        results['wav2vec_error'] = str(e)
else:
    results['wav2vec_error'] = f'dir not found: {wav_dir}'

# DistilBERT checkpoint
ckpt = 'distilbert_grid_best.pth'
if os.path.isfile(ckpt):
    try:
        sd = torch.load(ckpt, map_location='cpu')
        # find embedding size if present
        def find_tensor(d, name_sub='embeddings.word_embeddings.weight'):
            if isinstance(d, dict):
                for k,v in d.items():
                    if name_sub in k and torch.is_tensor(v): return v.shape[0]
                    res = find_tensor(v, name_sub)
                    if res: return res
            elif isinstance(d, (list,tuple)):
                for v in d:
                    res = find_tensor(v, name_sub)
                    if res: return res
            return None
        vocab_sz = find_tensor(sd) or None
        if vocab_sz is None:
            # fallback: sum all tensors
            def sum_params(obj):
                total=0
                if isinstance(obj, dict):
                    for v in obj.values(): total+=sum_params(v)
                elif isinstance(obj,(list,tuple)):
                    for v in obj: total+=sum_params(v)
                elif torch.is_tensor(obj): total+=obj.numel()
                return total
            results['distilbert_checkpoint_params'] = sum_params(sd)
            results['distilbert_note'] = 'vocab size not found; reported total params from checkpoint'
        else:
            config = DistilBertConfig(vocab_size=vocab_sz, num_labels=3)
            m2 = AutoModelForSequenceClassification.from_config(config).to(device)
            # pick state-dict
            sd2 = sd.get('model_state_dict') if isinstance(sd, dict) and 'model_state_dict' in sd else (sd.get('state_dict') if isinstance(sd, dict) and 'state_dict' in sd else sd)
            try:
                m2.load_state_dict(sd2, strict=False)
            except Exception as e:
                results['distilbert_load_warning'] = str(e)
            params2 = count_params(m2)
            tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
            inputs = tokenizer('This is a test.', return_tensors='pt')
            inputs = {k:v.to(device) for k,v in inputs.items()}
            tmean2, times2, peak2 = bench_model_forward(m2, inputs)
            results['distilbert'] = {'params':params2, 'time_mean_s':tmean2, 'times_s':times2, 'peak_gpu_bytes':peak2, 'vocab_size':vocab_sz}
            del m2, inputs; gc.collect(); torch.cuda.empty_cache()
    except Exception as e:
        results['distilbert_error'] = str(e)
else:
    results['distilbert_error'] = f'file not found: {ckpt}'

print('\\n--- RESULTS ---')
for k,v in results.items():
    print(k, ':', v)