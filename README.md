# DEFINE

Accent control for zero-shot TTS: a LoRA adapter on a frozen F5-TTS v1 Base, conditioned on an accent label or a few accent exemplar clips.

## Environment

```bash
conda create -n accentbridge python=3.11 -y && conda activate accentbridge
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

## Weights

Download the weights from [Google Drive](https://drive.google.com/file/d/1gZGYN7bUc0JgzCHO_BuKRDvEWMDLDJkp/view?usp=sharing) and extract them into this folder:

```bash
pip install gdown
gdown 1gZGYN7bUc0JgzCHO_BuKRDvEWMDLDJkp -O weights.zip
unzip weights.zip && rm weights.zip
```

All paths are set in `config.yaml` (relative to this folder). Expected weights layout:

```
weights/
  accentbridge.pt                                  # trained AccentBridge checkpoint
  F5TTS_v1_Base/model_1250000.safetensors          # SWivid/F5-TTS
  vocos-mel-24khz/{config.yaml,pytorch_model.bin}  # charactr/vocos-mel-24khz
  wav2vec2-xls-r-300m/                             # facebook/wav2vec2-xls-r-300m
```

Any entry that is missing is downloaded from the Hugging Face Hub into `hf_cache`.

## Inference

Accent by label (`canada, england, hong_kong, india, malaysia, new_zealand, scotland, singapore, southern_africa, us, spanish, turkish, chinese, nigerian, dutch, russian, polish, arabic, french, german, japanese, swedish, italian`):

```bash
CUDA_VISIBLE_DEVICES=0 python infer.py --ckpt weights/accentbridge.pt \
  --ref-wav prompt.wav --ref-text "transcript of prompt.wav" \
  --text "Text to speak in the target accent." --accent india --w 9 --out out/india.wav
```

Accent from exemplar clips (works for accents unseen in training):

```bash
CUDA_VISIBLE_DEVICES=0 python infer.py --ckpt weights/accentbridge.pt \
  --ref-wav prompt.wav --ref-text "transcript of prompt.wav" \
  --text "Text to speak in the target accent." --exemplars ex1.wav ex2.wav ex3.wav --w 9 --out out/exemplar.wav
```

`--w 0` disables the accent; larger `--w` gives a stronger accent.

## Data preparation

1. Common Voice English: download the corpus from commonvoice.mozilla.org, then extract the metadata into `data/commonvoice`:
   ```bash
   mkdir -p data/commonvoice
   tar -xzf cv-corpus-*-en.tar.gz -C data/commonvoice --wildcards '*/en/validated.tsv' '*/en/clip_durations.tsv'
   python prepare_cv.py --archive cv-corpus-*-en.tar.gz
   ```
2. In-house prompt bank (optional): `data/bank/metadata.csv` with columns `id,path,accent,gender[,age]`, one long read-speech clip per voice.
   ```bash
   CUDA_VISIBLE_DEVICES=0 python prepare_bank.py
   ```
3. Voice-converted pairs (optional, needs the bank and a Seed-VC checkout at `seedvc_dir`):
   ```bash
   git clone https://github.com/Plachta/Seed-VC third_party/seed-vc
   CUDA_VISIBLE_DEVICES=0 python prepare_aug.py
   ```
4. Evaluation grid (needs the bank for reference voices):
   ```bash
   python build_eval_grid.py
   ```

Outputs go to `data/manifest/{cv,bank,aug,eval_grid}.parquet`.

## Training

Stage 1 learns the accent table (lookup); stage 2 anchors the exemplar encoder to it (prototype).

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --mode lookup --steps 20000 --out ckpt/lookup
CUDA_VISIBLE_DEVICES=0 python train.py --mode proto --steps 30000 --init-from ckpt/lookup/accent_last.pt --out ckpt/proto
```

Add `--resume` to continue an interrupted run.

## Evaluation

```bash
CUDA_VISIBLE_DEVICES=0 python evaluate.py --ckpt weights/accentbridge.pt --mode exemplar --weights 0,9,15 --real --tag exemplar
CUDA_VISIBLE_DEVICES=0 python evaluate.py --ckpt weights/accentbridge.pt --mode lookup --weights 0,9,15 --tag lookup
CUDA_VISIBLE_DEVICES=0 python evaluate.py --ckpt weights/accentbridge.pt --no-adapter --weights 0 --tag f5
```

Scores are written to `data/eval/<tag>/{scores.parquet,summary.csv}` (accent probe accuracy, speaker similarity, WER, UTMOS per condition and weight).
