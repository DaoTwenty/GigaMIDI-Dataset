# Loop Generation
## Generating and evaluating loops using MIDI-GPT

Here we present the scripts for loop generation as well as evaluation scripts used for our paper. 

### MMM

This repository is the library to run and train our MIDI-GPT-Loop model. It is a fork of the `MMM` repository from the `Metacreation Lab` adapted for training and inference with loops. This library also uses a fork of `MidiTok` allowing the integration of loop tokens.

The main entry point is `inference.py`, where `generate` and `generate_batch`, may be used for MIDI generation

### MMM-Loop

This library holds the scripts used to generate the synthetic MIDI data, and evaluate the accuracy of the generation NOMML controls and the generated loops.

# Installation and Setup

> **Note:** The following scripts were created for computing on [Compute Canada](https://docs.alliancecan.ca/). Therefore, slight modifications (file organization, module imports, environment variables) may be needed in order to run these scripts elsewhere.

The dependencies installed are those needed for inference as well as training, therefore some may not be needed if you do not wish to train a model. The following setup is identical for both submodules, MMM and MMM-Loop

## On Compute Canada

On **Canada Canada**, load the correct modules and install dependencies:

```bash
cd MMM-Loop
bash slurm/install.sh
```

## Elsewhere

### 🧰 Dependencies

The code depends on several Python packages, some of which may require special installation steps or system libraries.

#### 📦 Key Python Dependencies

| Package                                     | Version    | Notes                                                     |
| ------------------------------------------- | ---------- | --------------------------------------------------------- |
| `python`                                    | 3.11       | Required for compatibility                                |
| `symusic`                                   | 0.5.0      | For symbolic music representations                        |
| `MidiTok`                                   | expressive | Custom fork installed from GitHub                         |
| `transformers`                              | 4.49.0     | Hugging Face Transformers                                 |
| `accelerate`                                | 1.4.0      | Hugging Face Accelerate                                   |
| `tensorboard`                               | 2.19.0     | TensorBoard for logging                                   |
| `flash-attn`                                | 2.5.7      | May require building from source (see instructions below) |
| `deepspeed`                                 | 0.14.4     | For model parallelism and training acceleration           |
| `datasets`                                  | 3.3.2      | Hugging Face Datasets                                     |
| `triton`                                    | 3.1.0      | Required for some GPU optimizations                       |
| `nvitop`, `pandas`, `matplotlib`, `sklearn` | Latest     | Utility and visualization tools                           |

### 💻 Installation Instructions

1. Create the virtual environment

Use Python 3.11:

```bash
cd MMM
```

or 

```bash
cd MMM-Loop
```

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

If `python3.11` is not available, install it via pyenv or your system's package manager.

2. Install dependencies

```bash
pip install --upgrade pip

# Required packages
pip install symusic==0.5.0
pip install git+https://github.com/DaoTwenty/MidiTok@expressive
pip install transformers==4.49.0 accelerate==1.4.0 tensorboard==2.19.0
pip install deepspeed==0.14.4
pip install datasets==3.3.2
pip install triton==3.1.0
pip install nvitop pandas matplotlib scikit-learn
```

### ⚡ Installing FlashAttention (Optional but Recommended)

FlashAttention often provides significant speedups for training Transformer models, but may require a manual installation from source depending on your system and GPU.

1. Clone FlashAttention

```bash
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention
git checkout v2.5.7
```

2. Install with `pip`

```bash
pip install .
```

> **Note:** You may need to have the following:

- A CUDA-capable GPU (Compute Capability ≥ 7.5)
- CUDA toolkit ≥ 11.8
- Compatible PyTorch version (typically latest stable)
- Refer to FlashAttention's official documentation for details.

### 📊 Verifying Installation

Run the following to verify installed versions:

```bash
pip freeze
```

### 💬 Notes

- On Compute Canada, system modules like `gcc`, `arrow`, and `rust` were required. These are **not needed** if you can build FlashAttention from source locally.
- If `arrow` or `rust` are required by specific packages, ensure they are available on your system (`brew`, `apt`, or via `conda`).

# 🎶 Usage

First, download the model via [https://1sfu-my.sharepoint.com/:u:/g/personal/pta63_sfu_ca/EbpBz06rnaJMtirT0DqvTFoBQSC2OqJ_gex88fenJ60CQQ?e=o24IMz](https://1sfu-my.sharepoint.com/:u:/g/personal/pta63_sfu_ca/EbpBz06rnaJMtirT0DqvTFoBQSC2OqJ_gex88fenJ60CQQ?e=o24IMz) and place it in ``MMM-Loop/models``

## Loop Generation

### 🧠 Environment Variables

The script (`slurm/ge_synthetic_data.sh`) expects these repositories to be present:

```bash
export PYTHONPATH=$PYTHONPATH:/path/to/MMM:/path/to/MMM-Loop
```
> Replace `/path/to/MMM`and `/path/to/MMM-Loop` with the actual path where the repository is cloned.

### 🛠 Running Locally (Without SLURM)

The SLURM batch loop can be mimicked using a local shell script or Python launcher. Here’s a basic loop for **manual local use**:

```bash
#!/bin/bash

source .venv/bin/activate

MODEL=MIDI-GPT-Loop-model
TOKENIZER=MMM_epl_mistral_tokenizer_with_acs.json
CONFIG=slurm/gen_config.json
NOMML=0
NUM_GEN=1000
BATCH=12
OUTPUT="./SYNTHETIC_DATA"
LABEL="V1"

python -m src.generate_loops \
    --config $CONFIG \
    --model models/$MODEL \
    --tokenizer models/$TOKENIZER \
    --num_generations $NUM_GEN \
    --max_attempts $NUM_GEN \
    --batch $BATCH \
    --nomml $EFFECTIVE_NOMML \
    --output_folder $OUTPUT \
    --label $LABEL \
    --rank 0
```

## 📈 Loop Evaluation

### 📝 Script Overview

The SLURM script (`slurm/eval_loops/sh`):
- Aggregates CSV result files (`RESULTS_*.csv`) from a directory.
- Merges them into a single `RESULTS.csv`.
- Runs the evaluation script via Python: `src.evaluate`.

```bash
#!/bin/bash

SOURCE="./SYNTHETIC_DATA"
LABEL="V1"
OUTFILE="$SOURCE/$LABEL/RESULTS.csv"

mkdir -p "$SOURCE/$LABEL"
mkdir -p "$SOURCE/$LABEL/MIDI"

# Clear existing merged file
> "$OUTFILE"

first=1
for file in "$SOURCE/$LABEL"/RESULTS_*.csv; do
  if [ -f "$file" ]; then
    echo "Processing $file"
    if [ $first -eq 1 ]; then
      cat "$file" >> "$OUTFILE"
      first=0
    else
      tail -n +2 "$file" >> "$OUTFILE"
    fi
  fi
done

# Run evaluation script
python -m src.evaluate --source "$SOURCE/$LABEL"
```

## Visualisation

Create Cross-entropy graph of the evaluation (`slurm/plot_results.sh`)

```bash
#!/bin/bash

SOURCE="../SYNTHETIC_DATA"
LABEL="V1"
INPUT_DIR="$SOURCE/$LABEL"

PLOT_ARGS=" \
    --input $INPUT_DIR \
    --output $INPUT_DIR \
    "

source .venv/bin/activate

python -m src.plot_results $PLOT_ARGS
```

### 🧾 Output

Results are merged to: `SYNTHETIC_DATA/V1/RESULTS.csv`

## ✂️ Trimming Loops

It is possible to trim the generated MIDI files to extract the loops only

```bash
#!/bin/bash

SOURCE="../SYNTHETIC_DATA"
LABEL="V1"
INPUT_DIR="$SOURCE/$LABEL"
OUTPUT_DIR="../LOOPS_ONLY"

ARGS=" \
    --input $INPUT_DIR \
    --output $OUTPUT_DIR \
    "

source .venv/bin/activate

python -m src.trim $ARGS
```