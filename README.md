# ASESUM summarisation


## Getting started

### Clone the repository

```bash
git clone https://git.ecdf.ed.ac.uk/s2236454/asesum.git
cd asesum
```

### Prepare environment
The following enviornment requires Linux system 
```bash
conda create -n asenv python=3.10
conda activate asenv
pip install -r requirements.txt
```

### Configuration file
Before start running the code, create a config file under the `script/config` folder for task configuration.
```
model_name: gpt-4o-mini  # the name of the model
model_path: ./qwen-7B    # the path to the model if open-sourced or local
prompt_path: argument_scheme.txt # the name of the prompt file in prompt folder
prompt_type: aspect     # the key in the prompt file if any
extract_column: text    # the column to perfrom argument extraction
```

We choose the [Qwen2.5-7B](https://huggingface.co/Qwen/Qwen2.5-7B) and [GPT-4o-mini](https://platform.openai.com/docs/models/gpt-4o-mini) as our backbone models.

## Summary Generation

### Argument Scheme Extraction

```
python -u  prompting_extract_LLMs_aspect.py --input_path path/to/you/input/data --config config/config file  --output_path path/to/save/the/output
```

### Baseline Extraction

```
python -u  baseline.py --input_path path/to/you/input/data --config config/baseline.yaml --context_length 128000 (the conext length of the model) --output_path path/to/save/the/output --batch
```

The parameter `batch` controls if the summaries for all products will be saved in a single file (specify it) or different files.

## Evaluation script

Run the [Summac](https://github.com/tingofurro/summac) score evaluation.

```
git clone https://github.com/tingofurro/summac.git
python -u evaluation_script.py --gt_dir path/to/dir/with/original/test/file --res_dir path/to/output/file --metric summac
```