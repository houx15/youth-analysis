# Agent Guidelines for Youth Analysis Codebase

## Project Overview

Python data analysis project for Chinese youth Weibo data: text analysis, sentiment analysis, user profiling, gender norms research, and NLP tasks.

## ⚠️ CRITICAL OPERATIONAL RULES

### 1. SLURM Usage (MANDATORY)
**This is a public server.** Long-running or resource-intensive tasks MUST be submitted via SLURM, never run directly:

```bash
# Submit job to SLURM
sbatch script.sh

# Example SLURM script
#!/bin/bash
#SBATCH --job-name=analysis
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
python analysis_script.py --year 2020
```

**Never run directly:**
- Training Word2Vec models
- Processing large datasets
- Long-running analyses
- Memory-intensive operations

### 2. Commit Changes
Always commit changes after modifying code:
```bash
git add <files>
git commit -m "descriptive message"
```

### 3. Large Data Handling
**NEVER attempt to read entire datasets into memory.** Always:
- Use `usecols` to load only needed columns
- Process data in chunks
- Filter early in the pipeline
- Use generators/iterators where possible

```python
# GOOD: Load only needed columns
data = pd.read_parquet(file_path, columns=["user_id", "gender"])

# BAD: Loading entire large dataset
data = pd.read_parquet("huge_file.parquet")  # DON'T DO THIS
```

## Build/Lint/Test Commands

### Running Scripts (Fire CLI)
```bash
# Short tests only - for SLURM jobs, create submission script
python <script_name>.py <command> --arg value
python ai_sentiment_analyzer.py sample --sample_size 1000
```

### Cython Compilation
```bash
python setup.py build_ext --inplace
rm -rf build/ *.so *.c && python setup.py build_ext --inplace
```

### Testing
```bash
python test.py
python -c "from utils.utils import sentence_cleaner; print(sentence_cleaner('text'))"
```

## Code Style Guidelines

### Imports (3 groups, alphabetically ordered)
```python
# Standard library
import json
import os
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

# Third-party
import fire
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from tqdm import tqdm

# Local
from configs.configs import OPENAI_API_KEY
from utils.utils import sentence_cleaner
```

### Module Documentation (Chinese)
```python
"""
微博数据分析脚本

功能：
1. 用户设备更换频率分析
2. 转发官方媒体情况分析

使用方法:
    python script_name.py command --arg value
"""
```

### Naming Conventions
```python
DATA_DIR = "cleaned_weibo_cov"                    # Constants: UPPER_SNAKE_CASE

def load_official_media_ids():                    # Functions: snake_case
    pass

class AISentimentAnalyzer:                        # Classes: PascalCase
    def _build_prompt(self, row):                 # Private: underscore prefix
        pass
```

### String Formatting & Error Handling
```python
# Prefer f-strings and double quotes (especially for Chinese)
print(f"已加载 {len(ids)} 个新闻账号ID")

# Proper error handling
try:
    data = pd.read_parquet(file_path, columns=needed_cols)
except FileNotFoundError:
    print(f"未找到数据文件: {file_path}")
    return None
```

### Data Handling (CRITICAL)
```python
# Always specify columns to reduce memory
data = pd.read_csv("user.csv", usecols=["user_id", "gender", "birthday"])

# Process large files in chunks
for chunk in pd.read_parquet("large.parquet", chunksize=10000):
    process(chunk)

# Filter early
data = data[data['year'] == 2020]  # Filter before further processing
```

### CLI Pattern with Fire
```python
import fire

class DataAnalyzer:
    def analyze(self, year: int = 2020):
        """分析数据"""
        pass

if __name__ == "__main__":
    fire.Fire(DataAnalyzer)
```

### Type Hints (Recommended)
```python
from typing import List, Dict, Optional

def load_source_ids(source_type: str = "news") -> set:
    pass

def process_data(data: pd.DataFrame, columns: List[str]) -> Dict[str, Any]:
    pass
```

## Project Structure
```
youth-analysis/
├── configs/              # Configuration (configs.example.py → configs.py)
├── utils/               # Utility functions
├── gender_norms/        # Gender norms submodule (see below)
│   ├── *.py             # All Python scripts for gender norm analysis
│   ├── wordlists/       # JSON wordlists
│   ├── results/         # Analysis results
│   └── newspaper_data/  # Newspaper analysis data (git ignored)
│       ├── *.json       # Mapping files
│       ├── *.csv        # Statistics
│       ├── logs/        # Log files
│       └── newspaper_corpus/  # Segmented corpus
├── *.sh                 # SLURM submission scripts (in root directory)
├── *.py                 # Analysis scripts
├── *.pyx                # Cython modules
└── test.py              # Ad-hoc tests
```

## 📁 File Organization Rules

### Python Scripts
- **Gender norms scripts**: Place in `gender_norms/` directory
  - Example: `newspaper_extractor.py`, `newspaper_corpus_builder.py`, etc.
  - These are part of the gender norms analysis pipeline

### Shell Scripts
- **SLURM scripts**: Place in project root directory (`*.sh`)
  - Example: `extract_newspapers.sh`, `build_corpus.sh`, etc.
  - Use relative paths: `python gender_norms/script_name.py`

### Data Directories
- **All data outputs**: Store in `gender_norms/newspaper_data/`
  - This directory is in `.gitignore` (data files not tracked)
  - Includes: mappings, statistics, logs, corpus files

### Documentation
- **One consolidated document**: `gender_norms/newspaper_data/README.md`
  - Consolidate all progress, changelogs, and status into single file
  - Update incrementally rather than creating multiple docs

### Example Paths
```bash
# In SLURM script
python gender_norms/newspaper_corpus_builder.py build

# Python script reading/writing data
DATA_DIR = "/lustre/home/2401111059/newspaper_data/pdf_txt"
OUTPUT_DIR = "gender_norms/newspaper_data/newspaper_corpus"
```

## Gender Norms Submodule (`gender_norms/`)

**Relatively independent submodule** for analyzing gender bias using Word2Vec embeddings.

### Workflow
1. **Train Word2Vec models** on large text corpus (requires SLURM)
2. **Load trained models** and wordlists from `gender_norms/wordlists/`
3. **Analyze gender bias** by calculating word associations
4. **Visualize results** in `gender_norms/results/`

### Key Components
- `gender_embedding_trainer.py` - Train Word2Vec models (resource-intensive, use SLURM)
- `gender_embedding_analyzer.py` - Analyze embeddings for gender bias
- `gender_norm_index_builder.py` - Build gender norm indices
- `wordlists/` - JSON files with gender/occupation/domestic work words

### Usage
```python
# Load pre-trained model
from gensim.models import KeyedVectors
model = KeyedVectors.load("gender_norms/gender_embedding/embedding_models/2020/model.kv")

# Load wordlists
from gender_norms.gender_embedding_analyzer import load_gender_words, load_occupation_words
gender_words = load_gender_words()
occupations = load_occupation_words()
```

## Key Dependencies
- **Data**: pandas, numpy
- **NLP**: gensim (Word2Vec), jieba (Chinese segmentation)
- **Stats**: scipy, sklearn
- **CLI**: fire
- **Viz**: matplotlib, seaborn
- **AI**: openai
- **I/O**: py7zr, orjson
- **Perf**: Cython

## Configuration
1. Copy `configs/configs.example.py` → `configs/configs.py`
2. Add API keys:
```python
OPENAI_API_KEY = "your-key-here"
OPENAI_BASE_URL = "https://api.example.com/v1"
```

## Data Files
- **Format**: Parquet (large), CSV (small)
- **Archives**: .7z files
- **Never read entire large files into memory**
- Data directories excluded from git

## Chinese Language
Primary language for documentation, comments, user messages, error messages, and log output.
