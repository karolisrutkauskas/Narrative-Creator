# script for finetuning on a clean machine

$currentDir = Get-Location

# activate python env
python -m venv "$currentDir/narr"
& "$currentDir/narr/Scripts/Activate.ps1"

# install requirements
pip install torch===1.7.0 torchvision===0.8.1 torchaudio===0.7.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install transformers
pip install sklearn

# run finetuning
python finetune.py "data/dataset.jsonl"