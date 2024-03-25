> **Music to Dance as Language Translation using Sequence Models**\
> André Correia, Luís A. Alexandre\
> Paper: 

## Pre-Processing

If you want to extract audio and pose features from a data set, you can find example code for AIST++ and PhantomDance data sets in the processing directory.

Otherwise, you can download the processed AIST++ and PhantomDance data bellow:

[Google Drive link](https://drive.google.com/drive/folders/16Uvp6rbyzLh0PFx_hG8v3sJTZqK5iWPC?usp=drive_link)

# Install Anaconda
```
sudo apt-get install libgl1-mesa-glx libegl1-mesa libxrandr2 libxrandr2 libxss1 libxcursor1 libxcomposite1 libasound2 libxi6 libxtst6

curl -O https://repo.anaconda.com/archive/Anaconda3-2024.02-1-Linux-x86_64.sh

bash Anaconda3-2024.02-1-Linux-x86_64.sh

conda init
```

# Create and activate environment
```
conda create --name MDLT

conda activate MDLT
```

# Install Dependencies
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

pip3 install numpy

pip3 install transformers

pip install causal-conv1d>=1.2.0

pip install mamba-ssm
```

# Train

Train Transformer on AIST++

```
python3 main_transformer_aist.py --genre 'all' --infer_every 1000 --K 20 --n_epochs 50000 --infer_every 5000 --n_layer 6 --d_model 128 --val_index 0
```

Train Transformer on AIST++ "mLH" genre

```
python3 main_transformer_aist.py --genre 'mLH' --infer_every 1000 --K 20 --n_epochs 50000 --infer_every 5000 --n_layer 6 --d_model 128 --val_index 0
```

Train Transformer on PhantomDance

```
python3 main_transformer_phantom.py  --infer_every 1000 --K 20 --n_epochs 50000 --infer_every 5000 --n_layer 6 --d_model 128 --val_type 0
```

## Citation

If you use this codebase, or otherwise found our work valuable, please cite MDLT:
```
TODO
```
