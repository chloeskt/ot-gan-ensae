# Project for the Optimal Transport course at ENSAE Paris

This repository contains the code done for the final project og the Optimal Transport course at ENSAE Paris. It has been
done by Mathilde Kaploun (ENSAE Paris), Thomas Doucet (ENSAE Paris) and Chloé Sekkat (ENSAE Paris - ENS Paris Saclay)

# Improving GANs using Optimal Transport
## OT-GAN from scratch

The goal is to implement a GAN variant that builds on Optimal Transport following the approach taken by [this paper](https://arxiv.org/abs/1803.05573),
which develops a OT GAN. 

# Installation 

After creating your virtual environment, please run 

```
pip install -r requirements.txt
```

This project uses Pytorch, please note that you will need to update manually the versions of ``torch``, ``torchaudio`` 
and ``torchvision`` depending on your hardware.

# About the package

This Python package is self-contained. A Colab link is given so that you can see its applications.

- [ ] Add final colab link. 

# Replicate experiments

To train the OT-GAN we implemented, you can use the `main` script. For instance, one can run:

````bash
python main.py \
        --seed 0 \
        --batch_size 24 \
        --data_path /mnt/hdd/ot-gan-ensae \
        --epochs 30 \
        --eval_steps 10000  \
        --patience 5 \
        --learning_rate 1e-4 \
        --weight_decay 0. \
        --eps_regularization 0.1 \
        --nb_sinkhorn_iterations 10 \
        --output_dir /mnt/hdd/ot-gan-ensae/models \
        --save True \
        --device cuda
````

To see all possible arguments, run:

```bash
python main.py --help
```

# Acknowledgments

The ``early_stopping_pytorch`` module has been taken from: https://github.com/Bjarten/early-stopping-pytorch 

# GANs and Optimal Transport

- [ ] TODO: add description, explanations etc
