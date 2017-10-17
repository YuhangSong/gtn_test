# pytorch-a3c

This is a PyTorch implementation of Asynchronous Advantage Actor Critic (A3C) from ["Asynchronous Methods for Deep Reinforcement Learning"](https://arxiv.org/pdf/1602.01783v1.pdf).

This implementation is inspired by [Universe Starter Agent](https://github.com/openai/universe-starter-agent).
In contrast to the starter agent, it uses an optimizer with shared statistics as in the original paper.

## A2C

I **highly recommend** to check a sychronous version and other algorithms: [pytorch-a2c-ppo-acktr](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr).

In my experience, A2C works better than A3C and ACKTR is better than both of them. Moreover, PPO is a great algorithm for continuous control. Thus, I recommend to try A2C/PPO/ACKTR first and use A3C only if you need it specifically for some reasons.

Also read [OpenAI blog](https://blog.openai.com/baselines-acktr-a2c/) for more information.

## Contributions

Contributions are very welcome. If you know how to make this code better, don't hesitate to send a pull request.

## Usage
```bash
sudo apt autoremove && sudo apt-get install -y tmux htop cmake golang libjpeg-dev git

source ~/.bashrc && source deactivate && conda remove --name gtn_env --all
conda create -n gtn_env && source ~/.bashrc && source activate gtn_env

pip install scipy universe six visdom "gym[atari]" matplotlib dill pygame imageio opencv-python 

rm -r gtn_env
mkdir -p gtn_env/project/ && cd gtn_env/project/

wget http://download.pytorch.org/whl/cu80/torch-0.2.0.post3-cp36-cp36m-manylinux1_x86_64.whl  && pip install http://download.pytorch.org/whl/cu80/torch-0.2.0.post3-cp36-cp36m-manylinux1_x86_64.whl && pip3 install torchvision

git clone https://github.com/YuhangSong/gtn.git && cd gtn

# Works only wih Python 3.
OMP_NUM_THREADS=1 python3 main.py --env-name "PongDeterministic-v4" --num-processes 16
```

This code runs evaluation in a separate thread in addition to 16 processes.

## Results

With 16 processes it converges for PongDeterministic-v4 in 15 minutes.
![PongDeterministic-v4](images/PongReward.png)

For BreakoutDeterministic-v4 it takes more than several hours.
