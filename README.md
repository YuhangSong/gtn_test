<img src="data/logo.jpg" width=25% align="right" />

# Baselines

OpenAI Baselines is a set of high-quality implementations of reinforcement learning algorithms.

These algorithms will make it easier for the research community to replicate, refine, and identify new ideas, and will create good baselines to build research on top of. Our DQN implementation and its variants are roughly on par with the scores in published papers. We expect they will be used as a base around which new ideas can be added, and as a tool for comparing a new approach against existing ones. 

You can install it by typing:

```
mkdir -p gtn_env/project/ && cd gtn_env/project/ && git clone https://github.com/YuhangSong/gtn.git && cd gtn && source ~/.bashrc && conda create -n gtn_env -y && source activate gtn_env && pip install -e . && pip install visdom matplotlib sklearn dill
```

- [A2C](baselines/a2c)
- [ACKTR](baselines/acktr)
- [DDPG](baselines/ddpg)
- [DQN](baselines/deepq)
- [PPO](baselines/ppo1)
- [TRPO](baselines/trpo_mpi)
