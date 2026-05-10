# ECE595_Verus

Our explainability project relies upon [`safe-autonomy-sims`](https://github.com/act3-ace/safe-autonomy-sims) package which requires Python 3.10

We use [`ray`](https://github.com/ray-project/ray) for training our RL expert policy which requires a unix interpreter. If you are on Windows, we reccomend you use [Windows Subsystem for Linux](https://learn.microsoft.com/en-us/windows/wsl/install) and install the Ubuntu-24.04 distribution.
## Installation

The easiest way to install our requirements is to use `requirements.txt` via `pip`:

```shell
pip install -r requirements.txt
```

## Usage



### Training

```shell
# from root of ECE595_Verus
python3 single_inspector1_dqn.py
```

## Team

Alex Atcitty,
Hector Valenzuela,
Trazon Jimerson

UNM Spring 2026 Semester