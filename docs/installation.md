# Installation

## Prerequisites

Install [uv](https://docs.astral.sh/uv/getting-started/installation/):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Main Environment

We use Isaac Gym for policy learning in simulation, which requires Python 3.8.
This is the existing Isaac Gym environment and should remain separate from the
Isaac Sim conversion environment.

```bash
# Create a virtual environment with Python 3.8
uv venv --python 3.8

# Auto-set LD_LIBRARY_PATH on activate (needed for Isaac Gym to find libpython3.8)
echo 'export LD_LIBRARY_PATH=$(python -c "import sysconfig; print(sysconfig.get_config_var(\"LIBDIR\"))"):$LD_LIBRARY_PATH' >> .venv/bin/activate

source .venv/bin/activate

# Install the project and all dependencies
uv pip install -e .

# Install Isaac Gym (included in repo, gitignored)
cd isaacgym/python
uv pip install -e .
cd -

# Install this repo's rl_games
cd rl_games
uv pip install -e .
cd -
```

## Isaac Sim Conversion Environment

Use a separate Python 3.11 environment for `isaacsim_conversion`. Do not reuse
the Isaac Gym `.venv` because the Isaac Sim stack conflicts with root package
pins used by the Python 3.8 Isaac Gym workflow.

See [isaacsim_conversion/README.md](../isaacsim_conversion/README.md) for the
full setup and usage instructions.

## Sim2Real Env

For real-world deployment, we need ROS1 Noetic. This can either be installed globally or using RoboStack.

### Global ROS

Follow the instructions at https://wiki.ros.org/noetic/Installation/Ubuntu to install ROS1 Noetic globally. Then source the ROS environment variables in your shell:

```bash
source /opt/ros/noetic/setup.bash
```

Now, you can use this global ROS setup with the virtual environment above.

You may need to run:

```bash
uv pip install rospkg
```

### RoboStack

Note that the RoboStack option requires a separate environment because it requires Python 3.11+. Follow the instructions at https://robostack.github.io/noetic.html to create an environment with ROS1 Noetic. Then follow the installation instructions above for the main environment (skip the venv creation and Isaac Gym installation).
