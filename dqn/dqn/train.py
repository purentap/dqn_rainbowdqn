from typing import Generator
import torch
import numpy as np
from copy import deepcopy
import argparse
import gym
import os
from tempfile import TemporaryDirectory
import warnings

from dqn.replaybuffer.uniform import UniformBuffer

from .model import DQN
from dqn.common import linear_annealing, exponential_annealing, PrintWriter, CSVwriter


class Trainer:
    """ Training class that organize evaluation, update, and transition gathering.

    Args:
        args (argparse.Namespace): CL arguments
        agent (DQN): DQN Agent
        opt (torch.optim.Optimizer): Optimizer for agents parameters
        env (gym.Env): Environment
    """

    def __init__(self, args: argparse.Namespace, agent: DQN, opt: torch.optim.Optimizer, env: gym.Env):

        self.env = env
        self.args = args
        self.agent = agent
        self.opt = opt

        self.train_rewards = []
        self.eval_rewards = []
        self.td_loss = []
        self.log_dir = args.log_dir
        self.device = args.device
        if self.log_dir is None:
            self.log_dir = TemporaryDirectory().name
            warnings.warn("Temporary Logging directory: {}".format(self.log_dir))
        self._writers = [PrintWriter(flush=True), CSVwriter(self.log_dir)]

        self.checkpoint_reward = -np.inf
        self.agent.to(args.device)

        if args.epsilon_decay is not None:
            self.epsilon = exponential_annealing(
                args.epsilon_init,
                args.epsilon_min,
                args.epsilon_decay
            )
        else:
            self.epsilon = linear_annealing(
                args.epsilon_init,
                args.epsilon_min,
                args.n_iterations if args.epsilon_range is None else args.epsilon_range
            )

    def __call__(self) -> None:
        """ Start training """
        for iteration, trans in enumerate(self):
            self.evaluation(iteration)
            self.agent.push_transition(trans)
            self.update(iteration)
            self.writer(iteration)

    def evaluation(self, iteration: int) -> None:
        """ Evaluate the agent if the index "iteration" equals to the evaluation period. 
        If "save_model" is given the current best model is saved based on the
        evaluation score. Evaluation score appended into the "eval_rewards"
        list to keep track of evaluation scores.

        Args:
            iteration (int): Training iteration

        Raises:
            FileNotFoundError:  If "save_model" is given in arguments and
                directory given by "model_dir" does not exist
        """
        if iteration % self.args.eval_period == 0:

            self.eval_rewards.append(
                self.agent.evaluate(self.args.eval_episode,
                                    self.env,
                                    self.args.device,
                                    self.args.render))
            if self.eval_rewards[-1] > self.checkpoint_reward and self.args.save_model:
                self.checkpoint_reward = self.eval_rewards[-1]
                model_id = "{}_{:6d}_{:6.3f}.b".format(
                    self.agent.__class__.__name__,
                    iteration,
                    self.eval_rewards[-1]).replace(" ", "0")
                if not os.path.exists(self.args.model_dir):
                    raise FileNotFoundError(
                        "No directory as {}".format(self.args.model_dir))
                torch.save(dict(
                    model=self.agent.state_dict(),
                    optim=self.opt.state_dict(),
                ), os.path.join(self.args.model_dir, model_id)
                )

    def update(self, iteration: int) -> None:
        """ One step updating function. Update the agent in training mode,
        clip gradient if "clip_grad" is given in args, and keep track of td
        loss. Check for the training index "iteration" to start the update.

        Append td loss to "self.td_loss" list

        Args:
            iteration (int): Training iteration
        """
        gamma = self.args.gamma
        batch_size = self.args.batch_size
        target_upt_period = self.args.target_update_period
        training_period = self.args.start_update
        self.agent.train()

        if iteration > training_period:
            if iteration % target_upt_period == 0 : 
                self.agent.update_target() #update the weights of the target network 
            
            batch = self.agent.buffer.sample(batch_size)
            if batch != None:
                loss = self.agent.loss(batch, gamma)
                self.td_loss.append(loss.detach().cpu().numpy())

                self.opt.zero_grad()
                loss.backward() #not sure abt this
                self.opt.step() 
            

    def writer(self, iteration: int) -> None:
        """ Simple writer function that feed PrintWriter with statistics 

        Args:
            iteration (int): Training iteration
        """
        if iteration % self.args.write_period == 0:
            for _writer in self._writers:
                _writer(
                    {
                        "Iteration": iteration,
                        "Train reward": np.mean(self.train_rewards[-20:]),
                        "Eval reward": self.eval_rewards[-1],
                        "TD loss": np.mean(self.td_loss[-100:]),
                        "Episode": len(self.train_rewards)
                    },
                )

    def __iter__(self) -> Generator[UniformBuffer.Transition, None, None]:
        """ Experience collector function that yields a transition at every
        iteration for "args.n_iterations" iterations by collecting experience
        from the environment. If the environment terminates, append the episodic
        reward and reset the environment. 

        Append episodic reward to "self.train_rewards" at every termination

        """
        iters = self.args.n_iterations
        epsilon = next(self.epsilon)
        state = torch.tensor(self.env.reset())
        episodic_reward = 0
        terminated = False

        for i in range(iters):
            
            action = self.agent.e_greedy_policy(state,epsilon)
            observation, reward, terminated,_ = self.env.step(action)
            episodic_reward += reward
            #print(reward)
            transition = self.agent.Transition(state.detach().cpu().numpy(), action, reward, observation, terminated)
            state = torch.tensor(observation, dtype= torch.float32, device= self.device) 

            yield transition

            if terminated:
                terminated = False
                self.train_rewards.append(episodic_reward)

                episodic_reward = 0
                state = torch.tensor(self.env.reset())
                epsilon = next(self.epsilon)

            #self.epsilon.send('restart')
