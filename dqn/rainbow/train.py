from typing import Generator
from collections import namedtuple, deque
from functools import reduce
import argparse
import gym
import torch

from .model import RainBow
from dqn.common import linear_annealing, exponential_annealing, PrintWriter
from dqn.dqn.train import Trainer as BaseTrainer


class Trainer(BaseTrainer):
    """ Training class that organize evaluation, update, and transition
    gathering.
        Arguments:
            - args: Parser arguments
            - agent: RL agent object
            - opt: Optimizer that optimizes agent's parameters
            - env: Gym environment
    """

    def __init__(self, args: argparse.Namespace, agent: RainBow, opt: torch.optim.Optimizer, env: gym.Env):
        """ Training class that organize evaluation, update, and transition gathering.

        Args:
            args (argparse.Namespace): CL arguments
            agent (RainBow): RainBow Agent
            opt (torch.optim.Optimizer): Optimizer for agents parameters
            env (gym.Env): Environment
        """
        super().__init__(args, agent, opt, env)
        # beta = 1 - prioritized_beta
        self.prioritized_beta = linear_annealing(
            init_value=1 - args.beta_init,
            min_value=0,
            decay_range=args.n_iterations
        )



    def update(self, iteration: int) -> None:
        """ One step updating function. Update the agent in training mode.
        - clip gradient if "clip_grad" is given in args.
        - keep track of td loss. Append td loss to "self.td_loss" list
        - Update target network.
        If the prioritized buffer is active:
            - Use the weighted average of the loss where the weights are 
            returned by the prioritized buffer
            - Update priorities of the sampled transitions
        If noisy-net is active:
            - reset noise for valuenet and targetnet
        Check for the training index "iteration" to start the update.

        Args:
            iteration (int): Training iteration
        """
        use_priority_buffer = not self.args.no_prioritized
        is_distributional = not self.args.no_dist
        is_noisy = not self.args.no_noisy
        gamma = self.args.gamma
        beta = self.args.beta_init
        batch_size = self.args.batch_size
        target_upt_period = self.args.target_update_period
        start_update = self.args.start_update

        self.agent.train()
        if iteration > start_update:
            if is_noisy:
                #reset noise for target and value networks (according to the hw notebook)
                self.agent.targetnet.layer5.reset_noise()
                self.agent.valuenet.layer5.reset_noise()
            
            if iteration % target_upt_period == 0:
                self.agent.update_target()

            if use_priority_buffer:    
                batch, indices, is_weights = self.agent.buffer.sample(batch_size, beta)
            else: 
                batch = self.agent.buffer.sample(batch_size)
            
            if is_distributional:
                td_errors = self.agent.distributional_loss(batch,gamma)
                loss = td_errors
            else:
                td_errors = self.agent.loss(batch, gamma)
            if use_priority_buffer:
                self.agent.buffer.update_priority(indices, td_errors.detach().cpu().numpy() )
                loss = torch.mean(td_errors * torch.tensor(is_weights))

            else: #uniform buffer
                if is_distributional == False:
                    loss = torch.mean(td_errors) 
   

            self.td_loss.append(loss.detach().cpu().numpy())

            self.opt.zero_grad()
            loss.backward() 
            if self.args.clip_grad:
                torch.nn.utils.clip_grad_norm_(self.agent.parameters(), 1.0)
            self.opt.step() 
            

        else:
            return super().update(iteration)


    def __iter__(self) -> Generator[RainBow.Transition, None, None]:
        """ n-step transition generator. Yield a transition with
         n-step look ahead. Use the greedy policy if noisy network 
         extension is activate.

        Yields:
            Generator[RainBow.Transition, None, None]: Transition of
            (s_t, a_t, \sum_{j=t}^{t+n}(\gamma^{j-t} r_j), done, s_{t+n})
        """
        n = self.args.n_steps
        is_noisy = not self.args.no_noisy

        if n== 1 and is_noisy:
            iters = self.args.n_iterations
            state = torch.tensor(self.env.reset())
            episodic_reward = 0
            terminated = False

            for i in range(iters):
                
                action = self.agent.greedy_policy(state)
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

                #self.epsilon.send('restart')
        elif  n == 1:
            yield from super().__iter__()

        else: #n-step learning
            iters = self.args.n_iterations
            epsilon = next(self.epsilon)
            terminated = False
            state = torch.tensor(self.env.reset())
            initial_state = state
            episodic_reward = 0
            n_buffer = deque(maxlen=n)
            for i in range(iters):
                n_step_reward = 0 
                n_buffer.clear()
                for step in range(n):
                    if is_noisy:
                       action = self.agent.greedy_policy(state)
                    else: 
                        action = self.agent.e_greedy_policy(state, epsilon)
            
                    observation, reward, terminated,_ = self.env.step(action)
                    n_step_reward += reward
                    state = torch.tensor(observation, dtype= torch.float32, device= self.device) 
                    n_buffer.append(state)

                    if len(n_buffer) == n or terminated:
                        transition = self.agent.Transition(initial_state.detach().cpu().numpy(), action, n_step_reward, observation, terminated)
                        yield transition
                    if terminated:
                        terminated = False
                        episodic_reward += n_step_reward
                        self.train_rewards.append(episodic_reward)
                        state = torch.tensor(self.env.reset())
                        initial_state = state
                        epsilon = next(self.epsilon)
                        episodic_reward = 0
                        n_buffer.clear()

                        break



                        

