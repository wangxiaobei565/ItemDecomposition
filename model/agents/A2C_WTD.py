import time
import copy
import torch
import torch.nn.functional as F
import numpy as np

import utils
from model.agents.BaseRLAgent import BaseRLAgent
    
class A2C_WTD(BaseRLAgent):
    @staticmethod
    def parse_model_args(parser):
        '''
        args:
        - episode_batch_size
        - batch_size
        - actor_lr
        - critic_lr
        - actor_decay
        - critic_decay
        - target_mitigate_coef
        - args from BaseRLAgent:
            - alpha
            - gamma
            - n_iter
            - train_every_n_step
            - initial_greedy_epsilon
            - final_greedy_epsilon
            - elbow_greedy
            - check_episode
            - with_eval
            - save_path
        '''
        parser = BaseRLAgent.parse_model_args(parser)
        parser.add_argument('--episode_batch_size', type=int, default=8, 
                            help='episode sample batch size')
        parser.add_argument('--batch_size', type=int, default=32, 
                            help='training batch size')
        parser.add_argument('--actor_lr', type=float, default=1e-4, 
                            help='learning rate for actor')
        parser.add_argument('--critic_lr', type=float, default=1e-4, 
                            help='decay rate for critic')
        parser.add_argument('--alpha', type=float, default=0.5, 
                            help='weighted reward')
        parser.add_argument('--actor_decay', type=float, default=1e-4, 
                            help='learning rate for actor')
        parser.add_argument('--critic_decay', type=float, default=1e-4, 
                            help='decay rate for critic')
        parser.add_argument('--target_mitigate_coef', type=float, default=0.01, 
                            help='mitigation factor')
        parser.add_argument('--advantage_bias', type=float, default=0, 
                            help='mitigation factor')
        parser.add_argument('--entropy_coef', type=float, default=0.1, 
                            help='mitigation factor')
        return parser
    
    
    def __init__(self, args, facade):
        '''
        self.gamma
        self.n_iter
        self.check_episode
        self.with_eval
        self.save_path
        self.facade
        self.exploration_scheduler
        '''
        super().__init__(args, facade)
        self.alpha = args.alpha
        self.episode_batch_size = args.episode_batch_size
        self.batch_size = args.batch_size
        
        self.actor_lr = args.actor_lr
        self.critic_lr = args.critic_lr
        self.actor_decay = args.actor_decay
        self.critic_decay = args.critic_decay
        
        self.actor = facade.actor
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.actor_lr, 
                                                weight_decay=args.actor_decay)

        self.critic = facade.critic
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=args.critic_lr, 
                                                 weight_decay=args.critic_decay)

        self.tau = args.target_mitigate_coef
        self.advantage_bias = args.advantage_bias
        self.entropy_coef = args.entropy_coef
        if len(self.n_iter) == 1:
            with open(self.save_path + ".report", 'w') as outfile:
                outfile.write(f"{args}\n")
        
    def action_before_train(self):
        '''
        Action before training:
        - facade setup:
            - buffer setup
        - run random episodes to build-up the initial buffer
        '''
        self.facade.initialize_train() # buffer setup
        prepare_step = 0
        # random explore before training
        initial_epsilon = 1.0
        observation = self.facade.reset_env({"batch_size": self.episode_batch_size})
        while not self.facade.is_training_available:
            observation = self.run_episode_step(0, initial_epsilon, observation, True)
            prepare_step += 1
        # training records
        self.training_history = {"critic_loss": [], "actor_loss": [], 'entropy_loss':[], 'advantage':[]}
        
        print(f"Total {prepare_step} prepare steps")
        
        
    def run_episode_step(self, *episode_args):
        '''
        One step of interaction
        '''
        episode_iter, epsilon, observation, do_buffer_update = episode_args
        with torch.no_grad():
            # sample action
            
#             import pdb
#             pdb.set_trace()
            policy_output = self.facade.apply_policy(observation, self.actor, epsilon, do_explore = True)
            # apply action on environment and update replay buffer
            next_observation, reward_, done, info = self.facade.env_step(policy_output)
            reward = info['response']
            # update replay buffer
            if do_buffer_update:
                self.facade.update_buffer(observation, policy_output, reward, done, next_observation, info)
        return next_observation
            
#     def run_an_episode(self, epsilon, initial_observation = None, with_train = False, pick_rows = None):
#         '''
#         Run episode for a batch of user
#         @input:
#         - epsilon: greedy epsilon for random exploration
#         - initial_observation
#         - with_train: apply batch training for each step of the episode
#         - pick_rows: pick certain rows of the data when reseting the environment
#         '''
#         # observation --> state, action
#         if initial_observation:
#             observation = initial_observation
#         elif pick_rows:
#             observation = self.facade.reset_env({"batch_size": len(pick_rows), 'pick_rows': pick_rows})
#         else:
#             observation = self.facade.reset_env({"batch_size": self.episode_batch_size})
#         step = 0
#         done = [False] * self.batch_size
#         train_report = None
#         while sum(done) < len(done):
#             step += 1
#             with torch.no_grad():
#                 # sample action
#                 policy_output = self.facade.apply_policy(observation, self.actor, epsilon, do_explore = True)
#                 # apply action on environment and update replay buffer
#                 next_observation, reward, done, info = self.facade.env_step(policy_output)
#                 # update replay buffer
#                 if not pick_rows:
#                     self.facade.update_buffer(observation, policy_output, reward, done, next_observation, info)
#                 # observate for the next step
#                 observation = next_observation
#             if with_train:
#                 train_report = self.step_train()
#         episode_reward = self.facade.get_total_reward()
#         return {'average_total_reward': np.mean(episode_reward['total_rewards']),
#                 'reward_variance': np.var(episode_reward['total_rewards']),
#                 'max_total_reward': np.max(episode_reward['total_rewards']),
#                 'min_total_reward': np.min(episode_reward['total_rewards']),
#                 'average_n_step': np.mean(episode_reward['n_step']),
#                 'step': step, 
#                 'buffer_size': self.facade.current_buffer_size}, train_report

    def step_train(self):
        observation, policy_output, reward, done_mask, next_observation = self.facade.sample_buffer(self.batch_size)
#         import pdb
#         pdb.set_trace()
#         reward = reward.to(float)
#         done_mask = done_mask.to(float)
        
        critic_loss, actor_loss, entropy_loss, advantage = self.get_a2c_loss(observation, policy_output, reward, done_mask, next_observation)
        self.training_history['actor_loss'].append(actor_loss.item())
        self.training_history['critic_loss'].append(critic_loss.item())
        self.training_history['entropy_loss'].append(entropy_loss.item())
        self.training_history['advantage'].append(advantage.item())

        # Update the frozen target models
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return {"step_loss": (self.training_history['actor_loss'][-1], 
                              self.training_history['critic_loss'][-1], 
                              self.training_history['entropy_loss'][-1], 
                              self.training_history['advantage'][-1])}
    
    def get_a2c_loss(self, observation, policy_output, reward, done_mask, next_observation, 
                      do_actor_update = True, do_critic_update = True):
        
        current_policy_output = self.facade.apply_policy(observation, self.actor)
        V_S = self.critic(current_policy_output)['v'].view(-1,1)
        
        # Compute the target Q value
        next_policy_output = self.facade.apply_policy(next_observation, self.actor_target)
#         next_policy_output = self.facade.apply_p[olicy(next_observation, self.actor)
        
        V_S_prime = self.critic_target(next_policy_output)['v'].detach()
#         V_S_prime = self.critic({'state_emb': S_prime})['v'].detach()
        
#         omiga = (self.alpha * reward + (1-self.alpha)) / (self.alpha * torch.sum(reward,dim=-1).view(-1,1) + (1-self.alpha)* reward.shape[1])
        
        Q_S = reward + self.gamma * (~done_mask * V_S_prime).view(-1,1)/ reward.shape[1]
#         import pdb
#         pdb.set_trace()
        advantage = torch.clamp((Q_S - V_S/ reward.shape[1]).detach(), -1, 1) # (B,)

        # Compute critic loss
        value_loss = F.mse_loss(V_S.squeeze(), Q_S.sum(-1)).mean()
        
        # Regularization loss
#         critic_reg = current_critic_output['reg']

        if do_critic_update and self.critic_lr > 0:
            # Optimize the critic
            self.critic_optimizer.zero_grad()
            value_loss.backward()
            self.critic_optimizer.step()

        # Compute actor loss
        
        current_policy_output = self.facade.apply_policy(observation, self.actor)
        A = policy_output['action']
#         logp = -torch.log(current_policy_output['action_prob'] + 1e-6) # (B,K)
        logp = -torch.log(torch.gather(current_policy_output['candidate_prob'],1,A-1) + 1e-6) # (B,K)
        # use log(1-p), p is close to zero when there are large number of items
#         logp = torch.log(-torch.gather(current_policy_output['candidate_prob'],1,A-1)+1) # (B,K)
#         actor_loss = torch.mean(torch.sum(logp * (advantage.view(-1,1) + self.advantage_bias), dim = 1))
        actor_loss = torch.mean(torch.sum(logp * advantage, dim=1 ))

        entropy_loss = -torch.sum(current_policy_output['candidate_prob'] \
                                  * torch.log(current_policy_output['candidate_prob']), dim = 1).mean()
        
        
        # Regularization loss
#         actor_reg = policy_output['reg']

        if do_actor_update and self.actor_lr > 0:
            # Optimize the actor 
            self.actor_optimizer.zero_grad()
            (actor_loss + self.entropy_coef * entropy_loss).backward()
            self.actor_optimizer.step()
            
        return value_loss, actor_loss, entropy_loss, torch.mean(advantage)


    def save(self):
        torch.save(self.critic.state_dict(), self.save_path + "_critic")
        torch.save(self.critic_optimizer.state_dict(), self.save_path + "_critic_optimizer")

        torch.save(self.actor.state_dict(), self.save_path + "_actor")
        torch.save(self.actor_optimizer.state_dict(), self.save_path + "_actor_optimizer")


    def load(self):
        self.critic.load_state_dict(torch.load(self.save_path + "_critic", map_location=self.device))
        self.critic_optimizer.load_state_dict(torch.load(self.save_path + "_critic_optimizer", map_location=self.device))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(self.save_path + "_actor", map_location=self.device))
        self.actor_optimizer.load_state_dict(torch.load(self.save_path + "_actor_optimizer", map_location=self.device))
        self.actor_target = copy.deepcopy(self.actor)

#     def log_iteration(self, step):
#         episode_report, train_report = self.get_report()
#         log_str = f"step: {step} @ episode report: {episode_report} @ step loss: {train_report['step_loss']}\n"
# #         if train_report:
# #             log_str = f"step: {step} @ episode report: {episode_report} @ step loss (actor,critic): {train_report['step_loss']}\n"
# #         else:
# #             log_str = f"step: {step} @ episode report: {episode_report}\n"
#         with open(self.save_path + ".report", 'a') as outfile:
#             outfile.write(log_str)
#         return log_str

#     def test(self):
#         t = time.time()
#         # Testing
#         self.facade.initialize_train()
#         self.load()
#         print("Testing:")
#         # for i in tqdm(range(self.n_iter)):
#         with torch.no_grad():
#             for i in tqdm(range(len(self.facade.env.reader) // self.batch_size)):
#                 pick_rows = [row for row in range(i * self.batch_size, (i + 1) * self.batch_size)]
#                 episode_report, _ = self.run_an_episode(self.exploration_scheduler.value(i), pick_rows = pick_rows)

#                 if (i+1) % 1 == 0:
#                     t_ = time.time()
#                     print(f"Episode {i+1}, time diff {t_ - t})")
#                     print(self.log_iteration(i, episode_report))
#                     t = t_