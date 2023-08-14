import time
import copy
import torch
import torch.nn.functional as F
import numpy as np

import utils
from model.agents.BaseRLAgent import BaseRLAgent
    
class SlateQ(BaseRLAgent):
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
        parser.add_argument('--actor_decay', type=float, default=1e-4, 
                            help='learning rate for actor')
        parser.add_argument('--critic_decay', type=float, default=1e-4, 
                            help='decay rate for critic')
        parser.add_argument('--target_mitigate_coef', type=float, default=0.01, 
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
        self.episode_batch_size = args.episode_batch_size
        self.batch_size = args.batch_size
        
        # self.actor_lr = args.actor_lr
        self.critic_lr = args.critic_lr
        # self.actor_decay = args.actor_decay
        self.critic_decay = args.critic_decay
        
        # self.actor = facade.actor
        # self.actor_target = copy.deepcopy(self.actor)
        # self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.actor_lr, 
        #                                         weight_decay=args.actor_decay)

        self.critic = facade.critic
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=args.critic_lr, 
                                                 weight_decay=args.critic_decay)

        self.tau = args.target_mitigate_coef
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
        self.training_history = {"critic_loss": []}
        
        print(f"Total {prepare_step} prepare steps")
        
        
    def run_episode_step(self, *episode_args):
        '''
        One step of interaction
        '''
        episode_iter, epsilon, observation, do_buffer_update = episode_args
        with torch.no_grad():
            # sample action
            policy_output = self.facade.apply_policy(observation, self.critic, epsilon, do_explore = True)
            # apply action on environment and update replay buffer
            next_observation, reward_, done, info = self.facade.env_step(policy_output)
            reward = info['response']
#             reward[reward == 0] -= 0.01 
            # update replay buffer
            if do_buffer_update:
                self.facade.update_buffer(observation, policy_output, reward, done, next_observation, info)
        return next_observation
            

    def step_train(self):
        observation, policy_output, reward, done_mask, next_observation = self.facade.sample_buffer(self.batch_size)
#         reward = torch.FloatTensor(reward)
#         done_mask = torch.FloatTensor(done_mask)
        
        critic_loss= self.get_slateq_loss(observation, policy_output, reward, done_mask, next_observation)
        
        self.training_history['critic_loss'].append(critic_loss.item())

        # Update the frozen target models
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        # for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
        #     target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return {"step_loss": (self.training_history['critic_loss'][-1])}
    
    def get_slateq_loss(self, observation, policy_output, reward, done_mask, next_observation, 
                    #   do_actor_update = True, do_critic_update = True):
                    do_critic_update = True):
        
        # Get current Q estimate 
        # TODO: include policy_output into the critic calculation
        # request-level Q(s_t, a_t): (B,)
        # item-level Q(s_t, i), i \in a_t: (B, K)
        current_critic_output = self.facade.apply_policy(observation, self.critic)
        
        current_Q = torch.gather(current_critic_output['q_all'], 1, policy_output['action']-1)
#         current_Q = torch.mean(current_Q, dim = 1).view(-1,1)
        
        # Compute the target Q value
        target_critic_output = self.facade.apply_policy(next_observation, self.critic_target)
        # (B,K)
        target_Q = target_critic_output['q']
        
        
        target_Q = reward + self.gamma * (~done_mask.view(-1) * torch.mean(target_Q, dim = 1)).view(-1,1)

        # Compute critic loss
        
        critic_loss = F.mse_loss(current_Q, target_Q).mean()
        
        # Regularization loss
#         critic_reg = current_critic_output['reg']

        if do_critic_update and self.critic_lr > 0:
            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

        
            
        return critic_loss


    def save(self):
        torch.save(self.critic.state_dict(), self.save_path + "_critic")
        torch.save(self.critic_optimizer.state_dict(), self.save_path + "_critic_optimizer")

        # torch.save(self.actor.state_dict(), self.save_path + "_actor")
        # torch.save(self.actor_optimizer.state_dict(), self.save_path + "_actor_optimizer")


    def load(self):
        self.critic.load_state_dict(torch.load(self.save_path + "_critic", map_location=self.device))
        self.critic_optimizer.load_state_dict(torch.load(self.save_path + "_critic_optimizer", map_location=self.device))
        self.critic_target = copy.deepcopy(self.critic)

        # self.actor.load_state_dict(torch.load(self.save_path + "_actor", map_location=self.device))
        # self.actor_optimizer.load_state_dict(torch.load(self.save_path + "_actor_optimizer", map_location=self.device))
        # self.actor_target = copy.deepcopy(self.actor)

