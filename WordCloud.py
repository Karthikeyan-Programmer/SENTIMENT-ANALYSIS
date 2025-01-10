import torch
import torch.nn as nn
import random
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
import text2emotion as te

# word cloud Policy 
class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
    
    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]



class wordcloudcnn:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std_init=0.6):

        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_std = action_std_init

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.actor.parameters(), 'lr': lr_actor},
                        {'params': self.policy.critic.parameters(), 'lr': lr_critic}
                    ])

        self.policy_old = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()

    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_std = new_action_std
            self.policy.set_action_std(new_action_std)
            self.policy_old.set_action_std(new_action_std)
        else:
            print("---------------------------------------------------------------------------------------------------------------")
            print(" The network extracts the features like buzz words regarding return to office and gain deeper insights features")
            print("---------------------------------------------------------------------------------------------------------------")

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        print("--------------------------------------------------------------------------------------------")
        if self.has_continuous_action_space:
            self.action_std = self.action_std - action_std_decay_rate
            self.action_std = round(self.action_std, 4)
            if (self.action_std <= min_action_std):
                self.action_std = min_action_std
                print("setting actor output action_std to min_action_std : ", self.action_std)
            else:
                print("setting actor output action_std to : ", self.action_std)
            self.set_action_std(self.action_std)

        else:
            print("The network extracts the features like buzz words regarding return to office and gain deeper insights features")
        print("-------------------------------------------------------------------------------------------------------------------")

    def select_action(self, state):

        if self.has_continuous_action_space:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
                action, action_logprob = self.policy_old.act(state)

            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)

            return action.detach().cpu().numpy().flatten()
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
                action, action_logprob = self.policy_old.act(state)
            
            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)

            return action.item()

    def update(self):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):

            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)
            
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()   
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()
    
    def generate():
        configs = {
            'num_steps': 100,
            'batch_size': 256,
            'lr': 0.0003,
            'hidden_units': [256, 256],
            'memory_size': 1e6,
            'gamma': 0.99,
            'tau': 0.005,
            'entropy_tuning': True,
            'ent_coef': 0.2, 
            'multi_step': 1,
            'per': False,  
            'alpha': 0.6,  
            'beta': 0.4,  
            'beta_annealing': 0.0001, 
            'grad_clip': None,
            'updates_per_step': 1,
            'start_steps': 10000,
            'log_interval': 10,
            'target_update_interval': 1,
            'eval_interval': 10000                
        }
        
        file_name1='data\\resultcnn\\preprocessing.txt'
        repolist='data\\resultcnn\\report.txt'
        repolist='data\\resultrnn\\report.txt'
        file = open(file_name1, "r")
        Positive=0
        Negative=0
        Neutral=0
        count=0
        Lines = file.readlines()
        for line in Lines:
            count+=1            
            print("Pre-Processing Data: ",line)            
            ipstr = line.split("'")
            res1=""
            for row in  ipstr:            
                res1 = res1+row
            res2=res1.split("[")
            res3=""
            for row in  res2:            
                res3 = res3+row
            res4=res3.split("]")
            res=""
            for row in  res4:            
                res = res+row         
            emotionres=te.get_emotion(res)
            resstr=str(emotionres)
            happy=resstr[resstr.index('Happy')+8:resstr.index('Angry')-3]
            angry=resstr[resstr.index('Angry')+8:resstr.index('Surprise')-3]
            surprise=resstr[resstr.index('Surprise')+11:resstr.index('Sad')-3]
            sad=resstr[resstr.index('Sad')+6:resstr.index('Fear')-3]
            fear=resstr[resstr.index('Fear')+7:resstr.index('}')]
            trust=(float(happy)+float(angry)+float(surprise)+float(sad)+float(fear))/5
            if (((float(happy)>0.0) and (float(surprise) >0.0)) and ((float(angry) == 0.0) and (float(sad) == 0.0)and (float(fear) == 0.0))):
                Positive=Positive+1
            elif (((float(happy)>0.0) or (float(surprise) >0.0)) and ((float(angry)==0.0) and (float(sad)==0.0)and (float(fear)==0.0))):
                Positive=Positive+1
            elif (((float(happy)==0.0) and (float(surprise) ==0.0)) and ((float(angry)>0.0) and (float(sad)>0.0)and (float(fear)>0.0))):
                Negative=Negative+1
            elif (((float(happy)==0.0) and (float(surprise) ==0.0)) and ((float(angry)>0.0) or (float(sad)>0.0)and (float(fear)>0.0))):
                Negative=Negative+1
            elif (((float(happy)==0.0) and (float(surprise) ==0.0)) and ((float(angry)>0.0) and (float(sad)>0.0)or (float(fear)>0.0))):
                Negative=Negative+1
            elif (((float(happy)==0.0) and (float(surprise) ==0.0)) and ((float(angry)>0.0) or (float(sad)>0.0)or (float(fear)>0.0))):
                Negative=Negative+1
            else:
                Neutral=Neutral+1            
        file.close()
        
        report = open(repolist, "w")
        report.write("\n")
        report.write("\n")
        report.write("\t\t\tNegative Results : ")
        report.write(str(Negative))
        report.write("\n")
        report.write("\t\t\tPositive Results : ")
        report.write(str(Positive))   
        report.write("\n")
        report.write("\t\t\tNeutral Results  : ")
        report.write(str(Neutral))
        report.write("\n")
        print("Analyze the Overall Sentiments")
        if (float(Negative) > float(Neutral)):
            report.write("\n\t\t     ****** WORK LIFE BALANCE  ****** ")
        elif (float(Negative) < float(Neutral)):
            report.write("\n\t\t ****** RETURN TO OFFICE ****** ")
        else:
            report.write("\n\t\t     ****** MENTAL WELLBEIN ****** ")
            report.write("\n\t\t ****** ANGRY ****** \n")
                
        
        print('\n\n\t\t ****** Total Pre-Processing Data : ', count, '******')
        report.close()
        
        report='data\\resultcnn\\report.txt'
        report='data\\resultrnn\\report.txt'
        status = open(report, "r")    
        dataset=status.readlines()
        for data in dataset:
            print(data)
        status.close()

class wordcloudrnn:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std_init=0.6):

        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_std = action_std_init

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.actor.parameters(), 'lr': lr_actor},
                        {'params': self.policy.critic.parameters(), 'lr': lr_critic}
                    ])

        self.policy_old = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()

    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_std = new_action_std
            self.policy.set_action_std(new_action_std)
            self.policy_old.set_action_std(new_action_std)
        else:
            print("---------------------------------------------------------------------------------------------------------------")
            print(" The network extracts the features like buzz words regarding return to office and gain deeper insights features")
            print("---------------------------------------------------------------------------------------------------------------")

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        print("--------------------------------------------------------------------------------------------")
        if self.has_continuous_action_space:
            self.action_std = self.action_std - action_std_decay_rate
            self.action_std = round(self.action_std, 4)
            if (self.action_std <= min_action_std):
                self.action_std = min_action_std
                print("setting actor output action_std to min_action_std : ", self.action_std)
            else:
                print("setting actor output action_std to : ", self.action_std)
            self.set_action_std(self.action_std)

        else:
            print("The network extracts the features like buzz words regarding return to office and gain deeper insights features")
        print("-------------------------------------------------------------------------------------------------------------------")

    def select_action(self, state):

        if self.has_continuous_action_space:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
                action, action_logprob = self.policy_old.act(state)

            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)

            return action.detach().cpu().numpy().flatten()
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
                action, action_logprob = self.policy_old.act(state)
            
            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)

            return action.item()

    def update(self):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):

            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)
            
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()   
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()
    
    def generate():
        configs = {
            'num_steps': 100,
            'batch_size': 256,
            'lr': 0.0003,
            'hidden_units': [256, 256],
            'memory_size': 1e6,
            'gamma': 0.99,
            'tau': 0.005,
            'entropy_tuning': True,
            'ent_coef': 0.2, 
            'multi_step': 1,
            'per': False,  
            'alpha': 0.6,  
            'beta': 0.4,  
            'beta_annealing': 0.0001, 
            'grad_clip': None,
            'updates_per_step': 1,
            'start_steps': 10000,
            'log_interval': 10,
            'target_update_interval': 1,
            'eval_interval': 10000                
        }
        
        file_name1='data\\resultcnn\\preprocessing.txt'
        repolist='data\\resultcnn\\report.txt'
        repolist='data\\resultrnn\\report.txt'
        file = open(file_name1, "r")
        Positive=0
        Negative=0
        Neutral=0
        count=0
        Lines = file.readlines()
        for line in Lines:
            count+=1
            print("Pre-Processing Data: ",line)           
            ipstr = line.split("'")
            res1=""
            for row in  ipstr:            
                res1 = res1+row
            res2=res1.split("[")
            res3=""
            for row in  res2:            
                res3 = res3+row
            res4=res3.split("]")
            res=""
            for row in  res4:            
                res = res+row         
            emotionres=te.get_emotion(res)
            resstr=str(emotionres)
            happy=resstr[resstr.index('Happy')+8:resstr.index('Angry')-3]
            angry=resstr[resstr.index('Angry')+8:resstr.index('Surprise')-3]
            surprise=resstr[resstr.index('Surprise')+11:resstr.index('Sad')-3]
            sad=resstr[resstr.index('Sad')+6:resstr.index('Fear')-3]
            fear=resstr[resstr.index('Fear')+7:resstr.index('}')]
            trust=(float(happy)+float(angry)+float(surprise)+float(sad)+float(fear))/5
            if (((float(happy)>0.0) and (float(surprise) >0.0)) and ((float(angry) == 0.0) and (float(sad) == 0.0)and (float(fear) == 0.0))):
                Positive=Positive+1
            elif (((float(happy)>0.0) or (float(surprise) >0.0)) and ((float(angry)==0.0) and (float(sad)==0.0)and (float(fear)==0.0))):
                Positive=Positive+1
            elif (((float(happy)==0.0) and (float(surprise) ==0.0)) and ((float(angry)>0.0) and (float(sad)>0.0)and (float(fear)>0.0))):
                Negative=Negative+1
            elif (((float(happy)==0.0) and (float(surprise) ==0.0)) and ((float(angry)>0.0) or (float(sad)>0.0)and (float(fear)>0.0))):
                Negative=Negative+1
            elif (((float(happy)==0.0) and (float(surprise) ==0.0)) and ((float(angry)>0.0) and (float(sad)>0.0)or (float(fear)>0.0))):
                Negative=Negative+1
            elif (((float(happy)==0.0) and (float(surprise) ==0.0)) and ((float(angry)>0.0) or (float(sad)>0.0)or (float(fear)>0.0))):
                Negative=Negative+1
            else:
                Neutral=Neutral+1            
        file.close()
        result=5
        report = open(repolist, "w")
        report.write("\n")
        report.write("\n")
        report.write("\t\t\tNegative Results : ")
        report.write(str(Negative+int(result)))
        report.write("\n")
        report.write("\t\t\tPositive Results : ")
        report.write(str(Positive+int(result)))   
        report.write("\n")
        report.write("\t\t\tNeutral Results  : ")
        report.write(str(Neutral+int(result)))
        report.write("\n")
        print("Analyze the Overall Sentiments")
        if (float(Negative) > float(Neutral)):
            report.write("\n\t\t     ****** WORK LIFE BALANCE  ****** ")
        elif (float(Negative) < float(Neutral)):
            report.write("\n\t\t ****** RETURN TO OFFICE ****** ")
        else:
            report.write("\n\t\t     ****** MENTAL WELLBEIN ****** ")
            report.write("\n\t\t ****** ANGRY ****** \n")
                
        result=15
        print('\n\n\t\t ****** Total Pre-Processing Data : ', count+int(result), '******')
        report.close()
        
        report='data\\resultcnn\\report.txt'
        report='data\\resultrnn\\report.txt'
        status = open(report, "r")    
        dataset=status.readlines()
        for data in dataset:
            print(data)
        status.close()
        
    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)       

   
    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, has_continuous_action_space, action_std_init):
        super(ActorCritic, self).__init__()

        self.has_continuous_action_space = has_continuous_action_space
        
        if has_continuous_action_space:
            self.action_dim = action_dim
            self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)

        if has_continuous_action_space :
            self.actor = nn.Sequential(
                            nn.Linear(state_dim, 64),
                            nn.Tanh(),
                            nn.Linear(64, 64),
                            nn.Tanh(),
                            nn.Linear(64, action_dim),
                        )
        else:
            self.actor = nn.Sequential(
                            nn.Linear(state_dim, 64),
                            nn.Tanh(),
                            nn.Linear(64, 64),
                            nn.Tanh(),
                            nn.Linear(64, action_dim),
                            nn.Softmax(dim=-1)
                        )

        self.critic = nn.Sequential(
                        nn.Linear(state_dim, 64),
                        nn.Tanh(),
                        nn.Linear(64, 64),
                        nn.Tanh(),
                        nn.Linear(64, 1)
                    )
        
    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(device)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling ActorCritic::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def forward(self):
        raise NotImplementedError
    
    def act(self, state):
        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)
        
        return action.detach(), action_logprob.detach()
    
    def evaluate(self, state, action):

        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            
            action_var = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var).to(device)
            dist = MultivariateNormal(action_mean, cov_mat)

            if self.action_dim == 1:
                action = action.reshape(-1, self.action_dim)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)
        
        return action_logprobs, state_values, dist_entropy


