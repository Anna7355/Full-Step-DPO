import wandb
import torch
import torch.nn.functional as F


class AlignRunner:
    
    def __init__(self, net, accelerator, batch_count = 0, stage="train",
                 optimizer=None,tokenizer = None, lr_scheduler=None, config = None):
        
        self.net, self.stage = net, stage
        self.optimizer, self.lr_scheduler = optimizer, lr_scheduler
        self.accelerator = accelerator 
        self.batch_count = batch_count
        self.tokenizer = tokenizer
        self.config = config
        self.generation_kwargs = { 
                                  "max_new_tokens": self.config.max_response_len, 
                                  "top_k": 30, 
                                  "top_p": 0.85, 
                                  "do_sample": True, 
                                  "num_return_sequences": 4,
                                  "pad_token_id": self.net.policy_tokenizer.pad_token_id,
                                  "eos_token_id": self.net.policy_tokenizer.eos_token_id,
                                  }
        if self.stage == 'train':
            self.net.train()
        else:
            self.net.eval()
        
    def __call__(self, batch):
        
        with self.accelerator.accumulate(self.net):

            inputs, rewards, prompt_len, weights = batch
            self.accelerator.wait_for_everyone()

            policy_logp = self.net.get_logits(inputs, prompt_len, self.net.policy) # [batch_size, seq_len]
            ref_logp = self.net.get_logits(inputs, prompt_len, self.net.ref)
            rewards =  torch.tensor(rewards).to("cuda").reshape(-1, 1).float() 

            token_gap = (policy_logp-ref_logp).detach()  # [batch_size, seq_len]
            
            logratios = (policy_logp.sum(-1,keepdims=True) - ref_logp.sum(-1,keepdims=True)) * self.config.beta # (batch_size, 1)
            logratios_gap = F.sigmoid(- logratios + logratios.transpose(1,0)).detach()

            pos_logp_sum = policy_logp.sum(-1,keepdims=True) #  # [batch_size, 1]
            
            if self.config.loss_type == "dpo":
                neg_logp_sum = (policy_logp).sum(-1,keepdims=True)

            elif self.config.loss_type == "fullstepdpo":
                pos_token_reward = torch.sigmoid(-token_gap * self.config.token_rate) # 
                neg_token_reward = torch.sigmoid(token_gap * self.config.token_rate)
                pos_step_reward = (weights*self.config.reward_temp - 999*(weights==0)).softmax(-1)*(weights!=0).sum(-1,keepdims=True)
                neg_step_reward = (-weights*self.config.reward_temp - 999*(weights==0)).softmax(-1)*(weights!=0).sum(-1,keepdims=True)
                
                masks = (weights != 0).float()
                neg_logp_sum = (policy_logp * neg_token_reward * neg_step_reward).sum(-1,keepdims=True) / masks.sum(-1, keepdims=True)
                pos_logp_sum = (policy_logp * pos_token_reward * pos_step_reward).sum(-1,keepdims=True) / masks.sum(-1, keepdims=True)

                loss = - logratios_gap * (pos_logp_sum - neg_logp_sum.transpose(1,0))

                reward_mask = (rewards - rewards.transpose(1,0)) > 0
                loss = (loss * reward_mask.float()).sum()/reward_mask.sum()

            if self.optimizer is not None and self.stage == "train":
                self.accelerator.backward(loss)
                if self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(self.net.parameters(), 1.0)
                self.optimizer.step()
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()
                self.optimizer.zero_grad()


        _loss = self.accelerator.gather(loss).mean() 
        _acc = self.accelerator.gather((logratios[0,0] > logratios[1,0]).float()).mean()
        _win = self.accelerator.gather(logratios[0]).mean() 
        _lose = self.accelerator.gather(logratios[1]).mean()
        _gap = _win-_lose

        step_metrics = {
                    "loss": _loss.item(),
                    "_win": _win.item(),
                    "_lose": _lose.item(),
                    "_gap": _gap.item(),
                    "_acc": _acc.item()
                }
        
        if self.accelerator.is_local_main_process:
            wandb.log(step_metrics)

        step_losses = {self.stage + "_loss": _loss.item()}

        if (self.batch_count+1) % self.config.save_per_step == 0:
            self.save_ckpt(self.config.ckpt_path+"_%s"%(self.batch_count+1),accelerator = self.accelerator)

        self.batch_count += 1
        return step_losses, step_metrics
        
    def save_ckpt(self, path, accelerator = None):

        if accelerator is None:
            accelerator = self.accelerator

        accelerator.wait_for_everyone()
        state_dict = accelerator.get_state_dict(self.net)
        
        if accelerator.is_main_process:
            policy_dict = {key[len("policy."):]: value for key, value in state_dict.items() if "policy" in key}
            self.net.policy.save_pretrained(path,state_dict=policy_dict,safe_serialization=False)
