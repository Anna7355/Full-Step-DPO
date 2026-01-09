import sys,datetime
from tqdm import tqdm
import torch
from utils import setup_seed

class DefaultRunner:
    def __init__(self, net, loss_fn, accelerator=None, stage = "train", metrics_dict = None, 
                 optimizer = None, lr_scheduler = None, **kwargs,
                 ):
        self.net,self.loss_fn,self.metrics_dict,self.stage = net,loss_fn,metrics_dict,stage
        self.optimizer,self.lr_scheduler = optimizer,lr_scheduler
        self.kwargs = kwargs
        self.accelerator = accelerator
        if self.stage=='train':
            self.net.train() 
        else:
            self.net.eval()
    
    def __call__(self, batch):
        features,labels = batch 
        
        #loss
        with self.accelerator.autocast():
            preds = self.net(features)
            loss = self.loss_fn(preds,labels)

        #backward()
        if self.stage=="train" and self.optimizer is not None:
            self.accelerator.backward(loss)
            if self.accelerator.sync_gradients:
                self.accelerator.clip_grad_norm_(self.net.parameters(), 1.0)
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            self.optimizer.zero_grad()
            
        all_loss = self.accelerator.gather(loss).sum()
        all_preds = self.accelerator.gather(preds)
        all_labels = self.accelerator.gather(labels)
        
        #losses (or plain metrics that can be averaged)
        step_losses = {self.stage+"_loss":all_loss.item()}
        
        #metrics (stateful metrics)
        step_metrics = {self.stage+"_"+name:metric_fn(all_preds, all_labels).item() 
                        for name,metric_fn in self.metrics_dict.items()}
        
        if self.stage=="train":
            if self.optimizer is not None:
                step_metrics['lr'] = self.optimizer.state_dict()['param_groups'][0]['lr']
            else:
                step_metrics['lr'] = 0.0
        return step_losses,step_metrics

class EpochRunner:
    def __init__(self,steprunner,quiet=False):
        self.steprunner = steprunner
        self.stage = steprunner.stage
        self.accelerator = steprunner.accelerator
        self.net = steprunner.net
        self.quiet = quiet
        
    def __call__(self,dataloader):
        n = dataloader.size  if hasattr(dataloader,'size') else len(dataloader)
        loop = tqdm(enumerate(dataloader,start=1), 
                    total=n,
                    file=sys.stdout,
                    disable=not self.accelerator.is_local_main_process or self.quiet,
                    ncols=100
                   )
        epoch_losses = {}
        
        for step, batch in loop: 
            with self.accelerator.accumulate(self.net):
                step_losses,step_metrics = self.steprunner(batch)   
                step_log = dict(step_losses,**step_metrics)
                for k,v in step_losses.items():
                    epoch_losses[k] = epoch_losses.get(k,0.0)+v
                    
                if step<n:
                    loop.set_postfix(**step_log)

                elif step==n:
                    epoch_metrics = step_metrics
                    epoch_losses = {k:v/step for k,v in epoch_losses.items()}
                    epoch_log = dict(epoch_losses,**epoch_metrics)
                    loop.set_postfix(**epoch_log)

                else:
                    break
        return epoch_log

class Trainer:
    
    StepRunner, EpochRunner = DefaultRunner, EpochRunner
    
    def __init__(self,net,accelerator,tokenizer,optimizer,config=None,lr_scheduler = None, init_batch = 0,**kwargs):
        super().__init__()
        self.net,self.accelerator,self.optimizer = net, accelerator, optimizer
        self.lr_scheduler, self.init_batch = lr_scheduler, init_batch
        self.config,self.tokenizer = config,tokenizer
        self.kwargs = kwargs
        
    def save_ckpt(self, ckpt_path=None):
        accelerator = self.accelerator
        net_dict = accelerator.get_state_dict(self.net)
        accelerator.save(net_dict,ckpt_path if ckpt_path is not None else self.ckpt_path)
    
    def fit(self, dataloader, ckpt_path = "checkpoint.pt", epochs=10):

        setup_seed(self.config.seed)
    
        self.net, self.optimizer, self.lr_scheduler, train_dataloader = self.accelerator.prepare(
            self.net, self.optimizer, self.lr_scheduler, dataloader
        )
        train_dataloader = self.accelerator.skip_first_batches(train_dataloader, self.init_batch)

        for key in self.kwargs:
            self.kwargs[key] = self.accelerator.prepare(self.kwargs[key])
        
        self.history = {}
                
        for epoch in range(0,epochs):

            nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            self.accelerator.print("\n"+"=========="*8 + "%s"%nowtime)
            self.accelerator.print("Epoch {0} / {1}".format(epoch, epochs)+"\n")

            # 1，train -------------------------------------------------  
            train_step_runner = self.StepRunner(
                    net = self.net,
                    accelerator = self.accelerator,
                    stage="train",
                    optimizer = self.optimizer,
                    lr_scheduler = self.lr_scheduler,
                    config = self.config,
                    tokenizer = self.tokenizer
            )

            train_epoch_runner = self.EpochRunner(train_step_runner)
            train_metrics = {'epoch':epoch}
            train_metrics.update(train_epoch_runner(train_dataloader))
            self.save_ckpt(ckpt_path,accelerator = self.accelerator)
            
        self.net = self.accelerator.unwrap_model(self.net)
        self.net.cpu()
        return train_metrics