import torch
import math
import os
import time
import copy
import numpy as np
import wandb
from lib.logger import get_logger
from lib.metrics import All_Metrics

class Trainer(object):
    def __init__(self, model, loss, optimizer, hyper_source, train_loader, val_loader, test_loader,
                 scaler, args, lr_scheduler=None):
        super(Trainer, self).__init__()
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.hyper_source = hyper_source
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.scaler = scaler
        self.args = args
        self.mm = 0
        self.lr_scheduler = lr_scheduler
        self.train_per_epoch = len(train_loader)
        if val_loader != None:
            self.val_per_epoch = len(val_loader)
        self.best_path = os.path.join(self.args.log_dir, 'best_model.pth')
        self.loss_figure_path = os.path.join(self.args.log_dir, 'loss.png')
        #log
        if os.path.isdir(args.log_dir) == False and not args.debug:
            os.makedirs(args.log_dir, exist_ok=True)
        self.logger = get_logger(args.log_dir, name=args.model, debug=args.debug)
        self.logger.info('Experiment log path in: {}'.format(args.log_dir))
        #if not args.debug:
        #self.logger.info("Argument: %r", args)
        # for arg, value in sorted(vars(args).items()):
        #     self.logger.info("Argument %s: %r", arg, value)

    def val_epoch(self, epoch, sparse_flag, val_dataloader):
        self.model.eval()
        total_val_loss = 0
        total_sp = 0
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(val_dataloader):
                
                data = data[..., :self.args.input_dim]
                label = target[..., :self.args.output_dim]
                output, masks = self.model(self.hyper_source, data, target, teacher_forcing_ratio=0.)
                if self.args.real_value:
                    label = self.scaler.inverse_transform(label)
                loss = self.loss(output.cuda(), label)
                #a whole batch of Metr_LA is filtered
                if not torch.isnan(loss):
                    total_val_loss += loss.item()
                sp = 0 if epoch<=self.mm else 1-masks.nonzero().size(0)/(masks.size(0)*masks.size(1)*masks.size(2)*masks.size(3))
                total_sp= total_sp+sp
                
        val_loss = total_val_loss / len(val_dataloader)
        val_sp = total_sp/len(val_dataloader)
       
        self.logger.info('**********Val Epoch {}: average Loss: {:.6f} | Sparsity: {:.6f} | '.format(epoch, val_loss, val_sp))
        wandb.log({"val averaged Loss": val_loss, "val sparsity": val_sp })
        return val_loss

    def train_epoch(self, epoch, alpha, sparse_flag):
        self.model.train()
        total_loss = 0
        total_sp = 0
        for batch_idx, (data, target) in enumerate(self.train_loader):
            
            data = data[..., :self.args.input_dim]
            label = target[..., :self.args.output_dim]  # (..., 1)
            self.optimizer.zero_grad()
            
            #teacher_forcing for RNN encoder-decoder model
            #if teacher_forcing_ratio = 1: use label as input in the decoder for all steps
            if self.args.teacher_forcing:
                global_step = (epoch - 1) * self.train_per_epoch + batch_idx
                teacher_forcing_ratio = self._compute_sampling_threshold(global_step, self.args.tf_decay_steps)
            else:
                teacher_forcing_ratio = 1.
            #data and target shape: B, T, N, F; output shape: B, T, N, F
            output, masks= self.model(self.hyper_source, data, target,  teacher_forcing_ratio=teacher_forcing_ratio)
            if self.args.real_value:
                label = self.scaler.inverse_transform(label)
            if epoch<=self.mm:
                loss  = self.loss(output.cuda(), label)
            else:
                #loss  = self.loss(output.cuda(), label)+self.model.regularization_cell_1()
                loss  = self.loss(output.cuda(), label)
            loss.backward()

            # add max grad clipping
            if self.args.grad_norm:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
            self.optimizer.step()
            total_loss += loss.item()
            
            sp = 0 if epoch<=self.mm else 1-masks.nonzero().size(0)/(masks.size(0)*masks.size(1)*masks.size(2)*masks.size(3))
            total_sp = total_sp+sp
            #log information
            if batch_idx % self.args.log_step == 0:
                self.logger.info('Train Epoch {}: {}/{} Loss: {:.6f} | Sparsity: {:.6f} | '.format(
                    epoch, batch_idx, self.train_per_epoch, loss.item(), sp))
        train_epoch_loss = total_loss/self.train_per_epoch
        train_sp = total_sp/self.train_per_epoch
        self.logger.info('**********Train Epoch {}: averaged Loss: {:.6f} , train_sp: {:.6f} , tf_ratio: {:.6f}'.format(epoch, train_epoch_loss, train_sp, teacher_forcing_ratio))
        wandb.log({"averaged Loss": train_epoch_loss, "train sparsity": train_sp })
        #learning rate decay
        if self.args.lr_decay:
            self.lr_scheduler.step()
        return train_epoch_loss

    def train(self):
        alpha =1
        best_model = None
        best_loss = float('inf')
        not_improved_count = 0
        train_loss_list = []
        val_loss_list = []
        start_time = time.time()
        for epoch in range(1, self.args.epochs + 1):
            #epoch_time = time.time()
            if epoch<=self.mm:
                #alpha = alpha*10.
                sparse_flag = False
            else:
                sparse_flag = True
            train_epoch_loss = self.train_epoch(epoch, alpha, sparse_flag)
            #print(time.time()-epoch_time)
            #exit()
            if self.val_loader == None:
                val_dataloader = self.test_loader
            else:
                val_dataloader = self.val_loader
            val_epoch_loss = self.val_epoch(epoch, sparse_flag, val_dataloader)

            #print('LR:', self.optimizer.param_groups[0]['lr'])
            train_loss_list.append(train_epoch_loss)
            val_loss_list.append(val_epoch_loss)
            if train_epoch_loss > 1e12:
                self.logger.warning('Gradient explosion detected. Ending...')
                break
            #if self.val_loader == None:
            #val_epoch_loss = train_epoch_loss
            if val_epoch_loss < best_loss:
                best_loss = val_epoch_loss
                not_improved_count = 0
                best_state = True
            else:
                not_improved_count += 1
                best_state = False
            # early stop
            if self.args.early_stop:
                if not_improved_count == self.args.early_stop_patience:
                    self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                    "Training stops.".format(self.args.early_stop_patience))
                    break
            # save the best state
            if best_state == True:
                self.logger.info('*********************************Current best model saved!')
                best_model = copy.deepcopy(self.model.state_dict())

        training_time = time.time() - start_time
        self.logger.info("Total training time: {:.4f}min, best loss: {:.6f}".format((training_time / 60), best_loss))

        #save the best model to file
        if not self.args.debug:
            torch.save(best_model, self.best_path)
            self.logger.info("Saving current best model to " + self.best_path)

        #test
        self.model.load_state_dict(best_model)
        #self.val_epoch(self.args.epochs, self.test_loader)
        self.test(self.model, self.args, self.hyper_source, self.test_loader, self.scaler, self.logger)

    def save_checkpoint(self):
        state = {
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': self.args
        }
        torch.save(state, self.best_path)
        self.logger.info("Saving current best model to " + self.best_path)

    @staticmethod
    def test(model, args, hyper_source, data_loader, scaler, logger, path=None):
        if path != None:
            check_point = torch.load(path)
            state_dict = check_point['state_dict']
            args = check_point['config']
            model.load_state_dict(state_dict)
            model.to(args.device)
        model.eval()
        mask_list =[]
        y_pred = []
        y_true = []
        total_sp = 0
     
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(data_loader):
                data = data[..., :args.input_dim]
                label = target[..., :args.output_dim]
                output, masks= model(hyper_source, data, target, teacher_forcing_ratio=0)
                y_true.append(label)
                y_pred.append(output)
                
                sp = 1-masks.nonzero().size(0)/(masks.size(0)*masks.size(1)*masks.size(2)*masks.size(3))
                
                mask_list.append(masks.to_sparse().cpu())
                total_sp= total_sp+sp
                
        
        test_sp = total_sp/len(data_loader)
      
        y_true = scaler.inverse_transform(torch.cat(y_true, dim=0))
        if args.real_value:
            y_pred = torch.cat(y_pred, dim=0)
        else:
            y_pred = scaler.inverse_transform(torch.cat(y_pred, dim=0))
        np.save('./{}_true.npy'.format(args.dataset), y_true.cpu().numpy())
        np.save('./{}_pred.npy'.format(args.dataset), y_pred.cpu().numpy())
        for t in range(y_true.shape[1]):
            mae, rmse, mape, _, _ = All_Metrics(y_pred[:, t, ...], y_true[:, t, ...],
                                                args.mae_thresh, args.mape_thresh)
            logger.info("Horizon {:02d}, MAE: {:.2f}, RMSE: {:.2f}, MAPE: {:.4f}%".format(
                t + 1, mae, rmse, mape*100))
        mae, rmse, mape, _, _ = All_Metrics(y_pred, y_true, args.mae_thresh, args.mape_thresh)
        logger.info("Average Horizon, MAE: {:.2f}, RMSE: {:.2f}, MAPE: {:.4f}% |Sparisty: {:.6f} ".format(
                    mae, rmse, mape*100, test_sp))
        wandb.log({"MAE": mae, "RMSE":rmse, "MAPE": mape*100, "Sparisty": test_sp})
        if args.if_save_mask:
            torch.save(mask_list,'MASK/'+args.dataset+'.pt')
    @staticmethod
    def _compute_sampling_threshold(global_step, k):
        """
        Computes the sampling probability for scheduled sampling using inverse sigmoid.
        :param global_step:
        :param k:
        :return:
        """
        return k / (k + math.exp(global_step / k))
