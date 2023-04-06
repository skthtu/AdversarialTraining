#ref:https://www.kaggle.com/competitions/nbme-score-clinical-patient-notes/discussion/323095#1777969

class Learner(pl.LightningModule):
    def __init__(self, model, num_train_steps, num_warmup_steps,
                 valid_df, wandb):
        super().__init__()
        self.model = model
        self.criterion = nn.BCEWithLogitsLoss(reduction="none")
        self.num_train_steps = num_train_steps
        self.num_warmup_steps = num_warmup_steps
        self.best_f1_score  = 0
        self.steps = 0
        self.automatic_optimization = False

    def training_step(self, batch, batch_idx):
        opt = self.optimizers(use_pl_optimizer=True)

        d_ = batch
        label = d_['label']
        out = self.model(d_['tokens'], d_['attention_mask'], d_["token_type_ids"])
        loss = self.criterion(out.view(-1, 1), label.view(-1, 1))
        loss = torch.masked_select(loss, label.view(-1, 1) != -1).mean()

        opt.zero_grad()
        self.manual_backward(loss)
        opt.step()

        if self.best_f1_score > 0.83:
            adv_loss = self.awp.attack_backward(d_['tokens'], d_['attention_mask'], d_["token_type_ids"],
                                                d_['label'], self.current_epoch)
            self.manual_backward(adv_loss)
            opt.step()
            self.awp._restore()

        self.log('Loss/train', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        sch = self.lr_schedulers()
        sch.step()
        self.steps += 1
        lr = float(sch.get_last_lr()[0])
        self.log('lr', lr, on_step=True, on_epoch=True, prog_bar=False, logger=True)


class AWP:
    def __init__(
        self,
        model,
        optimizer,
        adv_param="weight",
        adv_lr=1.0,
        adv_eps= 0.01,
        start_epoch=0,
        adv_step=1,
    ):
        self.model = model
        self.optimizer = optimizer
        self.adv_param = adv_param
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.start_epoch = start_epoch
        self.adv_step = adv_step
        self.backup = {}
        self.backup_eps = {}
        self.criterion = nn.BCEWithLogitsLoss(reduction="none")

    def attack_backward(self, tokens, attention_mask, token_type_ids, label, epoch):
        if (self.adv_lr == 0) or (epoch < self.start_epoch):
            return None

        self._save()
        for _ in range(self.adv_step):
            self._attack_step()
            with torch.cuda.amp.autocast():

                out = self.model(tokens, attention_mask=attention_mask, token_type_ids=token_type_ids).view(-1, 1)
                adv_loss = self.criterion(out, label.view(-1, 1))
                adv_loss = torch.masked_select(adv_loss, label.view(-1, 1) != -1).mean()

            self.optimizer.zero_grad()
        return adv_loss

    def _attack_step(self):
        e = 1e-6
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                norm1 = torch.norm(param.grad)
                norm2 = torch.norm(param.data.detach())
                if norm1 != 0 and not torch.isnan(norm1):
                    r_at = self.adv_lr * param.grad / (norm1 + e) * (norm2 + e)
                    param.data.add_(r_at)
                    param.data = torch.min(
                        torch.max(param.data, self.backup_eps[name][0]), self.backup_eps[name][1]
                    )
                # param.data.clamp_(*self.backup_eps[name])

    def _save(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                if name not in self.backup:
                    self.backup[name] = param.data.clone()
                    grad_eps = self.adv_eps * param.abs().detach()
                    self.backup_eps[name] = (
                        self.backup[name] - grad_eps,
                        self.backup[name] + grad_eps,
                    )

    def _restore(self):
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}
        self.backup_eps = {}
        
