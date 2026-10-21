"""Two-phase full-objective patience schedule; no parameter rewind."""
import math


class PlateauLR:
    def __init__(self, optimizer, patience=100):
        if int(patience)!=patience or patience<1:raise ValueError('Patience must be a positive integer')
        self.optimizer=optimizer;self.patience=int(patience)
        self.phase=0;self.best=math.inf;self.best_iteration=0
        self.rows=[];self.events=[]

    def observe(self, iteration, loss):
        loss=float(loss)
        if not math.isfinite(loss):raise FloatingPointError('Nonfinite plateau objective')
        if loss<self.best:self.best=loss;self.best_iteration=iteration
        self.rows.append(dict(iteration=iteration,phase=self.phase,loss=loss,
                              best=self.best,since_best=iteration-self.best_iteration))
        if iteration-self.best_iteration<self.patience:return None
        action='reduce' if self.phase==0 else 'stop'
        self.events.append(dict(iteration=iteration,action=action,loss=loss,
                                best=self.best,best_iteration=self.best_iteration))
        if action=='reduce':
            for group in self.optimizer.param_groups:group['lr']*=.1
            self.phase=1;self.best=loss;self.best_iteration=iteration
        return action
