# from pytorch3dunet

from torch import optim

def create_optimizer(optimizer_config, model):
    optim_name = optimizer_config.get('name', 'Adam')
    # common optimizer settings
    learning_rate = optimizer_config.get('learning_rate', 1e-3)
    weight_decay = optimizer_config.get('weight_decay', 0)
    optim_name = optim_name.lower()

    # grab optimizer specific settings and init
    # optimizer
    if optim_name == 'adadelta':
        rho = optimizer_config.get('rho', 0.9)
        optimizer = optim.Adadelta(model.parameters(), lr=learning_rate, rho=rho,
                                   weight_decay=weight_decay)
    elif optim_name == 'adagrad':
        lr_decay = optimizer_config.get('lr_decay', 0)
        optimizer = optim.Adagrad(model.parameters(), lr=learning_rate, lr_decay=lr_decay,
                                  weight_decay=weight_decay)
    elif optim_name == 'adamW':
        betas = tuple(optimizer_config.get('betas', (0.9, 0.999)))
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, betas=betas,
                                weight_decay=weight_decay)
    elif optim_name == 'sparseadam':
        betas = tuple(optimizer_config.get('betas', (0.9, 0.999)))
        optimizer = optim.SparseAdam(model.parameters(), lr=learning_rate, betas=betas)
    elif optim_name == 'adamax':
        betas = tuple(optimizer_config.get('betas', (0.9, 0.999)))
        optimizer = optim.Adamax(model.parameters(), lr=learning_rate, betas=betas,
                                 weight_decay=weight_decay)
    elif optim_name == 'asgd':
        lambd = optimizer_config.get('lambd', 0.0001)
        alpha = optimizer_config.get('alpha', 0.75)
        t0 = optimizer_config.get('t0', 1e6)
        optimizer = optim.Adamax(model.parameters(), lr=learning_rate, lambd=lambd,
                                 alpha=alpha, t0=t0, weight_decay=weight_decay)
    elif optim_name == 'lbfgs':
        max_iter = optimizer_config.get('max_iter', 20)
        max_eval = optimizer_config.get('max_eval', None)
        tolerance_grad = optimizer_config.get('tolerance_grad', 1e-7)
        tolerance_change = optimizer_config.get('tolerance_change', 1e-9)
        history_size = optimizer_config.get('history_size', 100)
        optimizer = optim.LBFGS(model.parameters(), lr=learning_rate, max_iter=max_iter,
                                max_eval=max_eval, tolerance_grad=tolerance_grad,
                                tolerance_change=tolerance_change, history_size=history_size)
    elif optim_name == 'nadam':
        betas = tuple(optimizer_config.get('betas', (0.9, 0.999)))
        momentum_decay = optimizer_config.get('momentum_decay', 4e-3)
        optimizer = optim.NAdam(model.parameters(), lr=learning_rate, betas=betas,
                                momentum_decay=momentum_decay,
                                weight_decay=weight_decay)
    elif optim_name == 'radam':
        betas = tuple(optimizer_config.get('betas', (0.9, 0.999)))
        optimizer = optim.RAdam(model.parameters(), lr=learning_rate, betas=betas,
                                weight_decay=weight_decay)
    elif optim_name == 'rmsprop':
        alpha = optimizer_config.get('alpha', 0.99)
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, alpha=alpha,
                                  weight_decay=weight_decay)
    elif optim_name == 'rprop':
        momentum = optimizer_config.get('momentum', 0)
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
    elif optim_name == 'sgd':
        momentum = optimizer_config.get('momentum', 0.99)
        dampening = optimizer_config.get('dampening', 0)
        nesterov = optimizer_config.get('nesterov', True)
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum,
                              dampening=dampening, nesterov=nesterov,
                              weight_decay=weight_decay)
        
    elif optim_name == 'stableadamw' : 
        from pytorch_optimizer import StableAdamW
        optimizer = StableAdamW(model.parameters(), weight_decay=weight_decay)
    
    elif optim_name == 'adabelief' : 
        from pytorch_optimizer import AdaBelief
        optimizer = AdaBelief(model.parameters(), weight_decay=weight_decay)

    elif optim_name == 'muon' : 
        from pytorch_optimizer import Muon
        optimizer = Muon(model.parameters(), weight_decay=weight_decay)
    
    elif optim_name == 'shampoo' :
        from pytorch_optimizer import Shampoo
        optimizer = Shampoo(model.parameters(), weight_decay=weight_decay)

    else:  # Adam is default
        betas = tuple(optimizer_config.get('betas', (0.9, 0.999)))
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=betas,
                               weight_decay=weight_decay)

    return optimizer
