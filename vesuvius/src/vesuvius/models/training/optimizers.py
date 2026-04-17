# from pytorch3dunet

from torch import nn, optim


def _iter_embedding_parameter_ids(model):
    embedding_param_ids = set()
    for module in model.modules():
        if isinstance(module, nn.Embedding):
            for parameter in module.parameters(recurse=False):
                embedding_param_ids.add(id(parameter))
    return embedding_param_ids


def _build_muon_param_groups(model):
    embedding_param_ids = _iter_embedding_parameter_ids(model)
    muon_params = []
    aux_adamw_params = []
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim >= 2 and id(parameter) not in embedding_param_ids:
            muon_params.append(parameter)
        else:
            aux_adamw_params.append(parameter)

    groups = []
    if muon_params:
        groups.append({"params": muon_params, "use_muon": True})
    if aux_adamw_params:
        groups.append({"params": aux_adamw_params, "use_muon": False})
    return groups



def create_optimizer(optimizer_config, model):
    optim_name = optimizer_config.get('name', 'Adam')
    learning_rate = optimizer_config.get('learning_rate', 1e-3)
    weight_decay = optimizer_config.get('weight_decay', 0)
    eps = optimizer_config.get('eps', None)
    optim_name = optim_name.lower()

    if optim_name == 'adadelta':
        rho = optimizer_config.get('rho', 0.9)
        optimizer = optim.Adadelta(model.parameters(), lr=learning_rate, rho=rho,
                                   weight_decay=weight_decay)
    elif optim_name == 'adagrad':
        lr_decay = optimizer_config.get('lr_decay', 0)
        optimizer = optim.Adagrad(model.parameters(), lr=learning_rate, lr_decay=lr_decay,
                                  weight_decay=weight_decay)
    elif optim_name == 'adamw':
        betas = tuple(optimizer_config.get('betas', (0.9, 0.999)))
        adamw_kwargs = {
            'lr': learning_rate,
            'betas': betas,
            'weight_decay': weight_decay,
        }
        if eps is not None:
            adamw_kwargs['eps'] = eps
        optimizer = optim.AdamW(model.parameters(), **adamw_kwargs)
    elif optim_name == 'sparseadam':
        betas = tuple(optimizer_config.get('betas', (0.9, 0.999)))
        optimizer = optim.SparseAdam(model.parameters(), lr=learning_rate, betas=betas)
    elif optim_name == 'adamax':
        betas = tuple(optimizer_config.get('betas', (0.9, 0.999)))
        adamax_kwargs = {'lr': learning_rate, 'betas': betas, 'weight_decay': weight_decay}
        if eps is not None:
            adamax_kwargs['eps'] = eps
        optimizer = optim.Adamax(model.parameters(), **adamax_kwargs)
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
        radam_kwargs = {'lr': learning_rate, 'betas': betas, 'weight_decay': weight_decay}
        if eps is not None:
            radam_kwargs['eps'] = eps
        optimizer = optim.RAdam(model.parameters(), **radam_kwargs)
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

    elif optim_name == 'muon':
        try:
            from pytorch_optimizer import Muon
        except ImportError as exc:
            raise ImportError(
                "optimizer.name='muon' requires the 'pytorch-optimizer' package. "
                "Install the project with the 'models' extra."
            ) from exc

        momentum = optimizer_config.get('momentum', 0.95)
        weight_decouple = optimizer_config.get('weight_decouple', True)
        nesterov = optimizer_config.get('nesterov', True)
        ns_steps = optimizer_config.get('ns_steps', 5)
        use_adjusted_lr = optimizer_config.get('use_adjusted_lr', False)
        adamw_lr = optimizer_config.get('adamw_lr', 3e-4)
        adamw_betas = tuple(optimizer_config.get('adamw_betas', (0.9, 0.95)))
        adamw_wd = optimizer_config.get('adamw_wd', weight_decay)
        adamw_eps = optimizer_config.get('adamw_eps', 1e-10)
        maximize = optimizer_config.get('maximize', False)
        param_groups = _build_muon_param_groups(model)
        if not param_groups:
            raise ValueError("Muon optimizer requires at least one trainable parameter")
        optimizer = Muon(
            param_groups,
            lr=optimizer_config.get('learning_rate', 2e-2),
            momentum=momentum,
            weight_decay=weight_decay,
            weight_decouple=weight_decouple,
            nesterov=nesterov,
            ns_steps=ns_steps,
            use_adjusted_lr=use_adjusted_lr,
            adamw_lr=adamw_lr,
            adamw_betas=adamw_betas,
            adamw_wd=adamw_wd,
            adamw_eps=adamw_eps,
            maximize=maximize,
        )
    
    elif optim_name == 'shampoo' :
        from pytorch_optimizer import Shampoo
        optimizer = Shampoo(model.parameters(), weight_decay=weight_decay)

    else:  # Adam is default
        betas = tuple(optimizer_config.get('betas', (0.9, 0.999)))
        adam_kwargs = {'lr': learning_rate, 'betas': betas, 'weight_decay': weight_decay}
        if eps is not None:
            adam_kwargs['eps'] = eps
        optimizer = optim.Adam(model.parameters(), **adam_kwargs)

    return optimizer
