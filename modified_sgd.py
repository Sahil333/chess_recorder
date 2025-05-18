import tensorflow as tf
from keras.optimizers.legacy import Optimizer


class Modified_SGD(Optimizer):
    """ Modified Stochastic gradient descent optimizer.

    Almost all this class is Keras SGD class code. I just reorganized it
    in this class to allow layer-wise momentum and learning-rate

    Includes support for momentum,
    learning rate decay, and Nesterov momentum.
    Includes the possibility to add multipliers to different
    learning rates in each layer.

    # Arguments
        lr: float >= 0. Learning rate.
        momentum: float >= 0. Parameter updates momentum.
        decay: float >= 0. Learning rate decay over each update.
        nesterov: boolean. Whether to apply Nesterov momentum.
        lr_multipliers: dictionary with learning rate for a specific layer
        for example:
            # Setting the Learning rate multipliers
            LR_mult_dict = {}
            LR_mult_dict['c1']=1
            LR_mult_dict['c2']=1
            LR_mult_dict['d1']=2
            LR_mult_dict['d2']=2
        momentum_multipliers: dictionary with momentum for a specific layer 
        (similar to the lr_multipliers)
    """

    def __init__(self, learning_rate=0.01, momentum=0., decay=0.,
                 nesterov=False, lr_multipliers=None, momentum_multipliers=None, name="Modified_SGD", **kwargs):
        super(Modified_SGD, self).__init__(name=name, **kwargs)
        self._set_hyper("learning_rate", learning_rate)
        self._set_hyper("decay", decay)
        self._set_hyper("momentum", momentum)
        self.nesterov = nesterov
        self.lr_multipliers = lr_multipliers
        self.momentum_multipliers = momentum_multipliers
        self.initial_decay = decay

    def _create_slots(self, var_list):
        # Create slots for the momentum variables
        for var in var_list:
            self.add_slot(var, "momentum")

    def _prepare_local(self, var_device, var_dtype, apply_state):
        super(Modified_SGD, self)._prepare_local(var_device, var_dtype, apply_state)
        apply_state[(var_device, var_dtype)]["momentum"] = tf.identity(
            self._get_hyper("momentum", var_dtype))

    def _resource_apply_dense(self, grad, var, apply_state=None):
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = ((apply_state or {}).get((var_device, var_dtype)) or
                       self._fallback_apply_state(var_device, var_dtype))
        
        # Apply learning rate multiplier if available
        lr_t = coefficients["lr_t"]
        if self.lr_multipliers is not None and var.name in self.lr_multipliers:
            lr_t = lr_t * self.lr_multipliers[var.name]
        
        # Apply momentum multiplier if available
        momentum = coefficients["momentum"]
        if self.momentum_multipliers is not None and var.name in self.momentum_multipliers:
            momentum = momentum * self.momentum_multipliers[var.name]
        
        # Get momentum accumulator
        momentum_var = self.get_slot(var, "momentum")
        
        # Use tf.cond instead of Python if-else
        def _momentum_case():
            momentum_buf = momentum_var * momentum - grad * lr_t
            momentum_var_update = momentum_var.assign(momentum_buf)
            
            with tf.control_dependencies([momentum_var_update]):
                if self.nesterov:
                    var_t = var + momentum_buf * momentum - grad * lr_t
                else:
                    var_t = var + momentum_buf
                var_update = var.assign(var_t)
            return var_update
            
        def _no_momentum_case():
            return var.assign_sub(lr_t * grad)
        
        return tf.cond(
            tf.greater(momentum, 0),
            _momentum_case,
            _no_momentum_case
        )

    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = ((apply_state or {}).get((var_device, var_dtype)) or
                        self._fallback_apply_state(var_device, var_dtype))
        
        # Apply learning rate multiplier if available
        lr_t = coefficients["lr_t"]
        if self.lr_multipliers is not None and var.name in self.lr_multipliers:
            lr_t = lr_t * self.lr_multipliers[var.name]
        
        # Apply momentum multiplier if available
        momentum = coefficients["momentum"]
        if self.momentum_multipliers is not None and var.name in self.momentum_multipliers:
            momentum = momentum * self.momentum_multipliers[var.name]
        
        # Get momentum accumulator
        momentum_var = self.get_slot(var, "momentum")
        
        # Use tf.cond instead of Python if-else
        def _momentum_case():
            momentum_buf = momentum_var.assign(momentum_var * momentum)
            
            with tf.control_dependencies([momentum_buf]):
                scatter = self._resource_scatter_add(momentum_var, indices, -grad * lr_t)
                
                with tf.control_dependencies([scatter]):
                    if self.nesterov:
                        var_update = var.assign_add(momentum_var * momentum)
                        with tf.control_dependencies([var_update]):
                            var_update = self._resource_scatter_add(var, indices, -grad * lr_t)
                    else:
                        var_update = self._resource_scatter_add(var, indices, momentum_var)
                    
                    return var_update
        
        def _no_momentum_case():
            return self._resource_scatter_add(var, indices, -grad * lr_t)
        
        return tf.cond(
            tf.greater(momentum, 0),
            _momentum_case,
            _no_momentum_case
        )

    def get_config(self):
        config = super(Modified_SGD, self).get_config()
        config.update({
            "learning_rate": self._serialize_hyperparameter("learning_rate"),
            "decay": self._serialize_hyperparameter("decay"),
            "momentum": self._serialize_hyperparameter("momentum"),
            "nesterov": self.nesterov,
            "lr_multipliers": self.lr_multipliers,
            "momentum_multipliers": self.momentum_multipliers
        })
        return config