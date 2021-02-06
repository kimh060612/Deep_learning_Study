from tensorflow import keras
import tensorflow as tf

# 별로 안 중요함. ==> 너네들이 만질 일이 거의 없음.
class CustomSGDOptimizer(keras.optimizers.Optimizer):
    def __init__(self, learning_rate = 0.001, name = "CustomSGDOptimizer", **kwargs):
        super().__init__(name, **kwargs)
        self._set_hyper("learning_rate", kwargs.get("lr", learning_rate))
        self._is_first = True

    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(var, "pv") # previous variable
        for vat in var_list:
            self.add_slot(var, "pg") # previous gradient
    
    @tf.function
    def _resource_apply_dense(self, grad, var):
        var_dtype = var.dtype.base_dtype
        lr_t = self._decayed_lr(var_dtype)

        new_var_m = var - lr_t * grad

        pv_var = self.get_slot(var, "pv")
        pg_var = self.get_slot(var, "pg")

        if self._is_first :
            self._is_first = False
            new_var = new_var_m
        else:
            cond = grad * pg_var >= 0
            avg_weight = (pv_var + var) / 2.0
            new_var = tf.where(cond, new_var_m, avg_weight)
        
        pv_var.assign(var)
        pg_var.assign(grad)

        var.assign(new_var)

    def _resource_apply_sparse(self, grad, var):
        raise NotImplementedError
    
    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "learning_rate" : self._serialize_hyperparameter("lr")
        }