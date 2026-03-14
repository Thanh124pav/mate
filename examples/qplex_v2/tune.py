from ray.tune.registry import get_trainable_cls, TRAINABLE_CLASS, _global_registry

# Lấy tất cả các trainable đã đăng ký
registered_trainables = _global_registry.keys(TRAINABLE_CLASS)
print(list(registered_trainables))