from .simple_vsr import SimpleVSR

def get_model(model_name, **kwargs):
    """获取模型"""
    if model_name == 'SimpleVSR':
        return SimpleVSR(**kwargs)
    else:
        raise ValueError(f"未知模型: {model_name}")