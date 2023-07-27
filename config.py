from yacs.config import CfgNode as CN


config = CN()
config.DB = 'D:/research/data/adfecgdb'
config.seg_len = 992
config.fs = 200
config.batch_size = 32
config.max_epoch = 80
config.lr = 0.0001

def cfg():
    return config