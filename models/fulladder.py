from nnUtils import *

model = Sequential([
#    DenseNandLayer(256,name='l1'),
    DenseNandLayer(16,name='l2'),
    BatchNormalization(center=False),
    DenseNandLayer(8,name='l3'),
    BatchNormalization(center=False),
    NandLayer(2,name='l4'),
    BatchNormalization(center=False),
])
