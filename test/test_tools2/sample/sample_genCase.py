import sys
import os

from tecoalNode import *
from nodes.embedding_forward_node import TecoalEmbeddingForwardNode

case = TecoalEmbeddingForwardNode()
indices_dims = [1, 2, 1, 1]
weights_dims = [123, 1, 1, 456]
out_dims = [1, 2, 1, 456]
case.node.input[0].dtype = 1
setTensorShape(case.node.input[0], indices_dims)
setTensorShape(case.node.input[1], weights_dims)
setTensorShape(case.node.input[2], out_dims)
if case.check():
    case.saveTo("./case_0.prototxt")
