from copy import deepcopy
from anytree import Node, RenderTree
import numpy as np

from environment import ArcEnvironment
from state import State

import dreamcoder.domains.arc.bidir.primitives.types as T
import dreamcoder.domains.arc.bidir.primitives.functions as F

from errors import *


class Action:
    def __init__(self):
        pass

    def update_environment_state(self, state, response):
        pass

    def get_copy(self):
        """ 
        Need a copy for storing in TaskTree
        """
        return deepcopy(self)

class ApplyPrimitive(Action):
    def __init__(self, state, node, primitive, forwardDirection):
        self.node = Node(node)  #this is the node to apply action on
        self.primitive = primitive #this is the primitive to apply to the object within the node
        self.forward = forwardDirection #add nodes forward from start if true or backward from end if false

    def update_environment_state(self, state, response):

        #get the object in the node
        newobject = self.primitive(self.node.name)

        #Now build the new state either in the forward direction
        if self.forward:
            
            #Check if the output of the action results in a tuple
            if isinstance(newobject,tuple):
                for x in newobject:
                    state.add_node_to_start(self, self.node, x, self.primitive)
            else:
                state.add_node_to_start(self, self.node, newobject, self.primitive)

        # or in the backward direction
        else:
            if isinstance(newobject,tuple):
                for x in newobject:
                    state.add_node_to_end(self, self.node, newobject, self.primitive)
            else:
                state.add_node_to_end(self, self.node, newobject, self.primitive)
            
#example
class RotateLeft(ApplyPrimitive):      
    def __init__(self, state, node, primitive, forwardDirection):
        self.node = Node(node)  #this is the node to apply action on
        self.primitive = F._rotate_cw(node.name)
        self.forward = forwardDirection


