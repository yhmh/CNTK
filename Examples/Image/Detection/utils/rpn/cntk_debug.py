# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

from cntk import output_variable
from cntk.ops.functions import UserFunction

class DebugLayer(UserFunction):
    def __init__(self, arg1, name='DebugLayer', debug_name=""):
        super(DebugLayer, self).__init__([arg1], name=name)
        self._debug_name = debug_name

    def infer_outputs(self):
        return [output_variable(self.inputs[0].shape, self.inputs[0].dtype, self.inputs[0].dynamic_axes, name=self._debug_name)]

    def forward(self, arguments, device=None, outputs_to_retain=None):
        values = arguments[0]
        print("Debug({}): values.shape: {}".format(self._debug_name, values.shape))

        return None, values

    def backward(self, state, root_gradients, variables):
        """This layer does not propagate gradients."""
        pass

    def clone(self, cloned_inputs):
        return DebugLayer(cloned_inputs[0], debug_name=self._debug_name)

    def serialize(self):
        internal_state = {}
        internal_state["debug_name"] = self._debug_name
        return internal_state

    @staticmethod
    def deserialize(inputs, name, state):
        debug_name = state['debug_name']
        return DebugLayer(inputs[0], name=name, debug_name=debug_name)

