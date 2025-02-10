# Copyright 2025 SqueezeBits, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pyright: reportAttributeAccessIssue=false, reportReturnType=false, reportArgumentType=false
import torch

from .aten_op import FinalATenOp
from .unary_elementwise import UnaryElementwise


class Activation(UnaryElementwise):
    """Base class for activation functions."""


@Activation.register(torch.ops.aten.elu.default)
class Elu(Activation, FinalATenOp):
    """Specialization for the ELU activation function."""


@Activation.register(torch.ops.aten.gelu.default)
class Gelu(Activation, FinalATenOp):
    """Specialization for the GELU activation function."""


@Activation.register(torch.ops.aten.hardsigmoid.default)
class HardSigmoid(Activation, FinalATenOp):
    """Specialization for the hard sigmoid activation function."""


@Activation.register(torch.ops.aten.leaky_relu.default)
class LeakyRelu(Activation, FinalATenOp):
    """Specialization for the leaky ReLU activation function."""


@Activation.register(torch.ops.aten.relu.default)
class Relu(Activation, FinalATenOp):
    """Specialization for the ReLU activation function."""


@Activation.register(torch.ops.aten.sigmoid.default)
class Sigmoid(Activation, FinalATenOp):
    """Specialization for the sigmoid activation function."""


@Activation.register(torch.ops.aten.softplus.default)
class Softplus(Activation, FinalATenOp):
    """Specialization for the softplus activation function."""


@Activation.register(torch.ops.aten.tanh.default)
class Tanh(Activation, FinalATenOp):
    """Specialization for the tanh activation function."""
