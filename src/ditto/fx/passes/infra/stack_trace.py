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

from torch.fx import Node
from torch.fx.graph import _parse_stack_trace, _ParsedStackTrace
from typing_extensions import Self

from ...nodes import NodeSpecialization


class StackTrace(_ParsedStackTrace):
    @property
    def raw(self) -> str:
        return f'File "{self.file}", line {self.lineno}, in {self.name}\n    Created by {self.code}'

    @classmethod
    def parse(cls, stack_trace: str | None) -> Self | None:
        if stack_trace is None or (_self := _parse_stack_trace(stack_trace)) is None:
            return None
        return cls(
            file=_self.file,
            lineno=_self.lineno,
            name=_self.name,
            code=_self.code,
        )


NodeOrSpecialization = Node | NodeSpecialization


def inject_stack_trace_from(
    node: NodeOrSpecialization,
    *others: NodeOrSpecialization,
    to: NodeOrSpecialization,
) -> None:
    if node.stack_trace is None:
        return
    if parsed_stack_trace := StackTrace.parse(to.stack_trace):
        code = parsed_stack_trace.code
        if others:
            code = f"{code} fusing ({', '.join(n.name for n in (node, *others))})"
        else:
            code = f"{code} substituting {node.name}"
        to.stack_trace = f"{node.stack_trace} -> {code}"
    else:
        to.stack_trace = node.stack_trace
