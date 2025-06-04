# pylint: disable=C0115

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

import pip
import setuptools
from setuptools.command.develop import develop
from setuptools.command.editable_wheel import editable_wheel
from setuptools.command.install import install


def install_torch_tensorrt_with_no_deps() -> None:
    """torch-tensorrt package should be installed with the --no-deps flag due to its broken dependencies."""
    if pip.main(["install", "torch-tensorrt==2.6.1", "--no-deps"]):
        raise SystemError("Failed to install torch-tensorrt")


class Develop(develop):
    def run(self) -> None:
        develop.run(self)
        install_torch_tensorrt_with_no_deps()


class EditableWheel(editable_wheel):
    def run(self) -> None:
        super().run()
        install_torch_tensorrt_with_no_deps()


class Install(install):
    def run(self) -> None:
        install.run(self)
        install_torch_tensorrt_with_no_deps()


setuptools.setup(
    cmdclass={
        "develop": Develop,
        "install": Install,
        "editable_wheel": EditableWheel,
    },
)
