# pylint: disable=missing-function-docstring
import argparse
import os

APACHE_HEADER = '''# Copyright 2025 SqueezeBits, Inc.
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

'''


def apply_copyright_header(file_path: str) -> None:
    assert os.path.exists(file_path)

    with open(file_path) as f:
        content = f.read()

    if 'Copyright 2025 SqueezeBits, Inc.' in content:
        return

    print(f"Applying copyright header to: {file_path}")
    with open(file_path, 'w') as f:
        f.write(APACHE_HEADER + content)


def apply_copyright_headers(target_dir: str) -> None:
    for root, _, files in os.walk(target_dir):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                apply_copyright_header(file_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--target-dir',
        type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '../src/ditto'),
        help='The directory to apply the copyright header to',
    )
    args = parser.parse_args()

    apply_copyright_headers(args.target_dir)
