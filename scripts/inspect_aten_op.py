import argparse
import os

import torch
from torch._ops import OpOverload, OpOverloadPacket, _OpNamespace


def main() -> None:
    available_aten_ops = {
        name: aten_op
        for name, aten_op in vars(torch.ops.aten).items()
        if isinstance(aten_op, OpOverloadPacket | _OpNamespace)
    }

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("ops", nargs="*")
    args = parser.parse_args()

    if not args.ops:
        print("Available ATen ops:\n")
        first_letter = None
        for name in sorted(available_aten_ops):
            # f = "__" if name.startswith("__") else ("_" if name.startswith("_") else name[0])
            # if first_letter != f:
            #     first_letter = f
            #     print_div(
            #         first_letter.upper() if first_letter.isalpha() else {"_": "private", "__": "dunder"}[first_letter]
            #     )
            print(name)
        return

    for op in args.ops:
        aten_op = available_aten_ops.get(op, None)
        if aten_op is None:
            print(f"[ERROR] No such op: {op}")
            continue

        display_op(aten_op)


def display_op(aten_op: OpOverloadPacket) -> None:
    print_div(f"{aten_op}")
    for overload_name in aten_op.overloads():
        op_overload = getattr(aten_op, overload_name, None)
        if isinstance(op_overload, OpOverload):
            print(op_overload._schema)
            continue
        print(f"[WARNING] unrecognized op overload: {type(op_overload)}")


def print_div(msg: str, token: str = "=") -> None:
    assert len(token) == 1
    try:
        terminal_width = os.get_terminal_size().columns
    except OSError:
        terminal_width = 128
    if len(msg) > terminal_width:
        print(msg)
        return
    pads = terminal_width - len(msg)
    lpads = pads // 2
    rpads = pads - lpads
    print(token * lpads + msg + token * rpads)


if __name__ == "__main__":
    main()
