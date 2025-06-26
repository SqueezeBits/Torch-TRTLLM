import argparse
import json

import numpy as np
import requests
import tensorrt_llm
import torch
from datasets import load_dataset
from PIL import Image
from tensorrt_llm import profiler
from tensorrt_llm._utils import str_dtype_to_torch
from tensorrt_llm.functional import RopeEmbeddingUtils, RotaryScalingType
from tensorrt_llm.layers import MropeParams
from tensorrt_llm.runtime import ModelRunnerCpp
from transformers import AutoConfig, AutoProcessor, AutoTokenizer
from transformers.models.qwen2_5_vl import Qwen2_5_VLConfig

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
DATASET_ID = "MMMU/MMMU"
IMAGE_SIZE = (504, 504)


def run(args: argparse.Namespace, model: ModelRunnerCpp, processor: AutoProcessor) -> None:
    device_id = tensorrt_llm.mpi_rank() % torch.cuda.device_count()
    torch.cuda.set_device(device_id)
    device = torch.device("cuda", index=device_id)

    hf_config = AutoConfig.from_pretrained(MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    vision_config = get_engine_config(args.engine_dir + "/vision")
    vision_dtype = str_dtype_to_torch(vision_config["builder_config"]["precision"])

    profiler.start("Load dataset")
    batch_size = args.batch_size
    # datasets = load_mmmu_dataset()
    # input_texts = [datasets[i][0] for i in range(batch_size)]
    # images = [datasets[i][1] for i in range(batch_size)]
    images = load_images(args.images)
    if len(images) < batch_size:
        images = images * (batch_size // len(images))
    else:
        images = images[:batch_size]
    input_texts = (
        ["Question: Describe this image. Answer:"] * len(images) if args.input_texts is None else args.input_texts
    )
    profiler.stop("Load dataset")

    profiler.start("e2e")
    profiler.start("Preprocess inputs")
    input_ids, attention_mask_llm, image_grid_thw = setup_inputs(processor, input_texts, images, config=hf_config)
    input_ids = input_ids.to(device)
    images = preprocess_images(processor, images).to(vision_dtype).to(device)
    mrope_args = get_mrope_args(input_ids, image_grid_thw, attention_mask_llm, config=hf_config)
    mrope_params = MropeParams(mrope_rotary_cos_sin=mrope_args[0], mrope_position_deltas=mrope_args[1])

    masks = (input_ids == hf_config.image_token_id) | (input_ids == hf_config.vision_token_id)
    cumulative_counts = masks.cumsum(dim=1)
    values = (hf_config.vocab_size - 1) + cumulative_counts
    input_ids[masks] = values[masks]
    profiler.stop("Preprocess inputs")

    # run model
    profiler.start("Run model")
    output_ids = model.generate(
        input_ids,
        mrope_params=mrope_params,
        encoder_input_features=torch.vsplit(images, images.shape[0]),
        sampling_config=None,
        prompt_table=None,
        prompt_tasks=None,
        max_new_tokens=128,
        end_id=tokenizer.eos_token_id,
        pad_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.all_special_ids[0],
        top_k=args.top_k,
        top_p=args.top_p,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
        num_beams=args.num_beams,
        output_sequence_lengths=False,
        return_dict=False,
    )
    profiler.stop("Run model")

    # decode outputs
    profiler.start("Decode outputs")
    input_lengths = torch.tensor([input_ids.shape[1]] * batch_size, dtype=torch.int32)
    output_beams_list = [
        tokenizer.batch_decode(output_ids[batch_idx, :, input_lengths[batch_idx] :], skip_special_tokens=True)
        for batch_idx in range(input_lengths.shape[0])
    ]
    output_texts = [
        [output_beams_list[batch_idx][beam_idx].strip() for beam_idx in range(args.num_beams)]
        for batch_idx in range(input_lengths.shape[0])
    ]
    profiler.stop("Decode outputs")
    profiler.stop("e2e")

    return input_texts, output_texts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine-dir", type=str, required=True, help="Path to the engine directory")
    parser.add_argument("--input-texts", type=str, nargs="+", default=None, help="Input texts")
    parser.add_argument("--images", type=str, nargs="+", default=None, help="File path or url to the image")
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--top-p", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-iterations", type=int, default=10)
    parser.add_argument("--run-profiling", action="store_true")
    return parser.parse_args()


def load_mmmu_dataset() -> list[tuple[str, np.ndarray]]:
    ds = load_dataset(DATASET_ID, "Design")
    datasets: list[tuple[str, np.ndarray]] = []

    for item in ds["test"]:
        if item["question_type"] == "multiple-choice":
            prompt = "Question: " + item["question"]
            prompt += "\nOptions: "
            options = [option.strip("'") for option in item["options"][1:-1].split(", ")]
            for i, option in enumerate(options):
                prompt += f"\n({chr(65 + i)}) {option}"
            prompt += "\nAnswer: "

            image = item["image_1"].convert("RGB").resize(IMAGE_SIZE)
            image = np.array(image)
            datasets.append((prompt, image))

    return datasets


def load_images(image_paths: list[str] | None = None) -> list[np.ndarray]:
    images: list[np.ndarray] = []

    if image_paths is not None:
        for image_path in image_paths:
            if image_path.startswith("http"):
                image = Image.open(requests.get(image_path, stream=True, timeout=5).raw)
            else:
                image = Image.open(image_path)
            image = image.convert("RGB").resize(IMAGE_SIZE)
            image = np.array(image)
            images.append(image)
    else:
        urls = [
            "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
            "https://storage.googleapis.com/sfr-vision-language-research/LAVIS/assets/merlion.png",
        ]
        for url in urls:
            image = Image.open(requests.get(url, stream=True, timeout=5).raw)
            image = image.convert("RGB").resize(IMAGE_SIZE)
            image = np.array(image)
            images.append(image)

    return images


def preprocess_images(processor: AutoProcessor, images: list[np.ndarray]) -> torch.Tensor:
    image_mean: list[float] = processor.image_processor.image_mean
    image_std: list[float] = processor.image_processor.image_std
    rescale_factor: float = processor.image_processor.rescale_factor

    for i in range(len(images)):
        images[i] = images[i] * rescale_factor
        images[i] = images[i] - image_mean
        images[i] = images[i] / image_std
        images[i] = images[i].transpose((2, 0, 1))

    return torch.tensor(images)


def setup_inputs(
    processor: AutoProcessor, input_texts: list[str], images: list[np.ndarray], *, config: Qwen2_5_VLConfig
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert len(input_texts) == len(images), "The number of input texts and images must be the same"
    messages = [
        [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": text},
                ],
            }
        ]
        for text, image in zip(input_texts, images)
    ]

    texts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages]
    inputs = processor(text=texts, images=images, padding=True, return_tensors="pt")

    return inputs["input_ids"], inputs["attention_mask"], inputs["image_grid_thw"]


def get_rope_index(
    input_ids: torch.Tensor, image_grid_thw: torch.Tensor, attention_mask: torch.Tensor, *, config: Qwen2_5_VLConfig
) -> tuple[torch.Tensor, torch.Tensor]:
    position_ids = torch.ones(3, input_ids.shape[0], input_ids.shape[1], dtype=input_ids.dtype, device=input_ids.device)
    total_input_ids = input_ids
    image_index = 0
    mrope_position_deltas = []

    for i, input_ids in enumerate(total_input_ids):
        input_ids = input_ids[attention_mask[i] == 1]
        image_nums = 0
        vision_start_indices = torch.argwhere(input_ids == config.vision_start_token_id).squeeze(1)
        vision_tokens = input_ids[vision_start_indices + 1]
        image_nums = (vision_tokens == config.image_token_id).sum()
        input_tokens = input_ids.tolist()
        llm_pos_ids_list = []
        st = 0
        remain_images = image_nums
        for _ in range(image_nums):
            if config.image_token_id in input_tokens and remain_images > 0:
                ed_image = input_tokens.index(config.image_token_id, st)
            else:
                ed_image = len(input_tokens) + 1

            t, h, w = (image_grid_thw[image_index][0], image_grid_thw[image_index][1], image_grid_thw[image_index][2])
            image_index += 1
            remain_images -= 1
            ed = ed_image

            llm_grid_t, llm_grid_h, llm_grid_w = (
                t.item(),
                h.item() // config.vision_config.spatial_merge_size,
                w.item() // config.vision_config.spatial_merge_size,
            )
            text_len = ed - st

            st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
            llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

            t_index = torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
            h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
            w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
            llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
            st = ed + llm_grid_t * llm_grid_h * llm_grid_w

        if st < len(input_tokens):
            st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
            text_len = len(input_tokens) - st
            llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

        llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
        position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
        mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))

    mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)

    return position_ids, mrope_position_deltas


def get_mrope_args(
    input_ids: torch.Tensor, image_grid_thw: torch.Tensor, attention_mask: torch.Tensor, *, config: Qwen2_5_VLConfig
) -> tuple[torch.Tensor, torch.Tensor]:
    mrope_position_ids, mrope_position_deltas = get_rope_index(input_ids, image_grid_thw, attention_mask, config=config)

    _, rotary_cos_sin = RopeEmbeddingUtils.create_sinusoidal_positions_for_attention_plugin(
        num_pos=config.max_position_embeddings,
        dim=int(config.hidden_size / config.num_attention_heads),
        theta=float(config.rope_theta),
        scale_type=RotaryScalingType.mrope,
    )
    rotary_cos_sin = torch.from_numpy(rotary_cos_sin)
    rotary_cos_sin = rotary_cos_sin.reshape(
        config.max_position_embeddings, int(config.hidden_size / config.num_attention_heads / 2), 2
    )
    cos_orig = rotary_cos_sin[:, :, 0]
    sin_orig = rotary_cos_sin[:, :, 1]

    mrope_position_ids = mrope_position_ids.transpose(1, 0)
    mrope_position_ids_padding = torch.zeros(
        mrope_position_ids.shape[:-1] + (config.max_position_embeddings,), dtype=torch.int32
    )
    mrope_position_ids_padding[:, :, : mrope_position_ids.shape[-1]] = mrope_position_ids
    cos = cos_orig[mrope_position_ids_padding]
    sin = sin_orig[mrope_position_ids_padding]

    mrope_section = config.rope_scaling["mrope_section"]
    cos = torch.cat([m[:, i % 3] for i, m in enumerate(cos.split(mrope_section, dim=-1))], dim=-1).unsqueeze(-1)
    sin = torch.cat([m[:, i % 3] for i, m in enumerate(sin.split(mrope_section, dim=-1))], dim=-1).unsqueeze(-1)
    concat_cos_sin = torch.cat((cos, sin), dim=-1)
    concat_cos_sin = concat_cos_sin.reshape(concat_cos_sin.shape[0], -1)

    return [concat_cos_sin, mrope_position_deltas]


def get_ptuning_args(
    prompt_table: torch.Tensor,
    input_ids: torch.Tensor,
    input_length: int,
    *,
    config: Qwen2_5_VLConfig,
    remove_input_padding: bool,
) -> list[torch.Tensor]:
    task_vocab_size = torch.tensor([prompt_table.shape[1]], dtype=torch.int32)
    prompt_table = prompt_table.view((prompt_table.shape[0] * prompt_table.shape[1], prompt_table.shape[2]))
    assert prompt_table.shape[1] == config.hidden_size, "Prompt table dimensions do not match hidden size"

    if remove_input_padding:
        tasks = torch.zeros([torch.sum(input_length)], dtype=torch.int32)
        tasks = tasks.unsqueeze(0)
    else:
        if not isinstance(input_ids, list):
            tasks = torch.zeros(input_ids.shape, dtype=torch.int32)
        else:
            max_length = max(input_id.size(-1) for input_id in input_ids)
            tasks = torch.zeros((len(input_ids), max_length), dtype=torch.int32)

    return [prompt_table, tasks, task_vocab_size]


def get_engine_config(engine_dir: str) -> dict:
    with open(engine_dir + "/config.json", "r") as f:
        config = json.load(f)
    return config


if __name__ == "__main__":
    args = parse_args()

    # initialize model runner
    model = ModelRunnerCpp.from_dir(
        args.engine_dir,
        rank=tensorrt_llm.mpi_rank(),
        debug_mode=False,
        is_enc_dec=True,
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID, use_fast=True)

    if args.run_profiling:
        for _ in range(3):
            run(args, model, processor)
        profiler.reset()

    num_iters = args.num_iterations if args.run_profiling else 1
    for _ in range(num_iters):
        input_texts, output_texts = run(args, model, processor)

    for input_text, output_text in zip(input_texts, output_texts):
        print(f"[Q] {input_text}")
        print(f"[A] {output_text}\n")

    if args.run_profiling:
        msec_per_batch = lambda name: 1000 * profiler.elapsed_time_in_sec(name) / num_iters
        print("---------------------------------------------------------")
        print("Latencies per batch (msec)")
        print("Load dataset: %.1f" % (msec_per_batch("Load dataset")))
        print("e2e generation: %.1f" % (msec_per_batch("e2e")))
        print(" " * 2 + "Preprocessing: %.1f" % (msec_per_batch("Preprocess inputs")))
        print(" " * 2 + "Run model: %.1f" % (msec_per_batch("Run model")))
        print("Decode outputs: %.1f" % (msec_per_batch("Decode outputs")))
        print("---------------------------------------------------------")
