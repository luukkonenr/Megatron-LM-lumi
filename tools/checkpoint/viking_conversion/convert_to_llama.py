####################################################################################################

# Copyright (c) 2021-, NVIDIA CORPORATION.  All rights reserved.
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

####################################################################################################

import argparse
import os
import re
import datetime
import torch
from transformers import AutoTokenizer, LlamaConfig
import sys

# change path to a relative path
sys.path.append("/scratch/project_462000319/rluukkon/Megatron-DeepSpeed-dev/")
sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__),
                    os.path.pardir)))

MODES = {'default':0, 'debug':1}
CUR_MODE = 'default'

def log(s, mode='default'):
    if MODES[mode]<=MODES[CUR_MODE]:
        now = str(datetime.datetime.now())
        print(f"{now.split('.')[:-1]}: {s}")

def recursive_print(name, val, spaces=0):
    # Format the message.
    if name is None:
        msg = None
    else:
        fmt = "." * max(0, spaces - 2) + "# {:" + str(50 - spaces) + "s}"
        msg = fmt.format(name)

    # Print and recurse (if needed).
    if isinstance(val, dict):
        if msg is not None:
            print(msg)
        for k in val.keys():
            recursive_print(k, val[k], spaces + 2)
    elif isinstance(val, torch.Tensor):
        print(msg, ":", val.size())
    else:
        print(msg, ":", val)


def fix_query_key_value_ordering(param, checkpoint_version, num_splits, num_heads, hidden_size):
    # log(f"Fix qkv_ord: num_splits: {num_splits} num_heads: {num_heads}, hidden_size: {hidden_size}", mode='debug')
    # Permutes layout of param tensor to [num_splits * num_heads * hidden_size, :]
    # for compatibility with later versions of NVIDIA Megatron-LM.
    # The inverse operation is performed inside Megatron-LM to read checkpoints:
    # https://github.com/NVIDIA/Megatron-LM/blob/v2.4/megatron/checkpointing.py#L209
    # If param is the weight tensor of the self-attention block, the returned tensor
    # will have to be transposed one more time to be read by HuggingFace GPT2.
    input_shape = param.size()
    if checkpoint_version == 1.0:
        # version 1.0 stores [num_heads * hidden_size * num_splits, :]
        saved_shape = (num_heads, hidden_size, num_splits) + input_shape[1:]
        param = param.view(*saved_shape)
        param = param.transpose(0, 2)
        param = param.transpose(1, 2).contiguous()
    elif checkpoint_version >= 2.0:
        # other versions store [num_heads * num_splits * hidden_size, :]
        saved_shape = (num_heads, num_splits, hidden_size) + input_shape[1:]
        param = param.view(*saved_shape)
        param = param.transpose(0, 1).contiguous()
        log("TODO: Delete this print unless models are catasthropically bad")
        # param = param.transpose(1, 0).contiguous()
    param = param.view(*input_shape)
    return param


####################################################################################################

def swap_mlp_layers_to_the_start(lm):
    swapped_lm = dict()
    h_to_4h_keys = [k for k in lm.keys() if "h_to_4h" in k]
    for k in h_to_4h_keys:
        swapped_lm[k] = lm.pop(k)
    for k, v in lm.items():
        swapped_lm[k] = v
    return swapped_lm


def convert_megatron_checkpoint(args, input_state_dict, config):
    # The converted output model.
    output_state_dict = {}

    # old versions did not store training args
    ds_args = input_state_dict.get("args", None)
    if ds_args is not None:
        # do not make the user write a config file when the exact dimensions/sizes are already in the checkpoint
        # from pprint import pprint
        # pprint(vars(ds_args))
        config.vocab_size = ds_args.padded_vocab_size
        config.max_position_embeddings = ds_args.max_position_embeddings
        config.hidden_size = ds_args.hidden_size
        config.num_hidden_layers = ds_args.num_layers
        config.num_attention_heads = ds_args.num_attention_heads
        # megatron_vars(ds_args).get("num_query_groups") 

        config.num_key_value_heads = 32 #ds_args.num_key_value_heads
        config.intermediate_size = ds_args.ffn_hidden_size
        config.untie_embeddings_and_output_weights = ds_args.untie_embeddings_and_output_weights
        # pprint(config)

    # The number of heads.
    heads = config.num_attention_heads
    log(f"Query heads {heads}", mode='debug')
    log(f"KV heads {config.num_key_value_heads}", mode='debug')
    
    # The hidden_size per head.
    hidden_size_per_head = config.hidden_size // config.num_attention_heads
    # Megatron-LM checkpoint version
    # if "checkpoint_version" in input_state_dict.keys():
    #     checkpoint_version = input_state_dict["checkpoint_version"]
    # else:
    #     checkpoint_version = 0.0
    checkpoint_version = 2
    # The model.
    model = input_state_dict["model"]
    # model = input_state_dict
    # The language model.
    lm = model["language_model"]
    # The embeddings.
    embeddings = lm["embedding"]

    # The word embeddings.
    word_embeddings = embeddings["word_embeddings"]["weight"]
    # Truncate the embedding table to vocab_size rows.
    word_embeddings = word_embeddings[: config.vocab_size, :]
    output_state_dict["model.embed_tokens.weight"] = word_embeddings

    # The transformer.
    transformer = lm["transformer"] if "transformer" in lm.keys() else lm["encoder"]
    transformer = swap_mlp_layers_to_the_start(transformer)
    # The regex to extract layer names.
    layer_re = re.compile(r"layers\.(\d+)\.([a-z0-9_.]+)\.([a-z]+)")

    # The simple map of names for "automated" rules.
    megatron_to_transformers = {
        "attention.dense": ".self_attn.o_proj.",
        "self_attention.dense": ".self_attn.o_proj.",
        "mlp.dense_h_to_4h": ".mlp.c_fc.",
        "mlp.dense_4h_to_h": ".mlp.down_proj.",
    }
    

    # Extract the layers.
    for key, val in transformer.items():
        log(key, mode='debug')
        # Match the name.
        m = layer_re.match(key)

        # Stop if that's not a layer
        if m is None:
            break

        # The index of the layer.
        layer_idx = int(m.group(1))
        # The name of the operation.
        op_name = m.group(2)
        # Is it a weight or a bias?
        weight_or_bias = m.group(3)

        log(f"{layer_idx} - {op_name} - {weight_or_bias}", mode='debug')
        # The name of the layer.
        layer_name = f"transformer.h.{layer_idx}"
        layer_name = f"model.layers.{layer_idx}"

        # For layernorm(s), simply store the layer norm.
        if op_name.endswith("norm"):
            #ln_name = "ln_1" if op_name.startswith("input") else "ln_2"
            ln_name = '_'.join(op_name.split('_')[:-1]) + "_layernorm"
            output_state_dict[layer_name + "." + ln_name + "." + weight_or_bias] = val

        # elif op_name.endswith("layernorm"):
        #     #ln_name = "ln_1" if op_name.startswith("input") else "ln_2"
        #     ln_name = op_name
        #     output_state_dict[layer_name + "." + ln_name + "." + weight_or_bias] = val
        # Transpose the QKV matrix.
        elif (
            op_name == "attention.query_key_value" or op_name == "self_attention.query_key_value"
        ) and weight_or_bias == "weight":
            # Insert a tensor of 1x1xDxD bias.
            #causal_mask = torch.tril(torch.ones((n_positions, n_positions), dtype=torch.float16)).view(
            #    1, 1, n_positions, n_positions
            #)
            #output_state_dict[layer_name + ".attn.bias"] = causal_mask

            # Insert a "dummy" tensor for masked_bias.
            #masked_bias = torch.tensor(-1e4, dtype=torch.float16)
            #output_state_dict[layer_name + ".attn.masked_bias"] = masked_bias

            out_val = fix_query_key_value_ordering(val, checkpoint_version, 3, heads, hidden_size_per_head)
            # Megatron stores (3*D) x D but transformers-GPT2 expects D x 3*D.
            #out_val = out_val.transpose(0, 1).contiguous()
            out_val = out_val.contiguous()
            # Store.
            #output_state_dict[layer_name + ".attn.c_attn.weight"] = out_val
            output_state_dict[layer_name + ".self_attn.q_proj.weight"] = out_val[:config.hidden_size, :]
            output_state_dict[layer_name + ".self_attn.k_proj.weight"] = out_val[config.hidden_size:config.hidden_size * 2, :]
            output_state_dict[layer_name + ".self_attn.v_proj.weight"] = out_val[config.hidden_size * 2 :, :]
        elif (
            op_name == "self_attention.query"
        ) and weight_or_bias == "weight":
            out_val = fix_query_key_value_ordering(val, checkpoint_version, 1, heads, hidden_size_per_head)
            #out_val = out_val.transpose(0, 1).contiguous()
            out_val = out_val.contiguous()
            output_state_dict[layer_name + ".self_attn.q_proj.weight"] = out_val
        elif (
            op_name == "self_attention.key_value"
        )   and weight_or_bias == "weight":
            #print(f">> key_value origin size: {val.size()}")
            size_per_weight = val.size(0) // 2
            kv_groups = config.num_attention_heads//config.num_key_value_heads
            out_val = fix_query_key_value_ordering(val, checkpoint_version, 2, heads//kv_groups, hidden_size_per_head)
            #print(f">> key_value output size: {out_val.size()}")
            out_val = out_val.contiguous()
            output_state_dict[layer_name + ".self_attn.k_proj.weight"] = out_val[:size_per_weight, :]
            output_state_dict[layer_name + ".self_attn.v_proj.weight"] = out_val[size_per_weight:, :]
        # Transpose the bias.
        elif (
            op_name == "attention.query_key_value" or op_name == "self_attention.query_key_value"
        ) and weight_or_bias == "bias":
            out_val = fix_query_key_value_ordering(val, checkpoint_version, 3, heads, hidden_size_per_head)
            # Store. No change of shape.
            output_state_dict[layer_name + ".attn.c_attn.bias"] = out_val
        elif op_name == "mlp.dense_h_to_4h":
            # this 2 lines for TP=1 (swiglu)
            if True:
            # if origin_tp_degree == 1:
                output_state_dict[layer_name + ".mlp.gate_proj.weight"] = val[:config.intermediate_size, :]
                output_state_dict[layer_name + ".mlp.up_proj.weight"] = val[config.intermediate_size:, :]
            elif origin_tp_degree == 2:
            # this 2 lines for TP=2 (swiglu)
                output_state_dict[layer_name + ".mlp.gate_proj.weight"] = torch.cat([val[:config.n_inner//2, :], val[config.n_inner:config.n_inner + config.n_inner // 2, :]])
                output_state_dict[layer_name + ".mlp.up_proj.weight"] = torch.cat([val[config.n_inner//2:config.n_inner, :], val[config.n_inner + config.n_inner // 2:, :]])
            else:
                raise ValueError("Not Implemented Yet for TP > 2.")
        # Transpose the weights.
        elif weight_or_bias == "weight":
            out_name = megatron_to_transformers[op_name]
            output_state_dict[layer_name + out_name + "weight"] = val#.transpose(0, 1)

        # Copy the bias.
        elif weight_or_bias == "bias":
            out_name = megatron_to_transformers[op_name]
            output_state_dict[layer_name + out_name + "bias"] = val

    # DEBUG.
    assert config.num_hidden_layers == layer_idx + 1

    # The final layernorm.
    #output_state_dict["transformer.ln_f.weight"] = transformer["final_layernorm.weight"]
    try:
        output_state_dict["model.norm.weight"] = transformer["final_norm.weight"]
    except:
        output_state_dict["model.norm.weight"] = transformer["final_layernorm.weight"]
    #output_state_dict["transformer.ln_f.bias"] = transformer["final_layernorm.bias"]

    # For LM head, transformers' wants the matrix to weight embeddings.
    output_state_dict["lm_head.weight"] = lm['output_layer']['weight']

    # transform the key for LLAMA2
    transform_dict = {
        "transformer.h": "model.layers",

    }
    # It should be done!
    return output_state_dict

def validate_args(args):
    assert not (args.path_to_merged_checkpoint and args.path_to_unmerged_checkpoint), "These flags are mutually exclusive"
    assert (args.path_to_merged_checkpoint or args.path_to_unmerged_checkpoint), "One of these flags is required!"
    assert not os.path.exists(args.output_dir), f"{args.output_dir} already exists and overwriting is not allowed"


####################################################################################################

def main():
    log("Starting the conversion")
    # Create the argument parser.
    parser = argparse.ArgumentParser()
    parser.add_argument("--print-checkpoint-structure", action="store_true")
    parser.add_argument(
        "--path_to_merged_checkpoint",
        type=str,
        help="Path to the checkpoint .pt file. \nMutually exclusive with --path_to_unmerged_checkpoint",
    )
    parser.add_argument(
        "--path_to_unmerged_checkpoint",
        type=str,
        help="Path to the checkpoint dir (checkpoint/path/iter_xxxxxxx/). Will first merge checkpoint into a single shard. \nMutually exclusive with --path_to_merged_checkpoint",
    )

    parser.add_argument(
        "--merged_save_path",
        type=str,
        help="If given, saves the intermediate merged megatron file to the given path",
        default=None
    )

    parser.add_argument(
        "--config_file",
        required=True,
        type=str,
        help="Config json file describing the pre-trained model.",
    )

    parser.add_argument(
        '--tokenizer',
        type=str,
        help="If given, saves the given tokenizer with the model"
    )

    parser.add_argument(
    "--output_dir",
    required=True,
    )

    parser.add_argument(
        "--debug",
        action='store_true'
    )
    
    args = parser.parse_args()

    if args.debug:
        global CUR_MODE
        CUR_MODE='debug'
    
    validate_args(args)

    if args.path_to_unmerged_checkpoint:
        from merge_partitions import merge_model, save_model, parse_output_path

        if args.merged_save_path:
            merged_output_path = parse_output_path(args.path_to_unmerged_checkpoint, args.merged_save_path)
            assert not os.path.exists(os.path.join(merged_output_path, "mp_rank_00")), "Merged output dir already exists!"

        log(f"Starting to merge model shards from {args.path_to_unmerged_checkpoint}")
        input_state_dict = merge_model(args.path_to_unmerged_checkpoint)
        log("...Done")
        
        if args.merged_save_path:
            log(f"Saving the merged megatron model to {merged_output_path}")
            save_model(input_state_dict, merged_output_path)

    else:
        log(f"Loading model shards from {args.path_to_merged_checkpoint}")
        input_state_dict = torch.load(args.path_to_merged_checkpoint, map_location="cpu")

    config = LlamaConfig.from_json_file(args.config_file)
    config.architectures = ["LLaMAForCausalLM"]

    # Convert.
    log("Converting to HF LLaMAForCausalLM")
    output_state_dict = convert_megatron_checkpoint(args, input_state_dict, config)
    log("...done")
    # Print the structure of converted state dict.
    if args.print_checkpoint_structure:
        recursive_print(None, output_state_dict)

    # Store the config to file.
    log("Saving config")
    config.save_pretrained(args.output_dir)
    log("...done")

    # Save tokenizer based on args
    if args.tokenizer:
        log(f"Saving tokenizer from {args.tokenizer} to {args.output_dir}")
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        tokenizer.save_pretrained(args.output_dir)
        log("...Done")
    
    # Store the state_dict to file.
    output_checkpoint_file = os.path.join(args.output_dir, "pytorch_model.bin")
    log(f'Saving checkpoint to "{args.output_dir}"')
    torch.save(output_state_dict, output_checkpoint_file)
    log("...Done")
    log("Model conversion ready!")

####################################################################################################

if __name__ == "__main__":
    main()

####################################################################################################
