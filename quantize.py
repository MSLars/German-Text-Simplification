import os
import argparse
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
from huggingface_hub import HfApi, Repository

from gts.data_loader import get_dataloaders, get_gptq


def awq_quantize_model(model_path, result_awq_path, data_path, load_to_hf, hf_path):
    quant_config = { "zero_point": False, "q_group_size": 128, "w_bit": 4, "version": "Marlin" }

    # Load model
    model = AutoAWQForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    if tokenizer.sep_token_id is None:
        tokenizer.add_special_tokens({'sep_token': '|<SEP>|'})
        print("Set SEP-Token to |<SEP>|")

    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({'pad_token': '|<PAD>|'})
        print("Set PAD-Token to |<PAD>|")


    # Get data loaders and gptq_samples
    gptq_samples = get_gptq(
        tokenizer=tokenizer,
        data_path=data_path,
        format="{bos_token}{context}{sep_token}{input}{sep_token}|<START_LOSS>|{target}{eos_token}",
    )

    # Quantize
    model.quantize(tokenizer, quant_config=quant_config, calib_data=gptq_samples)

    # Save quantized model
    model.save_quantized(result_awq_path)
    tokenizer.save_pretrained(result_awq_path)
    print(f'Model is quantized and saved at "{result_awq_path}"')

    if load_to_hf:
        api = HfApi()
        # Check if the repository exists, and create it if it doesn't
        try:
            api.repo_info(repo_id=hf_path, repo_type="model")
            print(f"Repository '{hf_path}' already exists.")
        except:
            api.create_repo(repo_id=hf_path, exist_ok=True, repo_type="model")
            print(f"Repository '{hf_path}' created.")

        # Upload files to the repository
        for filename in os.listdir(result_awq_path):
            path_in_repo = os.path.join(result_awq_path, filename)
            api.upload_file(
                path_or_fileobj=path_in_repo,
                path_in_repo=filename,
                repo_id=hf_path,
            )
        print(f'Model is uploaded to HF at "{hf_path}"')


def main():
    parser = argparse.ArgumentParser(description="AWQ Quantization CLI")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the fine-tuned HF model directory")
    parser.add_argument('--result_awq_path', type=str, required=True, help="Path to save the AWQ quantized model")
    parser.add_argument('--data_path', type=str, required=True, help="Path to the config data")
    parser.add_argument('--load_to_hf', type=bool, required=True, help="Boolean to decide if the model should be uploaded to HF")
    parser.add_argument('--hf_path', type=str, required=False, help="Name or path for HF upload")

    args = parser.parse_args()

    awq_quantize_model(
        model_path=args.model_path,
        result_awq_path=args.result_awq_path,
        data_path=args.data_path,
        load_to_hf=args.load_to_hf,
        hf_path=args.hf_path
    )


if __name__ == "__main__":
    main()
