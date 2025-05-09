import argparse
import pickle
import transformers
import torch
from tqdm import tqdm
import datasets
import os
import json

from utils import apply_template


def generate_samples(model_name_or_path, num_samples, generation_config, dataset_name, dataset_split, device='cpu'):
    """
    Generate samples from a pre-trained model.
    Args:
        model_name_or_path (str): Path to the pre-trained model.
        num_samples (int): Number of samples to use from the dataset.
        generation_config: Configuration for text generation.
        dataset_name (str): Dataset to use for generation.
        dataset_split (str): Dataset split to use for generation.
        device (str): Device to use for generation.
    Returns:
        List of dicts.
        {
            "prompt": str: the story prompt,
            "generated": str: the generated text
        }
    """
    model = transformers.AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.bfloat16)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer.pad_token_id = tokenizer.eos_token_id if tokenizer.pad_token_id is None else tokenizer.pad_token_id

    model = model.to(device)
    model.eval()

    dataset = datasets.load_dataset(dataset_name, split=dataset_split)
    #fix seed
    dataset = dataset.shuffle(seed=2025)
    
    samples = []
    for i in tqdm(range(num_samples)):
        prompt = apply_template(dataset[i]["prompt"], model_name_or_path)
        
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        prompt_length = inputs.input_ids.shape[1]
        
        output_ids = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            generation_config=generation_config,
            pad_token_id=tokenizer.pad_token_id,
        )
        
        new_tokens = output_ids[:, prompt_length:]
        generated_text = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

        samples.append({
            "prompt": prompt,
            "generated": generated_text,
        })
    return samples


def forward_and_get_logits(model_name_or_path, num_samples, dataset_name, dataset_split, max_length, device='cpu'):
    """
    Forward the data samples throught models and get logit vectors.
    Args:
        model_name_or_path (str): Path to the pre-trained model.
        num_samples (int): Number of samples to generate.
        dataset_name (str): Dataset to use for generation.
        dataset_split (str): Dataset split to use for generation.
        max_length (int): Maximum length of story tokens, where we obtain logits upon.
        device (str): Device to use for generation.
    Returns:
        List of dicts.
        {
            "prompt": str: the story prompt,
            "generated_logits": List of generated logits
        }
    """
    model = transformers.AutoModelForCausalLM.from_pretrained(model_name_or_path)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path)

    model = model.to(device)
    model.eval()

    dataset = datasets.load_dataset(dataset_name, split=dataset_split)
    dataset = dataset.shuffle(seed=2025)
    
    samples = []
    all_entropy = []
    
    for i in tqdm(range(num_samples)):
        prompt, story = apply_template(dataset[i]["prompt"], model_name_or_path), dataset[i]["story"] # HACK hardcoded the column name for now
        prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        story_ids = tokenizer(story, return_tensors="pt", truncation=True, max_length=max_length).input_ids.to(device)
        prompt_len = prompt_ids.shape[1]

        input_ids = torch.cat([prompt_ids, story_ids], dim=1)

        with torch.no_grad():
            outputs = model(input_ids)
        story_logits = outputs.logits[0, prompt_len-1:, :].squeeze(0).detach().cpu()

        # TODO top k entropy
        p = torch.softmax(story_logits, dim=-1)
        token_entropy = -torch.sum(p * torch.log(p + 1e-10), dim=-1)
        avg_entropy = torch.mean(token_entropy).item()
        all_entropy.append(avg_entropy)

        samples.append({
            "prompt": prompt,
            "story": story,
            "generated_logits": story_logits,
        })

    return samples, all_entropy

        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="model_outputs")
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--dataset_name", type=str, default="euclaise/writingprompts")
    parser.add_argument("--dataset_split", type=str, default="validation")
    parser.add_argument("--max_length", type=int, default=50, help="Maximum length of story tokens, where we obtain logits upon.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--mode", type=str, choices=["generate", "get_logits"], default="generate",
                        help="Mode to run: generate samples or get logits")
    
    # Generation parameters
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Maximum number of new tokens to generate")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for sampling")
    parser.add_argument("--num_return_sequences", type=int, default=1, help="Number of sequences to return")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.output_dir):
        print(f"Creating output directory {args.output_dir}")
        os.makedirs(args.output_dir)
    
    model_list = [
        'EleutherAI/pythia-2.8b',
        'ContextualAI/archangel_sft_pythia2-8b',
        'ContextualAI/archangel_sft-ppo_pythia2-8b',
        'ContextualAI/archangel_sft-dpo_pythia2-8b',
    ]
    
    for model_name_or_path in model_list:
        print(f"Processing {model_name_or_path} in {args.mode} mode")
        
        if args.mode == "get_logits":
            samples, all_entropy = forward_and_get_logits(
                model_name_or_path, args.num_samples, args.dataset_name, 
                args.dataset_split, args.max_length, device=args.device
            )
            
            # report average entropy
            avg_entropy = sum(all_entropy) / len(all_entropy)
            print(f"Average entropy: {avg_entropy}")

            output_file = f"{args.output_dir}/{model_name_or_path.replace('/', '_')}_{args.dataset_name.replace('/', '_')}_{args.dataset_split}_samples_{args.num_samples}.pkl"
            with open(output_file, "wb") as f:
                pickle.dump(samples, f)
            print(f"Logit samples saved to {output_file}")
            
        elif args.mode == "generate":
            generation_config = transformers.GenerationConfig(
                num_return_sequences=args.num_return_sequences,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                do_sample=True,
            )
            
            samples = generate_samples(
                model_name_or_path, args.num_samples, generation_config,
                args.dataset_name, args.dataset_split, device=args.device
            )
            
            # Save generations in JSON format
            output_file = f"{args.output_dir}/{model_name_or_path.replace('/', '_')}_{args.dataset_name.replace('/', '_')}_{args.dataset_split}_generations_{args.num_samples}_nseq{args.num_return_sequences}.json"
            
            with open(output_file, "w") as f:
                json.dump(samples, f, indent=2)
            print(f"Generated samples saved to {output_file}")
        
if __name__ == "__main__":
    main()