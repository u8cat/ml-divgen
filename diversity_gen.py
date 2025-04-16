import argparse
import pickle
import transformers
from tqdm import tqdm
import datasets


def generate_samples(model_name_or_path, num_samples, dataset, max_length,device='cpu'):
    """
    Generate samples from a pre-trained model.
    Args:
        model_name_or_path (str): Path to the pre-trained model.
        num_samples (int): Number of samples to generate.
        dataset (str): Dataset to use for generation.
        max_length (int): Maximum length of generated text.
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
    dataset = datasets.load_dataset(dataset, split="test")
    #fix seed
    dataset = dataset.shuffle(seed=2025)
    
    samples = []

    for i in tqdm(range(num_samples)):
        prompt = dataset[i]["story"]
        #Get the first 5 words of the story
        prompt_list = prompt.split(" ")
        all_output=[]
        for i in range(5,max_length+5):
            curr_prompt = " ".join(prompt_list[:i])
            input_ids = tokenizer(curr_prompt, return_tensors="pt").input_ids.cuda() if device=='cuda' else tokenizer(curr_prompt, return_tensors="pt").input_ids
            outputs = model.generate(input_ids, max_new_tokens=4, do_sample=True, return_dict_in_generate=True, output_scores=True)
            all_output.append(outputs)
        samples.append({"prompt": prompt, "generated_logits": all_output})
    return samples

def main():
    parser = argparse.ArgumentParser()
    #parser.add_argument("--model_name_or_path", type=str, default="EleutherAI/pythia-1.4b")
    parser.add_argument("--output_dir", type=str, default="model_outputs")
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--dataset", type=str, default="euclaise/writingprompts")
    parser.add_argument("--max_length", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda")
    
    
    args = parser.parse_args()
    
    model_list = [
        'EleutherAI/pythia-2.8b',
        'ContextualAI/archangel_sft_pythia2-8b'
    ]
    for model_name_or_path in model_list:
        print(f"Generating samples for {model_name_or_path}")
        samples = generate_samples(model_name_or_path, args.num_samples, args.dataset, args.max_length,device=args.device)
        output_file = f"{args.output_dir}/{model_name_or_path.replace('/', '_')}_{args.dataset.replace('/', '_')}_samples_{args.num_samples}.pkl"
        with open(output_file, "wb") as f:
            pickle.dump(samples, f)
        print(f"Samples saved to {output_file}")
        
if __name__ == "__main__":
    main()