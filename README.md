## This is the code base for the final project of ML2025 Spring

### To do
- [ ] Check the generated logits
- [ ] Generate on 1k samples
- [ ] Plots 
  
### Data format
- File names: 
  - SFT  generations: "ContextualAI_archangel_sft_pythia2-8b_euclaise_writingprompts_validation_samples_100.pkl"
  - Base generations: "EleutherAI_pythia-2.8b_euclaise_writingprompts_validation_samples_100.pkl
- In each files, there are 100 generations conditioned on the 100 different prompts.
- Each generation is a dictionary with the following keys:
  - "prompt": str the prompt used to generate the text
  - "story": str human written story conditioned on the prompt
  - "generated_logits": torch.tensor size(51,50304). Each column vector corresponds to the next-token logit distribution of the last position in the prompt tokens and the first 50 positions in the story tokens. For instance, the 10th column in this matrix corresponds to the next token that succeeds the prompt and the first 9 tokens in the story.

### Updates and Acknowledgements
* 2025-04-21: Added the generation mode in `diversity_gen.py`. 
* 2025-04-19: Thx Bale for generating logits and beautifully written code for plots (Bale: in colab with copilot XD).
