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
  - "generated_logits": torch.tensor size(50,50304) [@Bale, please briefly describe how you generate the logits]

### Updates and Acknowledgements
* 2025-04-19: Thx Bale for generating logits and beautifully written code for plots.