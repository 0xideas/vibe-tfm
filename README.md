## VIBE CODE A TRANSACTION FOUNDATION MODEL, A vibe-tfm

### Background

You really shouldn't vibe code a foundation model from scratch. Unless you use [sequifier](https://github.com/0xideas/sequifier), a framework for configurable transformer model training. I would pair it with a frontier or near-frontier model such as Astra and GPT-5.6, then I think it's actually a viable approach, for a v1.

### Approach

Put the data you want to train on in `data`, install sequifier using `pip install sequifier` then paste the prompt below into Codex or CC.

### Prompt

Your goal is to build a v1 transaction foundation model that learns informative representations from transaction data, using sequifier.

Please preprocess the transaction data you find in `data` to prepare it to serve as input data to `sequifier preprocess`. Find the file https://github.com/0xideas/sequifier/blob/main/documentation/consolidated-docs.md for the sequifier version that is installed and use it as reference.

You need to achieve the following:
 1. Identify what current and derived variables should go into the v1 of the transaction foundation model, as input and target variables. Check what input data are available, the quality of each (completeness, outliers), potential problems with their distribution or cardinality, and adopt mitigations for these potential problems (for reference, check resources/shaping-the-problem.md). Document your analysis, decision process and decision outcomes in documentation/thoughts-on-preprocessing.md. When in doubt over specific tradeoffs, check the wider literature. 
 2. Write a python script that preprocesses the input data into the desired output, including cleaned original variables and derived variables. Run the script and check that the output matches your expectations/does not indicate any issue for downstream use. If the script fails or the output is unsatisfactory in some way, add these findings to documentation/thoughts-on-preprocessing.md (but don't delete anything), adapt the script, and run it again, until the script succeeds and the data looks clean.
 3. Adapt the preprocessing config at `configs/preprocess.yaml` to match the input and target variables specified in documentation/thoughts-on-preprocessing.md. Run `sequifier preprocess` and check that it succeeds. Configure 3 split_ratios, so that we have a held out test set. If it doesn't, add the findings to documentation/thoughts-on-preprocessing.md, rerun the custom preprocessing script, and run `sequifier preprocess`, until it succeeds.
 4. Configure a v1 model in `configs/train.yaml`. Train a causal model unless something different was specified elsewhere, and ensure that the parameter count is appropriate for the amount of data that is available. Go with a safe, fairly standard approach to the specific model configuration, unless you have strong reasons to deviate from that. Do not use mixed precision training, and train at default precision. Ensure that the disk has sufficient space for the expected disk footprint of the configured checkpoints, by a factor of 2. Run `sequifier train`. If it fails, fix the issue in the training config and try again, until it succeeds. If that doesn't work, go back to step 1.
 5. Configure model inference at `configs/infer.yaml` appropriately, and to infer on the test set data (3rd split). Run `sequifier infer`. If it fails, fix `infer.yaml` until it succeeds. If that doesn't work, go back to step 1.
 6. Analyse the model outputs extensively, using any interesting internal or external (if provided in the original input data) validation of the model outputs. Summarise your findings and output them in a nicely formatted, concise documentation/v1-analysis.md. Think about what experiments could make sense, and add these ideas to that output.

 Overall, try not to overengineer and keep it simple, for now. This is a v1. We mainly want to learn whether this data is suitable for foundation model training, not to get a final result in one session. At the same time, be thorough so that we can make evidence-based decisions. 
