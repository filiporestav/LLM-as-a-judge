---
title: LLM as a Judge
emoji: 🧐
colorFrom: yellow
colorTo: purple
sdk: gradio
sdk_version: 5.0.1
app_file: app.py
pinned: false
---

This is a space where you can compare two models using the technique "LLM as a Judge". LLM as a Judge uses a LLM itself for judging the response from two LLMs, and compare them based on certain evaluation metrics which are relevant for the task.

In this space, our default placeholder repos and models compare two LLMs finetuned on the same base model, [Llama 3.2 3B parameter model](unsloth/Llama-3.2-3B-Instruct). Both of them are finetuned on the FineTome-100k dataset, but they have been finetuned on a different amount of data.

The models were finetuned using [Unsloth](https://unsloth.ai/), a framework which allows finetuning, training and inference with LLMs 2x faster.

## Default models and their hyperparameters

### forestav/LoRA-2000

Finetuned on 2000 steps.\
Quantization method: `float16`

### KolumbusLindh/LoRA-4100

Finetuned on 4100 steps.\
Quantization method: `float16`

### Hyperparameters

Both models used the same hyperparameters during training.\
`per_device_train_batch_size = 2`\
`gradient_accumulation_steps=4`\
`learning_rate=2e-4`\
`optim="adamw_8bit"`\
`weight_decay=0.01`\
`lr_scheduler_type="linear"`

We chose float16 as the quantization method as it has the fastest conversion and retains 100% accuracy. However, it is slow and memory hungry which is a disadvantage.
Source: https://github.com/unslothai/unsloth/wiki

## Judge

We are using the KolumbusLindh/LoRA-4100 model as a judge. However, for better accuracy one should use a stronger model such as GPT-4, which can evaluate the responses more thoroughly.

## Evaluation using GPT-4

To better evaluate our fine-tuned models, we let GPT-4 be our judge, when the respective model answered the following prompts:

1. Describe step-by-step how to set up a tent in a windy environment.

2. How-To Guidance: "Explain how to bake a chocolate cake without using eggs."

3. Troubleshooting: "Provide instructions for troubleshooting a laptop that won’t turn on."

4. Educational Explanation: "Teach a beginner how to solve a Rubik’s Cube in simple steps."

5. DIY Project: "Give detailed instructions for building a birdhouse using basic tools."

6. Fitness Routine: "Design a beginner-friendly 15-minute workout routine that requires no equipment."

7. Cooking Tips: "Explain how to properly season and cook a medium-rare steak."

8. Technical Guidance: "Write a step-by-step guide for setting up a local Git repository and pushing code to GitHub."

9. Emergency Response: "Provide instructions for administering first aid to someone with a sprained ankle."

10. Language Learning: "Outline a simple plan for a beginner to learn Spanish in 30 days."

### Results

#### Prompt 1: Describe step-by-step how to set up a tent in a windy environment.
