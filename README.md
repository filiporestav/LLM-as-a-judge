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

In this space, our default placeholder repos and models compare two LLMs finetuned on the same base model, [Llama 3.2 3B parameter model](unsloth/Llama-3.2-3B-Instruct). Both of them are finetuned on the [FineTome-100k dataset](https://huggingface.co/datasets/mlabonne/FineTome-100k), but they have been finetuned on a different amount of data.

The models were finetuned using [Unsloth](https://unsloth.ai/), a framework which allows finetuning, training and inference with LLMs 2x faster.

## Default models and their hyperparameters

Both models were trained with a [Tesla T4 GPU](https://www.nvidia.com/en-us/data-center/tesla-t4/) with 16GB of GDDR6 memory and 2560 CUDA cores.

### [forestav/LoRA-2000](https://huggingface.co/forestav/LoRA-2000)

Finetuned on 2000 steps.\
Quantization method: `float16`

### [KolumbusLindh/LoRA-6150](https://huggingface.co/KolumbusLindh/LoRA-6150)

Finetuned on 6150 steps.\
Quantization method: `float16`

### Hyperparameters

Both models used the same hyperparameters during training.\
`lora_alpha=16`: Scaling factor for low-rank matrices' contribution. Higher increases influence, speeds up convergence, risks instability/overfitting. Lower gives small effect, but may require more training steps.\
`lora_dropout=0`: Probability of zeroing out elements in low-rank matrices for regularization. Higher gives more regularization but may slow training and degrade performance.\
`per_device_train_batch_size=2`:\
`gradient_accumulation_steps=4`: The number of steps to accumulate gradients before performing a backpropagation update. Higher accumulates gradients over multiple steps, increasing the batch size without requiring additional memory. Can improve training stability and convergence if you have a large model and limited hardware.\
`learning_rate=2e-4`: Rate at which the model updates its parameters during training. Higher gives faster convergence but risks overshooting optimal parameters and instability. Lower requires more training steps but better performance.\
`optim="adamw_8bit"`\: Using the Adam optimizer, a gradient descent method with momentum.
`weight_decay=0.01`: Penalty to add to the weights during training to prevent overfitting. The value is proportional to the magnitude of the weights to the loss function.\
`lr_scheduler_type="linear"`: We decrease the learning rate linearly.

These hyperparameters are [suggested as default](https://docs.unsloth.ai/tutorials/how-to-finetune-llama-3-and-export-to-ollama) when using Unsloth. However, to experiment with them we also tried to finetune a third model by changing the hyperparameters, keeping some of of the above but changing to:

`lora_dropout=0.3`\
`per_device_train_batch_size=20`\
`gradient_accumulation_steps=40`\
`learning_rate=2e-2`

The effects of this were evident. One step took around 10 minutes due to the increased `gradient_accumulation_steps`, and it required significant amount of memory from the GPU due to `per_device_train_batch_size=20`. It also overfitted just in 15 steps, achieving `loss=0`, due to the high learning rate. We wanted to try if the dropout could prevent overfitting while at the same time having a high learning rate, but it could not.

Both models have a max sequence length of 2048 tokens. This means that they only process the 2048 first tokens in the input.

We chose float16 as the quantization method as it according to [Unsloth wiki](https://github.com/unslothai/unsloth/wiki) has the fastest conversion and retains 100% accuracy. However, it is slow and memory hungry which is a disadvantage.

## Judge

We are using the KolumbusLindh/LoRA-6150 model as a judge. However, for better accuracy one should use a stronger model such as GPT-4, which can evaluate the responses more thoroughly.

## Evaluation using GPT-4

To better evaluate our fine-tuned models, we let GPT-4 be our judge, when the respective model answered the following prompts with different kinds of instructions:

1. Describe step-by-step how to set up a tent in a windy environment.

2. Explain how to bake a chocolate cake without using eggs.

3. Provide instructions for troubleshooting a laptop that won’t turn on.

4. Teach a beginner how to solve a Rubik’s Cube in simple steps.

5. Give detailed instructions for building a birdhouse using basic tools.

6. Design a beginner-friendly 15-minute workout routine that requires no equipment.

7. Explain how to properly season and cook a medium-rare steak.

8. Write a step-by-step guide for setting up a local Git repository and pushing code to GitHub.

9. Provide instructions for administering first aid to someone with a sprained ankle.

10. Outline a simple plan for a beginner to learn Spanish in 30 days.

Each model was evaluated by GPT using the following prompt:

```
Prompt:

Response 1:

Response 2:

Response 3:

Response 4:

Evaluation Criteria: Relevance, Coherence and Completeness

Please evaluate the responses based on the selected criteria. For each criterion, rate the response on a scale from 1 to 4. Answer only with the total sum for each response.
```

### Results

![image/png](https://cdn-uploads.huggingface.co/production/uploads/6601e305a4d296af0703f56a/-dy-a44LT_U2FEqap3Zri.png)
**p1** : `temperature=0.5` and `min_p=0.05` during inference\
**p2**: `temperature=1.5` and `min_p=0.1` `during inference

### Discussion

We can see on the results that all models perform relatively good. However, we see that changing inference parameters significantly influenced the results. Setting `min_p=0.1` and `temperature=0.5` improved the quality of the answers, being more creative but still relevant. We saw this in this [Tweet](https://x.com/menhguin/status/1826132708508213629) as well.

#### The temperature

The temperature essentially controls how creative the generation should be, by scaling the logits (raw output probabilities) before they are converted into probabilities via the softmax function. A higher temperature (>1) allows the model to explore less common words or phrases, whereas a lower temperature (<1) makes the output more deterministic and favoring the most probable tokens.

#### The minimum p parameter ("top-p" or nucleus sampling parameter)

This parameter controls which subset of tokens that are considered when generating the next token, with only the most probable tokens being in the sample pool. The cumulative probabilities of the tokens are calculated and sorted, and tokens are included in the sampling pool until the cumulative probability exceeds min_p. Then, the model samples from this reduced set of tokens. A higher min_p (closer to 1) allows more tokens into the pool, leading to a more diverse output. Lower min_p values (e.g. 0.1) restricts the pool to a few of the most probable tokens.

The combination of a relatively high temperature but a small nucleus sampling parameter makes the model generate a diverse and somewhat random output due to the high temperature, while still limiting it to only the most topmost likely tokens within a cumulative probability of 10%.

#### More steps needed to get more significant difference

Something to mention, however, is that the different between the model trained on 2000 steps versus 6150 steps is not that different regarding the evaluation results. We believe that we still need more data to finetune this model for it to make a more signifiant difference. Since the model trained on 2000 steps is capable of providing instructions (and the learning rate increases the most in the beginning), just training the model on more instructions will have a diminishing return. We likely need to train the model on 2 epochs or something like that in order for us to really see a large difference.

#### Further improvements

For further improvement, we should finetuned the model on more data. We should do at least 1-2 epochs on the FineTome-100k dataset, and then watch out closely for overfitting.

For even further improvement, the entire [The Tome](https://huggingface.co/datasets/arcee-ai/The-Tome) dataset (which the FineTome-100k is a subset of), with almost 20x the amount of data should be used for finetuning. However, this requires substantial time and/or more computational resources.
