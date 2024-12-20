import gradio as gr
from llama_cpp import Llama
from huggingface_hub import hf_hub_download

# Load a user-specified model
def load_user_model(repo_id, model_file):
    print(f"Downloading model {model_file} from repository {repo_id}...")
    local_path = hf_hub_download(repo_id=repo_id, filename=model_file)
    print(f"Model downloaded to: {local_path}")
    return Llama(model_path=local_path, n_ctx=2048, n_threads=8)

# Generate a response using the specified model and prompt
def generate_response(model, prompt):
    response = model(prompt, max_tokens=1024, temperature=1.5, min_p=0.1)
    return response["choices"][0]["text"]

# Evaluate responses using the LoRA evaluation model
def evaluate_responses(prompt, repo_a, model_a, repo_b, model_b):

    # Load models
    model_a_instance = load_user_model(repo_a, model_a)
    model_b_instance = load_user_model(repo_b, model_b)
    
    # Generate responses
    response_a = generate_response(model_a_instance, prompt)
    response_b = generate_response(model_b_instance, prompt)
    
    # Display generated responses
    print(f"Response A: {response_a}")
    print(f"Response B: {response_b}")
    
    # Format the evaluation prompt
    evaluation_prompt = f"""
Prompt: {prompt}

Response A: {response_a}
Response B: {response_b}

Evaluation Criteria: Relevance, Coherence and Completeness

Please evaluate the responses based on the selected criteria. For each criterion, rate both responses on a scale from 1 to 4 and provide a justification. Finally, declare the winner (or 'draw' if they are equal).
"""
    # Use the LoRA model to evaluate the responses
    evaluation_response = lora_model.create_completion(
        prompt=evaluation_prompt,
        max_tokens=512,
        temperature=1.5,
        min_p=0.1
    )
    evaluation_results = evaluation_response["choices"][0]["text"]
    
    # Combine results for display
    final_output = f"""
{evaluation_results}
"""
    return final_output, response_a, response_b

# Load the LoRA evaluation model
def load_lora_model():
    repo_id = "KolumbusLindh/LoRA-6150"
    model_file = "unsloth.F16.gguf"
    print(f"Downloading LoRA evaluation model from repository {repo_id}...")
    local_path = hf_hub_download(repo_id=repo_id, filename=model_file)
    print(f"LoRA evaluation model downloaded to: {local_path}")
    return Llama(model_path=local_path, n_ctx=2048, n_threads=8)

lora_model = load_lora_model()
print("LoRA evaluation model loaded successfully!")

# Gradio interface
with gr.Blocks(title="LLM as a Judge") as demo:
    gr.Markdown("## LLM as a Judge 🧐")
    gr.Markdown("Welcome to the LLM as a Judge demo! This application uses a finetuned LLM to evaluate responses generated by two different models based on Relevance, Coherence and Completeness. The model will then evaluate the responses based on the criteria and determine the winner.")
    gr.Markdown("The default models are models we have finetuned on the FineTome-100k dataset, using Llama 3.2 3B as the base model. You can also specify your own models by entering the Hugging Face repository name and model filename for Model A and Model B. Just make sure they are in GGUF format.")

    # Model inputs
    repo_a_input = gr.Textbox(label="Model A Repository", placeholder="Enter the Hugging Face repo name for Model A...", value="forestav/LoRA-2000")
    model_a_input = gr.Textbox(label="Model A File Name", placeholder="Enter the model filename for Model A...", value="unsloth.F16.gguf")
    repo_b_input = gr.Textbox(label="Model B Repository", placeholder="Enter the Hugging Face repo name for Model B...", value="KolumbusLindh/LoRA-6150")
    model_b_input = gr.Textbox(label="Model B File Name", placeholder="Enter the model filename for Model B...", value="unsloth.F16.gguf")

    # Prompt and criteria inputs
    prompt_input = gr.Textbox(label="Enter Prompt", placeholder="Enter the prompt here...", lines=3)

    # Button and outputs
    evaluate_button = gr.Button("Evaluate Models")

    with gr.Row():
        with gr.Column():
            response_a = gr.Textbox(
                label="Response A",
                placeholder="The response from Model A will appear here...",
                lines=20,
                interactive=False
            )

        with gr.Column():
            response_b = gr.Textbox(
                label="Response B",
                placeholder="The response from Model B will appear here...",
                lines=20,
                interactive=False
            )

    gr.Markdown("### The LLMs are evaluated based on the criterion of Relevance, Coherence and Completeness.")
    evaluation_output = gr.Textbox(
        label="Evaluation Results",
        placeholder="The evaluation results will appear here...",
        lines=20,
        interactive=False
    )

    # Link evaluation function
    evaluate_button.click(
        fn=evaluate_responses,
        inputs=[prompt_input, repo_a_input, model_a_input, repo_b_input, model_b_input],
        outputs=[evaluation_output, response_a, response_b]
    )

# Launch app
if __name__ == "__main__":
    demo.launch()
