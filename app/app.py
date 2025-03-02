import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load the trained model and tokenizer
model_path = "minmarn/dpo_best_model_gpt_neo"  # Ensure this model is accessible
model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Ensure model is in evaluation mode
model.eval()

def generate_response(prompt, max_tokens=100):
    try:
        # Format input as a dialogue
        formatted_prompt = f"Human: {prompt}\n\nAssistant:"
        
        # Tokenize the input
        input_ids = tokenizer(formatted_prompt, return_tensors="pt").input_ids
        
        # Generate a response
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        # Decode and clean response
        full_response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        response = full_response.replace(formatted_prompt, "").strip()
        
        return response
    except Exception as e:
        return f"Error generating response: {str(e)}"

# Streamlit UI setup
st.title("Chat with DPO Assistant-GPT-Neo")
st.write("Product by MMK")

# User input
user_input = st.text_input("Ask something:", "")

if st.button("Get Response"):
    if user_input:
        response = generate_response(user_input)
        st.subheader("Response:")
        st.write(response)
