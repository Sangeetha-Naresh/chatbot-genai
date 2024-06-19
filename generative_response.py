from transformers import AutoModelForCausalLM, AutoTokenizer

# Load pre-trained model and tokenizer from Hugging Face
model_name = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

def generate_bot_response(user_input):
    # Encode input and generate response
    input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt")
    
    # Generate response with the model using better control parameters
    output_ids = model.generate(
        input_ids,
        max_length=100,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        top_k=50,
        top_p=0.95,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id,
        early_stopping=True,
    )
    
    # Decode the output to text
    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    # Post-process the response to make it more suitable for chatbot interaction
    response = response.split('\n')[0]  # Take only the first line
    response = response.split('.')[0]  # Take only the first sentence
    response = response.strip()  # Remove any leading/trailing whitespace
    
    # Remove the user input from the response if it appears at the start
    if response.lower().startswith(user_input.lower()):
        response = response[len(user_input):].strip()

    # Ensure the response ends appropriately
    if response and response[-1] not in ['.', '!', '?']:
        response += '.'
    
    return response