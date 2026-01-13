# -------------------------------
# UAE Central Bank Rulebook QA Assistant
# Gradio v2 (Clean Demo UI)
# -------------------------------

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# -------------------------------
# Load model
# -------------------------------

MODEL_ID = "rajeshthangaraj1/uae_rule_book_QA_assistant"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    torch_dtype=torch.float16
)

print("✅ Model loaded successfully")

# -------------------------------
# Chat function
# -------------------------------

def chat_with_model(message, history):
    try:
        system_prompt = (
            "You are an assistant specialized in the UAE Central Bank Rulebook. "
            "Only answer using the UAE Central Bank Rulebook content. "
            "If the answer is not present, reply: 'Not found in UAE Rulebook'. "
            "Keep answers concise (max 3–4 sentences)."
        )

        messages = [{"role": "system", "content": system_prompt}]

        # Rebuild conversation history
        for user_msg, bot_msg in history:
            messages.append({"role": "user", "content": user_msg})
            messages.append({"role": "assistant", "content": bot_msg})

        # Add new user message
        messages.append({"role": "user", "content": message})

        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        inputs.pop("token_type_ids", None)

        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            temperature=0.1,
            top_p=0.95
        )

        new_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
        answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        return answer

    except Exception as e:
        return f"⚠️ Error: {str(e)}"

# -------------------------------
# Gradio UI
# -------------------------------

with gr.Blocks() as demo:

    gr.Markdown("""
    # 🇦🇪 UAE Central Bank Rulebook AI Assistant  
    Ask questions strictly based on official UAE banking regulations.
    """)

    gr.Markdown("""
    **Example questions you can try:**
    """)

    gr.Examples(
        examples=[
            "What percentage of a company's capital must be owned by UAE or GCC nationals or entities wholly owned by them?",
            "What is the role of the Board under Corporate Governance regulations?",
            "Which regulations govern bank licensing in the UAE?",
            "What are the responsibilities of senior management under UAE banking laws?"
        ],
        inputs=gr.Textbox()
    )

    gr.ChatInterface(
        fn=chat_with_model,
        title="UAE Central Bank Rulebook QA Assistant"
    )

    gr.Markdown("""
    ⚠️ **Disclaimer:**  
    This tool is for informational purposes only and does not constitute legal advice.
    Always refer to the official UAE Central Bank Rulebook for compliance decisions.
    """)

# -------------------------------
# Launch app
# -------------------------------

demo.launch(
    debug=False,
    share=False
)
