import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def load_model():
    model_id = "rajeshthangaraj1/uae_rule_book_QA_assistant"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.float16
    )
    return tokenizer, model

def chat_with_model(message, history):
    try:
        print("🔹 Incoming user message:", message)
        print("🔹 History object:", history)

        messages = [
            {"role": "system", "content":
             "You are an assistant specialized in the UAE Central Bank Rulebook. "
             "Only answer based on the UAE Rulebook. "
             "If the answer is not in the Rulebook, reply 'Not found in UAE Rulebook'. "
             "Keep answers concise (max 3-4 sentences)."}
        ]

        # Parse chat history
        for turn in history:
            if len(turn) == 2:
                user_msg, bot_msg = turn
                messages.append({"role": "user", "content": user_msg})
                messages.append({"role": "assistant", "content": bot_msg})
            elif len(turn) == 1:
                user_msg = turn[0]
                messages.append({"role": "user", "content": user_msg})

        # Add current user message
        messages.append({"role": "user", "content": message})

        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        print("🔹 Prompt preview:\n", prompt[:300], "...")

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

        print("🔹 Model answer:", answer)
        return answer

    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"⚠️ Error: {str(e)}"


def main():
    global tokenizer, model
    tokenizer, model = load_model()

    with gr.Blocks() as demo:
        gr.ChatInterface(fn=chat_with_model, title="UAE Rulebook QA Assistant")

    demo.launch(
        debug=True,
        share=False,
        show_error=True
    )

if __name__ == "__main__":
    main()
