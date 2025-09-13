# UAE Central Bank Rulebook QA Assistant

A specialized AI assistant fine-tuned on UAE Central Bank regulations and rulebook content, capable of answering complex banking and regulatory questions with high accuracy and domain-specific knowledge.

## 📋 Overview

This project implements a complete end-to-end pipeline for creating a domain-specific AI assistant specialized in UAE Central Bank rulebook content. The system generates training data, fine-tunes a language model, and deploys it as an interactive web application.

### 🎯 Key Features

- **Domain-Specific Training**: Fine-tuned on 168,711 Q&A pairs from UAE banking regulations
- **Efficient Training**: Uses LoRA adapters with only 0.78% parameter training
- **Memory Optimized**: 4-bit quantization for reduced memory footprint
- **Interactive Interface**: Gradio-based web application with chat functionality
- **Strict Compliance**: Only answers based on UAE rulebook content
- **Fast Inference**: 2x faster inference with Unsloth optimization

## 🏗️ Architecture

```
Data Generation → Model Training → Deployment
      ↓              ↓              ↓
   Q&A Pairs    Fine-tuned Model   Web App
```

### Pipeline Components

1. **Data Generation** (`qa_generate_using_llm.py`)
   - Automated Q&A pair creation using Google Gemini
   - Batch processing with error recovery and checkpointing
   - Context-aware question generation from rulebook text

2. **Data Upload** (`push_json_to_huggingface.py`)
   - Converts JSONL data to HuggingFace Dataset format
   - Uploads to HuggingFace Hub for model training

3. **Model Training** (`Fintune_Liquid_LFM2_(1_2B).ipynb`)
   - LoRA fine-tuning on Liquid LFM2 1.2B model
   - Memory-efficient 4-bit quantization
   - Response-only training for improved accuracy

4. **Model Deployment** (`main.py`)
   - Gradio web interface for user interaction
   - Specialized system prompts for UAE banking context
   - Conversation history and error handling

## 📊 Dataset Statistics

- **Total Q&A Pairs**: 168,711
- **Dataset Size**: 44MB (JSONL format)
- **Source**: UAE Central Bank rulebook content
- **Processing**: 500-character text chunks
- **Quality**: Context-aware, domain-specific questions

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)
- HuggingFace account and token
- Google Gemini API key

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/uae_bank_rulebook_finetune_domainbase.git
   cd uae_bank_rulebook_finetune_domainbase
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install additional dependencies for training**
   ```bash
   pip install unsloth
   pip install trl
   pip install peft
   pip install gradio
   ```

4. **Set up environment variables**
   ```bash
   # Create .env file
   echo "HF_TOKEN=your_huggingface_token" >> .env
   echo "GEMINI_MODEL=your_gemini_model" >> .env
   ```

### Usage

#### 1. Data Generation (Optional - if you have your own data)

```bash
python qa_generate_using_llm.py
```

#### 2. Upload Data to HuggingFace

```bash
python push_json_to_huggingface.py
```

#### 3. Model Training

Open and run the Jupyter notebook:
```bash
jupyter notebook Fintune_Liquid_LFM2_(1_2B) (1).ipynb
```

#### 4. Deploy the Model

```bash
python main.py
```

The web interface will be available at `http://localhost:7860`

## 🔧 Technical Details

### Model Specifications

- **Base Model**: Liquid LFM2 1.2B (4-bit quantized)
- **Training Method**: LoRA (Low-Rank Adaptation)
- **Trainable Parameters**: 9,142,272 (0.78% of total)
- **Training Time**: ~1.19 minutes on A100 GPU
- **Memory Usage**: 1.252 GB peak (3.165% of 40GB)

### Training Configuration

```python
# LoRA Configuration
r = 16
lora_alpha = 16
lora_dropout = 0
bias = "none"

# Training Parameters
per_device_train_batch_size = 2
gradient_accumulation_steps = 4
learning_rate = 2e-4
max_steps = 60
```

### Performance Metrics

- **Training Speed**: 2x faster with Unsloth
- **Memory Efficiency**: 4-bit quantization
- **Accuracy**: Specialized for UAE banking regulations
- **Response Quality**: Context-aware, domain-specific answers

## 📁 Project Structure

```
uae_bank_rulebook_finetune_domainbase/
├── data/
│   ├── qa_merged_50.jsonl          # Generated Q&A dataset (44MB)
│   └── qa_merged_50.zip            # Compressed dataset (8.8MB)
├── main.py                         # Gradio web interface
├── qa_generate_using_llm.py        # Data generation script
├── push_json_to_huggingface.py     # Data upload script
├── Fintune_Liquid_LFM2_(1_2B) (1).ipynb  # Training notebook
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## 🤖 Model Usage

### Web Interface

1. Start the application: `python main.py`
2. Open your browser to `http://localhost:7860`
3. Ask questions about UAE banking regulations
4. The model will provide accurate, rulebook-based answers

### Example Questions

- "What is the minimum capital requirement for a commercial bank in the UAE?"
- "What are the regulations for foreign exchange transactions?"
- "What is the process for obtaining a banking license in the UAE?"

### API Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load the fine-tuned model
model_id = "rajeshthangaraj1/uae_rule_book_QA_assistant"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.float16
)

# Generate response
messages = [
    {"role": "system", "content": "You are an assistant specialized in the UAE Central Bank Rulebook."},
    {"role": "user", "content": "Your question here"}
]

prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.1)
answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

## 🔍 Model Capabilities

- **Regulatory Compliance**: Answers based strictly on UAE rulebook content
- **Domain Expertise**: Specialized knowledge in UAE banking regulations
- **Context Awareness**: Understands complex regulatory relationships
- **Accurate Responses**: High-quality, factually correct answers
- **Conversation Support**: Maintains context across multiple questions

## 📈 Performance

- **Training Efficiency**: 0.78% parameter training with LoRA
- **Memory Usage**: Optimized for 4-bit quantization
- **Inference Speed**: 2x faster with Unsloth optimization
- **Response Quality**: Domain-specific, accurate answers
- **Scalability**: Efficient deployment on various hardware

## 🛠️ Customization

### Adding New Data

1. Place your rulebook text in `scraped_section.txt`
2. Run `qa_generate_using_llm.py` to generate Q&A pairs
3. Update the training notebook with your new dataset
4. Retrain the model with the additional data

### Modifying System Prompts

Edit the system prompt in `main.py`:

```python
messages = [
    {"role": "system", "content": "Your custom system prompt here"}
]
```

## 📚 Dependencies

- **Core ML**: transformers, torch, datasets
- **Training**: unsloth, trl, peft, bitsandbytes
- **Data Processing**: pandas, numpy
- **Web Interface**: gradio
- **API Integration**: agno (for Gemini)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Unsloth**: For efficient fine-tuning capabilities
- **HuggingFace**: For model hosting and dataset management
- **Google Gemini**: For data generation
- **UAE Central Bank**: For regulatory content

## 📞 Support

For questions, issues, or contributions:

- Create an issue in the GitHub repository
- Contact: [your-email@domain.com]
- Documentation: [Link to detailed docs]

## 🔗 Links

- **Model on HuggingFace**: [rajeshthangaraj1/uae_rule_book_QA_assistant](https://huggingface.co/rajeshthangaraj1/uae_rule_book_QA_assistant)
- **Dataset**: [rajeshthangaraj1/uae-banking-rulebook-qa](https://huggingface.co/datasets/rajeshthangaraj1/uae-banking-rulebook-qa)
- **Unsloth Documentation**: [https://docs.unsloth.ai/](https://docs.unsloth.ai/)

---

**Note**: This model is specialized for UAE Central Bank rulebook content and should not be used for general banking advice. Always consult official regulatory sources for compliance matters.
