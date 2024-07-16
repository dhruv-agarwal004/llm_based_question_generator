import fitz  
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Trainer, TrainingArguments
from torch.utils.data import Dataset
import gc

def extract_text_from_pdf(pdf_path):

    doc = fitz.open(pdf_path)
    text = ""

    for page_num in range(len(doc)):
        page = doc.load_page(page_num) 
        text += page.get_text()  
    return text


pdf_path = "JAVA_PROGRAMMING.pdf"  
pdf_text = extract_text_from_pdf(pdf_path)
print("extarction done")



#####TRAINING#######



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

saved_model_path = "./flan-t5-base-finetuned"
model = AutoModelForSeq2SeqLM.from_pretrained(saved_model_path, low_cpu_mem_usage=True)
model.to(device)




max_input_length = 512  
stride = 128

def prepare_data(text):
    inputs = []
    for i in range(0, len(text), stride):
        chunk = text[i:i+max_input_length]
        inputs.append(f"Summarize: {chunk}")
    return inputs  

inputs = prepare_data(pdf_text)

class CustomDataset(Dataset):
    def __init__(self, text_inputs, tokenizer, max_length):
        self.text_inputs = text_inputs
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __getitem__(self, idx):
        tokenized = self.tokenizer(self.text_inputs[idx], 
                                   padding="max_length", 
                                   truncation=True, 
                                   max_length=self.max_length, 
                                   return_tensors="pt")
        item = {key: val.squeeze(0) for key, val in tokenized.items()}
        item['labels'] = item['input_ids'].clone()
        return item

    def __len__(self):
        return len(self.text_inputs)

dataset = CustomDataset(inputs, tokenizer, max_input_length)

# Training configuration
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=10,  
    per_device_train_batch_size=4,  
    gradient_accumulation_steps=2,  
    warmup_steps=0,  
    weight_decay=0.01,
    logging_dir='./logs',
    fp16=True,
    logging_steps=10,
    save_steps=500,
    eval_steps=500,
    save_total_limit=1,
    prediction_loss_only=True,
    remove_unused_columns=False,
    gradient_checkpointing=False,  
   
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

print("Starting training...")
trainer.train()
print("Training complete.")

if torch.cuda.is_available():
    torch.cuda.empty_cache()
gc.collect()

output_dir = "./flan-t5-base-finetuned"
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"Model saved to {output_dir}")

def load_finetuned_model(model_path):
    loaded_model = AutoModelForSeq2SeqLM.from_pretrained(model_path, low_cpu_mem_usage=True)
    loaded_model.to(device)
    loaded_tokenizer = AutoTokenizer.from_pretrained(model_path)
    return loaded_model, loaded_tokenizer

loaded_model, loaded_tokenizer = load_finetuned_model(output_dir)

def generate_text(prompt, max_length=100):
    input_ids = loaded_tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    outputs = loaded_model.generate(input_ids, max_length=max_length)
    return loaded_tokenizer.decode(outputs[0], skip_special_tokens=True)

prompt = "Summarize the key points of the text you were trained on."
generated_text = generate_text(prompt)
print(f"Generated text: {generated_text}")
