from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from transformers import DataCollatorWithPadding
from transformers import set_seed
from datasets import load_dataset

def main():
	set_seed(42)
	
	dataset = load_dataset("jakelever/debug_drugprot2")
	
	model_name = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"

	tokenizer = AutoTokenizer.from_pretrained(model_name)
	model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=13, problem_type="multi_label_classification")
	
	data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
		
	training_args = TrainingArguments(
		output_dir = "awesome_model",
		learning_rate=1e-3,
		per_device_train_batch_size=4,
		per_device_eval_batch_size=4,
		num_train_epochs=16,
		weight_decay=0,
		save_strategy="no",
	)
	
	trainer = Trainer(
		model=model,
		args=training_args,
		train_dataset=dataset['train'],
		tokenizer=tokenizer,
		data_collator=data_collator
	)
	
	trainer.train()	
	
if __name__ == '__main__':
	main()
