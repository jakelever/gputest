
import argparse
from datasets import load_dataset
from transformers import AutoModelForTokenClassification, AutoTokenizer, DataCollatorForTokenClassification
from transformers import TrainingArguments, Trainer
	
def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--learning_rate',required=True,type=float)
	parser.add_argument('--batch_size',required=True,type=int)
	parser.add_argument('--num_epochs',required=True,type=int)
	args = parser.parse_args()

	dataset = load_dataset("conll2003")
	dataset = dataset.rename_column('ner_tags','labels')

	tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

	def tokenize_function(examples):
		return tokenizer(examples["tokens"], padding="max_length", truncation=True, is_split_into_words=True)

	tokenized_datasets = dataset.map(tokenize_function, batched=True)

	model = AutoModelForTokenClassification.from_pretrained("bert-base-cased", num_labels=9)

	data_collator = DataCollatorForTokenClassification(tokenizer)

	training_args = TrainingArguments(
		output_dir="test_trainer",
		learning_rate=args.learning_rate,
		per_device_train_batch_size=args.batch_size,
		per_device_eval_batch_size=args.batch_size,
		num_train_epochs=args.num_epochs,
		save_strategy="no"
	)

	trainer = Trainer(
		model=model,
		args=training_args,
		train_dataset=tokenized_datasets['train'],
		eval_dataset=tokenized_datasets['test'],
		data_collator=data_collator
	)

	trainer.train()

if __name__ == '__main__':
	main()
