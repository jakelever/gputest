
import argparse
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import TrainingArguments, Trainer
	
def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--learning_rate',required=True,type=float)
	parser.add_argument('--batch_size',required=True,type=int)
	parser.add_argument('--num_epochs',required=True,type=int)
	args = parser.parse_args()

	dataset = load_dataset("rotten_tomatoes")

	tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

	def tokenize_function(examples):
		return tokenizer(examples["text"], padding="max_length", truncation=True)

	tokenized_datasets = dataset.map(tokenize_function, batched=True)

	model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2)

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
		eval_dataset=tokenized_datasets['test']
	)

	trainer.train()

if __name__ == '__main__':
	main()
