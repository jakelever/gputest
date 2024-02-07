from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from transformers import DataCollatorWithPadding
from transformers import set_seed
from datasets import load_dataset

def main():
	set_seed(42)
	
	model_name = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"

	tokenizer = AutoTokenizer.from_pretrained(model_name)
	tokenizer.add_tokens(["[E1]", "[/E1]", "[E2]", "[/E2]"])

	dataset = load_dataset("jakelever/debug_drugprot")
	
	# Code to replace token_type_ids with zeros
	def replaceTokenTypeIds(x):
		x['token_type_ids'] = [ 0 for _ in x['input_ids'] ]
		return x
	dataset = dataset.map(replaceTokenTypeIds)

	data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
	
	reltypes = ['ACTIVATOR', 'AGONIST', 'AGONIST-ACTIVATOR', 'AGONIST-INHIBITOR', 'ANTAGONIST', 'DIRECT-REGULATOR', 'INDIRECT-DOWNREGULATOR', 'INDIRECT-UPREGULATOR', 'INHIBITOR', 'PART-OF', 'PRODUCT-OF', 'SUBSTRATE', 'SUBSTRATE_PRODUCT-OF']

	model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(reltypes), problem_type="multi_label_classification")
		
	training_args = TrainingArguments(
		output_dir = "awesome_model",
		learning_rate=1e-3,
		per_device_train_batch_size=4,
		per_device_eval_batch_size=4,
		num_train_epochs=16,
		weight_decay=0,
		#evaluation_strategy="epoch",
		save_strategy="no",
		#load_best_model_at_end=True,
	)
	
	trainer = Trainer(
		model=model,
		args=training_args,
		train_dataset=dataset['train'],
		#eval_dataset=val_dataset,
		tokenizer=tokenizer,
		data_collator=data_collator
	)
	
	print("\n\nTraining:\n\n")
	trainer.train()
		
	print()
	print("Done.")

if __name__ == '__main__':
	main()
