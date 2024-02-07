
import argparse

from scipy.special import expit

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from transformers import DataCollatorWithPadding
from transformers import set_seed

from transformers import AutoModel
from transformers.modeling_outputs import TokenClassifierOutput
import torch

from datasets import load_dataset

from sklearn.metrics import f1_score

class RelationExtractionModel(torch.nn.Module):
	def __init__(self, model_name, num_labels, vocab_size, loss_pos_weight=None):
		super().__init__()
		self.base_model = AutoModel.from_pretrained(model_name)
		self.base_model.resize_token_embeddings(vocab_size)

		self.num_labels = num_labels
		
		representation_size = 768

		self.linear = torch.nn.Linear(representation_size, num_labels)
		self.loss_func = torch.nn.BCEWithLogitsLoss(pos_weight=loss_pos_weight)

	def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, e1_start_index=None, e2_start_index=None, e1_end_index=None, e2_end_index=None, entity_mask=None):
		output = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
		
		representation = output.last_hidden_state[:,0,:]

		logits = self.linear(representation)

		loss = None
		if labels is not None: # If we're provided with labels, use them to calculate the loss
			loss = self.loss_func(logits.reshape(-1,self.num_labels), labels.reshape(-1,self.num_labels))

		return TokenClassifierOutput(loss=loss, logits=logits)
			
def compute_metrics(logits_and_preds):
	logits, labels = logits_and_preds
	predictions = expit(logits).round()
	
	macro_f1 = f1_score(labels, predictions, average='macro', zero_division=0.0)
	micro_f1 = f1_score(labels, predictions, average='micro', zero_division=0.0)
	   
	return {'macro_f1':macro_f1, 'micro_f1':micro_f1}

def main():
	parser = argparse.ArgumentParser('Train and evaluate a BERT-based model')
	parser.add_argument('--learning_rate',required=True,type=float)
	parser.add_argument('--batch_size',required=True,type=int)
	parser.add_argument('--max_epochs',required=True,type=int)
	args = parser.parse_args()
		
	print(f"Running with args={vars(args)}")
	
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

	#model = RelationExtractionModel(
	#	model_name = model_name,
	#	num_labels = len(reltypes),
	#	vocab_size = len(tokenizer))
	
	model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(reltypes), problem_type="multi_label_classification")
		
	training_args = TrainingArguments(
		output_dir = "awesome_model",
		learning_rate=args.learning_rate,
		per_device_train_batch_size=args.batch_size,
		per_device_eval_batch_size=args.batch_size,
		num_train_epochs=args.max_epochs,
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
		data_collator=data_collator,
		compute_metrics=compute_metrics
	)
	
	print("\n\nTraining:\n\n")
	trainer.train()
		
	print()
	print("Done.")


if __name__ == '__main__':
	main()
