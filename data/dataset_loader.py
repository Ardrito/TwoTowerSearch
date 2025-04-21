from datasets import load_dataset

# Load MS Marco v1.1 from Hugging Face
dataset = load_dataset("ms_marco", "v1.1")
print(dataset)
print(dataset.keys())

# Look at a sample from the training split
if 'train' in dataset:
    print(f"Training examples: {len(dataset['train'])}")
    print(dataset['train'][0])

# Check validation and test splits
for split in ['validation', 'test']:
    if split in dataset:
        print(f"{split} examples: {len(dataset[split])}")
