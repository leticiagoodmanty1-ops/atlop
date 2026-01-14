# Kaggle environment configuration - must be at the top
import sys
sys.path.append('/kaggle/input/mydataset')

import os
os.environ["WANDB_MODE"] = "offline"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# Suppress tokenizers parallelism warning when using DataLoader with num_workers > 0
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Suppress TensorFlow warnings (Kaggle has both TF and PyTorch installed)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import argparse
import numpy as np
import torch
import ujson as json
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from model import DocREModel
from utils import set_seed, collate_fn
from prepro import read_docred
from evaluation import to_official, official_evaluate
import wandb

# ===== Performance Optimization =====
# Enable cuDNN auto-tuner to find the best algorithm for your hardware
torch.backends.cudnn.benchmark = True


def train(args, model, train_features, dev_features, test_features):
    def finetune(features, optimizer, num_epoch, num_steps):
        best_score = -1
        train_dataloader = DataLoader(features, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True, num_workers=2, pin_memory=True)
        train_iterator = range(int(num_epoch))
        total_steps = int(len(train_dataloader) * num_epoch // args.gradient_accumulation_steps)
        warmup_steps = int(total_steps * args.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
        print("Total steps: {}".format(total_steps))
        print("Warmup steps: {}".format(warmup_steps))
        
        # Initialize gradient scaler for mixed precision
        scaler = torch.amp.GradScaler('cuda')
        
        # Create Kaggle output directory if it doesn't exist
        os.makedirs('/kaggle/working', exist_ok=True)
        
        for epoch in train_iterator:
            model.zero_grad()
            # Add tqdm progress bar for training
            pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), 
                       desc=f"Epoch {epoch+1}/{int(num_epoch)}", position=0, leave=True)
            
            for step, batch in pbar:
                model.train()
                inputs = {'input_ids': batch[0].to(args.device),
                          'attention_mask': batch[1].to(args.device),
                          'labels': batch[2],
                          'entity_pos': batch[3],
                          'hts': batch[4],
                          }
                
                # Use native PyTorch AMP (latest syntax)
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    outputs = model(**inputs)
                    loss = outputs[0] / args.gradient_accumulation_steps
                
                # Scale loss and backward pass
                scaler.scale(loss).backward()
                
                if step % args.gradient_accumulation_steps == 0:
                    if args.max_grad_norm > 0:
                        # Unscale gradients before clipping
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    
                    # Optimizer step with scaler
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    model.zero_grad()
                    num_steps += 1
                
                # Update progress bar with current loss
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
                
                # Log to wandb
                wandb.log({"loss": loss.item()}, step=num_steps)
                
                if (step + 1) == len(train_dataloader) - 1 or (args.evaluation_steps > 0 and num_steps % args.evaluation_steps == 0 and step % args.gradient_accumulation_steps == 0):
                    dev_score, dev_output = evaluate(args, model, dev_features, tag="dev")
                    wandb.log(dev_output, step=num_steps)
                    print(dev_output)
                    if dev_score > best_score:
                        best_score = dev_score
                        pred = report(args, model, test_features)
                        # Save to Kaggle working directory
                        result_path = "/kaggle/working/result.json"
                        with open(result_path, "w") as fh:
                            json.dump(pred, fh)
                        if args.save_path != "":
                            # Update save path to Kaggle working directory
                            save_path = os.path.join("/kaggle/working", os.path.basename(args.save_path))
                            torch.save(model.state_dict(), save_path)
            
            pbar.close()
            
            print(f"[Epoch {epoch+1}] Completed")
        
        return num_steps

    # ===== Differential Learning Rate Strategy =====
    # Three parameter groups with different learning rates:
    # 1. BERT pretrained: Small LR (3e-5) - protect pretrained knowledge
    # 2. Reasoner module (new): Large LR (3e-4) - accelerate convergence
    # 3. Classifiers (new): Medium LR (1e-4) - standard new layer training
    
    # Define parameter name patterns for each group
    reasoner_modules = ["reasoner"]  # SemanticReasoner (new IER architecture)
    classifier_modules = ["bilinear", "bridge_classifier", "context_proj"]  # Classification heads
    
    # Separate parameters into groups
    bert_params = []
    reasoner_params = []
    classifier_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(module in name for module in reasoner_modules):
            reasoner_params.append(param)
        elif any(module in name for module in classifier_modules):
            classifier_params.append(param)
        else:
            bert_params.append(param)
    
    # Print parameter group sizes for verification
    print(f"Parameter groups: BERT={len(bert_params)}, Reasoner={len(reasoner_params)}, "
          f"Classifier={len(classifier_params)}")
    
    optimizer_grouped_parameters = [
        {"params": bert_params, "lr": 3e-5},           # BERT: small LR
        {"params": reasoner_params, "lr": 3e-4},       # Reasoner: large LR
        {"params": classifier_params, "lr": 1e-4},    # Classifier: medium LR
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    num_steps = 0
    set_seed(args)
    model.zero_grad()
    finetune(train_features, optimizer, args.num_train_epochs, num_steps)


def evaluate(args, model, features, tag="dev"):

    dataloader = DataLoader(features, batch_size=args.test_batch_size, shuffle=False, collate_fn=collate_fn, drop_last=False)
    preds = []
    
    # Add tqdm progress bar for evaluation
    pbar = tqdm(dataloader, desc=f"Evaluating {tag}", position=0, leave=False)
    
    for batch in pbar:
        model.eval()

        inputs = {'input_ids': batch[0].to(args.device),
                  'attention_mask': batch[1].to(args.device),
                  'entity_pos': batch[3],
                  'hts': batch[4],
                  }

        with torch.no_grad():
            pred, *_ = model(**inputs)
            pred = pred.cpu().numpy()
            pred[np.isnan(pred)] = 0
            preds.append(pred)
    
    pbar.close()

    preds = np.concatenate(preds, axis=0).astype(np.float32)
    ans = to_official(preds, features)
    if len(ans) > 0:
        best_f1, _, best_f1_ign, _ = official_evaluate(ans, args.data_dir)
    output = {
        tag + "_F1": best_f1 * 100,
        tag + "_F1_ign": best_f1_ign * 100,
    }
    return best_f1, output


def report(args, model, features):

    dataloader = DataLoader(features, batch_size=args.test_batch_size, shuffle=False, collate_fn=collate_fn, drop_last=False)
    preds = []
    
    # Add tqdm progress bar for report generation
    pbar = tqdm(dataloader, desc="Generating report", position=0, leave=False)
    
    for batch in pbar:
        model.eval()

        inputs = {'input_ids': batch[0].to(args.device),
                  'attention_mask': batch[1].to(args.device),
                  'entity_pos': batch[3],
                  'hts': batch[4],
                  }

        with torch.no_grad():
            pred, *_ = model(**inputs)
            pred = pred.cpu().numpy()
            pred[np.isnan(pred)] = 0
            preds.append(pred)
    
    pbar.close()

    preds = np.concatenate(preds, axis=0).astype(np.float32)
    preds = to_official(preds, features)
    return preds


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default="/kaggle/input/mydataset/dataset/docred", type=str)
    parser.add_argument("--transformer_type", default="bert", type=str)
    parser.add_argument("--model_name_or_path", default="/kaggle/input/mydataset/model/bert-base-cased", type=str)

    parser.add_argument("--train_file", default="train.json", type=str)
    parser.add_argument("--dev_file", default="dev.json", type=str)
    parser.add_argument("--test_file", default="test.json", type=str)
    parser.add_argument("--save_path", default="", type=str)
    parser.add_argument("--load_path", default="", type=str)

    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--max_seq_length", default=512, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")

    parser.add_argument("--train_batch_size", default=1, type=int,
                        help="Batch size for training.")
    parser.add_argument("--test_batch_size", default=1, type=int,
                        help="Batch size for testing.")
    parser.add_argument("--gradient_accumulation_steps", default=4, type=int,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--num_labels", default=4, type=int,
                        help="Max number of labels in prediction.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--warmup_ratio", default=0.06, type=float,
                        help="Warm up ratio for Adam.")
    parser.add_argument("--num_train_epochs", default=30.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--evaluation_steps", default=-1, type=int,
                        help="Number of training steps between evaluations.")
    parser.add_argument("--seed", type=int, default=66,
                        help="random seed for initialization")
    parser.add_argument("--num_class", type=int, default=97,
                        help="Number of relation types in dataset.")
    
    # ===== IER Architecture Arguments =====
    parser.add_argument("--num_reasoning_layers", default=2, type=int,
                        help="Number of stacked reasoning layers in SemanticReasoner.")

    args, unknown = parser.parse_known_args()  # Use parse_known_args for Jupyter/Kaggle compatibility
    
    # Print architecture configuration
    print("=" * 60)
    print("IER (Semantic Reasoner) Architecture Configuration:")
    print(f"  num_reasoning_layers: {args.num_reasoning_layers}")
    print(f"  num_slots: 3 (fixed: Head/Tail/Bridge)")
    print("=" * 60)
    
    wandb.init(project="DocRED")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=args.num_class,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
    )

    read = read_docred

    train_file = os.path.join(args.data_dir, args.train_file)
    dev_file = os.path.join(args.data_dir, args.dev_file)
    test_file = os.path.join(args.data_dir, args.test_file)
    train_features = read(train_file, tokenizer, max_seq_length=args.max_seq_length)
    dev_features = read(dev_file, tokenizer, max_seq_length=args.max_seq_length)
    test_features = read(test_file, tokenizer, max_seq_length=args.max_seq_length)

    # import random
    # random.seed(42) 
    
    # num_dev_samples = len(dev_features)
    # num_leak = int(num_dev_samples * 0.05) 
    
    # leak_indices = random.sample(range(num_dev_samples), num_leak)
    
    # dev_sample = [dev_features[i] for i in leak_indices]
    
    # train_features = train_features + dev_sample


    model = AutoModel.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
    )

    config.cls_token_id = tokenizer.cls_token_id
    config.sep_token_id = tokenizer.sep_token_id
    config.transformer_type = args.transformer_type

    set_seed(args)
    
    # ===== Create Model with IER Architecture =====
    model = DocREModel(
        config, 
        model, 
        num_labels=args.num_labels,
        num_reasoning_layers=args.num_reasoning_layers,
    )
    model.to(0)
    
    # Print model parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    if args.load_path == "":  # Training
        train(args, model, train_features, dev_features, test_features)
    else:  # Testing
        model.load_state_dict(torch.load(args.load_path))
        dev_score, dev_output = evaluate(args, model, dev_features, tag="dev")
        print(dev_output)
        pred = report(args, model, test_features)
        # Save to Kaggle working directory
        os.makedirs('/kaggle/working', exist_ok=True)
        result_path = "/kaggle/working/result.json"
        with open(result_path, "w") as fh:
            json.dump(pred, fh)


if __name__ == "__main__":
    main()
