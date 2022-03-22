import copy 
import torch
import math
from transformers import (MODEL_MAPPING, AutoConfig, AutoTokenizer, AutoModel)
from functools import partial
import os 
import random
import logging
from densephrases import MIPS
from transformers import (AdamW, get_linear_schedule_with_warmup)


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize model variables 
model_type='bert'
device = 'cpu'            #Change based on machine
pretrained_name_or_path = 'SpanBERT/spanbert-base-cased'
load_dir = 'princeton-nlp/densephrases-multi'


#Load Phrase Index 
dump_dir = '/scratch/nishantraj_umass_edu/dense_phrases/outputs/densephrases-multi_wiki-20181220/dump'
phrase_dir = 'phrase'
index_name = 'start/1048576_flat_OPQ96_small'
index_path = 'index.faiss'
idx2id_path = 'idx2id.hdf5'
cuda = False
ignore_logging= True

# Optimizer params
learning_rate  = 3e-5
adam_epsilon   = 1e-8

#QA FirstHop File
file = "/gypsum/scratch1/dagarwal/multihop_dense_retrieval/data/hotpot/hotpot_train_firsthop.json"
# Load QA
do_lower_case = True
shuffle = True

#Params for finetuning 
per_gpu_train_batch_size = 12
num_train_warm_epochs = 1 #Specify number of epochs for warmup 
gradient_accumulation_steps = 1
num_warmup_steps = 0 #Warmup for our first warming up model 


#Load config and a pretrained model from load_dir in hugging face
config = AutoConfig.from_pretrained(pretrained_name_or_path)
tokenizer = AutoTokenizer.from_pretrained(pretrained_name_or_path)
pretrained = None
hf_model_path = f"https://huggingface.co/{load_dir}/resolve/main/pytorch_model.bin"
load_class = partial(Encoder.from_pretrained, pretrained_model_name_or_path=hf_model_path)
# Takes significant time to load
model = load_class(config=config,tokenizer=tokenizer, transformer_cls=MODEL_MAPPING[config.__class__], pretrained=copy.deepcopy(pretrained) if pretrained is not None else None, cache_dir=None, lambda_kl=2, lambda_neg=4,lambda_flt=1)
model.to(device)
pretrained_encoder = model 
# Copy the encoder module 
target_encoder = copy.deepcopy(pretrained_encoder)

# Load MIPS from densephrase indexes
phrase_dump_dir = os.path.join(dump_dir, phrase_dir)
index_dir = os.path.join(dump_dir, index_name)
index_path = os.path.join(index_dir, index_path)
idx2id_path = os.path.join(index_dir, idx2id_path)
mips = MIPS(phrase_dump_dir=phrase_dump_dir, index_path=index_path, idx2id_path=idx2id_path, cuda= cuda, logging_level=logging.WARNING if ignore_logging else (logging.DEBUG if args.verbose_logging else logging.INFO))

# Function to check if training param is word embeddings type and if yes don't train it's weights
def is_train_param(name):
        if name.endswith(".embeddings.word_embeddings.weight"):
            logger.info(f'freezing {name}')
            return False
        return True

# Mostly draft and Truecasing not required (check later)
# Unchecked function below, check indentation (only this fn)
def load_qa_pairs(file, do_lowercase=do_lowercase, shuffle=shuffle):
  """
  Function to get question answer titles 
  """
  q_ids = []
  questions = []
  answers = []
  titles = []
  data = json.load(open(file))['data']

  #Unchecked Code below
  for data_idx, item in enumerate(data):
      q_id = item['id']
      # ['START_ENT'] required by SPANBERT ?
      question = '[START_ENT] '+ item['question'] + '[END_ENT]'
      answer  = item['answers']
      # Ask Dhruv to get titles in data preparation
      title = item.get('titles', [''])
      if len(answer) == 0:
              continue
      q_ids.append(q_id)
      questions.append(question)
      answers.append(answer)
      titles.append(title)

  questions = [query[:-1] if query.endswith('?') else query for query in questions]
  if do_lower_case:
    logger.info(f'Lowercasing queries')
    questions = [query.lower() for query in questions]

  if shuffle:
    qa_pairs = list(zip(q_ids, questions, answers, titles))
    random.shuffle(qa_pairs)
    q_ids, questions, answers, titles = zip(*qa_pairs)
    logger.info(f'Shuffling QA pairs')
  
  return q_ids, questions, answers, titles

no_decay = ["bias", "LayerNorm.weight"]
# Assign decay for bias and layernorm weights by grouping optimization params
optimizer_grouped_parameters = [{"params": [p for n, p in target_encoder.named_parameters() if not any(nd in n for nd in no_decay) and is_train_param(n)],"weight_decay": 0.01,}, {"params": [p for n, p in target_encoder.named_parameters() if any(nd in n for nd in no_decay) and is_train_param(n)], "weight_decay": 0.0}]
optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)

# Get steps for the first epoch 
step_per_epoch = math.ceil(len(load_qa_pairs(file, do_lowercase, shuffle)[1]) / per_gpu_train_batch_size)

# Get steps where update is taking place accounting for grad accumulation
t_total = int(step_per_epoch // args.gradient_accumulation_steps * args.num_train_epochs)
logger.info(f"Train for {t_total} iterations")

# Scheduler instantiation
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps , num_training_steps=t_total)


  

  


