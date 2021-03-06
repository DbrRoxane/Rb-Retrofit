pretrained_embs =  ['./data/crawl-300d-2M.vec'] # ['./data/wgc_g_zip.txt'] ['./data/wgc_w_zip.txt'] #,
synonyms_graph = ['./data/synonyms_triplet.txt'] #['./data/cn_relations_orig.txt'] ['./data/cnet_graph_score.txt', './data/wordnet_graph_score.txt', './data/ppdb_graph_score.txt']
antonyms_graph = ['./data/antonyms_triplet.txt']
entities_file = './data/vocab_Ins_wgc.txt'
relations_file = ['./data/cnetrellist.txt', './data/wordnetrellist.txt', './data/ppdbrellist.txt']
evaluations_file = ['./data/evaluation/MEN/MEN_dataset_natural_form_full',
                    'data/evaluation/SimLex-999/SimLex-999.txt',
                    './data/evaluation/SimVerb-3500.txt']

params_dataset = {'batch_size': 1024,
                  'shuffle': True,
                  'num_workers': 0}

train_prop, valid_prop = 0.7, 0.15


params_network = {'embedding_dim': 300,
                  'number_models': len(pretrained_embs)}

optimizer_other_params = {'lr': 3e-1}#,
                    #'weight_decay': 1e-1}

optimizer_embeddings_params = {'lr': 3e-1}#,
                   # 'weight_decay': 1e-1}

device = 0
epoch = 30
dir_experiment = './experiment/experiment_retrofit_2002_finetune15'
embedding_dim = 300
nb_false = 0
