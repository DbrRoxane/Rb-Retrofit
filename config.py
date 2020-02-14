pretrained_embs =  ['./data/crawl-300d-2M.vec'] # ['./data/wgc_g_zip.txt'] ['./data/wgc_w_zip.txt'] #,
graphs = ['./data/cn_relations_orig.txt'] #['./data/cnet_graph_score.txt', './data/wordnet_graph_score.txt', './data/ppdb_graph_score.txt']
entities_file = './data/vocab_Ins_wgc.txt'
relations_file = ['./data/cnetrellist.txt', './data/wordnetrellist.txt', './data/ppdbrellist.txt']

params_dataset = {'batch_size': 1024,
                  'shuffle': True,
                  'num_workers': 0}

train_prop, valid_prop = 0.7, 0.15


params_network = {'embedding_dim': 300,
                  'number_models': len(pretrained_embs)}

optimizer_other_params = {'lr': 1e-4,
                    'weight_decay': 1e-1}

optimizer_embeddings_params = {'lr': 1e-4,
                    'weight_decay': 1e-1}

device = 0
epoch = 10
dir_experiment = './experiment/experiment_retrofit_1202_finetune6'
embedding_dim = 300
nb_false = 2
