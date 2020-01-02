pretrained_embs = ['./data/wgc_w_zip.txt', './data/wgc_g_zip.txt']
graphs = ['./data/cnet_graph_score.txt', './data/wordnet_graph_score.txt', './data/ppdb_graph_score.txt']
entities_file = './data/vocab_Ins_wgc.txt'
relations_file = ['./data/cnetrellist.txt', './data/wordnetrellist.txt', './data/ppdbrellist.txt']

params_dataset = {'batch_size': 1024,
                  'shuffle': True,
                  'num_workers': 0}

train_prop, valid_prop = 0.7, 0.15

params_network = {'embedding_dim': 300,
                  'number_models': len(pretrained_embs)}

params_optimizer = {'lr': 1e-2,
                    'momentum': 0.9,
                    'weight_decay': 1e-3}

device = 0
epoch = 10
dir_experiment = './experiment_5'
