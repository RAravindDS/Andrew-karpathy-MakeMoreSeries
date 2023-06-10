from charLLM import NPLM


text_path = "names.txt"
model_parameters = {
    "block_size" :3, 
    "train_size" :0.8, 
    'epochs' :10000, 
    'batch_size' :32, 
    'hidden_layer' :100, 
    'embedding_dimension' :50,
    'learning_rate' :0.1 
}

obj = NPLM(text_path, model_parameters)
obj.train_model()
obj.sampling()