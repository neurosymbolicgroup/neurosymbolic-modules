from data import *
from nn import *
from params import *

def main():
    # Step 1: Build parameters
    model_params = DeepCoderModelParams(num_vals, val_embedding_dim, input_length, output_length, num_dsl_ops, dsl_embedding_dim, ngram_length, hidden_layer_dim)
    test_params = DeepCoderTestParams(save_path)

    # Step 2: Build neural net
    model = DeepCoderModel(model_params)

    # Step 3: Read dataset
    if input_length != output_length:
        raise Exception('Input and output lengths must be equal!')
    dataset = read_deep_coder_train_dataset(deep_coder_test_dataset_filename, deep_coder_funcs_filename, deep_coder_fo_filename, num_vals, input_length)
    dataset = np.array(dataset).T.tolist()
    dataset = [np.array(dataset[:5]), np.array(dataset[5:10]), np.array(dataset[10:15]), np.array(dataset[15]), np.array(dataset[16])]

    model.test(np.array(dataset[0]), np.array(dataset[1]), np.array(dataset[2]), np.array(dataset[3]), np.array(dataset[4]), test_params)
    
if __name__ == '__main__':
    main()
