import os
import yaml

import torch
import torch.nn as nn

from models.Transformer import TransformerNet

if __name__ == "__main__":
    config_path = './src/config/default.yaml'
    config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)

    # MODEL PATH
    checkpoint_dir = config['training']['checkpoint_dir']

    # MODEL PARAMETERS
    vocab_size = config['model']['vocab_size']
    d_model = config['model']['d_model']
    nhead = config['model']['n_head']
    num_layers = config['model']['num_layers']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("\nCreating new Transformer seq-to-seq model")
    model = TransformerNet(vocab_size, d_model, nhead, num_layers).to(device)

    print("\nLoading trained model state")
    checkpoint_path = os.path.join(checkpoint_dir, "transformer_seq_model.pt")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))

    model.eval()

    print("\nModel loaded successfully")

    # src = torch.tensor([[1, 4,5,6,7,6,5,4, 2]], 
    #                dtype=torch.int64).to(device)
    # # should predict 5,6,7,8,7,6,5
    # tgt_in = torch.tensor([[1]], dtype=torch.int64).to(device)
    # t_mask = nn.Transformer.generate_square_subsequent_mask(tgt_in.size(1))

    # with torch.no_grad():
    #   prediction = model(src, tgt_in, t_mask)
    # print("\nInput: ", src)
    # print("\npredected pseudo-probs: ", prediction)
    # pred_tokens = torch.argmax(prediction, dim=2)
    # print("\nfirst predicted output token: ", pred_tokens)

    # print("\nEnd Transformer seq-to-seq demo")