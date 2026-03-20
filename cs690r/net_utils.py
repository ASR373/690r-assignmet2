import gc
import copy
import numpy as np

import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F

from .accNet import Resnet

# For reproducibility
torch.manual_seed(0)
# torch.set_deterministic(True)  
# torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def get_torch_device(verbose=True):
    if torch.cuda.is_available():
        device = torch.device('cuda', 0)
        if verbose:
            print(f'CUDA VERSION: {torch.backends.cudnn.version()}')
        return device
    else:
        return 'cpu'
    
    
def train_model(device, model, X_tr, X_val, y_tr, y_val, loss_func,
    optimizer, epochs=20, batch_size=int(2**10)):
    
    # Deploy model on the device
    model = model.to(device=device)
    
    # Save loss history during training
    train_loss_history = np.zeros(epochs)
    val_loss_history = np.zeros(epochs)

    # Train
    for e in range(epochs):
        # Model training
        model.train()
        permutation = torch.randperm(X_tr.size()[0])
        train_losses = []
        for i in range(0, X_tr.size()[0], batch_size):
            indices = permutation[i:i+batch_size]
            batch_x, batch_y = X_tr[indices, :, :], y_tr[indices]
            optimizer.zero_grad()
            scores = model(batch_x)
            loss = loss_func(scores, batch_y)
            train_losses.append(loss.item())
            loss.backward()
            optimizer.step()
            del indices, batch_x, batch_y, scores, loss
        train_loss_history[e] = np.mean(train_losses)
        del train_losses

        # Model validation
        with torch.no_grad():
            model.eval()
            scores = model(X_val)
            loss = loss_func(scores, y_val)
            val_loss_history[e] = loss.item()
            del scores, loss
        gc.collect()
    return train_loss_history, val_loss_history


def eval_model(device, model, X, batch_size=int(2**8)):
    n_samples = X.shape[0]
    with torch.no_grad():
        model = model.to(device=device)
        model.eval()
        if batch_size > n_samples:
            return model(X)
        else:
            scores = []
            for i in range(0, X.size()[0], batch_size):
                batch_x = X[i:i+batch_size]
                scores[i:i+batch_size] = model(batch_x)
                del batch_x
            gc.collect()
            return torch.vstack(scores)


def determine_folds(subj, n_val=1, seed=0):
    train_masks = []
    val_masks = []
    test_masks = []
    
    np.random.seed(seed)
    unique_subject_random = np.unique(subj)
    np.random.shuffle(unique_subject_random)
    
    for i, s in enumerate(unique_subject_random):
        test_mask = (subj == s)
        val_mask = np.zeros(test_mask.shape, dtype=bool)
        for j in range(n_val):
            val_mask |= (subj == unique_subject_random[(i+j+1)%len(unique_subject_random)])
        train_mask = ~(test_mask | val_mask)
        train_masks.append(train_mask)
        val_masks.append(val_mask)
        test_masks.append(test_mask)

    train_masks = np.array(train_masks, dtype=bool)
    val_masks = np.array(val_masks, dtype=bool)
    test_masks = np.array(test_masks, dtype=bool)
    
    return train_masks, val_masks, test_masks


def make_model():
    model = Resnet(
        output_size=5,
        resnet_version=1,
        epoch_len=5,
        is_eva=True
    )
    return model


def load_weights(weight_path, model, my_device):
    # only need to change weights name when
    # the model is trained in a distributed manner

    pretrained_dict = torch.load(weight_path, map_location=my_device)
    pretrained_dict_v2 = copy.deepcopy(
        pretrained_dict
    )  # v2 has the right para names

    # distributed pretraining can be inferred from the keys' module. prefix
    head = next(iter(pretrained_dict_v2)).split(".")[
        0
    ]  # get head of first key
    if head == "module":
        # remove module. prefix from dict keys
        pretrained_dict_v2 = {
            k.partition("module.")[2]: pretrained_dict_v2[k]
            for k in pretrained_dict_v2.keys()
        }

    if hasattr(model, "module"):
        model_dict = model.module.state_dict()
        multi_gpu_ft = True
    else:
        model_dict = model.state_dict()
        multi_gpu_ft = False

    # 1. filter out unnecessary keys such as the final linear layers
    #    we don't want linear layer weights either
    pretrained_dict = {
        k: v
        for k, v in pretrained_dict_v2.items()
        if k in model_dict and k.split(".")[0] != "classifier"
    }

    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)

    # 3. load the new state dict
    if multi_gpu_ft:
        model.module.load_state_dict(model_dict)
    else:
        model.load_state_dict(model_dict)
    # print("%d Weights loaded" % len(pretrained_dict))


def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find("BatchNorm1d") != -1:
        m.eval()