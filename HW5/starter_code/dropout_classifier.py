import time
import numpy as np

import torch as t
from torch import nn
from torch import optim
from torch.nn import functional as F
import matplotlib.pyplot as plt
import pickle

from data import prepare_cifar_data, get_dataloader

#######################################################################################
# AUXILIARY CLASSES
#######################################################################################

def divide_no_nan(a, b):
    """
    a/b where the resulted NaN or Inf are replaced by 0.
    """
    result = a / b
    result[result != result] = .0
    result[result == np.inf] = .0
    return result


class WeightNormConstrainer(object):

    def __init__(self, norm):
        self.norm = norm

    def __call__(self, module):
        if hasattr(module, 'weight'):
            w = module.weight.data
            wn = t.norm(w, p=2, dim=1).detach()
            ind = t.gt(wn, self.norm)
            div = (divide_no_nan(wn, self.norm) * ind) + (1 * t.logical_not(ind))
            div = div.unsqueeze_(1)
            w.div_(div)


class _DropoutClassifier(nn.Module):
    def __init__(self, layers):
        super(_DropoutClassifier, self).__init__()
        self.input_layer = nn.Sequential(*layers['input_layer'])
        self.hidden_layers = nn.Sequential(*layers['hidden_layers'])
        self.output_layer = nn.Sequential(*layers['output_layer'])

    def layer_features(self, x):
        input_layer = self.input_layer(x)
        hidden = self.hidden_layers(input_layer)
        logits = self.output_layer(hidden)
        layer_features = {'input_layer': input_layer.data.cpu().numpy(),
                          'hidden': hidden.data.cpu().numpy(),
                          'logits': logits.data.cpu().numpy()}
        return layer_features

    def forward(self, x):
        input_layer = self.input_layer(x)
        hidden = self.hidden_layers(input_layer)
        logits = self.output_layer(hidden)
        return logits

#######################################################################################
# DROPOUT CLASSIFIER
#######################################################################################


params = {'model': 'dclf',
          'display_step': 1_000, # Do not decrease this, can make the training extremelly slow.
          'batch_size': 256,
          'iterations': 100_000,
          'initial_lr': 0.1,
          'lr_decay': 0.5,
          'adjust_lr_step': 10_000,
          'initial_momentum': 0.5,
          'final_momentum': 0.95,
          'momentum_change_steps': 20_000,
          'adjust_momentum_step': 2_000,
          'apply_weight_norm': True,
          'weight_norm': 3.5,
          'adjust_norm_step': 5_000,
          'in_features': 3072,
          'out_features': 10,
          'input_activation': 'logistic',
          'input_dropout_prob': 0.5,
          'output_l2_decay': 0.03,
          'hidden_layers': [1024, 1024, 2048],
          'hidden_dropout_prob': 0.5,  #TODO: change this parameter
          'hidden_activation': 'relu',
          'random_seed': 42}

class DropoutClassifier(object):
    def __init__(self, params):
        
        super().__init__()
        self.params = params
        self.device = t.device('cuda' if t.cuda.is_available() else 'cpu')
        self.activations = {'logistic': nn.Sigmoid(), 'relu': nn.ReLU()}
        
        # Instantiate model
        t.manual_seed(self.params['random_seed'])
        np.random.seed(self.params['random_seed'])
        layers = self._initialize_network()
        self.model = _DropoutClassifier(layers).to(self.device)

    def _initialize_network(self):
    	# TODO: complete this function
        # Input
        input_layer = [
            nn.Linear(params['in_features'], params['hidden_layers'][0]),
            self.activations[self.params['input_activation']],
            nn.Dropout(self.params['input_dropout_prob'])
        ]  #TODO: Write the input_layer using torch modules - Done
        # Hidden
        hidden_layers = []
        in_features = self.params['hidden_layers'][0]
        for layer, out_features in enumerate(self.params['hidden_layers'][1:]):
            layer = [
                nn.Linear(in_features, out_features),
                self.activations[self.params['hidden_activation']],
                nn.Dropout(self.params['hidden_dropout_prob'])
            ]   # TODO: Write the hidden_layers using torch modules - Done
            in_features = out_features   # This helps to define input and outputs of layers
            hidden_layers += layer
        
        # Output
        output_layer = [
            nn.Linear(in_features, self.params['out_features'])
        ]  #TODO: Write the output_layer using torch modules - Done
        
        layers = {'input_layer': input_layer,
                  'hidden_layers': hidden_layers,
                  'output_layer': output_layer}
        return layers
    
    def evaluate_accuracy(self, dataloader):
        self.model.eval()
        with t.no_grad():
            n_correct = 0
            n_samples = 0
            for batch in iter(dataloader):
                batch_x = t.flatten(batch[0].to(self.device), start_dim=1)
                batch_y = batch[1].to(self.device)
                
                logits = self.model(batch_x)
                y_hat = t.argmax(logits, dim=1)
                correct = t.sum(y_hat==batch_y)

                n_correct += correct.data.cpu().numpy()
                n_samples += len(batch_x)
        accuracy = (n_correct/n_samples) * 100
        self.model.train()
        return accuracy

    def evaluate_cross_entropy(self, dataloader):
        self.model.eval()
        loss_func = nn.CrossEntropyLoss()        
        with t.no_grad():
            n_samples = 0
            total_loss = 0
            for batch in iter(dataloader):
                # TODO: Write the code to measure the cross_entropy 
                # (Hint, look at the evaluate_accuracy method)
                # Be careful with the eval and train modes of the model - Done
                batch_x = t.flatten(batch[0].to(self.device), start_dim=1)
                batch_y = batch[1].to(self.device)
                
                logits = self.model(batch_x)
                loss = loss_func(logits, batch_y) 
                
                total_loss += loss.item() * len(batch_x)
                n_samples += len(batch_x)
        
        cross_entropy = total_loss/n_samples
        self.model.train()
        return cross_entropy
    
    def adjust_lr(self, optimizer, lr_decay):
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * lr_decay
        
    def adjust_momentum(self, optimizer, step, momentum_change_steps,
                        initial_momentum, final_momentum):
        mcs = momentum_change_steps
        s = min(step, mcs)
        momentum = (initial_momentum) * ((mcs-s)/mcs) + \
                   (final_momentum) * (s/mcs)
        for param_group in optimizer.param_groups:
            param_group['momentum'] = momentum

    def get_filters(self):
        filters = self.model.input_layer[0].weight.data.cpu().numpy()
        filters = filters.reshape(self.params['hidden_layers'][0], 32, 32, 3)
        return filters
    
    def save_weights(self, path):
        t.save(self.model.state_dict(), path)

    def load_weights(self, path):
        self.model.load_state_dict(t.load(path,
                                          map_location=t.device(self.device)))
        self.model.eval()

    def fit(self, insample_dataloader, outsample_dataloader):        
        # Instantiate optimization tools
        loss = nn.CrossEntropyLoss()        
        optimizer = optim.SGD([{'params': self.model.input_layer.parameters()},
                               {'params': self.model.hidden_layers.parameters()},
                               {'params': self.model.output_layer.parameters(),
                                'weight_decay': self.params['output_l2_decay']}],
                               lr=self.params['initial_lr'],
                               momentum=self.params['initial_momentum'])
        
        constrainer = WeightNormConstrainer(norm=self.params['weight_norm'])

        # Initialize counters and trajectories
        step = 0
        epoch = 0
        metric_trajectories = {'step':  [],
                               'epoch':  [],
                               'insample_accuracy': [],
                               'outsample_accuracy': [],
                               'insample_cross_entropy': [],
                               'outsample_cross_entropy': []
                               }

        print('\n'+'='*36+' Fitting DCLF '+'='*36)
        while step <= self.params['iterations']:

            # Train
            epoch += 1
            self.model.train()
            for batch in iter(insample_dataloader):
                step+=1
                if step > self.params['iterations']:
                    continue
                                
                batch_x = t.flatten(batch[0].to(self.device), start_dim=1)
                batch_y = batch[1].to(self.device)
                
                optimizer.zero_grad()

                # TODO: make predictions, compute the cross entropy loss and perform backward propagation - Done
                logits = self.model(batch_x)
                cross_entropy = loss(logits, batch_y) 
                cross_entropy.backward()
                

                t.nn.utils.clip_grad_norm_(self.model.parameters(), 20)
                optimizer.step()

                # Evaluate metrics
                if (step % params['display_step'] == 0):
                    in_cross_entropy   = self.evaluate_cross_entropy(insample_dataloader)
                    out_cross_entropy  = self.evaluate_cross_entropy(outsample_dataloader)
                    in_accuracy        = self.evaluate_accuracy(insample_dataloader)
                    out_accuracy       = self.evaluate_accuracy(outsample_dataloader)

                    print('Epoch:', '%d,' % epoch,
                          'Step:', '%d,' % step,
                          'In Loss: {:.7f},'.format(in_cross_entropy),
                          'Out Loss: {:.7f},'.format(out_cross_entropy),
                          'In Acc: {:03.3f},'.format(in_accuracy),
                          'Out Acc: {:03.3f}'.format(out_accuracy))
                    
                    metric_trajectories['insample_cross_entropy'].append(in_cross_entropy)
                    metric_trajectories['outsample_cross_entropy'].append(out_cross_entropy)
                    metric_trajectories['insample_accuracy'].append(in_accuracy)
                    metric_trajectories['outsample_accuracy'].append(out_accuracy)
                    metric_trajectories['step'].append(step)
                    metric_trajectories['epoch'].append(epoch)

                # Update optimizer learning rate
                if step % self.params['adjust_lr_step'] == 0:
                    self.adjust_lr(optimizer=optimizer, lr_decay=self.params['lr_decay'])
                
                # Update optimizer momentum
                if step % self.params['adjust_momentum_step'] == 0 and \
                    step < self.params['momentum_change_steps']:
                    self.adjust_momentum(optimizer=optimizer, step=step,
                                         momentum_change_steps=self.params['momentum_change_steps'],
                                         initial_momentum=self.params['initial_momentum'],
                                         final_momentum=self.params['final_momentum'])
              
                # Constraint max_norm of weights
                if self.params['apply_weight_norm'] and \
                  (step % self.params['adjust_norm_step'] == 0):
                    self.model.apply(constrainer)

        # Store trajectories
        print('\n'+'='*35+' Finished Train '+'='*35)
        self.trajectories = metric_trajectories

def main():
    data = prepare_cifar_data(n_train=10_000, n_val=10_000, n_test=10_000)

    X_train, y_train, X_val, y_val, _, _ = data.values()

    insample_dataloader = get_dataloader(X_train, y_train, batch_size=params['batch_size'])
    outsample_dataloader = get_dataloader(X_val, y_val, batch_size=params['batch_size'])

    DROPOUT = 0.0 # [0.0, 0.20.5, 0.8]
    params['hidden_dropout_prob'] = DROPOUT
    clf = DropoutClassifier(params)
    clf.fit(insample_dataloader, outsample_dataloader)

    # To avoid unnecesary pain, we recommend you to save your classifiers
    clf.save_weights(f'./model_dropout_{DROPOUT}.pth')
    clf.load_weights(f'./model_dropout_{DROPOUT}.pth')

    # TODO: save trajectories and/or plot trajectories - Done
    with open(f'./results_dropout_{DROPOUT}.pkl', 'wb') as file:
        pickle.dump(clf.trajectories, file)
        
    plt.figure(0)
    plt.plot(clf.trajectories['step'], clf.trajectories['insample_cross_entropy'], linestyle='-.', color='b', label='Train Loss')
    plt.plot(clf.trajectories['step'], clf.trajectories['outsample_cross_entropy'], linestyle='-', color='orange', label='Validation Loss')
    plt.title('Loss vs Gradient Steps')
    plt.ylabel('Loss')
    plt.xlabel('Gradient Step')
    plt.legend(loc='best')
    plt.savefig(f'./loss_dropout_{DROPOUT}.png')
    plt.show()
    
    plt.figure(0)
    plt.plot(clf.trajectories['step'], clf.trajectories['insample_accuracy'], linestyle='-.', color='b', label='Train Accuracy')
    plt.plot(clf.trajectories['step'], clf.trajectories['outsample_accuracy'], linestyle='-', color='orange', label='Validation Accuracy')
    plt.title('Accuracy vs Gradient Steps')
    plt.ylabel('Accuracy')
    plt.xlabel('Gradient Step')
    plt.legend(loc='best')
    plt.savefig(f'./acc_dropout_{DROPOUT}.png')
    plt.show()

if __name__ == '__main__':
    main()
