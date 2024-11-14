import os
import sys
# package_path = os.path.join(os.getcwd(), 'pykan')
# sys.path.append(package_path)
from kan import *
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import timeit
import math
from kan.utils import SYMBOLIC_LIB
from kan.utils import ex_round

#print(os.getcwd())

class Wind_KAN(MultKAN):
    ################################################################
    ######################## INITIALIZATION ########################
    ################################################################

    #outputs: train_input, train_label, validation_input, validation_label, test_input, test_label
    def __init__(self, width=[4,4,1], grid=3, k=3, seed=42, *args, **kwargs):
        super().__init__(width=width,grid=grid, k=k, seed=seed, *args, **kwargs)
        # predefined: width=[4,4,1], grid=3, k=3, seed=42, 
        self.winddata_path = 'No path yet'
        self.save_act = False
        self.first_plot = False
        self.batch_size = -1
        self.epochs = 100
        # history elements: 
        # fit_iteration, prune_iteration, grid, train_time, mse, mae, r2, train_loss, test_loss
        self.fit_iteration = 0
        self.prune_iteration = 0
        self.grid = grid
        self.train_time = 0
        self.mse = 0
        self.mae = 0
        self.r2 = 0
        self.train_loss = 0
        self.test_loss = 0
        self.stat_history = []


    def get_winddataset(self, winddata_path, train_val_split = 0.1, train_test_split = 0.1, out=False):
        self.winddata_path = winddata_path
        data = pd.read_csv(self.winddata_path)
        Y = data.iloc[:, 2].values.astype(np.float64)
        X = data.iloc[:, 3:7].values.astype(np.float64)

        scaler = MinMaxScaler()
        X= scaler.fit_transform(X)
        split1 = int((1-train_test_split) * train_val_split * len(X))
        split2 = int((1-train_test_split) * len(X))

        # for comparison with other models
        split1 = int(0.7 * len(X))
        split2 = int(0.85 * len(X))
        ##########################################

        X_train, y_train = X[:split1], Y[:split1]
        X_val, y_val = X[split1:split2], Y[split1:split2]
        X_test, y_test = X[split2:], Y[split2:]

        ndarray_data = [X_train, y_train, X_val, y_val, X_test, y_test]
        tensor_data_list = [torch.from_numpy(x) for x in ndarray_data]

        tensor_data_list[1] = tensor_data_list[1].unsqueeze(1)  # y_train
        tensor_data_list[3] = tensor_data_list[3].unsqueeze(1)  # y_val
        tensor_data_list[5] = tensor_data_list[5].unsqueeze(1)  # y_test

        # The KAN Class trains on train/validation called train_input/test_input
        self.dataset = { 'train_input': tensor_data_list[0].float(),
                    'train_label': tensor_data_list[1].float(),
                    'test_input': tensor_data_list[2].float(),
                    'test_label': tensor_data_list[3].float(),
                    'true_test_input': tensor_data_list[4].float(),
                    'true_test_label': tensor_data_list[5].float()  }
    
    ################################################################
    ###################### FITTING/TUNING HP #######################
    ################################################################
    
    def update_history(self):
        predictions = self(self.dataset['true_test_input'])
        if predictions.isnan().any():
            raise ValueError(f'nan detected {error_place} iteration {self.fit_iteration}')
        current_values = [self.fit_iteration, self.prune_iteration, self.grid, self.train_time,\
                            self.mse, self.mae, self.r2, self.train_loss, self.test_loss]   
        self.stat_history.append(current_values)
        return

    def update_metrics(self):
        predictions = (self(self.dataset['true_test_input'])).detach().numpy()
        y_test = (self.dataset['true_test_label'])
        self.mse = mean_squared_error(y_test, predictions, squared=False)
        self.mae = mean_absolute_error(y_test, predictions)
        self.r2 = r2_score(y_test, predictions)
            
    def fit_model(self, epochs=20, batch_size=-1, **kwargs):
        train_data_length = self.dataset['train_input'].shape[0]
        if batch_size != -1 or batch_size > train_data_length:
            steps = train_data_length*epochs/batch_size
        else:
            steps = epochs
        start = timeit.default_timer()
        print('steps: ', steps); print('batchsize: ', batch_size)
        results = super().fit(dataset=self.dataset, steps=math.ceil(steps), batch=batch_size, **kwargs)
        print('fitted')
        stop = timeit.default_timer()
        train_time = stop-start
        last_train_loss = results['train_loss'][-1]
        last_test_loss = results['test_loss'][-1]
        self.train_time = train_time
        self.fit_iteration += 1
        self.train_loss = last_train_loss
        self.test_loss = last_test_loss
        self.update_metrics()
        self.update_history()
        return results
    
    def prune_model(self, iterations=1, epochs=100, batch_size=-1):
        self.prune_model()
        self.prune_iteration += 1
        results = self.fit_model()
        return results
    
    def refine_model(self, new_grid, iterations=1, epochs=100, batch_size=-1):
        self.refine(new_grid)
        self.grid = new_grid
        results = self.fit_model()
        return results
    
    def plot_model(self, **kwargs):
        if self.first_plot == False:
            self(self.dataset['train_input'])
        super().plot()
    
    def print_history(self, iterations='all'):
        hist = self.stat_history
        if iterations == 'all':
            iterations = len(hist)
        elif iterations > len(hist):
            iterations = len(hist)
        for i in reversed(range(iterations)):
            current_m = hist[-i-1]
            print('fit iter:', current_m[-7], '  prune iter:', current_m[-6], '  grid:', current_m[-5],\
                 '  train_time:', round(current_m[-4],2),\
                 '  mse:', round(current_m[-3],3), '  mae:', round(current_m[-2],3), '  r2:', round(current_m[-1],3))


    def optimize_prune_refine(self, grid_increase=5,\
                        min_prune_improvement, min_fit_improvement, min_refine_improvement,\
                        max_prune_iters=10, max_refine_iters=10, max_fit_iters=10):
    # opt_history: all losses including during fitting
    # self.stat_history: all changes after calling fit

    opt_history = {'train_losses': [], 'test_losses': [], 'prune_location': [], 'refine_location': []}
    # initiating gammas
    fit_improvement = 1000000
    prune_improvement = 1000000
    refine_improvement = 1000000
    # iteration counters
    refine_iter = 0
    while refine_improvement >= min_refine_improvement and refine_iter <= max_refine_iters:
        prune_iter = 0
        while prune_improvement >= min_prune_improvement and prune_iter <= max_prune_iters:
            fit_iter = 0
            # fitting loop
            while fit_improvement >= min_fit_improvement and fit_iter <= max_fit_iters:
                results = self.fit_model()
                fit_loss_0 = opt_history['train_losses'][-1]
                opt_history['train_losses'].extend(results['train_loss'])
                opt_history['test_losses'].extend(results['test_loss'])
                zeroes = [0 for x in range(len(results['train_loss']))]
                opt_history['prune_location'].extend(zeroes)
                opt_history['refine_location'].extend(zeroes)
                fit_improvement = (fit_loss_0 - results['train_loss'][-1])/fit_loss_0
                fit_iter += 1
            # prune after fitted
            results = self.prune_model()
            prune_loss_0 = results['train_loss'][-1]
            opt_history['train_losses'].extend(results['train_loss'])
            opt_history['test_losses'].extend(results['test_loss'])
            zeroes = [0 for x in range(len(results['train_loss']))]
            prune_indices = [1 if x==0 else 0 for x in range(len(results['train_loss']))]
            opt_history['prune_location'].extend(prune_indices)
            opt_history['refine_location'].extend(zeroes)
            prune_iter += 1
            prune_improvement = (prune_loss_0 - opt_history['train_losses'][-1])/prune_loss_0
        # refine (increase grid) after pruned and fitted
        grid += grid_increase
        results = self.refine_model(new_grid=grid)
        refine_loss_0 = results['train_losses']
        opt_history['train_losses'].extend(results['train_loss'])
        opt_history['test_losses'].extend(results['test_loss'])
        zeroes = [0 for x in range(len(results['train_loss']))]
        refine_indices = [1 if x==0 else 0 for x in range(len(results['train_loss']))]
        opt_history['prune_location'].extend(zeroes)
        opt_history['refine_location'].extend(refine_indices)
        refine_iter += 1
        refine_gamma = (refine_loss_0 - opt_history['train_losses'][-1])/refine_loss_0

    ################################################################
    ############################ FORMULA ###########################
    ################################################################

    def get_formula(self, ignore_forms=['0']):

        my_symbolics = SYMBOLIC_LIB.copy()
        for formula in ignore_forms:
            my_symbolics.pop(formula, None)
        self.auto_symbolic(lib=my_symbolics)
        print(ex_round(self.symbolic_formula()[0][0],2))
