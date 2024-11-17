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
        self.save_act = True
        self.first_plot = False
        self.batch_size = -1
        self.epochs = 100
        self.train_predictions = 0
        # history elements: 
        # Wind_fit_iter, Winde_prune_iter, Wind_grid, train_time, mse, mae, r2, Wind_train_loss, Wind_test_loss
        self.Wind_fit_iter = [0, []]
        self.Winde_prune_iter = [0, []]
        self.Wind_grid = [grid, []]
        self.train_time = [0, []]
        self.mse = [0, []]
        self.mae = [0, []]
        self.r2 = [0, []]
        self.Wind_train_loss = [0, []]
        self.Wind_test_loss = [0, []]


    def get_winddataset(self, winddata_path, set_float=True, \
                        train_val_split = 0.1, train_test_split = 0.1, out=False):
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

        # The KAN Class trains on train/validation called train_input/test_input.

        # There is a really weird inconsistency with .float32 being the explicit datatype depending on whether
        # the tensors are set to it. We need it to be non specific
        # the output when printing should be:
        # tensor([...]) NOT tensor([...], dtype=torch.float64)
        # change set_float if this is not the case

        if set_float == True:
            # The KAN Class trains on train/validation called train_input/test_input
            self.dataset = { 'train_input': tensor_data_list[0].float(),
                        'train_label': tensor_data_list[1].float(),
                        'test_input': tensor_data_list[2].float(),
                        'test_label': tensor_data_list[3].float(),
                        'true_test_input': tensor_data_list[4].float(),
                        'true_test_label': tensor_data_list[5].float()  }
        elif set_float == False:
            self.dataset = { 'train_input': tensor_data_list[0],
                        'train_label': tensor_data_list[1],
                        'test_input': tensor_data_list[2],
                        'test_label': tensor_data_list[3],
                        'true_test_input': tensor_data_list[4],
                        'true_test_label': tensor_data_list[5]  }
    
    ################################################################
    ###################### FITTING #######################
    ################################################################
            
    def fit_model(self, epochs=20, batch_size=-1, track_time=False, **kwargs):
        
        train_data_length = self.dataset['train_input'].shape[0]
        if batch_size != -1 or batch_size > train_data_length:
            steps = train_data_length*epochs/batch_size
        else:
            steps = epochs

        # the actual fitting
        start = timeit.default_timer()
        results = self.fit(dataset=self.dataset, steps=math.ceil(steps), batch=batch_size, **kwargs)
        stop = timeit.default_timer()
        train_time = stop-start
        last_train_loss = [float(x) for x in results['train_loss']]
        last_test_loss = [float(x) for x in results['test_loss']]

        # check for nans in train dataset predictions
        predictions = self(self.dataset['true_test_input'])
        if predictions.isnan().any():
            raise ValueError(f'nan detected {error_place} iteration {self.Wind_fit_iter}')
        self.train_predictions = predictions.detach().numpy()

        # update all metrics and history of the model
        self.Wind_train_loss[0] = last_train_loss[-1]
        self.Wind_train_loss[1].append(last_train_loss)
        self.Wind_test_loss[0] = last_test_loss[-1]
        self.Wind_test_loss[1].append(last_test_loss)

        self.Wind_fit_iter[0] += 1
        self.Wind_fit_iter[1].append([self.Wind_fit_iter[0] for x in last_train_loss])

        self.Winde_prune_iter[1].append([self.Winde_prune_iter[0] for x in last_train_loss])
        self.Wind_grid[1].append([self.Wind_grid[0] for x in last_train_loss])

        self.train_time[0] += train_time
        self.train_time[1].append([train_time for x in last_train_loss])

        y_test = (self.dataset['true_test_label'])
        
        self.mse[0] = mean_squared_error(y_test, self.train_predictions, squared=False)
        self.mse[1].append([self.mse[0] for x in last_train_loss])

        self.mae[0] = mean_absolute_error(y_test, self.train_predictions)
        self.mae[1].append([self.mae[0] for x in last_train_loss])

        self.r2[0] = r2_score(y_test, self.train_predictions)
        self.r2[1].append([self.r2[0] for x in last_train_loss])
        return results

    def prune_model(self):
        model = self.prune()
        results = self.fit_model()
        self.Winde_prune_iter[0] +=1
        return results
    
    def refine_model(self, new_grid):
        model = self.refine(new_grid)
        self.Wind_grid[0] = new_grid
        results = self.fit_model()
        return results
    
    def plot_model(self, **kwargs):
        if self.first_plot == False:
            self(self.dataset['train_input'])
        super().plot()
    
    # def print_history(self, iterations='all'):
    #     hist = self.stat_history
    #     if iterations == 'all':
    #         iterations = len(hist)
    #     elif iterations > len(hist):
    #         iterations = len(hist)
    #     for i in reversed(range(iterations)):
    #         current_m = hist[-i-1]
    #         print(type(current_m), '\n', current_m)
    #         print('fit iter:', current_m[-7], '  prune iter:', current_m[-6], '  grid:', current_m[-5],\
    #              '  train_time:', round(current_m[-4],2),\
    #              '  mse:', round(current_m[-3],3), '  mae:', round(current_m[-2],3), '  r2:', round(current_m[-1],3))


    ################################################################
    ######################### OPTIMIZATION #########################
    ################################################################

    def optimize_prune_refine(self, grid_increase=5,\
                        min_prune_improvement=0.01, min_fit_improvement=0.01, min_refine_improvement=0.01,\
                        max_prune_iters=10, max_refine_iters=10, max_fit_iters=10):

        # initiating losses
        fit_loss_new = 100000
        prune_loss_new = 100000
        refine_loss_new = 100000
        # initiating gammas
        fit_improvement = 1000000
        prune_improvement = 1000000
        refine_improvement = 1000000

        # iteration counters
        refine_iter = 0
        while refine_improvement >= min_refine_improvement and refine_iter <= max_refine_iters:
            refine_loss_old = refine_loss_new

            # prune loop
            prune_iter = 0
            while prune_improvement >= min_prune_improvement and prune_iter <= max_prune_iters:
                prune_loss_old = prune_loss_new

                # fitting loop
                fit_iter = 0
                while fit_improvement >= min_fit_improvement and fit_iter <= max_fit_iters:
                    fit_loss_old = fit_loss_new
                    print('fitting iteration: ', fit_iter)
                    results = self.fit_model()
                    fit_loss_new = results['train_loss'][-1]
                    fit_improvement = (fit_loss_old - fit_loss_new)/fit_loss_old
                    fit_iter += 1

                # prune after fitted 
                print('prune iteration: ', prune_iter)
                results = self.prune_model()
                prune_loss_new = results['train_loss'][-1]
                prune_improvement = (prune_loss_old - prune_loss_new)/prune_loss_old
                prune_iter += 1

            # refine (increase grid) after pruned and fitted
            print('refine iteration: ', refine_iter)
            new_grid += grid_increase
            results = self.refine_model(new_grid)
            refine_loss_new = results['train_loss'][-1]
            refine_improvement = (refine_loss_old - refine_loss_new)/refine_loss_old
            refine_iter += 1
        return

    ################################################################
    ############################ FORMULA ###########################
    ################################################################

    def get_formula(self, ignore_forms=['0']):

        my_symbolics = SYMBOLIC_LIB.copy()
        for formula in ignore_forms:
            my_symbolics.pop(formula, None)
        self.auto_symbolic(lib=my_symbolics)
        print(ex_round(self.symbolic_formula()[0][0],2))
