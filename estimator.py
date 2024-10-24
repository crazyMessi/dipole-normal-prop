# import torch
# from options import get_parser
# import options
# import util
# from util import orient_center
# from inference_utils import load_model_from_file, fix_n_filter, voting_policy

# # estimator基类
# class Estimator:
#     def __init__(self):
#         pass
    
#     def estimate(self, input_pc: torch.Tensor):
#         pass
    
#     def save(self, output_path):
#         pass
    
#     def load(self, input_path):
#         pass

# class PointCNN_Estimator(Estimator):
#     def __init__(self,opt,init_estimator = None):
#         self.opt = opt
#         self.device = torch.device(torch.cuda.current_device() if torch.cuda.is_available() else torch.device('cpu'))
#         softmax = torch.nn.Softmax(dim=-1)
#         self.n_models = len(opt.models)
#         self.models = [load_model_from_file(opt.models[i], self.device) for i in range(self.n_models)]
#         self.estimate_iter = opt.iters
#         self.propagation_iters = opt.propagation_iters
        
    