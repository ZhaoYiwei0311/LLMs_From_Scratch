import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gptModel import ExampleDeepNeuralNetwork, print_gradients


layer_sizes = [3, 3, 3, 3, 3, 1] 
sample_input = torch.tensor([[1., 0., -1.]]) 
torch.manual_seed(123) 
model_without_shortcut = ExampleDeepNeuralNetwork( 
    layer_sizes, use_shortcut=False 
)
print_gradients(model_without_shortcut, sample_input)

torch.manual_seed(123) 
model_with_shortcut = ExampleDeepNeuralNetwork( 
    layer_sizes, use_shortcut=True 
) 
print_gradients(model_with_shortcut, sample_input)