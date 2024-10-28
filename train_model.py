#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import argparse
import time
import os
import smdebug.pytorch as smd
from smdebug import modes
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class DebuggerHookModel(nn.Module):
    """Wrapper class for model with SMDebugger hook integration"""
    def __init__(self, model, hook):
        super(DebuggerHookModel, self).__init__()
        self.model = model
        self.hook = hook
        
    def forward(self, x):
        outputs = self.model(x)
        # Save forward pass outputs
        if self.hook is not None:
            self.hook.save_tensor("outputs", outputs)
        return outputs

def test(model, test_loader, criterion, num_classes, hook):
    model.eval()
    if hook:
        hook.set_mode(modes.EVAL)
    
    running_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            output = model(data)
            loss = criterion(output, target)
            
            # Calculate running statistics
            running_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            # Calculate current metrics
            current_loss = running_loss / (batch_idx + 1)
            current_accuracy = 100. * correct / total
            
            # Save validation metrics
            if hook:
                hook.save_tensor("validation_loss", torch.tensor(current_loss))
                hook.save_tensor("validation_accuracy", torch.tensor(current_accuracy))
    
    final_loss = running_loss / len(test_loader)
    final_accuracy = 100. * correct / total
    print(f'\nTest set: Average loss: {final_loss:.4f}, Accuracy: {correct}/{total} ({final_accuracy:.2f}%)\n')
    return final_loss, final_accuracy

def train(model, train_loader, criterion, optimizer, epochs=3, num_classes=10, hook=None):
    if hook:
        hook.set_mode(modes.TRAIN)
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            # Calculate running statistics
            running_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            # Calculate current metrics
            current_loss = running_loss / (batch_idx + 1)
            current_accuracy = 100. * correct / total
            
            # Save tensors with the hook
            if hook:
                # Save metrics as tensors
                hook.save_tensor("train_loss", torch.tensor(current_loss))
                hook.save_tensor("train_accuracy", torch.tensor(current_accuracy))
                
                # Save weights and gradients every 100 steps
                if batch_idx % 100 == 0:
                    for name, param in model.named_parameters():
                        if param.requires_grad:
                            if param.grad is not None:
                                hook.save_tensor(f"gradients/{name}", param.grad)
                            hook.save_tensor(f"weights/{name}", param.data)
            
            if batch_idx % 100 == 0:
                print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                      f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {current_loss:.6f}')
    
    return model

def net(num_classes):
    model = models.resnet18(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    return model

def create_data_loaders(data_dir, batch_size):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = torchvision.datasets.ImageFolder(root=data_dir, transform=transform)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True), len(dataset.classes)

def main(args):
    # Create the hook
    hook = smd.Hook.create_from_json_file()
    
    train_data_path = os.path.join(args.data, 'train')
    test_data_path = os.path.join(args.data, 'test')
    
    train_loader, num_classes = create_data_loaders(train_data_path, args.batch_size)
    test_loader, _ = create_data_loaders(test_data_path, args.batch_size)
    
    print(f"Number of classes detected: {num_classes}")
    
    # Create model and wrap it with debugger hook
    base_model = net(num_classes)
    model = DebuggerHookModel(base_model, hook) if hook else base_model
    
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Register collections
    if hook:
        hook.register_module(model)
        hook.register_loss(loss_criterion)
        
        # Save hyperparameters
        hook.save_tensor("hyperparameters/learning_rate", torch.tensor(args.learning_rate))
        hook.save_tensor("hyperparameters/batch_size", torch.tensor(args.batch_size))
        
    start_time = time.time()
    model = train(model, train_loader, loss_criterion, optimizer, epochs=3, 
                 num_classes=num_classes, hook=hook)
    end_time = time.time()
    print(f'Training completed in {end_time - start_time:.2f} seconds')
    
    test_loss, test_accuracy = test(model, test_loader, loss_criterion, num_classes, hook)
    
    # Save final metrics
    if hook:
        hook.save_tensor("final_test_loss", torch.tensor(test_loss))
        hook.save_tensor("final_test_accuracy", torch.tensor(test_accuracy))
    
    # Save the model
    if isinstance(model, DebuggerHookModel):
        torch.save(model.model.state_dict(), os.path.join(args.model_dir, 'model.pth'))
    else:
        torch.save(model.state_dict(), os.path.join(args.model_dir, 'model.pth'))
    print(f"Model saved at {os.path.join(args.model_dir, 'model.pth')}")

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--model_dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--output_dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    args = parser.parse_args()

    main(args)
