import torch
import os

class Harness:
    def __init__(self, model, optimizer, loss_fn, scheduler = None, device = None):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        print(f'Initialized Harness with device: {self.device}')
        
    def _train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        for _, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)

            loss = self.loss_fn(output, target)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(train_loader)
    
    def validate(self, val_loader):
        self.model.eval()
        val_loss = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                val_loss += self.loss_fn(output, target).item()
        return val_loss / len(val_loader)

    def train(self, train_loader, val_loader, epochs, early_stopping=None):
        best_val_loss = float('inf')
        best_model = None
        epochs_no_improve = 0
        for epoch in range(epochs):
            train_loss = self._train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            epochs_no_improve += 1
            additional_message = ''
            if self.scheduler:
                self.scheduler.step(val_loss)
            if val_loss < best_val_loss:
                epochs_no_improve = 0
                best_val_loss = val_loss
                best_model = self.model.state_dict().copy()
                additional_message = '<--- New Best'
            if early_stopping and epochs_no_improve >= early_stopping:
                print(f'Early stopping at epoch {epoch}')
                break
            print(f'Epoch: {epoch} Train Loss: {train_loss:.4f} Val Loss: {val_loss:.4f} {additional_message}')
        self.model.load_state_dict(best_model)

    def predict(self, input):
        self.model.eval()
        with torch.no_grad():
            input = input.to(self.device)
            return self.model(input).cpu().numpy()
        
    def load_model(self, path):
        if not os.path.exists(path):
            print(f"Cannot load model as {path} does not exist.")
            return
        self.model.load_state_dict(torch.load(path, weights_only=True))
        
    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
    