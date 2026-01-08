from collections import namedtuple
import torch
from torch.utils.data import DataLoader
from models import MultimodalSentimentModel, MultimodalTrainer


def test_logging():
    Batch= namedtuple('batch', ['text_inputs','video_frames', 'audio_features',
                          'emotion_label', 'sentiment_label'])
    mock_batch = {
        'text_inputs': {
            'input_ids': torch.ones(1, 128),
            'attention_mask': torch.ones(1, 128)
        },
        'video_frames': torch.ones(1, 3, 30, 224, 224),
        'audio_features': torch.ones(1, 1, 64, 300),
        'emotion_label': torch.tensor([0]),
        'sentiment_label': torch.tensor([0])
    }

    mock_loader = DataLoader([mock_batch], batch_size=1, collate_fn=lambda x: x[0])

    model = MultimodalSentimentModel()
    trainer = MultimodalTrainer(model, mock_loader, mock_loader)
    
    training_losses = { 'total': 2.5, 'emotion': 1.0, 'sentiment': 1.5 }
    
    trainer.log_metrics(training_losses, phase="train")
    
    val_losses = { 'total': 1.5, 'emotion': 0.5, 'sentiment': 1.0 }
    
    val_metrics = {
        'emotion_precision': 0.65,
        'emotion_accuracy': 0.75,
        'sentiment_precision': 0.85,
        'sentiment_accuracy': 0.95
    }
    
    trainer.log_metrics(val_losses, val_metrics, phase="val")
    
    
if __name__ == "__main__":
    test_logging()