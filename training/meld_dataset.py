from torch.utils.data import Dataset, DataLoader, default_collate
import pandas as pd
from transformers import AutoTokenizer
import os
import cv2
import numpy as np
import torch
import subprocess
import torchaudio

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
class MELDDataset(Dataset):
    def __init__(self, csv_path, video_dir):
        self.data = pd.read_csv(csv_path)
        self.video_dir = video_dir
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.emotion_map={
            'anger':0,
            'disgust':1,
            'fear':2,
            'joy':3,
            'neutral':4,
            'sadness':5,
            'surprise':6
        }
        self.sentiment_map = {
            'negative':0,
            'neutral':1,
            'positive':2
        }
    def _load_video_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames =[]
    
        
        try:
            if not cap.isOpened():
                raise ValueError(f"Video not found: {video_path}")
            ret, frame= cap.read()
            if not ret or frame is None:
                raise ValueError(f"Video not found: {video_path}")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            while len(frames) < 30 and cap.isOpened():
                ret, frame =cap.read()
                if not ret:
                    break
                frame = cv2.resize(frame, (224, 224))
                frame = frame/ 255.0
                frames.append(frame)

        except Exception as e:
            raise ValueError(f"Error loading video frames from {str(e)}")
        finally:
            cap.release()
        if (len(frames)==0):
            raise ValueError(f"No frames extracted from video")
        #pad or trancate frames to 30
        if len(frames) <30:
            frames+=[np.zeros_like(frames[0])] * (30 -len(frames))
        else:
            frames = frames[:30]
        # Before permute: [Frames,heaight,width,channels]
        #After permute: [Frames,channels,height,width]
        return torch.FloatTensor(np.array(frames)).permute(0,3,1,2)
            
    def _extract_audio_features(self, video_path):
        try:
            cmd = [
                'ffmpeg',
                '-i', video_path,
                '-vn',
                '-ac', '1',
                '-ar', '16000',
                '-f', 'f32le',
                'pipe:1'
            ]

            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                check=True
            )

            audio = np.frombuffer(result.stdout, dtype=np.float32)
            waveform = torch.from_numpy(audio).unsqueeze(0)  # [1, T]

            mel_spectrogram = torchaudio.transforms.MelSpectrogram(
                sample_rate=16000,
                n_mels=64,
                n_fft=1024,
                hop_length=512
            )

            mel_spec = mel_spectrogram(waveform)

            mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-6)

            if mel_spec.size(2) < 300:
                mel_spec = torch.nn.functional.pad(
                    mel_spec, (0, 300 - mel_spec.size(2))
                )
            else:
                mel_spec = mel_spec[:, :, :300]

            return mel_spec

        except Exception as e:
            raise ValueError(f"Audio extraction failed: {str(e)}")

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if isinstance(idx, torch.Tensor):
            idx = idx.item()
        row = self.data.iloc[idx]
        
        try:
            video_filename = f"""dia{row['Dialogue_ID']}_utt{
                row['Utterance_ID']}.mp4"""
            path = os.path.join(self.video_dir, video_filename)
            video_path_exists = os.path.exists(path)

            if video_path_exists == False:
                raise FileNotFoundError(f"Video file not found: {path}")
            text_inputs = self.tokenizer(row['Utterance'],
                                        padding='max_length',
                                        truncation=True,
                                        max_length=128,
                                        return_tensors='pt'
                                        )
            video_frames = self._load_video_frames(path)
            audio_features = self._extract_audio_features(path)
            #map sentiment and emotion to labels
            emotion_label = self.emotion_map.get(row['Emotion'].lower())
            sentiment_label = self.sentiment_map.get(row['Sentiment'].lower())
            
            return {
                'text_inputs':{
                    'input_ids': text_inputs['input_ids'].squeeze(),
                    'attention_mask': text_inputs['attention_mask'].squeeze()
                },
                'video_frames': video_frames,
                'audio_features': audio_features,
                'emotion_label': torch.tensor(emotion_label),
                'sentiment_label': torch.tensor(sentiment_label)
            }
        except Exception as e:
            print(f"Error processing index {path}: {str(e)}")
            return None
        
def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    return default_collate(batch)

def prepare_dataloaders(train_csv, train_video_dir, 
                        dev_csv, dev_video_dir, 
                        test_csv, test_video_dir, batch_size=32):
    train_dataset = MELDDataset(train_csv, train_video_dir)
    dev_dataset = MELDDataset(dev_csv, dev_video_dir)
    test_dataset = MELDDataset(test_csv, test_video_dir)
    
    train_loader = DataLoader(train_dataset,
                            batch_size=batch_size,
                            shuffle=True
                            ,collate_fn=collate_fn)
    
    dev_loader = DataLoader(dev_dataset,
                            batch_size=batch_size,
                            collate_fn=collate_fn)
    
    test_loader = DataLoader(test_dataset,
                            batch_size=batch_size,
                            collate_fn=collate_fn)
    return train_loader, dev_loader, test_loader

if __name__ == "__main__":
    train_loader, dev_loader, test_loader = prepare_dataloaders(
        '../dataset/train/train_sent_emo.csv',
        '../dataset/train/train_splits',
        '../dataset/dev/dev_sent_emo.csv',
        '../dataset/dev/dev_splits_complete',
        '../dataset/test/test_sent_emo.csv',
        '../dataset/test/output_repeated_splits_test',
    )

    for batch in train_loader:
        if batch is None:
            continue
        print(batch['text_inputs'])
        print(batch['video_frames'].shape)
        print(batch['audio_features'].shape)
        print(batch['emotion_label'])
        print(batch['sentiment_label'])
        break
