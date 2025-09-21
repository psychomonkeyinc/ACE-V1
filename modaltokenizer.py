# c:\ace3\modaltokenizer.py

import torch
import torch.nn as nn
import numpy as np
import cv2

class VectorQuantizer(nn.Module):
    """
    This is the core of the tokenizer. It holds the "vocabulary" of sensory experiences.
    It takes a continuous vector and finds the closest "word" in its codebook.
    """
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        
        # The codebook, or "vocabulary of experience"
        # MODIFIED: Added .half() to ensure the embedding weights are float16, matching the input.
        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim).half()
        
        # Initialize the codebook weights
        self.embedding.weight.data.uniform_(-1/self.num_embeddings, 1/self.num_embeddings)

    def forward(self, inputs):
        # Flatten input tensor
        flat_input = inputs.view(-1, self.embedding_dim)
        
        # Calculate distances to find the closest codebook vector
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self.embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self.embedding.weight.t()))
            
        # Get the index of the closest "word" in the vocabulary
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        
        # The output is the token ID (the index)
        return encoding_indices

class ModalTokenizer:
    """
    A true tokenizer that converts raw sensory data into sequences of discrete affective tokens.
    """
    def __init__(self, device, unified_dim=1024, vocab_size=512):
        self.device = device
        self.unified_dim = unified_dim # This is now the feature dimension before quantization
        
        # We create separate "vocabularies" for audio and video experiences
        self.audio_quantizer = VectorQuantizer(num_embeddings=vocab_size, 
                                               embedding_dim=self.unified_dim, 
                                               commitment_cost=0.25).to(device)
                                               
        self.video_quantizer = VectorQuantizer(num_embeddings=vocab_size, 
                                               embedding_dim=self.unified_dim, 
                                               commitment_cost=0.25).to(device)

        # Simple feature extractors to prepare data for quantization
        # This in_features needs to match the chunk size from audioin.py * sample_rate
        # 0.2s chunk * 44100 Hz = 8820 samples
        self.audio_feature_extractor = nn.Linear(8820, self.unified_dim).to(device).half()
        self.video_feature_extractor = nn.Linear(64*64*3, self.unified_dim).to(device).half()


    def process_audio(self, audio_chunk: np.ndarray) -> torch.Tensor:
        """
        Processes an audio chunk and returns a single affective audio token.
        """
        if audio_chunk is None:
            return torch.tensor([0], device=self.device, dtype=torch.long)
        
        flat_audio = torch.from_numpy(audio_chunk.flatten()).float().unsqueeze(0).to(self.device)

        # Pad or truncate to the required input size for the linear layer
        if flat_audio.shape[1] < self.audio_feature_extractor.in_features:
            padding = torch.zeros(1, self.audio_feature_extractor.in_features - flat_audio.shape[1], device=self.device)
            flat_audio = torch.cat([flat_audio, padding], dim=1)
        elif flat_audio.shape[1] > self.audio_feature_extractor.in_features:
            flat_audio = flat_audio[:, :self.audio_feature_extractor.in_features]

        with torch.no_grad():
            features = self.audio_feature_extractor(flat_audio.half())
            token_indices = self.audio_quantizer(features)
            
        return token_indices.flatten()


    def process_video(self, video_frame: np.ndarray) -> torch.Tensor:
        """
        Processes a video frame and returns a single affective video token.
        """
        if video_frame is None:
            return torch.tensor([0], device=self.device, dtype=torch.long)

        frame = cv2.resize(video_frame, (64, 64))
        flat_frame = torch.from_numpy(frame.flatten()).float().unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            features = self.video_feature_extractor(flat_frame.half())
            token_indices = self.video_quantizer(features)
            
        return token_indices.flatten()