import numpy as np
import torch
import torch.nn as nn
from vispy import app, scene
import threading
import time

class FieldLatentBridge:
    def __init__(self):
        # Field space representation
        self.field_dim = 64
        self.latent_dim = 32
        
        # Create encoder/decoder networks
        self.encoder = nn.Sequential(
            nn.Linear(self.field_dim, 128),
            nn.ReLU(),
            nn.Linear(128, self.latent_dim * 2)  # Mean and variance
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, self.field_dim)
        )
        
        # Visualization setup
        self.canvas = scene.SceneCanvas(keys='interactive', size=(1200, 800))
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = 'turntable'
        
        # Create initial scatter plot
        pos = np.zeros((1000, 3))
        color = np.ones((1000, 4))
        self.scatter = scene.visuals.Markers()
        self.scatter.set_data(pos, edge_color=None, face_color=color, size=5)
        self.view.add(self.scatter)
        
        self.theta = 0
        
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def field_to_latent(self, field_state):
        # Convert field state to latent representation
        encoded = self.encoder(field_state)
        mu, log_var = encoded.chunk(2, dim=-1)
        latent = self.reparameterize(mu, log_var)
        return latent
        
    def latent_to_field(self, latent_state):
        # Convert latent state back to field representation
        decoded = self.decoder(latent_state)
        return decoded
        
    def generate_transition_points(self, field, latent, reconstructed):
        # Create flowing patterns between states
        t = np.linspace(0, 2*np.pi, 1000)
        field_np = field.detach().numpy()
        latent_np = latent.detach().numpy()
        
        # Reshape for broadcasting
        t = t.reshape(-1, 1)
        field_np = np.mean(field_np) * np.ones_like(t)
        latent_np = np.mean(latent_np) * np.ones_like(t)
        
        # Field space (wave-like)
        field_points = np.column_stack([
            np.sin(t + self.theta) * field_np,
            np.cos(t + self.theta) * field_np,
            np.sin(2*t + self.theta) * field_np
        ])
        
        # Latent space (geometric)
        latent_points = np.column_stack([
            latent_np * np.cos(t - self.theta),
            latent_np * np.sin(t - self.theta),
            latent_np * np.sin(2*t - self.theta)
        ])
        
        # Combine with smooth transition
        points = np.vstack([field_points, latent_points])
        
        # Add some dynamic movement
        points += 0.1 * np.sin(self.theta) * np.random.randn(*points.shape)
        
        return points
        
    def generate_colors(self, points):
        # Generate colors based on position and time
        colors = np.zeros((len(points), 4))
        
        # Normalize positions for coloring
        pos_norm = (points - points.min()) / (points.max() - points.min())
        
        # Create shifting colors
        colors[:, 0] = 0.5 + 0.5 * np.sin(pos_norm[:, 0] * np.pi + self.theta)
        colors[:, 1] = 0.5 + 0.5 * np.sin(pos_norm[:, 1] * np.pi - self.theta)
        colors[:, 2] = 0.5 + 0.5 * np.sin(pos_norm[:, 2] * np.pi + self.theta * 0.5)
        colors[:, 3] = 0.8  # Alpha channel
        
        return colors
        
    def update(self, event):
        # Generate new field state
        field_state = torch.randn(self.field_dim)
        
        # Transform through latent space
        latent = self.field_to_latent(field_state)
        reconstructed = self.latent_to_field(latent)
        
        # Generate visualization points
        points = self.generate_transition_points(field_state, latent, reconstructed)
        colors = self.generate_colors(points)
        
        # Update scatter plot
        self.scatter.set_data(points, edge_color=None, face_color=colors, size=5)
        
        # Update rotation
        self.theta += 0.02

    def run(self):
        timer = app.Timer(interval=1.0/60, connect=self.update, start=True)
        self.canvas.show()
        app.run()

if __name__ == "__main__":
    bridge = FieldLatentBridge()
    bridge.run()