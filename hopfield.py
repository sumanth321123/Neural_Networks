from PIL import Image
import numpy as np
class HopfieldNetwork:
    def __init__(self, pattern_size):
        self.pattern_size = pattern_size
        self.weights = np.zeros((pattern_size, pattern_size))
    
    def train(self, patterns):
        for i in range(self.pattern_size):
            for j in range(self.pattern_size):
                if i != j:
                    for pattern in patterns:
                        self.weights[i, j] += pattern[i] * pattern[j]
        np.fill_diagonal(self.weights, 0)
    
    def recall(self, noisy_pattern, max_iterations=10):
        for _ in range(max_iterations):
            activations = np.dot(noisy_pattern, self.weights)
            noisy_pattern = np.sign(activations)
        return noisy_pattern

image_paths = ["3.png", "8.png"]
clean_images = [Image.open(path).convert("1") for path in image_paths]
patterns = [np.array(image).flatten() * 2 - 1 for image in clean_images]
pattern_size = len(patterns[0])
hopfield_net = HopfieldNetwork(pattern_size)
hopfield_net.train(patterns)

noisy_image = Image.open("3n.png").convert("1")
noisy_pattern = np.array(noisy_image).flatten() * 2 - 1
noisy_pattern = noisy_pattern.reshape(1, -1)
denoised_pattern = hopfield_net.recall(noisy_pattern)

denoised_image = Image.fromarray(
    ((denoised_pattern.reshape(clean_images[0].size[1], clean_images[0].size[0]) + 1) * 0.5 * 255).astype(np.uint8)
)

noisy_image.show()
denoised_image.show()
