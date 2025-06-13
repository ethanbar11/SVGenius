# Refer https://torchmetrics.readthedocs.io/en/stable/image/frechet_inception_distance.html
# from torchmetrics.image.fid import FrechetInceptionDistance
from PIL import Image
from .base_metric import BaseMetric
import torch
from torchvision import transforms
import clip
from torch.nn.functional import adaptive_avg_pool2d
from inception import InceptionV3
import numpy as np
from tqdm import tqdm
from scipy import linalg
import torchvision.transforms as TF

class FIDCalculator(BaseMetric): 
    def __init__(self, model_name = 'InceptionV3'):
        super().__init__()
        self.class_name = self.__class__.__name__
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_name = model_name
        if self.model_name == 'ViT-B/32':
            self.dims = 512
            model, preprocess = clip.load('ViT-B/32', device=self.device)
            
        elif self.model_name == 'InceptionV3':
            self.dims = 2048
            block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[self.dims]
            model = InceptionV3([block_idx]).to(self.device)
            preprocess = TF.Compose([
                TF.Resize((299, 299)),  # InceptionV3 standard input size
                TF.ToTensor(),
                TF.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        
        self.model = model.to(self.device)
        self.preprocess = preprocess

    def calculate_frechet_distance(self, mu1, sigma1, mu2, sigma2, eps=1e-6):
        """Numpy implementation of the Frechet Distance.
        The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
        and X_2 ~ N(mu_2, C_2) is
                d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

        Stable version by Dougal J. Sutherland.

        Params:
        -- mu1   : Numpy array containing the activations of a layer of the
                inception net (like returned by the function 'get_predictions')
                for generated samples.
        -- mu2   : The sample mean over activations, precalculated on an
                representative data set.
        -- sigma1: The covariance matrix over activations for generated samples.
        -- sigma2: The covariance matrix over activations, precalculated on an
                representative data set.

        Returns:
        --   : The Frechet Distance.
        """

        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)

        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        assert mu1.shape == mu2.shape, \
            'Training and test mean vectors have different lengths'
        assert sigma1.shape == sigma2.shape, \
            'Training and test covariances have different dimensions'

        diff = mu1 - mu2

        # Product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = ('fid calculation produces singular product; '
                'adding %s to diagonal of cov estimates') % eps
            print(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError('Imaginary component {}'.format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        return (diff.dot(diff) + np.trace(sigma1)
                + np.trace(sigma2) - 2 * tr_covmean)

    def get_activations(self, images):
        # Use enhanced image preprocessing
        processed_images = []
        for img in images:
            processed_img = self.process_image(img)
            if processed_img is None or not torch.isfinite(processed_img).all():
                print(f"Image processing failed or contains invalid values, skipping image")
                continue

            if processed_img is not None:
                processed_images.append(processed_img)
            
        if not processed_images:
            raise ValueError("All images are invalid after preprocessing")
            
        dataset = ImageDataset(processed_images, processor=None)  # Already preprocessed
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=50, shuffle=False, num_workers=4)
        
        pred_arr = np.empty((len(processed_images), self.dims))
        start_idx = 0
        
        for batch in tqdm(dataloader):
            batch = batch.to(self.device)
            
            with torch.no_grad():
                if self.model_name == 'ViT-B/32':
                    pred = self.model.encode_image(batch).cpu().numpy()
                elif self.model_name == 'InceptionV3':
                    pred = self.model(batch)[0]

                    # If model output is not scalar, apply global spatial average pooling.
                    # This happens if you choose a dimensionality not equal 2048.
                    if pred.size(2) != 1 or pred.size(3) != 1:
                        pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

                    pred = pred.squeeze(3).squeeze(2).cpu().numpy()
                    
                pred_arr[start_idx:start_idx + pred.shape[0]] = pred
                start_idx = start_idx + pred.shape[0]

        return pred_arr

    def calculate_activation_statistics(self, images):
        act = self.get_activations(images)
        mu = np.mean(act, axis=0)
        sigma = np.cov(act, rowvar=False)
        return mu, sigma

    def process_image(self, img):
        """Process various types of image inputs and return normalized tensor"""
        try:
            # Check image type
            if isinstance(img, torch.Tensor):
                # If already a tensor, check shape and value range
                if img.dim() == 4:  # Batch tensor
                    if img.shape[0] != 1:
                        raise ValueError(f"Expected batch size 1, got {img.shape[0]}")
                    img = img.squeeze(0)
                
                if img.dim() != 3:
                    raise ValueError(f"Expected tensor dimension 3 (C,H,W), got {img.dim()}")
                
                # Check channel order
                if img.shape[0] != 3 and img.shape[-1] == 3:
                    # Might be HWC format, convert to CHW
                    img = img.permute(2, 0, 1)
                
                # Check value range and normalize
                if img.max() > 1.0 + 1e-6:
                    img = img / 255.0
                
                # Apply model-specific preprocessing
                if self.model_name == 'InceptionV3':
                    # For InceptionV3, need to resize and normalize
                    img = TF.Resize((299, 299))(img)
                    img = TF.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)
                    
                return img
                
            elif isinstance(img, np.ndarray):
                # Numpy array
                if img.ndim == 4:  # Batch array
                    if img.shape[0] != 1:
                        raise ValueError(f"Expected batch size 1, got {img.shape[0]}")
                    img = img[0]
                
                # Check color channels
                if img.ndim == 3:
                    # Ensure RGB format
                    if img.shape[0] == 3:  # CHW format
                        pass
                    elif img.shape[-1] == 3:  # HWC format
                        img = np.transpose(img, (2, 0, 1))
                    else:
                        raise ValueError(f"Image should have 3 color channels, got shape {img.shape}")
                else:
                    # Convert grayscale to RGB
                    img = np.stack([img, img, img], axis=0)
                
                # Normalize
                if img.max() > 1.0 + 1e-6:
                    img = img.astype(np.float32) / 255.0
                
                # Convert to tensor and apply preprocessing
                img = torch.from_numpy(img).float()
                return self.process_image(img)  # Recursively process converted tensor
                
            elif isinstance(img, Image.Image):
                # PIL image
                # Ensure RGB mode
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Apply preprocessing
                return self.preprocess(img)
                
            elif isinstance(img, str):
                # Image path
                try:
                    return self.process_image(Image.open(img))
                except Exception as e:
                    print(f"Cannot open image file {img}: {e}")
                    return None
                    
            else:
                print(f"Unsupported image type: {type(img)}")
                return None
                
        except Exception as e:
            print(f"Error processing image: {e}")
            return None

    def calculate_score(self, batch, update=True):
        # Validate input
        if 'gt_im' not in batch or 'gen_im' not in batch:
            raise ValueError("Batch must contain 'gt_im' and 'gen_im' keys")
            
        if len(batch['gt_im']) == 0 or len(batch['gen_im']) == 0:
            raise ValueError("Reference or generated image list is empty")
            
        print(f"Calculating FID: Processing {len(batch['gt_im'])} reference images and {len(batch['gen_im'])} generated images")
            
        m1, s1 = self.calculate_activation_statistics(batch['gt_im'])
        m2, s2 = self.calculate_activation_statistics(batch['gen_im'])
        fid_value = float(self.calculate_frechet_distance(m1, s1, m2, s2))
        
        # For FID, we only have one overall value, not per image pair values
        if hasattr(self, 'meter'):
            if update:
                self.meter.update(fid_value, 1)
                return self.meter.avg, [fid_value]
            else:
                return fid_value, [fid_value]
        else:
            # If no meter attribute (might not properly inherit BaseMetric), return compatible format
            return fid_value, [fid_value]

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, images, processor=None):
        self.images = images
        self.processor = processor

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        img = self.images[i]
        if self.processor is not None:
            img = self.processor(img)
        return img

if __name__ == '__main__':
    calculator = FIDCalculator()
    # Example 1: Calculate score for individual images
    batch = {
        'gen_im': [Image.open(p) for p in ['example1.png']], 
        'gt_im': [Image.open(p) for p in ['example2.png']]
    }
    avg_score, all_scores = calculator.calculate_score(batch)
    print(f"Average FID score: {avg_score}")