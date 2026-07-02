import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from dataset import inference_transform as transform, remove_background
from model import OpenWorldEncoder


def discover_watchlist_classes(reference_dir: str | Path) -> list[str]:
    root = Path(reference_dir)
    if not root.exists():
        raise FileNotFoundError(f"Reference directory not found: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Reference path is not a directory: {root}")

    return sorted(
        child.name
        for child in root.iterdir()
        if child.is_dir() and not child.name.startswith(".")
    )


class BatchInferenceModel:
    """Model wrapper for efficient batch inference with ensemble of multiple reference sets"""
    def __init__(self, model_path, reference_dir, image_size=224, batch_size=32, num_workers=4, 
                 similarity_threshold=0.9, max_reference_per_class=None, median_threshold=0.1, 
                 variance_threshold=0.01, num_ensembles=5, l2_distance_threshold=2.0):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.median_threshold = median_threshold
        self.variance_threshold = variance_threshold
        self.l2_distance_threshold = l2_distance_threshold  # Add L2 distance threshold
        self.num_ensembles = num_ensembles  # Number of different reference sets to use
        
        # Load model
        self.model = OpenWorldEncoder(
            model_path=model_path,
            image_size=image_size
        )
        
        # Setup parameters
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.similarity_threshold = similarity_threshold
        self.max_reference_per_class = max_reference_per_class
        
        # Extract multiple sets of reference features
        self.reference_features_ensemble = []
        self.reference_features_ensemble_norm = []  # Store normalized features
        for i in range(self.num_ensembles):
            print(f"Extracting reference features for ensemble {i+1}/{self.num_ensembles}")
            reference_features, reference_features_norm = self._extract_reference_features(reference_dir)
            self.reference_features_ensemble.append(reference_features)
            self.reference_features_ensemble_norm.append(reference_features_norm)

        self.classes = sorted(list(self.reference_features_ensemble[0].keys()))
        self.watchlist_classes = discover_watchlist_classes(reference_dir)
        self.watchlist_set = set(self.watchlist_classes)

    @staticmethod
    def _build_confidence_bin(similarity):
        clipped_similarity = min(max(float(similarity), 0.0), 0.999999)
        bin_idx = min(int(clipped_similarity * 10), 9)
        return f"{bin_idx * 0.1:.1f}-{(bin_idx + 1) * 0.1:.1f}"

    @staticmethod
    def _select_best_class(class_stats):
        best_class = None
        for class_name, stats in class_stats.items():
            if best_class is None:
                best_class = class_name
                continue

            best_stats = class_stats[best_class]
            if stats['l2_min'] < best_stats['l2_min']:
                best_class = class_name
            elif stats['l2_min'] == best_stats['l2_min'] and stats['max'] > best_stats['max']:
                best_class = class_name

        return best_class

    def _extract_reference_features(self, reference_dir):
        """Extract features from reference images for each class with random sampling"""
        print(f"Extracting reference features from {reference_dir}")
        reference_features = {}
        reference_features_norm = {}  # Store normalized features
        
        # Ensure model is in evaluation mode
        self.model.eval()
        
        # Get all class directories
        for class_name in os.listdir(reference_dir):
            class_dir = os.path.join(reference_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            
            # Get all images in this class directory
            images = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
            if not images:
                print(f"No images found for class {class_name}")
                continue
            
            # Random sampling for this ensemble
            if self.max_reference_per_class is not None:
                import random
                random.shuffle(images)
                images = images[:self.max_reference_per_class]
                
            # Extract features for all images
            class_features = []
            class_features_norm = []  # Store normalized features
            for img_file in images:
                image_path = os.path.join(class_dir, img_file)
                try:
                    img = Image.open(image_path)
                    img = self._remove_background(img)
                    img_tensor = transform(img).unsqueeze(0).to(self.device)
                    
                    with torch.no_grad():
                        # Extract features using online network
                        online_features = self.model.online_backbone(img_tensor)
                        online_features = self.model.online_projector(online_features)
                        class_features.append(online_features.clone())

                        # Extract features using target network for normalization
                        target_features = self.model.target_backbone(img_tensor)
                        target_features = self.model.target_projector(target_features)
                        features_norm = F.normalize(target_features, dim=1, p=2)
                        class_features_norm.append(features_norm)
                except Exception as e:
                    print(f"Error extracting features from {image_path}: {e}")
            
            if class_features:
                class_features = torch.cat(class_features, dim=0)
                class_features_norm = torch.cat(class_features_norm, dim=0)
                reference_features[class_name] = class_features
                reference_features_norm[class_name] = class_features_norm
                print(f"Extracted features for class {class_name} from {len(class_features)} images")
        
        return reference_features, reference_features_norm

    def _remove_background(self, img):
        return remove_background(img)

    def process_batch(self, batch):
        """Process a batch of images with simplified prediction logic based on similarity and class stats"""
        # Ensure model is in evaluation mode
        self.model.eval()
        
        # Extract images from batch
        images = batch['image'].to(self.device)
        paths = batch['path']
        true_classes = batch['true_class']
        
        ensemble_results = []
        
        # Get predictions from each ensemble member
        for i in range(self.num_ensembles):
            reference_features_norm = self.reference_features_ensemble_norm[i]
            reference_features = self.reference_features_ensemble[i]
            
            batch_results = []
            
            # Extract features
            with torch.no_grad():
                features = self.model.online_backbone(images)
                
                # Compare with reference features
                for j in range(len(images)):
                    img_feature = features[j:j+1]
                    prediction = self._compare_with_references(img_feature, reference_features_norm, reference_features)
                    prediction['path'] = paths[j]
                    prediction['true_class'] = true_classes[j]
                    prediction['correct'] = prediction['predicted_class'] == true_classes[j]
                    batch_results.append(prediction)
            
            ensemble_results.append(batch_results)
        
        # Combine ensemble predictions with simplified unknown detection
        final_results = []
        for i in range(len(images)):
            # Collect predictions and metrics from all ensemble members
            all_class_similarities = defaultdict(list)
            all_class_l2_distances = defaultdict(list)  # Initialize for L2 distances
            full_similarities = defaultdict(list)  # Store full similarity distributions
            full_l2_distances = defaultdict(list)  # Store full L2 distance distributions
            all_l2_means = defaultdict(list)  # Store L2 means
            all_l2_medians = defaultdict(list)  # Store L2 medians
            all_l2_variances = defaultdict(list)  # Store L2 variances
            all_l2_mins = defaultdict(list)  # Store L2 minimums
            all_l2_maxs = defaultdict(list)  # Store L2 maximums
            all_l2_stds = defaultdict(list)  # Store L2 standard deviations
            
            # Collect all similarity scores and L2 distances for each class
            for member_results in ensemble_results:
                result = member_results[i]
                # Collect basic metrics
                for class_name, sim in result['all_similarities'].items():
                    all_class_similarities[class_name].append(sim)
                for class_name, l2_dist in result['all_l2_distances'].items():
                    all_class_l2_distances[class_name].append(l2_dist)
                # Collect full distributions
                for class_name, sim_list in result['full_similarities'].items():
                    full_similarities[class_name].extend(sim_list)
                for class_name, l2_list in result['full_l2_distances'].items():
                    full_l2_distances[class_name].extend(l2_list)
                # Collect L2 statistics
                for class_name in result['all_l2_means'].keys():
                    all_l2_means[class_name].append(result['all_l2_means'][class_name])
                    all_l2_medians[class_name].append(result['all_l2_medians'][class_name])
                    all_l2_variances[class_name].append(result['all_l2_variances'][class_name])
                    all_l2_mins[class_name].append(result['all_l2_mins'][class_name])
                    all_l2_maxs[class_name].append(result['all_l2_maxs'][class_name])
                    all_l2_stds[class_name].append(result['all_l2_stds'][class_name])
            
            # Calculate class-wise statistics
            class_stats = {}
            for class_name, similarities in all_class_similarities.items():
                l2_distances = all_class_l2_distances[class_name]  # Get L2 distances for this class
                stats = {
                    'mean': np.mean(similarities),
                    'max': np.max(similarities),
                    'min': np.min(similarities),
                    'median': np.median(similarities),
                    'std': np.std(similarities),
                    'variance': np.var(similarities),
                    'l2_mean': float(np.mean(all_l2_means[class_name])),  # Convert to scalar
                    'l2_max': float(np.max(all_l2_maxs[class_name])),
                    'l2_min': float(np.min(all_l2_mins[class_name])),
                    'l2_median': float(np.median(all_l2_medians[class_name])),
                    'l2_std': float(np.mean(all_l2_stds[class_name])),
                    'l2_variance': float(np.mean(all_l2_variances[class_name]))
                }
                class_stats[class_name] = stats
            
            # Select the best candidate by minimum L2 distance first, then
            # break ties with maximum cosine similarity.
            best_class = self._select_best_class(class_stats)
            best_similarity = class_stats[best_class]['max']
            best_l2_min = class_stats[best_class]['l2_min']  # min L2 distance of the best class
            
            # Count classes with high max similarity
            high_confidence_count = sum(1 for _, stats in class_stats.items() 
                                     if stats['max'] > self.similarity_threshold and 
                                     stats['l2_min'] < self.l2_distance_threshold)  # check using the minimum L2 distance
            
            # Make prediction
            if best_similarity < self.similarity_threshold or best_l2_min > self.l2_distance_threshold:
                final_pred = "unknown"
                # keep the best class's max similarity and its min L2 distance
                final_sim = best_similarity
                final_l2 = best_l2_min
            else:
                final_pred = best_class
                final_sim = best_similarity
                final_l2 = best_l2_min  # use the best class's min L2 distance (ensures same class)
            
            # Create final result with simplified metadata
            final_result = {
                'path': paths[i],
                'true_class': true_classes[i],
                'predicted_class': final_pred,
                'similarity': float(final_sim),  # Convert to scalar
                'l2_distance': float(final_l2),  # Convert to scalar
                'correct': final_pred.lower() == true_classes[i].lower(),
                'class_stats': class_stats,
                'all_similarities': {k: float(v[0]) if isinstance(v, list) else float(v) 
                                   for k, v in all_class_similarities.items()},  # Convert to scalar
                'all_l2_distances': {k: float(v[0]) if isinstance(v, list) else float(v) 
                                   for k, v in all_class_l2_distances.items()},  # Convert to scalar
                'full_similarities': {k: [float(x) for x in v] for k, v in full_similarities.items()},
                'full_l2_distances': {k: [float(x) for x in v] for k, v in full_l2_distances.items()},
                'all_l2_means': {k: float(np.mean(v)) for k, v in all_l2_means.items()},
                'all_l2_medians': {k: float(np.median(v)) for k, v in all_l2_medians.items()},
                'all_l2_variances': {k: float(np.mean(v)) for k, v in all_l2_variances.items()},
                'all_l2_mins': {k: float(np.min(v)) for k, v in all_l2_mins.items()},
                'all_l2_maxs': {k: float(np.max(v)) for k, v in all_l2_maxs.items()},
                'all_l2_stds': {k: float(np.mean(v)) for k, v in all_l2_stds.items()}
            }
            
            final_results.append(final_result)
        
        return final_results
    
    def _compare_with_references(self, feature, reference_features_norm, reference_features):
        """Compare features with reference class features using both cosine similarity and L2 distance"""
        # Ensure model is in evaluation mode
        self.model.eval()
        
        # Pass features through online projector and predictor
        with torch.no_grad():
            online_proj = self.model.online_projector(feature)
            online_proj_features = online_proj.clone()
            online_pred = self.model.online_predictor(online_proj)
            # Store raw features before normalization
            # Normalize the online prediction for cosine similarity
            online_pred_norm = F.normalize(online_pred, dim=1, p=2)
        
        # Compare with reference features using both metrics
        similarities, full_similarities = {}, {}
        l2_distances, full_l2_distances = {}, {}
        
        for class_name, class_reference_features_norm in reference_features_norm.items():
            # Get both normalized and raw reference features
            class_reference_features = reference_features[class_name]
            
            # Calculate cosine similarity
            similarity_list = []
            for i in range(class_reference_features_norm.size(0)):
                target_proj_norm = class_reference_features_norm[i:i+1]  # Already normalized
                cos_sim = (online_pred_norm * target_proj_norm).sum(dim=1)
                similarity_list.append(cos_sim.item())
            
            # Calculate L2 distances
            l2_distance_list = []
            for i in range(class_reference_features.size(0)):
                target_proj = class_reference_features[i:i+1]
                l2_dist = torch.norm(online_proj_features - target_proj, dim=1, p=2)
                l2_distance_list.append(l2_dist.item())
            
            # Store both metrics
            full_similarities[class_name] = similarity_list
            similarities[class_name] = max(similarity_list)
            full_l2_distances[class_name] = l2_distance_list
            l2_distances[class_name] = min(l2_distance_list)  # Use minimum distance
         
        # Compute means, medians, variances and quartiles from full similarities
        means = {class_name: np.mean(similarity_list) for class_name, similarity_list in full_similarities.items()}
        medians = {class_name: np.median(similarity_list) for class_name, similarity_list in full_similarities.items()}
        variances = {class_name: np.var(similarity_list) for class_name, similarity_list in full_similarities.items()}
        
        # Calculate quartiles and IQR for each class
        quartiles = {}
        iqrs = {}
        for class_name, similarity_list in full_similarities.items():
            q1, q2, q3 = np.percentile(similarity_list, [25, 50, 75])
            quartiles[class_name] = {'q1': q1, 'q2': q2, 'q3': q3}
            iqrs[class_name] = q3 - q1  # Interquartile Range
        
        # Calculate L2 distance statistics
        l2_means = {class_name: np.mean(dist_list) for class_name, dist_list in full_l2_distances.items()}
        l2_medians = {class_name: np.median(dist_list) for class_name, dist_list in full_l2_distances.items()}
        l2_variances = {class_name: np.var(dist_list) for class_name, dist_list in full_l2_distances.items()}
        l2_mins = {class_name: np.min(dist_list) for class_name, dist_list in full_l2_distances.items()}
        l2_maxs = {class_name: np.max(dist_list) for class_name, dist_list in full_l2_distances.items()}
        l2_stds = {class_name: np.std(dist_list) for class_name, dist_list in full_l2_distances.items()}
        
        # Filter candidates using both cosine similarity and L2 distance criteria
        candidates = {}
        for class_name, mean in means.items():
            median = medians[class_name]
            variance = variances[class_name]
            q1 = quartiles[class_name]['q1']
            q3 = quartiles[class_name]['q3']
            iqr = iqrs[class_name]
            l2_mean = l2_means[class_name]
            l2_median = l2_medians[class_name]
            
            # Enhanced criteria using both metrics:
            # 1. Median similarity is above threshold
            # 2. Q1 is reasonably high
            # 3. Small IQR
            # 4. Mean similarity is above threshold
            # 5. Variance is below threshold
            # 6. L2 distance is reasonably small
            if (median > self.median_threshold and 
                q1 > self.similarity_threshold * 0.95 and
                iqr < self.variance_threshold * 2 and
                mean > self.similarity_threshold and
                variance < self.variance_threshold and
                l2_mean < self.l2_distance_threshold):  # Add L2 distance threshold
                # Score calculation incorporating both metrics
                similarity_score = (mean * 0.3 +
                                median * 0.3 +
                                q1 * 0.2 +
                                (1 - iqr) * 0.2)
                l2_score = 1 - (l2_mean / self.l2_distance_threshold)  # Normalize L2 score
                # Combine both scores
                final_score = similarity_score * 0.7 + l2_score * 0.3  # Weight more on similarity
                candidates[class_name] = final_score
        
        if candidates:
            # Select the candidate with highest composite score
            predicted_class, _ = max(candidates.items(), key=lambda x: x[1])
        else:
            # Try relaxed criteria with more emphasis on Q3 and less on Q1
            relaxed_candidates = {}
            for class_name, mean in means.items():
                median = medians[class_name]
                variance = variances[class_name]
                q1 = quartiles[class_name]['q1']
                q3 = quartiles[class_name]['q3']
                iqr = iqrs[class_name]
                l2_mean = l2_means[class_name]
                
                # Relaxed criteria including L2 distance
                if (median > self.median_threshold * 1.02 and
                    q3 > self.similarity_threshold * 1.05 and
                    q1 > self.similarity_threshold * 0.85 and
                    iqr < self.variance_threshold * 3 and
                    mean > self.similarity_threshold * 1.02 and
                    l2_mean < self.l2_distance_threshold * 1.2):  # Relaxed L2 threshold
                    # Modified score calculation for relaxed criteria
                    similarity_score = (mean * 0.25 +
                                    median * 0.25 +
                                    q3 * 0.3 +
                                    q1 * 0.1 +
                                    (1 - iqr) * 0.1)
                    l2_score = 1 - (l2_mean / (self.l2_distance_threshold * 1.2))
                    final_score = similarity_score * 0.7 + l2_score * 0.3
                    relaxed_candidates[class_name] = final_score
            
            if relaxed_candidates:
                predicted_class, _ = max(relaxed_candidates.items(), key=lambda x: x[1])
            else:
                predicted_class = "unknown"  # No valid candidates even with relaxed criteria
        
        return {
            'predicted_class': predicted_class,
            'similarity': means[predicted_class] if predicted_class != "unknown" else 0,
            'l2_distance': l2_means[predicted_class] if predicted_class != "unknown" else float('inf'),
            'median': medians[predicted_class] if predicted_class != "unknown" else 0,
            'variance': variances[predicted_class] if predicted_class != "unknown" else 0,
            'all_similarities': similarities,
            'all_l2_distances': l2_distances,
            'full_similarities': full_similarities,
            'full_l2_distances': full_l2_distances,
            'all_means': means,
            'all_medians': medians,
            'all_variances': variances,
            'all_l2_means': l2_means,
            'all_l2_medians': l2_medians,
            'all_l2_variances': l2_variances,
            'all_l2_mins': l2_mins,
            'all_l2_maxs': l2_maxs,
            'all_l2_stds': l2_stds,
            'quartiles': quartiles,
            'iqrs': iqrs
        }
    
