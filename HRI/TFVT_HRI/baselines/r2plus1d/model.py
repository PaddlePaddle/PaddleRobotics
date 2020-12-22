import numpy as np
from time import time
import torch
import torch.nn as nn
import torch.cuda as cuda
from sklearn.metrics import accuracy_score
from utils_cv.action_recognition.model import VideoLearner
from utils_cv.common.gpu import num_devices, torch_device


class R2plus1D(VideoLearner):
    def __init__(self,
                 dataset,
                 base_model='ig65m'):
        num_classes = len(dataset.classes)
        super(R2plus1D, self).__init__(dataset, num_classes, base_model)

    def evaluate(self,
                 report_every=100,
                 train_or_test='test'):
        assert train_or_test in ["train", "test"]
        # set device and num_gpus
        num_gpus = num_devices()
        device = torch_device()
        torch.backends.cudnn.benchmark = True if cuda.is_available() else False

        # init model with gpu (or not)
        self.model.to(device)
        if num_gpus > 1:
            self.model = nn.DataParallel(self.model)
        self.model.eval()

        # set train or test
        ds = (
            self.dataset.test_ds
            if train_or_test == "test"
            else self.dataset.train_ds
        )

        ds.dataset.num_samples = 1

        # Loop over all examples in the test set and compute accuracies
        ret = dict(
            infer_times=[],
            clip_preds=[],
            clip_trues=[],
            pred_scores=[],
        )

        # inference
        with torch.no_grad():
            for i in range(
                1, len(ds)
            ):  # [::10]:  # Skip some examples to speed up accuracy computation
                if i % report_every == 0:
                    print(
                        f"Processsing {i} of {len(self.dataset.test_ds)} samples.."
                    )

                # Get model inputs
                inputs, label = ds[i]
                inputs = torch.unsqueeze(inputs, 0)
                inputs = inputs.to(device, non_blocking=True)

                # Run inference
                start_time = time()
                outputs = self.model(inputs)
                outputs = outputs.cpu().numpy()
                infer_time = time() - start_time
                ret["infer_times"].append(infer_time)

                # Store results
                ret["clip_preds"].extend(outputs.argmax(axis=1))
                ret["clip_trues"].extend([label])
                ret["pred_scores"].extend(
                    [get_softmax_scores(o) for o in list(outputs)])

        tp = fp = fn = 0
        nll = 0  # TODO: update criterion in training
        for clip_true, clip_pred, scores in zip(
                ret['clip_trues'], ret['clip_preds'], ret['pred_scores']):
            nll += -np.log(scores[clip_true])
            if clip_true > 0 and clip_pred > 0:
                tp += 1
            elif clip_true == 0 and clip_pred > 0:
                fp += 1
            elif clip_true > 0 and clip_pred == 0:
                fn += 1

        eps = 1e-6
        precision = (tp + eps) / (tp + fp + eps)
        recall = (tp + eps) / (tp + fn + eps)

        print(
            f"Avg. inference time per video ({len(ds)} clips) =",
            round(np.array(ret["infer_times"]).mean() * 1000, 2),
            "ms",
        )
        print(
            "Clip prediction accuracy =",
            round(accuracy_score(ret["clip_trues"], ret["clip_preds"]), 2),
        )
        print("Precision =", round(precision, 4))
        print("Recall =", round(recall, 4))
        print("Avg NLL =", round(nll / len(ret['clip_trues']), 4))
        return ret


def get_softmax_scores(output):
    logits = np.exp(output)
    return logits / np.sum(logits)
