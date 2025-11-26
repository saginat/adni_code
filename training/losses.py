import numpy as np
import torch
import torch.nn as nn


def choose_labels(labels_dict, chosen_labels):
    """Filters the labels dictionary to include only the chosen labels."""
    return {k: v for k, v in labels_dict.items() if k in chosen_labels}


def calculate_balanced_weights(imageID_to_labels, tasks_types, device):
    """Calculates class weights for handling imbalanced datasets."""
    task_class_counts = {
        task: {} for task, t_type in tasks_types.items() if t_type != "regression"
    }

    for labels in imageID_to_labels.values():
        for task in task_class_counts.keys():
            if task in labels and not np.isnan(labels[task]):
                label = int(labels[task])
                task_class_counts[task][label] = (
                    task_class_counts[task].get(label, 0) + 1
                )

    task_weights = {}
    for task, counts in task_class_counts.items():
        if not counts:
            continue

        if tasks_types[task] == "binary":
            neg = counts.get(0, 0)
            pos = counts.get(1, 0)
            # Handle case where a class is not present
            if pos == 0:
                weight = torch.tensor(1.0, device=device)
            else:
                weight = torch.tensor(neg / pos, device=device)
            task_weights[task] = weight
        elif tasks_types[task] == "categorical":
            num_classes = len(counts)
            total_samples = sum(counts.values())

            class_weights = []
            for i in sorted(counts.keys()):
                weight = total_samples / (num_classes * counts[i] + 1e-6)
                class_weights.append(weight)
            task_weights[task] = torch.tensor(class_weights, device=device)

    return task_weights


def get_loss_fns(config, class_weights):
    """Creates a dictionary of loss functions for each task."""
    loss_fns = {}
    tasks_with_loss = config.TASKS + ["Reconstruction"]
    for task in tasks_with_loss:
        task_type = config.TASKS_TYPES.get(task)
        if task_type == "binary":
            loss_fns[task] = nn.BCEWithLogitsLoss(
                reduction="none", pos_weight=class_weights.get(task)
            )
        elif task_type == "categorical":
            loss_fns[task] = nn.CrossEntropyLoss(
                reduction="none", weight=class_weights.get(task)
            )
        elif task_type == "regression":
            loss_fns[task] = nn.MSELoss(reduction="none")
    return loss_fns


def handle_null_and_dtypes(outputs, target, loss_fns):
    """Filters out NaN targets and ensures correct data types for loss calculation."""
    filtered_outputs, filtered_targets, valid_masks = {}, {}, {}
    for task_name in loss_fns.keys():
        if task_name not in outputs or task_name not in target:
            continue

        task_output = outputs[task_name]
        task_target = target[task_name]

        valid_mask = ~torch.isnan(task_target)
        if not valid_mask.any():
            continue

        filtered_outputs[task_name] = task_output[valid_mask]
        filtered_targets[task_name] = task_target[valid_mask]
        valid_masks[task_name] = valid_mask

        if isinstance(loss_fns[task_name], nn.BCEWithLogitsLoss):
            filtered_targets[task_name] = (
                filtered_targets[task_name].unsqueeze(1).float()
            )
        elif isinstance(loss_fns[task_name], nn.CrossEntropyLoss):
            filtered_targets[task_name] = filtered_targets[task_name].long()

    return filtered_outputs, filtered_targets, valid_masks


def calculate_task_losses(outputs, target, loss_fns):
    """Calculates the mean loss for each task."""
    task_losses = {}
    for task_name, task_output in outputs.items():
        if task_name in target:
            task_target = target[task_name]
            task_loss_fn = loss_fns[task_name]
            task_losses[task_name] = task_loss_fn(task_output, task_target).mean()
    return task_losses
