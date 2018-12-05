import argparse


import os
import torch
import tqdm


def _get_predicts(predicts, coefficients):
    return torch.einsum("ij,j->ij", (predicts, coefficients))


def _get_labels_distribution(predicts, coefficients):
    predicts = _get_predicts(predicts, coefficients)
    labels = predicts.argmax(dim=-1)
    counter = torch.bincount(labels, minlength=predicts.shape[1])
    return counter


def _compute_score_with_coefficients(predicts, coefficients):
    counter = _get_labels_distribution(predicts, coefficients).float()
    counter = counter * 100 / len(predicts)
    max_scores = torch.ones(len(coefficients)).cuda().float() * 100 / len(coefficients)
    result, _ = torch.min(torch.cat([counter.unsqueeze(0), max_scores.unsqueeze(0)], dim=0), dim=0)

    return float(result.sum().cpu())


def _find_best_coefficients(predicts, coefficients, alpha=0.001, iterations=100):
    best_coefficients = coefficients.clone()
    best_score = _compute_score_with_coefficients(predicts, coefficients)

    for _ in tqdm.trange(iterations):
        counter = _get_labels_distribution(predicts, coefficients)
        label = int(torch.argmax(counter).cpu())
        coefficients[label] -= alpha
        score = _compute_score_with_coefficients(predicts, coefficients)
        if score > best_score:
            best_score = score
            best_coefficients = coefficients.clone()

    return best_coefficients


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path")
    parser.add_argument("output_path")
    parser.add_argument("--start_alpha", type=float, default=0.01)
    parser.add_argument("--min_alpha", type=float, default=0.0001)

    args = parser.parse_args()

    with open(args.input_path, "rb") as fin:
        y = torch.load(fin).cuda()

    alpha = args.start_alpha

    coefs = torch.ones(y.shape[1]).cuda().float()
    last_score = _compute_score_with_coefficients(y, coefs)
    print("Start score", last_score)

    while alpha >= args.min_alpha:
        coefs = _find_best_coefficients(y, coefs, iterations=3000, alpha=alpha)
        new_score = _compute_score_with_coefficients(y, coefs)

        if new_score <= last_score:
            alpha *= 0.5

        last_score = new_score
        print("Score: {}, alpha: {}".format(last_score, alpha))

    predicts = _get_predicts(y, coefs)

    with open(args.output_path, "wb") as fout:
        torch.save(predicts.cpu(), fout)


if __name__ == "__main__":
    main()

